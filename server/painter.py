import warp as wp
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
wp.init()

# Genome layout: [c, bw, x0, y0, x1, y1, a1, d1, a2, d2]
# (x0,y0)→(x1,y1) set the origin and initial direction.
# a1/d1 encode the second knot: angle offset (±MAX_TURN_RAD from dir 0→1)
# and distance scale (×len01, range 0.3–2.0).  a2/d2 do the same for the third knot
# relative to dir 1→2.  Capping the angles prevents aggressive U-turns.
N_GENES      = 10
GENE_MAX     = 255
MAX_TURN_RAD = 1.0472   # 60° – hard cap on per-knot turning angle
_CHECK_EVERY = 1   # sort + track best every N GA generations (fewer CPU↔GPU syncs)

# ── GA kernels ─────────────────────────────────────────────────────────────────

@wp.kernel
def kernel_init_residual(
    canvas   : wp.array3d(dtype=wp.float32),
    target   : wp.array3d(dtype=wp.float32),
    residual : wp.array2d(dtype=wp.float32),
):
    flat = wp.tid()
    h = residual.shape[0]; w = residual.shape[1]
    if flat >= h * w:
        return
    py = flat // w; px = flat % w
    dr = target[py, px, 0] - canvas[py, px, 0]
    dg = target[py, px, 1] - canvas[py, px, 1]
    db = target[py, px, 2] - canvas[py, px, 2]
    residual[py, px] = dr*dr + dg*dg + db*db

@wp.kernel
def kernel_eval_chain_residual(
    genomes  : wp.array2d(dtype=wp.int32),
    palette  : wp.array2d(dtype=wp.float32),
    canvas   : wp.array3d(dtype=wp.float32),
    target   : wp.array3d(dtype=wp.float32),
    residual : wp.array2d(dtype=wp.float32),
    fitness  : wp.array(dtype=wp.float32),
    img_h    : int,
    img_w    : int,
    max_hw   : int,
    alpha    : float,
):
    """Evaluate a chained triplet of strokes; fitness = -(sum of error deltas on base canvas)."""
    ind = wp.tid()
    if ind >= genomes.shape[0]:
        return

    n_colors = palette.shape[0]

    # Decode shared color + brush width, then 4 chained endpoints
    c    = int(genomes[ind, 0]) % n_colors
    bw   = int(genomes[ind, 1])
    x0   = float(int(genomes[ind, 2])) / 255.0 * float(img_w - 1)
    y0   = float(int(genomes[ind, 3])) / 255.0 * float(img_h - 1)
    x1   = float(int(genomes[ind, 4])) / 255.0 * float(img_w - 1)
    y1   = float(int(genomes[ind, 5])) / 255.0 * float(img_h - 1)
    hw   = float(1) + float(bw) / 255.0 * float(max_hw - 1)

    # Derive x2,y2 and x3,y3 from angle-offset + distance genes (±60° cap).
    dx01  = x1 - x0; dy01 = y1 - y0
    len01 = wp.sqrt(dx01*dx01 + dy01*dy01)
    x2 = x1; y2 = y1; x3 = x1; y3 = y1      # degenerate defaults
    ux12 = float(1.0); uy12 = float(0.0)      # keep compiler happy
    if len01 >= float(0.5):
        inv01 = float(1.0) / len01
        ux01  = dx01 * inv01; uy01 = dy01 * inv01
        a1    = (float(int(genomes[ind, 6])) / float(255.0) * float(2.0) - float(1.0)) * float(1.0472)
        d1    = (float(0.3) + float(int(genomes[ind, 7])) / float(255.0) * float(1.7)) * len01
        c1    = wp.cos(a1); s1 = wp.sin(a1)
        ux12  = ux01*c1 - uy01*s1; uy12 = ux01*s1 + uy01*c1
        x2    = wp.clamp(x1 + ux12*d1, float(0.0), float(img_w - 1))
        y2    = wp.clamp(y1 + uy12*d1, float(0.0), float(img_h - 1))
        a2    = (float(int(genomes[ind, 8])) / float(255.0) * float(2.0) - float(1.0)) * float(1.0472)
        d2    = (float(0.3) + float(int(genomes[ind, 9])) / float(255.0) * float(1.7)) * len01
        c2    = wp.cos(a2); s2 = wp.sin(a2)
        ux23  = ux12*c2 - uy12*s2; uy23 = ux12*s2 + uy12*c2
        x3    = wp.clamp(x2 + ux23*d2, float(0.0), float(img_w - 1))
        y3    = wp.clamp(y2 + uy23*d2, float(0.0), float(img_h - 1))

    cr = palette[c, 0]; cg = palette[c, 1]; cb = palette[c, 2]

    total_delta = float(0.0)

    # ── Stroke 1: (x0,y0)→(x1,y1) ────────────────────────────────────────────
    dx = x1 - x0; dy = y1 - y0
    seg = wp.sqrt(dx*dx + dy*dy)
    if seg >= float(0.5):
        inv = float(1.0) / seg
        ux = dx*inv; uy = dy*inv
        vx = -uy; vy = ux
        cxc = (x0+x1)*float(0.5); cyc = (y0+y1)*float(0.5)
        hl  = seg*float(0.5)
        ex  = hl*wp.abs(ux) + hw*wp.abs(vx)
        ey  = hl*wp.abs(uy) + hw*wp.abs(vy)
        bx0 = int(wp.max(float(0.0), cxc - ex))
        bx1 = int(wp.min(float(img_w - 1), cxc + ex))
        by0 = int(wp.max(float(0.0), cyc - ey))
        by1 = int(wp.min(float(img_h - 1), cyc + ey))
        for py in range(by0, by1 + 1):
            for px in range(bx0, bx1 + 1):
                qx = float(px) - cxc; qy = float(py) - cyc
                du = qx*ux + qy*uy; dv = qx*vx + qy*vy
                if wp.abs(du) <= hl and wp.abs(dv) <= hw:
                    br = canvas[py,px,0]*(float(1.0)-alpha) + cr*alpha
                    bg = canvas[py,px,1]*(float(1.0)-alpha) + cg*alpha
                    bb = canvas[py,px,2]*(float(1.0)-alpha) + cb*alpha
                    dr2 = target[py,px,0]-br; dg2 = target[py,px,1]-bg; db2 = target[py,px,2]-bb
                    total_delta = total_delta + dr2*dr2+dg2*dg2+db2*db2 - residual[py,px]

    # ── Stroke 2: (x1,y1)→(x2,y2) ────────────────────────────────────────────
    dx = x2 - x1; dy = y2 - y1
    seg = wp.sqrt(dx*dx + dy*dy)
    if seg >= float(0.5):
        inv = float(1.0) / seg
        ux = dx*inv; uy = dy*inv
        vx = -uy; vy = ux
        cxc = (x1+x2)*float(0.5); cyc = (y1+y2)*float(0.5)
        hl  = seg*float(0.5)
        ex  = hl*wp.abs(ux) + hw*wp.abs(vx)
        ey  = hl*wp.abs(uy) + hw*wp.abs(vy)
        bx0 = int(wp.max(float(0.0), cxc - ex))
        bx1 = int(wp.min(float(img_w - 1), cxc + ex))
        by0 = int(wp.max(float(0.0), cyc - ey))
        by1 = int(wp.min(float(img_h - 1), cyc + ey))
        for py in range(by0, by1 + 1):
            for px in range(bx0, bx1 + 1):
                qx = float(px) - cxc; qy = float(py) - cyc
                du = qx*ux + qy*uy; dv = qx*vx + qy*vy
                if wp.abs(du) <= hl and wp.abs(dv) <= hw:
                    br = canvas[py,px,0]*(float(1.0)-alpha) + cr*alpha
                    bg = canvas[py,px,1]*(float(1.0)-alpha) + cg*alpha
                    bb = canvas[py,px,2]*(float(1.0)-alpha) + cb*alpha
                    dr2 = target[py,px,0]-br; dg2 = target[py,px,1]-bg; db2 = target[py,px,2]-bb
                    total_delta = total_delta + dr2*dr2+dg2*dg2+db2*db2 - residual[py,px]

    # ── Stroke 3: (x2,y2)→(x3,y3) ────────────────────────────────────────────
    dx = x3 - x2; dy = y3 - y2
    seg = wp.sqrt(dx*dx + dy*dy)
    if seg >= float(0.5):
        inv = float(1.0) / seg
        ux = dx*inv; uy = dy*inv
        vx = -uy; vy = ux
        cxc = (x2+x3)*float(0.5); cyc = (y2+y3)*float(0.5)
        hl  = seg*float(0.5)
        ex  = hl*wp.abs(ux) + hw*wp.abs(vx)
        ey  = hl*wp.abs(uy) + hw*wp.abs(vy)
        bx0 = int(wp.max(float(0.0), cxc - ex))
        bx1 = int(wp.min(float(img_w - 1), cxc + ex))
        by0 = int(wp.max(float(0.0), cyc - ey))
        by1 = int(wp.min(float(img_h - 1), cyc + ey))
        for py in range(by0, by1 + 1):
            for px in range(bx0, bx1 + 1):
                qx = float(px) - cxc; qy = float(py) - cyc
                du = qx*ux + qy*uy; dv = qx*vx + qy*vy
                if wp.abs(du) <= hl and wp.abs(dv) <= hw:
                    br = canvas[py,px,0]*(float(1.0)-alpha) + cr*alpha
                    bg = canvas[py,px,1]*(float(1.0)-alpha) + cg*alpha
                    bb = canvas[py,px,2]*(float(1.0)-alpha) + cb*alpha
                    dr2 = target[py,px,0]-br; dg2 = target[py,px,1]-bg; db2 = target[py,px,2]-bb
                    total_delta = total_delta + dr2*dr2+dg2*dg2+db2*db2 - residual[py,px]

    fitness[ind] = -total_delta

@wp.kernel
def kernel_paint_stroke_textured(
    canvas : wp.array3d(dtype=wp.float32),
    tex    : wp.array2d(dtype=wp.float32),
    cr: float, cg: float, cb: float,
    x0f: float, y0f: float, x1f: float, y1f: float,
    hw: float, alpha: float,
    img_h: int, img_w: int,
):
    flat = wp.tid()
    if flat >= img_h * img_w:
        return
    py = flat // img_w; px = flat % img_w
    dx = x1f - x0f; dy = y1f - y0f
    seg = wp.sqrt(dx*dx + dy*dy)
    if seg < float(0.5):
        return
    inv = float(1.0) / seg
    ux = dx*inv; uy = dy*inv
    vx = -uy; vy = ux
    cxc = (x0f + x1f) * float(0.5); cyc = (y0f + y1f) * float(0.5)
    hl  = seg * float(0.5)
    qx  = float(px) - cxc; qy = float(py) - cyc
    du  = qx*ux + qy*uy; dv = qx*vx + qy*vy
    if wp.abs(du) > hl or wp.abs(dv) > hw:
        return
    u_tex = (du + hl) / (float(2.0) * hl)
    v_tex = (dv + hw) / (float(2.0) * hw)
    tex_h = tex.shape[0]; tex_w = tex.shape[1]
    tx = wp.clamp(int(u_tex * float(tex_w)), 0, tex_w - 1)
    ty = wp.clamp(int(v_tex * float(tex_h)), 0, tex_h - 1)
    tex_alpha = tex[ty, tx] * alpha
    if tex_alpha < float(0.01):
        return
    canvas[py, px, 0] = canvas[py, px, 0] * (float(1.0) - tex_alpha) + cr * tex_alpha
    canvas[py, px, 1] = canvas[py, px, 1] * (float(1.0) - tex_alpha) + cg * tex_alpha
    canvas[py, px, 2] = canvas[py, px, 2] * (float(1.0) - tex_alpha) + cb * tex_alpha

@wp.kernel
def kernel_update_residual(
    canvas   : wp.array3d(dtype=wp.float32),
    target   : wp.array3d(dtype=wp.float32),
    residual : wp.array2d(dtype=wp.float32),
    x0f: float, y0f: float, x1f: float, y1f: float,
    hw: float,
    img_h: int, img_w: int,
):
    flat = wp.tid()
    if flat >= img_h * img_w:
        return
    py = flat // img_w; px = flat % img_w
    dx = x1f - x0f; dy = y1f - y0f
    seg = wp.sqrt(dx*dx + dy*dy)
    if seg < float(0.5):
        return
    inv = float(1.0) / seg
    ux = dx*inv; uy = dy*inv
    vx = -uy; vy = ux
    cxc = (x0f+x1f)*float(0.5); cyc = (y0f+y1f)*float(0.5)
    ex  = seg*float(0.5)*wp.abs(ux) + hw*wp.abs(vx)
    ey  = seg*float(0.5)*wp.abs(uy) + hw*wp.abs(vy)
    if (float(px) < cxc - ex or float(px) > cxc + ex or
        float(py) < cyc - ey or float(py) > cyc + ey):
        return
    dr = target[py, px, 0] - canvas[py, px, 0]
    dg = target[py, px, 1] - canvas[py, px, 1]
    db = target[py, px, 2] - canvas[py, px, 2]
    residual[py, px] = dr*dr + dg*dg + db*db

@wp.kernel
def kernel_next_gen(
    parents  : wp.array2d(dtype=wp.int32),
    children : wp.array2d(dtype=wp.int32),
    seeds    : wp.array(dtype=wp.uint32),
    N        : int,
    gmax     : int,
    elite_n  : int,
    p_size   : int,
    mut_rate : int,
):
    idx = wp.tid()
    if idx >= p_size:
        return
    if idx < elite_n:
        for g in range(N):
            children[idx, g] = parents[idx, g]
        return
    s = seeds[idx]
    s = s ^ (s << wp.uint32(13)); s = s ^ (s >> wp.uint32(17)); s = s ^ (s << wp.uint32(5))
    pa = int(s % wp.uint32(elite_n))
    s = s ^ (s << wp.uint32(13)); s = s ^ (s >> wp.uint32(17)); s = s ^ (s << wp.uint32(5))
    pb = int(s % wp.uint32(elite_n))
    cp = int(s % wp.uint32(N))
    for g in range(N):
        children[idx, g] = parents[pa, g] if g < cp else parents[pb, g]
    for g in range(N):
        s = s ^ (s << wp.uint32(13)); s = s ^ (s >> wp.uint32(17)); s = s ^ (s << wp.uint32(5))
        if int(s % wp.uint32(mut_rate)) == 0:
            s = s ^ (s << wp.uint32(13)); s = s ^ (s >> wp.uint32(17)); s = s ^ (s << wp.uint32(5))
            children[idx, g] = int(s % wp.uint32(gmax + 1))
    seeds[idx] = s

# ── Particle-layer kernel ──────────────────────────────────────────────────────

@wp.kernel
def k_apply_stroke_to_particles(
    x0f        : float,
    y0f        : float,
    x1f        : float,
    y1f        : float,
    hw         : float,
    alpha      : float,
    cr         : float,
    cg         : float,
    cb         : float,
    brush_tex  : wp.array2d(dtype=wp.float32),
    p_active   : wp.array(dtype=wp.int32),
    p_xyz      : wp.array(dtype=wp.vec3),
    p_rgb      : wp.array(dtype=wp.vec3),
    canvas     : wp.array3d(dtype=wp.float32),
    img_h      : int,
    img_w      : int,
    impasto_dz : float,
):
    flat = wp.tid()
    if flat >= img_h * img_w:
        return
    py = flat // img_w
    px = flat % img_w

    dx = x1f - x0f
    dy = y1f - y0f
    seg = wp.sqrt(dx * dx + dy * dy)
    if seg < float(0.5):
        return
    inv = float(1.0) / seg
    ux = dx * inv; uy = dy * inv
    vx = -uy;      vy = ux
    cxc = (x0f + x1f) * float(0.5)
    cyc = (y0f + y1f) * float(0.5)
    hl  = seg * float(0.5)
    qx  = float(px) - cxc
    qy  = float(py) - cyc
    du  = qx * ux + qy * uy
    dv  = qx * vx + qy * vy

    if wp.abs(du) > hl or wp.abs(dv) > hw:
        return

    u_tex = (du + hl) / (float(2.0) * hl)
    v_tex = (dv + hw) / (float(2.0) * hw)
    bh = brush_tex.shape[0]; bw = brush_tex.shape[1]
    tx = wp.clamp(int(u_tex * float(bw)), 0, bw - 1)
    ty = wp.clamp(int(v_tex * float(bh)), 0, bh - 1)
    eff_alpha = brush_tex[ty, tx] * alpha
    if eff_alpha < float(0.01):
        return

    pos     = p_xyz[flat]
    old_rgb = p_rgb[flat]

    new_r01 = float(0.0)
    new_g01 = float(0.0)
    new_b01 = float(0.0)

    if p_active[flat] == 0:
        p_active[flat] = wp.int32(1)
        new_r01 = cr * eff_alpha
        new_g01 = cg * eff_alpha
        new_b01 = cb * eff_alpha
    else:
        old_r01 = old_rgb[0] / float(255.0)
        old_g01 = old_rgb[1] / float(255.0)
        old_b01 = old_rgb[2] / float(255.0)
        new_r01 = old_r01 * (float(1.0) - eff_alpha) + cr * eff_alpha
        new_g01 = old_g01 * (float(1.0) - eff_alpha) + cg * eff_alpha
        new_b01 = old_b01 * (float(1.0) - eff_alpha) + cb * eff_alpha
        p_xyz[flat] = wp.vec3(pos[0], pos[1], pos[2] + impasto_dz * brush_tex[ty, tx])

    p_rgb[flat] = wp.vec3(new_r01 * float(255.0), new_g01 * float(255.0), new_b01 * float(255.0))
    canvas[py, px, 0] = new_r01
    canvas[py, px, 1] = new_g01
    canvas[py, px, 2] = new_b01

# ── Spline rendering kernels ───────────────────────────────────────────────────

@wp.kernel
def k_apply_spline_to_particles(
    spline_pts : wp.array2d(dtype=wp.float32),
    cum_len    : wp.array(dtype=wp.float32),
    n_pts      : int,
    total_len  : float,
    hw         : float,
    alpha      : float,
    cr         : float,
    cg         : float,
    cb         : float,
    brush_tex  : wp.array2d(dtype=wp.float32),
    p_active   : wp.array(dtype=wp.int32),
    p_xyz      : wp.array(dtype=wp.vec3),
    p_rgb      : wp.array(dtype=wp.vec3),
    canvas     : wp.array3d(dtype=wp.float32),
    img_h      : int,
    img_w      : int,
    impasto_dz : float,
):
    """Per-pixel closest-point on Catmull-Rom polyline -> texture UV -> paint."""
    flat = wp.tid()
    if flat >= img_h * img_w:
        return
    py  = flat // img_w
    px  = flat % img_w
    fpx = float(px)
    fpy = float(py)

    min_dist_sq = float(1.0e18)
    best_t      = float(0.0)
    best_dv     = float(0.0)

    for i in range(n_pts - 1):
        ax = spline_pts[i,     0];  ay = spline_pts[i,     1]
        bx = spline_pts[i + 1, 0];  by = spline_pts[i + 1, 1]
        dx = bx - ax;  dy = by - ay
        seg_len = wp.sqrt(dx*dx + dy*dy)
        if seg_len >= float(0.01):
            inv   = float(1.0) / seg_len
            ux    = dx * inv;   uy = dy * inv
            vx    = -uy;        vy = ux
            qx    = fpx - ax;   qy = fpy - ay
            du    = qx*ux + qy*uy
            t_seg = wp.clamp(du * inv, float(0.0), float(1.0))
            cpx   = ax + t_seg * dx
            cpy   = ay + t_seg * dy
            dist_sq = (fpx-cpx)*(fpx-cpx) + (fpy-cpy)*(fpy-cpy)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_t  = (cum_len[i] + t_seg * seg_len) / total_len
                best_dv = (fpx-cpx)*vx + (fpy-cpy)*vy

    if wp.sqrt(min_dist_sq) > hw:
        return

    u_tex = wp.clamp(best_t, float(0.0), float(1.0))
    v_tex = wp.clamp((best_dv + hw) / (float(2.0) * hw), float(0.0), float(1.0))
    bh    = brush_tex.shape[0];  bw = brush_tex.shape[1]
    tx    = wp.clamp(int(u_tex * float(bw)), 0, bw - 1)
    ty    = wp.clamp(int(v_tex * float(bh)), 0, bh - 1)
    eff_alpha = brush_tex[ty, tx] * alpha
    if eff_alpha < float(0.01):
        return

    pos     = p_xyz[flat]
    old_rgb = p_rgb[flat]
    new_r   = float(0.0);  new_g = float(0.0);  new_b = float(0.0)

    if p_active[flat] == 0:
        p_active[flat] = wp.int32(1)
        new_r = cr * eff_alpha
        new_g = cg * eff_alpha
        new_b = cb * eff_alpha
    else:
        old_r = old_rgb[0] / float(255.0)
        old_g = old_rgb[1] / float(255.0)
        old_b = old_rgb[2] / float(255.0)
        new_r = old_r * (float(1.0)-eff_alpha) + cr * eff_alpha
        new_g = old_g * (float(1.0)-eff_alpha) + cg * eff_alpha
        new_b = old_b * (float(1.0)-eff_alpha) + cb * eff_alpha
        p_xyz[flat] = wp.vec3(pos[0], pos[1], pos[2] + impasto_dz * brush_tex[ty, tx])

    p_rgb[flat]       = wp.vec3(new_r * float(255.0), new_g * float(255.0), new_b * float(255.0))
    canvas[py, px, 0] = new_r
    canvas[py, px, 1] = new_g
    canvas[py, px, 2] = new_b


@wp.kernel
def kernel_update_residual_bbox(
    canvas   : wp.array3d(dtype=wp.float32),
    target   : wp.array3d(dtype=wp.float32),
    residual : wp.array2d(dtype=wp.float32),
    bx0      : int,
    bx1      : int,
    by0      : int,
    by1      : int,
    img_h    : int,
    img_w    : int,
):
    flat = wp.tid()
    bw   = bx1 - bx0 + 1
    if flat >= bw * (by1 - by0 + 1):
        return
    py = by0 + flat // bw
    px = bx0 + flat % bw
    if py >= img_h or px >= img_w:
        return
    dr = target[py, px, 0] - canvas[py, px, 0]
    dg = target[py, px, 1] - canvas[py, px, 1]
    db = target[py, px, 2] - canvas[py, px, 2]
    residual[py, px] = dr*dr + dg*dg + db*db

# ── CPU seeding helper ─────────────────────────────────────────────────────────

def _seed_population(
    residual_np : np.ndarray,
    palette_np  : np.ndarray,
    img_rgb     : np.ndarray,
    P           : int,
    H           : int,
    W           : int,
    seed_frac   : float,
    explore_r   : float,
    rng         : np.random.Generator,
) -> np.ndarray:
    """Vectorised population seeding for 3-stroke chain genomes."""
    prob = residual_np.flatten().astype(np.float64)
    prob += 1e-8
    prob /= prob.sum()

    n_seed = int(P * seed_frac)
    flat_idx = rng.choice(H * W, size=n_seed, p=prob)
    py_s = flat_idx // W
    px_s = flat_idx % W

    # Vectorised nearest-palette colour for each stroke origin
    def _best_color(ys, xs):
        colors = img_rgb[ys, xs].astype(np.float32) / 255.0   # (n, 3)
        diffs  = palette_np[None] - colors[:, None]            # (n, K, 3)
        return np.argmin((diffs**2).sum(-1), axis=-1).astype(np.int32)

    def _endpoint(cx, cy, ex_frac, ey_frac):
        dx = rng.uniform(-W * ex_frac, W * ex_frac, size=n_seed).astype(int)
        dy = rng.uniform(-H * ey_frac, H * ey_frac, size=n_seed).astype(int)
        ex = np.clip(cx + dx, 0, W - 1)
        ey = np.clip(cy + dy, 0, H - 1)
        return ex, ey

    # Single color chosen at chain origin; one brush width for the whole chain
    c        = _best_color(py_s, px_s)
    bw       = rng.integers(0, 256, size=n_seed, dtype=np.int32)
    x1s, y1s = _endpoint(px_s, py_s, explore_r, explore_r)
    # Genes 6-9 are angle-offset and distance-scale for knots 2 and 3.
    # Random in [0,255] → uniform exploration of the ±60° cone; GA refines from here.
    a1s = rng.integers(0, 256, size=n_seed, dtype=np.int32)
    d1s = rng.integers(0, 256, size=n_seed, dtype=np.int32)
    a2s = rng.integers(0, 256, size=n_seed, dtype=np.int32)
    d2s = rng.integers(0, 256, size=n_seed, dtype=np.int32)

    def _enc(xs, dim): return np.clip((xs / max(dim - 1, 1) * 255), 0, 255).astype(np.int32)

    genomes = np.empty((P, N_GENES), dtype=np.int32)
    genomes[:n_seed] = np.stack([
        c, bw, _enc(px_s, W), _enc(py_s, H), _enc(x1s, W), _enc(y1s, H),
        a1s, d1s, a2s, d2s,
    ], axis=1)
    genomes[n_seed:] = rng.integers(0, 256, size=(P - n_seed, N_GENES), dtype=np.int32)
    return genomes

# ── Catmull-Rom spline helper ─────────────────────────────────────────────────

def _sample_catmull_rom(
    pts      : np.ndarray,
    n_samples: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a Catmull-Rom spline through 4 control points.

    pts: (4, 2) float array of (x, y) control points.
    Returns (samples (N, 2), cum_arc_len (N,)) both float32.
    The curve passes through all 4 points; phantom endpoints are reflected
    to ensure C1 continuity at the tips.
    """
    p0, p1, p2, p3 = pts[0], pts[1], pts[2], pts[3]
    pm1 = 2.0 * p0 - p1   # phantom before p0
    p4  = 2.0 * p3 - p2   # phantom after p3

    # Three segments: p0->p1, p1->p2, p2->p3
    segs = [
        (pm1, p0, p1, p2),
        (p0,  p1, p2, p3),
        (p1,  p2, p3,  p4),
    ]
    n_per = max(2, (n_samples + 2) // 3)
    parts = []
    for qa, qb, qc, qd in segs:
        ts = np.linspace(0.0, 1.0, n_per, endpoint=False)
        t2 = ts * ts;  t3 = t2 * ts
        x = 0.5*((2*qb[0]) + (-qa[0]+qc[0])*ts
                 + (2*qa[0]-5*qb[0]+4*qc[0]-qd[0])*t2
                 + (-qa[0]+3*qb[0]-3*qc[0]+qd[0])*t3)
        y = 0.5*((2*qb[1]) + (-qa[1]+qc[1])*ts
                 + (2*qa[1]-5*qb[1]+4*qc[1]-qd[1])*t2
                 + (-qa[1]+3*qb[1]-3*qc[1]+qd[1])*t3)
        parts.append(np.stack([x, y], axis=1))
    parts.append(p3.reshape(1, 2))            # final endpoint

    samples = np.concatenate(parts, axis=0).astype(np.float32)
    diffs   = np.diff(samples, axis=0)
    dists   = np.sqrt((diffs**2).sum(axis=1))
    cum     = np.concatenate([[0.0], np.cumsum(dists)]).astype(np.float32)
    return samples, cum

# ── Painter class ──────────────────────────────────────────────────────────────

class Painter:
    """
    GPU-accelerated impasto painter using a genetic algorithm.

    Each update() step the GA finds the best chain of 3 linked brushstrokes
    that jointly reduce the error to the target image, then applies them
    sequentially. Strokes are chained endpoint-to-endpoint so they read
    as a continuous mark.

    Workflow
    --------
    1. painter = Painter(...)
    2. painter.init_image(img_bgr)
    3. painter.init_brush(path)
    4. painter.init_canvas()
    5. loop: painter.update()
    6. Pass painter.xyz, painter.rgb, painter.r to raytrace_sphere
    """

    def __init__(
        self,
        canvas_w        : int   = 1024,
        canvas_h        : int   = 768,
        population      : int   = 1024,
        n_strokes       : int   = 300,
        gens_per_stroke : int   = 60,
        elite_n         : int   = 20,
        mutation_rate   : float = 0.05,
        n_colors        : int   = 256,
        max_brush_hw    : int   = 32,
        stroke_alpha    : float = 0.95,
        seed_frac       : float = 0.6,
        explore_r       : float = 0.20,
        max_dim         : int   = 512,
        impasto_dz_frac : float = 0.3,
        device          : str   = "cuda",
        seed            : int   = 42,
    ):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h

        self.P               = population
        self.n_strokes       = n_strokes
        self.gens_per_stroke = gens_per_stroke
        self.elite_n         = elite_n
        self.mut_rate_k      = max(1, int(1.0 / mutation_rate))
        self.alpha           = stroke_alpha
        self.seed_frac       = seed_frac
        self.explore_r       = explore_r

        self.n_colors        = n_colors
        self.max_hw          = max_brush_hw
        self.max_dim         = max_dim
        self.impasto_dz_frac = impasto_dz_frac

        self.device = device
        self.rng    = np.random.default_rng(seed)

        self.stroke_list : list[dict] = []
        self._step        = 0

        self.img_rgb    : np.ndarray | None = None
        self.palette_np : np.ndarray | None = None
        self.brush_base : np.ndarray | None = None
        self.H = self.W = 0
        self.HH = self.HW = 0
        self.scale : float = 1.0

        self.canvas_lo_a : wp.array | None = None
        self.target_a    : wp.array | None = None
        self.residual_a  : wp.array | None = None
        self.fitness_a   : wp.array | None = None
        self.palette_a   : wp.array | None = None
        self.genome_a    : wp.array | None = None
        self.children_a  : wp.array | None = None
        self.seeds_a     : wp.array | None = None

        self.xyz       : wp.array | None = None
        self.rgb       : wp.array | None = None
        self.rot       : wp.array | None = None
        self.p_active  : wp.array | None = None
        self.r         : float = 0.0
        self.z_active  : float = 0.0
        self.impasto_dz: float = 0.0
        self.particles : int   = 0

    # ── Init ──────────────────────────────────────────────────────────────────

    def init_image(self, img_bgr: np.ndarray) -> None:
        if img_bgr is None:
            raise ValueError("img_bgr is None")
        if not isinstance(img_bgr, np.ndarray):
            raise TypeError(f"img_bgr must be a numpy array, got {type(img_bgr)}")
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError(f"img_bgr must have shape (H, W, 3), got {img_bgr.shape}")

        canvas_aspect = self.canvas_w / self.canvas_h

        if canvas_aspect >= 1.0:
            work_w = self.max_dim
            work_h = max(1, int(round(self.max_dim / canvas_aspect)))
        else:
            work_h = self.max_dim
            work_w = max(1, int(round(self.max_dim * canvas_aspect)))

        orig_h, orig_w = img_bgr.shape[:2]
        orig_aspect    = orig_w / orig_h

        scale  = work_h / orig_h if orig_aspect >= canvas_aspect else work_w / orig_w
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        sw     = max(1, int(round(orig_w * scale)))
        sh     = max(1, int(round(orig_h * scale)))
        img_bgr = cv2.resize(img_bgr, (sw, sh), interpolation=interp)

        cy = (sh - work_h) // 2
        cx = (sw - work_w) // 2
        img_bgr = img_bgr[cy:cy + work_h, cx:cx + work_w]

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)

        pixels = img_rgb.reshape(-1, 3).astype(np.float32) / 255.0
        km = MiniBatchKMeans(n_clusters=self.n_colors, random_state=42, n_init=3)
        km.fit(pixels)

        self.img_rgb    = img_rgb
        self.palette_np = km.cluster_centers_.astype(np.float32)
        self.H, self.W  = work_h, work_w

        print(f"[Painter] Image loaded → working {self.W}×{self.H}  "
              f"canvas aspect {canvas_aspect:.3f}  palette {self.palette_np.shape}")

    def init_brush(self, path: str | None = None, invert: bool = False) -> None:
        if path is not None:
            raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if raw is not None:
                tex = 1.0 - (raw.astype(np.float32) / 255.0)
                if invert:
                    tex = 1.0 - tex
                self.brush_base = np.clip(tex, 0.0, 1.0)
                print(f"[Painter] Brush loaded → {tex.shape}")
                return

        print("[Painter] Brush not found — using synthetic fallback.")
        th, tw = 64, 256
        tex = np.zeros((th, tw), dtype=np.float32)
        cy2, cx2 = th / 2.0, tw / 2.0
        for y in range(th):
            for x in range(tw):
                nx = (x - cx2) / (tw * 0.48)
                ny = (y - cy2) / (th * 0.40)
                v = max(0.0, 1.0 - nx*nx - ny*ny)
                tex[y, x] = v ** 0.5
        self.brush_base = tex

    def init_canvas(self) -> None:
        if self.img_rgb is None:
            raise RuntimeError("Call init_image() before init_canvas().")
        if self.brush_base is None:
            raise RuntimeError("Call init_brush() before init_canvas().")

        H, W   = self.H, self.W
        HH, HW = self.canvas_h, self.canvas_w
        self.HH, self.HW = HH, HW
        self.scale = self.canvas_w / W

        scale_y_check = self.canvas_h / H
        if abs(self.scale - scale_y_check) > 0.5:
            import warnings
            warnings.warn(
                f"[Painter] Scale mismatch: scale_x={self.scale:.3f} "
                f"scale_y={scale_y_check:.3f}",
                stacklevel=2,
            )

        canvas_lo = np.zeros((H, W, 3), dtype=np.float32)
        target_np = self.img_rgb.astype(np.float32) / 255.0

        self.canvas_lo_a = wp.array(canvas_lo, dtype=wp.float32,
                                    shape=(H, W, 3), device=self.device)
        self.target_a    = wp.array(target_np, dtype=wp.float32,
                                    shape=(H, W, 3), device=self.device)
        self.residual_a  = wp.zeros((H, W),  dtype=wp.float32, device=self.device)
        self.fitness_a   = wp.zeros(self.P,   dtype=wp.float32, device=self.device)
        self.palette_a   = wp.array(self.palette_np, dtype=wp.float32,
                                    shape=self.palette_np.shape, device=self.device)

        genome_np   = self.rng.integers(0, 256, size=(self.P, N_GENES), dtype=np.int32)
        children_np = np.zeros_like(genome_np)
        seeds_np    = self.rng.integers(1, 2**32, size=self.P, dtype=np.uint32)
        self.genome_a   = wp.array(genome_np,   dtype=wp.int32,
                                   shape=(self.P, N_GENES), device=self.device)
        self.children_a = wp.array(children_np, dtype=wp.int32,
                                   shape=(self.P, N_GENES), device=self.device)
        self.seeds_a    = wp.array(seeds_np,    dtype=wp.uint32,
                                   shape=(self.P,),          device=self.device)

        wp.launch(kernel_init_residual, dim=H * W,
                  inputs=[self.canvas_lo_a, self.target_a, self.residual_a],
                  device=self.device)

        N = H * W
        self.particles = N

        h_cell          = 1.0 / max(W, H)
        self.r          = 0.5 * h_cell
        self.z_active   = h_cell + self.r
        self.impasto_dz = self.impasto_dz_frac * self.r

        px_idx = np.arange(W, dtype=np.float32)
        py_idx = np.arange(H, dtype=np.float32)
        px_grid, py_grid = np.meshgrid(px_idx, py_idx)

        x_world = (px_grid + 0.5) * h_cell
        y_world = (H - 1 - py_grid + 0.5) * h_cell
        z_world = np.full((H, W), self.z_active, dtype=np.float32)

        xyz_np = np.stack([x_world, y_world, z_world], axis=-1).reshape(-1, 3).astype(np.float32)
        rgb_np = np.zeros((N, 3), dtype=np.float32)
        rot_np = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (N, 1))
        act_np = np.zeros(N, dtype=np.int32)

        self.xyz      = wp.array(xyz_np, dtype=wp.vec3,  device=self.device)
        self.rgb      = wp.array(rgb_np, dtype=wp.vec3,  device=self.device)
        self.rot      = wp.array(rot_np, dtype=wp.vec3,  device=self.device)
        self.p_active = wp.array(act_np, dtype=wp.int32, device=self.device)

        self._step = 0
        print(f"[Painter] Ready  lo {W}×{H}  hi {HW}×{HH}  "
              f"scale ×{self.scale:.2f}  {N:,} particles  "
              f"r={self.r:.5f}  z_floor={self.z_active:.5f}  "
              f"impasto_dz={self.impasto_dz:.6f}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def update(self) -> dict | None:
        """Paint one chain of 3 linked brushstrokes. Returns stroke info, or None when done."""
        if self._step >= self.n_strokes:
            return None
        self._step += 1
        return self._paint_stroke()

    # ── Output ────────────────────────────────────────────────────────────────

    def get_canvas_bgr(self) -> np.ndarray:
        """Lo-res canvas upscaled to canvas dimensions (BGR uint8)."""
        lo = (self.canvas_lo_a.numpy() * 255).clip(0, 255).astype(np.uint8)
        lo_bgr = cv2.cvtColor(lo, cv2.COLOR_RGB2BGR)
        return cv2.resize(lo_bgr, (self.HW, self.HH), interpolation=cv2.INTER_LINEAR)

    def get_canvas_lo_bgr(self) -> np.ndarray:
        rgb = (self.canvas_lo_a.numpy() * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def get_error_bgr(self) -> np.ndarray:
        res = self.residual_a.numpy()
        res_norm = (res / (res.max() + 1e-8) * 255).clip(0, 255).astype(np.uint8)
        return cv2.applyColorMap(res_norm, cv2.COLORMAP_HOT)

    @property
    def done(self) -> bool:
        return self._step >= self.n_strokes

    @property
    def mse(self) -> float:
        return float(self.residual_a.numpy().sum()) / (self.H * self.W * 3)

    @property
    def active_count(self) -> int:
        return int(self.p_active.numpy().sum())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _make_tex_array(self, seg_px: float, hw_px: float) -> wp.array:
        tex_w = max(1, int(round(seg_px)))
        tex_h = max(1, int(round(hw_px * 2.0)))
        tex   = cv2.resize(self.brush_base, (tex_w, tex_h),
                           interpolation=cv2.INTER_LINEAR)
        return wp.array(np.clip(tex, 0.0, 1.0).astype(np.float32),
                        dtype=wp.float32, shape=(tex_h, tex_w),
                        device=self.device)

    def _apply_stroke(
        self,
        x_a: float, y_a: float,
        x_b: float, y_b: float,
        hw: float,
        cr: float, cg: float, cb: float,
    ) -> None:
        """Apply one stroke to particles + canvas_lo, then update residual."""
        seg = float(np.hypot(x_b - x_a, y_b - y_a))
        tex_a = self._make_tex_array(seg, hw)
        wp.launch(k_apply_stroke_to_particles, dim=self.H * self.W,
                  inputs=[x_a, y_a, x_b, y_b, hw, self.alpha, cr, cg, cb,
                          tex_a, self.p_active, self.xyz, self.rgb,
                          self.canvas_lo_a, self.H, self.W, self.impasto_dz],
                  device=self.device)
        wp.launch(kernel_update_residual, dim=self.H * self.W,
                  inputs=[self.canvas_lo_a, self.target_a, self.residual_a,
                          x_a, y_a, x_b, y_b, hw, self.H, self.W],
                  device=self.device)

    def _apply_spline_stroke(
        self,
        pts: np.ndarray,
        hw : float,
        cr : float, cg: float, cb: float,
    ) -> None:
        """Render one Catmull-Rom spline stroke through 4 control points.

        Brush texture is UV-mapped along the curve:
          u = arc-length position (0=start, 1=end)  → brush columns (tapered tips)
          v = signed perpendicular distance          → brush rows (cross-section profile)
        """
        samples, cum = _sample_catmull_rom(pts)
        total_len    = float(cum[-1])
        if total_len < 0.5:
            return

        # Texture width = total arc length so one brush column maps to one pixel
        tex_w = max(1, int(round(total_len)))
        tex_h = max(1, int(round(hw * 2.0)))
        tex   = cv2.resize(self.brush_base, (tex_w, tex_h),
                           interpolation=cv2.INTER_LINEAR)
        tex_a = wp.array(np.clip(tex, 0.0, 1.0).astype(np.float32),
                         dtype=wp.float32, shape=(tex_h, tex_w),
                         device=self.device)

        n_pts      = len(samples)
        samples_a  = wp.array(samples, dtype=wp.float32,
                               shape=(n_pts, 2), device=self.device)
        cum_a      = wp.array(cum, dtype=wp.float32,
                               shape=(n_pts,), device=self.device)

        wp.launch(k_apply_spline_to_particles, dim=self.H * self.W,
                  inputs=[samples_a, cum_a, n_pts, total_len,
                          hw, self.alpha, cr, cg, cb, tex_a,
                          self.p_active, self.xyz, self.rgb,
                          self.canvas_lo_a, self.H, self.W, self.impasto_dz],
                  device=self.device)

        # Update residual over the spline bounding box (padded by hw)
        bx0 = max(0,         int(np.min(samples[:, 0]) - hw) - 1)
        bx1 = min(self.W-1,  int(np.max(samples[:, 0]) + hw) + 1)
        by0 = max(0,         int(np.min(samples[:, 1]) - hw) - 1)
        by1 = min(self.H-1,  int(np.max(samples[:, 1]) + hw) + 1)
        bbox_area = max(1, (bx1-bx0+1) * (by1-by0+1))
        wp.launch(kernel_update_residual_bbox, dim=bbox_area,
                  inputs=[self.canvas_lo_a, self.target_a, self.residual_a,
                          bx0, bx1, by0, by1, self.H, self.W],
                  device=self.device)

    def _paint_stroke(self) -> dict:
        # Coarse-to-fine: large strokes early, fine detail later
        t = (self._step - 1) / max(1, self.n_strokes - 1)
        eff_max_hw = max(4, int(self.max_hw * (1.0 - t) ** 0.5))

        residual_np = self.residual_a.numpy()
        genome_np   = _seed_population(
            residual_np, self.palette_np, self.img_rgb,
            self.P, self.H, self.W,
            self.seed_frac, self.explore_r, self.rng,
        )
        self.genome_a = wp.array(genome_np, dtype=wp.int32,
                                 shape=(self.P, N_GENES), device=self.device)

        best_f     = -1e18
        best_genes = genome_np[0].copy()

        for gen in range(self.gens_per_stroke):
            wp.launch(kernel_eval_chain_residual, dim=self.P,
                      inputs=[self.genome_a, self.palette_a, self.canvas_lo_a,
                              self.target_a, self.residual_a, self.fitness_a,
                              self.H, self.W, eff_max_hw, self.alpha],
                      device=self.device)

            # Sync every _CHECK_EVERY gens (or on the last gen) to sort + track best
            if (gen % _CHECK_EVERY) == (_CHECK_EVERY - 1) or gen == self.gens_per_stroke - 1:
                fit_np    = self.fitness_a.numpy()
                order     = np.argsort(-fit_np)
                genome_np = self.genome_a.numpy()[order]
                if fit_np[order[0]] > best_f:
                    best_f     = float(fit_np[order[0]])
                    best_genes = genome_np[0].copy()
                self.genome_a = wp.array(genome_np, dtype=wp.int32,
                                         shape=(self.P, N_GENES), device=self.device)

            wp.launch(kernel_next_gen, dim=self.P,
                      inputs=[self.genome_a, self.children_a, self.seeds_a,
                              N_GENES, GENE_MAX, self.elite_n, self.P, self.mut_rate_k],
                      device=self.device)
            self.genome_a, self.children_a = self.children_a, self.genome_a

        # ── Decode winning chain ───────────────────────────────────────────────
        g  = best_genes
        nc = self.palette_np.shape[0]

        cid = int(g[0]) % nc
        bw  = int(g[1])
        x0  = float(g[2]) / 255.0 * (self.W - 1)
        y0  = float(g[3]) / 255.0 * (self.H - 1)
        x1  = float(g[4]) / 255.0 * (self.W - 1)
        y1  = float(g[5]) / 255.0 * (self.H - 1)
        hw  = 1.0 + float(bw) / 255.0 * (eff_max_hw - 1)

        dx01 = x1 - x0; dy01 = y1 - y0
        len01 = float(np.hypot(dx01, dy01))
        if len01 >= 0.5:
            ux01 = dx01 / len01; uy01 = dy01 / len01
            a1   = (float(g[6]) / 255.0 * 2.0 - 1.0) * MAX_TURN_RAD
            d1   = (0.3 + float(g[7]) / 255.0 * 1.7) * len01
            c1   = float(np.cos(a1)); s1 = float(np.sin(a1))
            ux12 = ux01*c1 - uy01*s1; uy12 = ux01*s1 + uy01*c1
            x2   = float(np.clip(x1 + ux12*d1, 0, self.W - 1))
            y2   = float(np.clip(y1 + uy12*d1, 0, self.H - 1))
            a2   = (float(g[8]) / 255.0 * 2.0 - 1.0) * MAX_TURN_RAD
            d2   = (0.3 + float(g[9]) / 255.0 * 1.7) * len01
            c2   = float(np.cos(a2)); s2 = float(np.sin(a2))
            ux23 = ux12*c2 - uy12*s2; uy23 = ux12*s2 + uy12*c2
            x3   = float(np.clip(x2 + ux23*d2, 0, self.W - 1))
            y3   = float(np.clip(y2 + uy23*d2, 0, self.H - 1))
        else:
            x2 = x1; y2 = y1; x3 = x1; y3 = y1

        # ── Render as a single Catmull-Rom spline through all 4 control points ─
        cr  = float(self.palette_np[cid, 0])
        cg  = float(self.palette_np[cid, 1])
        cb  = float(self.palette_np[cid, 2])
        pts = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]], dtype=np.float32)
        self._apply_spline_stroke(pts, hw, cr, cg, cb)

        mse = float(self.residual_a.numpy().sum()) / (self.H * self.W * 3)
        stroke = dict(
            step=self._step,
            eff_max_hw=eff_max_hw,
            ctrl_pts=[(x0, y0), (x1, y1), (x2, y2), (x3, y3)],
            hw=hw,
            color_id=cid,
            color_rgb=(cr, cg, cb),
            genes=best_genes.tolist(),
            mse=mse,
            active_particles=self.active_count,
        )
        self.stroke_list.append(stroke)
        return stroke
