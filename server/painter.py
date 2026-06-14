import warp as wp
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
wp.init()

N_GENES  = 6
GENE_MAX = 255

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
def kernel_eval_stroke_residual(
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
    ind = wp.tid()
    if ind >= genomes.shape[0]:
        return
    color_id  = int(genomes[ind, 0]) % palette.shape[0]
    bw_gene   = int(genomes[ind, 1])
    p0x_i = int(genomes[ind, 2]); p0y_i = int(genomes[ind, 3])
    p1x_i = int(genomes[ind, 4]); p1y_i = int(genomes[ind, 5])

    x0f = float(p0x_i) / 255.0 * float(img_w - 1)
    y0f = float(p0y_i) / 255.0 * float(img_h - 1)
    x1f = float(p1x_i) / 255.0 * float(img_w - 1)
    y1f = float(p1y_i) / 255.0 * float(img_h - 1)
    hw  = float(1) + float(bw_gene) / 255.0 * float(max_hw - 1)

    cr = palette[color_id, 0]; cg = palette[color_id, 1]; cb = palette[color_id, 2]

    dx = x1f - x0f; dy = y1f - y0f
    seg_len = wp.sqrt(dx*dx + dy*dy)
    if seg_len < float(0.5):
        fitness[ind] = float(-1e9)
        return
    inv_len = float(1.0) / seg_len
    ux = dx * inv_len; uy = dy * inv_len
    vx = -uy; vy = ux
    cxc = (x0f + x1f) * float(0.5); cyc = (y0f + y1f) * float(0.5)
    hl  = seg_len * float(0.5)
    ex  = hl * wp.abs(ux) + hw * wp.abs(vx)
    ey  = hl * wp.abs(uy) + hw * wp.abs(vy)
    bx0 = int(wp.max(float(0.0), cxc - ex))
    bx1 = int(wp.min(float(img_w - 1), cxc + ex))
    by0 = int(wp.max(float(0.0), cyc - ey))
    by1 = int(wp.min(float(img_h - 1), cyc + ey))

    delta = float(0.0)
    for py in range(by0, by1 + 1):
        for px in range(bx0, bx1 + 1):
            qx = float(px) - cxc; qy = float(py) - cyc
            du = qx*ux + qy*uy; dv = qx*vx + qy*vy
            res_before = residual[py, px]
            if wp.abs(du) <= hl and wp.abs(dv) <= hw:
                blended_r = canvas[py, px, 0] * (float(1.0) - alpha) + cr * alpha
                blended_g = canvas[py, px, 1] * (float(1.0) - alpha) + cg * alpha
                blended_b = canvas[py, px, 2] * (float(1.0) - alpha) + cb * alpha
                dr2 = target[py, px, 0] - blended_r
                dg2 = target[py, px, 1] - blended_g
                db2 = target[py, px, 2] - blended_b
                res_after = dr2*dr2 + dg2*dg2 + db2*db2
            else:
                res_after = res_before
            delta = delta + res_after - res_before
    fitness[ind] = -delta

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
    cr         : float,   # palette color [0,1]
    cg         : float,
    cb         : float,
    brush_tex  : wp.array2d(dtype=wp.float32),  # [0,1]: 1=full paint, 0=no paint
    p_active   : wp.array(dtype=wp.int32),
    p_xyz      : wp.array(dtype=wp.vec3),
    p_rgb      : wp.array(dtype=wp.vec3),   # stored as [0,255] to match raytrace_sphere
    canvas     : wp.array3d(dtype=wp.float32),  # [0,1] for GA residual
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

    # Brush shape: map OBB coords → UV → texture sample
    u_tex = (du + hl) / (float(2.0) * hl)
    v_tex = (dv + hw) / (float(2.0) * hw)
    bh = brush_tex.shape[0]; bw = brush_tex.shape[1]
    tx = wp.clamp(int(u_tex * float(bw)), 0, bw - 1)
    ty = wp.clamp(int(v_tex * float(bh)), 0, bh - 1)
    eff_alpha = brush_tex[ty, tx] * alpha   # 0 at brush edges, alpha at brush centre
    if eff_alpha < float(0.01):
        return

    pos     = p_xyz[flat]
    old_rgb = p_rgb[flat]   # [0,255]

    new_r01 = float(0.0)
    new_g01 = float(0.0)
    new_b01 = float(0.0)

    if p_active[flat] == 0:
        # First paint: blend from black using brush opacity so edges are soft.
        p_active[flat] = wp.int32(1)
        new_r01 = cr * eff_alpha
        new_g01 = cg * eff_alpha
        new_b01 = cb * eff_alpha
    else:
        # Re-paint: blend colour in [0,1] space, raise z (impasto accumulation).
        old_r01 = old_rgb[0] / float(255.0)
        old_g01 = old_rgb[1] / float(255.0)
        old_b01 = old_rgb[2] / float(255.0)
        new_r01 = old_r01 * (float(1.0) - eff_alpha) + cr * eff_alpha
        new_g01 = old_g01 * (float(1.0) - eff_alpha) + cg * eff_alpha
        new_b01 = old_b01 * (float(1.0) - eff_alpha) + cb * eff_alpha
        p_xyz[flat] = wp.vec3(pos[0], pos[1], pos[2] + impasto_dz * brush_tex[ty, tx])

    # [0,255] → raytrace_sphere compatibility
    p_rgb[flat] = wp.vec3(new_r01 * float(255.0), new_g01 * float(255.0), new_b01 * float(255.0))
    # [0,1]   → GA residual tracking
    canvas[py, px, 0] = new_r01
    canvas[py, px, 1] = new_g01
    canvas[py, px, 2] = new_b01

# ── CPU seeding helper ─────────────────────────────────────────────────────────

def _seed_population(
    residual_np: np.ndarray,
    palette_np: np.ndarray,
    img_rgb: np.ndarray,
    P: int, H: int, W: int,
    max_hw: int, seed_frac: float, explore_r: float,
    rng: np.random.Generator,
) -> np.ndarray:
    prob = residual_np.flatten().astype(np.float64)
    prob += 1e-8
    prob /= prob.sum()

    genomes = np.zeros((P, N_GENES), dtype=np.int32)
    n_seed = int(P * seed_frac)

    flat_idx = rng.choice(H * W, size=n_seed, p=prob)
    py_s = flat_idx // W; px_s = flat_idx % W

    for i in range(n_seed):
        cx = px_s[i]; cy = py_s[i]
        color_id = int(
            np.argmin(np.sum(
                (palette_np - img_rgb[cy, cx].astype(np.float32) / 255.0) ** 2,
                axis=1,
            ))
        )
        bw = rng.integers(0, 256)
        dx = int(rng.uniform(-W * explore_r, W * explore_r))
        dy = int(rng.uniform(-H * explore_r, H * explore_r))
        x1 = np.clip(cx + dx, 0, W - 1); y1 = np.clip(cy + dy, 0, H - 1)
        genomes[i] = [
            color_id, bw,
            int(cx / (W - 1) * 255), int(cy / (H - 1) * 255),
            int(x1  / (W - 1) * 255), int(y1  / (H - 1) * 255),
        ]

    genomes[n_seed:] = rng.integers(0, 256, size=(P - n_seed, N_GENES), dtype=np.int32)
    return genomes

# ── Painter class ──────────────────────────────────────────────────────────────

class Painter:
    """
    GPU-accelerated impasto painter using a genetic algorithm.

    All particles start at z = h + r (floor level) with black RGB, so they are
    invisible against a black background without any masking. The GA finds brush
    strokes that minimise the error to a target image; each stroke colours the
    particles it covers and raises their z on subsequent re-paints (impasto).

    Interface compatible with Simulator: exposes xyz, rgb, rot, r, particles.
    rgb is stored in [0,255] so it can be passed directly to raytrace_sphere.

    Workflow
    --------
    1. painter = Painter(...)
    2. painter.init_image(img_bgr)
    3. painter.init_brush(path)          # optional texture; falls back to synthetic
    4. painter.init_canvas()
    5. loop: painter.update()            # one stroke per call → dict or None when done
    6. Pass painter.xyz, painter.rgb, painter.r to raytrace_sphere / Grid.build
    """

    def __init__(
        self,
        canvas_w        : int   = 1024,
        canvas_h        : int   = 768,
        population      : int   = 1024,
        n_strokes       : int   = 1000,
        gens_per_stroke : int   = 60,
        elite_n         : int   = 20,
        mutation_rate   : float = 0.05,
        n_colors        : int   = 256,
        max_brush_hw    : int   = 32,
        stroke_alpha    : float = 0.95,
        seed_frac       : float = 0.6,
        explore_r       : float = 0.20,
        max_dim         : int   = 512,
        # z raise per re-paint expressed as a fraction of the particle radius
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

        # GA GPU arrays
        self.canvas_lo_a : wp.array | None = None
        self.canvas_hi_a : wp.array | None = None
        self.target_a    : wp.array | None = None
        self.residual_a  : wp.array | None = None
        self.fitness_a   : wp.array | None = None
        self.palette_a   : wp.array | None = None
        self.genome_a    : wp.array | None = None
        self.children_a  : wp.array | None = None
        self.seeds_a     : wp.array | None = None

        # 3D particle arrays — same interface as Simulator
        self.xyz       : wp.array | None = None  # (N,) vec3  world positions
        self.rgb       : wp.array | None = None  # (N,) vec3  [0,255]
        self.rot       : wp.array | None = None  # (N,) vec3  always (0,0,1)
        self.p_active  : wp.array | None = None  # (N,) int32 0=black, 1=painted
        self.r         : float = 0.0
        self.z_active  : float = 0.0
        self.impasto_dz: float = 0.0
        self.particles : int   = 0

    # ── Init ──────────────────────────────────────────────────────────────────

    def init_image(self, img_bgr: np.ndarray) -> None:
        """Load target image, scale-to-fill canvas aspect ratio, build k-means palette."""
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
        """Load (or synthesise) greyscale brush texture. Black = full paint."""
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
        """
        Allocate all GPU buffers. Call after init_image and init_brush.

        Particle layout
        ---------------
        N = H × W particles, one per canvas pixel.
        All start at z = h + r (floor level) with rgb = (0,0,0) (black).
        Black particles render invisibly against a black background — no masking
        or active-flag bookkeeping is needed by the raytracer.
        When a stroke covers a pixel the particle gets coloured and rises on
        subsequent re-paints (impasto accumulation).

        rgb is stored in [0,255] to match raytrace_sphere's uint8 convention.
        """
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

        # ── GA 2D buffers ─────────────────────────────────────────────────────
        canvas_lo = np.zeros((H,  W,  3), dtype=np.float32)
        canvas_hi = np.zeros((HH, HW, 3), dtype=np.float32)
        target_np = self.img_rgb.astype(np.float32) / 255.0

        self.canvas_lo_a = wp.array(canvas_lo, dtype=wp.float32,
                                    shape=(H,  W,  3), device=self.device)
        self.canvas_hi_a = wp.array(canvas_hi, dtype=wp.float32,
                                    shape=(HH, HW, 3), device=self.device)
        self.target_a    = wp.array(target_np, dtype=wp.float32,
                                    shape=(H,  W,  3), device=self.device)
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

        # ── 3D particle grid ──────────────────────────────────────────────────
        N = H * W
        self.particles = N

        h_cell         = 1.0 / max(W, H)
        self.r         = 0.5 * h_cell
        self.z_active  = h_cell + self.r          # floor level, same as sim.py
        self.impasto_dz = self.impasto_dz_frac * self.r

        px_idx = np.arange(W, dtype=np.float32)
        py_idx = np.arange(H, dtype=np.float32)
        px_grid, py_grid = np.meshgrid(px_idx, py_idx)   # (H, W)

        # All particles at floor level, black — invisible against black background
        x_world = (px_grid + 0.5) * h_cell
        y_world = (py_grid + 0.5) * h_cell
        z_world = np.full((H, W), self.z_active, dtype=np.float32)

        xyz_np = np.stack([x_world, y_world, z_world], axis=-1).reshape(-1, 3)
        xyz_np = xyz_np.astype(np.float32)
        rgb_np = np.zeros((N, 3), dtype=np.float32)          # black [0,255]
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
        """
        Paint one brushstroke.  Returns a stroke info dict, or None when done.
        Each call:
          a) Runs GA to find the best stroke for the current residual map
          b) Colours / re-paints particles in the stroke OBB
          c) Syncs canvas_lo for the next GA iteration
          d) Updates residual
        """
        if self._step >= self.n_strokes:
            return None
        self._step += 1
        return self._paint_stroke()

    # ── Output ────────────────────────────────────────────────────────────────

    def get_canvas_bgr(self) -> np.ndarray:
        """Hi-res textured 2D preview as uint8 BGR."""
        rgb = (self.canvas_hi_a.numpy() * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def get_canvas_lo_bgr(self) -> np.ndarray:
        """Lo-res particle-colour 2D canvas as uint8 BGR (debug / error overlay)."""
        rgb = (self.canvas_lo_a.numpy() * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def get_error_bgr(self) -> np.ndarray:
        """Residual heat-map as BGR (bright = high error)."""
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

    def _paint_stroke(self) -> dict:
        """
        Run one GA-optimised stroke and apply it to the particle layer.

        GA fitness is evaluated on canvas_lo (OBB blending, consistent with
        the particle application), then the winning stroke is written to the
        3D particle array (p_xyz, rgb) and canvas_lo simultaneously.
        """
        residual_np = self.residual_a.numpy()
        genome_np   = _seed_population(
            residual_np, self.palette_np, self.img_rgb,
            self.P, self.H, self.W,
            self.max_hw, self.seed_frac, self.explore_r, self.rng,
        )
        self.genome_a = wp.array(genome_np, dtype=wp.int32,
                                 shape=(self.P, N_GENES), device=self.device)

        best_f     = -1e18
        best_genes = genome_np[0].copy()

        for _ in range(self.gens_per_stroke):
            wp.launch(kernel_eval_stroke_residual, dim=self.P,
                      inputs=[self.genome_a, self.palette_a, self.canvas_lo_a,
                              self.target_a, self.residual_a, self.fitness_a,
                              self.H, self.W, self.max_hw, self.alpha],
                      device=self.device)

            fit_np = self.fitness_a.numpy()
            order  = np.argsort(-fit_np)
            if fit_np[order[0]] > best_f:
                best_f     = float(fit_np[order[0]])
                best_genes = self.genome_a.numpy()[order[0]].copy()

            genome_np = self.genome_a.numpy()[order]
            self.genome_a = wp.array(genome_np, dtype=wp.int32,
                                     shape=(self.P, N_GENES), device=self.device)
            wp.launch(kernel_next_gen, dim=self.P,
                      inputs=[self.genome_a, self.children_a, self.seeds_a,
                              N_GENES, GENE_MAX, self.elite_n, self.P, self.mut_rate_k],
                      device=self.device)
            self.genome_a, self.children_a = self.children_a, self.genome_a

        # ── decode best genome ─────────────────────────────────────────────────
        g        = best_genes
        color_id = int(g[0]) % self.palette_np.shape[0]
        bw_gene  = int(g[1])

        x0_lo = float(g[2]) / 255.0 * float(self.W - 1)
        y0_lo = float(g[3]) / 255.0 * float(self.H - 1)
        x1_lo = float(g[4]) / 255.0 * float(self.W - 1)
        y1_lo = float(g[5]) / 255.0 * float(self.H - 1)
        hw_lo = 1.0 + float(bw_gene) / 255.0 * float(self.max_hw - 1)
        seg_lo = float(np.hypot(x1_lo - x0_lo, y1_lo - y0_lo))

        s      = self.scale
        x0_hi  = x0_lo * s;  y0_hi = y0_lo * s
        x1_hi  = x1_lo * s;  y1_hi = y1_lo * s
        hw_hi  = hw_lo * s
        seg_hi = seg_lo * s

        cr = float(self.palette_np[color_id, 0])
        cg = float(self.palette_np[color_id, 1])
        cb = float(self.palette_np[color_id, 2])

        # ── apply to 3D particles + canvas_lo ─────────────────────────────────
        tex_lo_a = self._make_tex_array(seg_lo, hw_lo)
        wp.launch(k_apply_stroke_to_particles, dim=self.H * self.W,
                  inputs=[
                      x0_lo, y0_lo, x1_lo, y1_lo,
                      hw_lo, self.alpha, cr, cg, cb,
                      tex_lo_a,
                      self.p_active, self.xyz, self.rgb,
                      self.canvas_lo_a,
                      self.H, self.W,
                      self.impasto_dz,
                  ],
                  device=self.device)

        # ── recompute residual in the stroke bounding box ─────────────────────
        wp.launch(kernel_update_residual, dim=self.H * self.W,
                  inputs=[self.canvas_lo_a, self.target_a, self.residual_a,
                          x0_lo, y0_lo, x1_lo, y1_lo, hw_lo,
                          self.H, self.W],
                  device=self.device)

        # ── hi-res 2D textured preview (canvas_hi only, for get_canvas_bgr) ──
        tex_hi_a = self._make_tex_array(seg_hi, hw_hi)
        wp.launch(kernel_paint_stroke_textured, dim=self.HH * self.HW,
                  inputs=[self.canvas_hi_a, tex_hi_a, cr, cg, cb,
                          x0_hi, y0_hi, x1_hi, y1_hi, hw_hi, self.alpha,
                          self.HH, self.HW],
                  device=self.device)

        mse = float(self.residual_a.numpy().sum()) / (self.H * self.W * 3)
        stroke = dict(
            step=self._step,
            color_id=color_id, color_rgb=(cr, cg, cb),
            hw=hw_lo, x0=x0_lo, y0=y0_lo, x1=x1_lo, y1=y1_lo,
            seg=seg_lo, active_particles=self.active_count,
            genes=best_genes.tolist(), mse=mse,
        )
        self.stroke_list.append(stroke)
        return stroke
