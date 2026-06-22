import cv2
import numpy as np
import warp as wp
wp.init()    

@wp.func
def clamp_position(
    xyz: wp.vec3,
    h: float, G: wp.vec3i
) -> wp.vec3:
    return wp.vec3(
        wp.clamp(xyz[0], h, float(G.x-1)*h),
        wp.clamp(xyz[1], h, float(G.y-1)*h),
        wp.clamp(xyz[2], h, float(G.z-1)*h)
    )

@wp.func
def trilerp(
    f: wp.array(dtype=float),
    x: float, y: float, z: float,
    G: wp.vec3i
) -> float:
    i0 = wp.clamp(int(x), 0, G.x-2); i1 = i0+1
    j0 = wp.clamp(int(y), 0, G.y-2); j1 = j0+1
    k0 = wp.clamp(int(z), 0, G.z-2); k1 = k0+1
    s1 = x-float(i0); s0 = 1.0-s1
    t1 = y-float(j0); t0 = 1.0-t1
    r1 = z-float(k0); r0 = 1.0-r1
    S = G.x; P = G.x*G.y
    return (s0*(t0*(r0*f[i0+S*j0+P*k0] + r1*f[i0+S*j0+P*k1]) +
                t1*(r0*f[i0+S*j1+P*k0] + r1*f[i0+S*j1+P*k1])) +
            s1*(t0*(r0*f[i1+S*j0+P*k0] + r1*f[i1+S*j0+P*k1]) +
                t1*(r0*f[i1+S*j1+P*k0] + r1*f[i1+S*j1+P*k1])))

@wp.func
def sample_velocity(
    xyz: wp.vec3,
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    h: float, G: wp.vec3i
) -> wp.vec3:
    x = xyz[0]/h
    y = xyz[1]/h
    z = xyz[2]/h
    return wp.vec3(
        trilerp(u, x, y, z, G),
        trilerp(v, x, y, z, G),
        trilerp(w, x, y, z, G)
    )

@wp.kernel
def k_add_source(x: wp.array(dtype=float), s: wp.array(dtype=float), dt: float):
    tid = wp.tid()
    x[tid] = x[tid] + dt * s[tid]

@wp.kernel
def k_advect(
    d: wp.array(dtype=float), d0: wp.array(dtype=float),
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    dt0: float, G: wp.vec3i
):
    i = wp.tid() % G.x
    j = (wp.tid() // G.x) % G.y
    k = wp.tid() // (G.x*G.y)
    if i < 1 or i >= G.x-1 or j < 1 or j >= G.y-1 or k < 1 or k >= G.z-1:
        return
    idx = i + G.x*j + G.x*G.y*k
    x = wp.clamp(float(i) - dt0*u[idx], 0.5, float(G.x-1)+0.5)
    y = wp.clamp(float(j) - dt0*v[idx], 0.5, float(G.y-1)+0.5)
    z = wp.clamp(float(k) - dt0*w[idx], 0.5, float(G.z-1)+0.5)
    d[idx] = trilerp(d0, x, y, z, G)

@wp.kernel
def k_divergence(
    div: wp.array(dtype=float),
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    h: float, G: wp.vec3i
):
    i = wp.tid() % G.x
    j = (wp.tid() // G.x) % G.y
    k = wp.tid() // (G.x*G.y)
    if i < 1 or i >= G.x-1 or j < 1 or j >= G.y-1 or k < 1 or k >= G.z-1:
        return
    S = G.x; P = G.x*G.y; idx = i + S*j + P*k
    div[idx] = -0.5 * h * (u[idx+1]-u[idx-1] + v[idx+S]-v[idx-S] + w[idx+P]-w[idx-P])

@wp.kernel
def k_solve_pressure(
    p: wp.array(dtype=float), div: wp.array(dtype=float),
    G: wp.vec3i
):
    i = wp.tid() % G.x
    j = (wp.tid() // G.x) % G.y
    k = wp.tid() // (G.x*G.y)
    if i < 1 or i >= G.x-1 or j < 1 or j >= G.y-1 or k < 1 or k >= G.z-1:
        return
    S = G.x; P = G.x*G.y; idx = i + S*j + P*k
    p[idx] = (div[idx] + p[idx-1]+p[idx+1] + p[idx-S]+p[idx+S] + p[idx-P]+p[idx+P]) / 6.0

@wp.kernel
def k_solve_pressure_rb(
    p: wp.array(dtype=float), div: wp.array(dtype=float),
    G: wp.vec3i, color: int
):
    i = wp.tid() % G.x
    j = (wp.tid() // G.x) % G.y
    k = wp.tid() // (G.x*G.y)
    if i < 1 or i >= G.x-1 or j < 1 or j >= G.y-1 or k < 1 or k >= G.z-1:
        return
    if (i + j + k) % 2 != color:
        return
    S = G.x; P = G.x*G.y; idx = i + S*j + P*k
    p[idx] = (div[idx] + p[idx-1]+p[idx+1] + p[idx-S]+p[idx+S] + p[idx-P]+p[idx+P]) / 6.0

@wp.kernel
def k_subtract_gradient(
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    p: wp.array(dtype=float),
    h: float, G: wp.vec3i
):
    i = wp.tid() % G.x
    j = (wp.tid() // G.x) % G.y
    k = wp.tid() // (G.x*G.y)
    if i < 1 or i >= G.x-1 or j < 1 or j >= G.y-1 or k < 1 or k >= G.z-1:
        return
    S = G.x; P = G.x*G.y; idx = i + S*j + P*k
    u[idx] -= 0.5 * (p[idx+1]-p[idx-1]) / h
    v[idx] -= 0.5 * (p[idx+S]-p[idx-S]) / h
    w[idx] -= 0.5 * (p[idx+P]-p[idx-P]) / h

@wp.kernel
def k_advect_particles_rk2(
    xyz: wp.array(dtype=wp.vec3),
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float), invmass: wp.array(dtype=float),
    dt: float,
    h: float, G: wp.vec3i
):
    pid = wp.tid()
    p = xyz[pid]
    v1 = sample_velocity(p, u, v, w, h, G)
    v2 = sample_velocity(p + v1*(dt*0.5), u, v, w, h, G)
    xyz[pid] = clamp_position(p + invmass[pid] * v2*dt, h, G)

@wp.kernel
def k_inject_mouse(
    u_prev: wp.array(dtype=float), v_prev: wp.array(dtype=float), w_prev: wp.array(dtype=float),
    m: wp.vec3, mv: wp.vec3, mradius: float, strength: float,
    G: wp.vec3i
):
    tid = wp.tid()
    i = tid % G.x
    j = (tid // G.x) % G.y
    k = tid // (G.x*G.y)
    wx = (float(i) - 0.5) / float(G.x)
    wy = (float(j) - 0.5) / float(G.y)
    dx = wx - m.x
    dy = wy - m.y
    dist2 = dx*dx + dy*dy
    if dist2 < mradius * mradius:
        q  = wp.sqrt(dist2) / mradius
        wt = (1.0 - q) * (1.0 - q)
        u_prev[tid] += mv.x * strength * wt
        v_prev[tid] += mv.y * strength * wt
        w_prev[tid] += mv.z * strength * wt

@wp.func
def smoothstep01(x: float) -> float:
    t = wp.clamp(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

@wp.func
def sample_bands_linear(bands: wp.array(dtype=float), n: int, u: float) -> float:
    uu = wp.clamp(u, 0.0, 0.9999)
    x = uu * float(n - 1)
    i0 = int(x)
    i1 = wp.min(i0 + 1, n - 1)
    t = x - float(i0)
    return (1.0 - t) * bands[i0] + t * bands[i1]

@wp.kernel
def k_inject_audio_field(
    u_prev: wp.array(dtype=float),
    v_prev: wp.array(dtype=float),
    w_prev: wp.array(dtype=float),
    bands: wp.array(dtype=float),
    num_bands: int,
    flux: float,
    strength: float,
    G: wp.vec3i
):
    tid = wp.tid()
    i = tid % G.x
    j = (tid // G.x) % G.y
    k = tid // (G.x * G.y)

    x = (float(i) + 0.5) / float(G.x)
    y = (float(j) + 0.5) / float(G.y)
    z = (float(k) + 0.5) / float(G.z)

    fu = wp.clamp(x + 0.15 * (z - 0.5), 0.0, 1.0)

    a0 = sample_bands_linear(bands, num_bands, fu)

    eps = 1.0 / float(num_bands)
    aL = sample_bands_linear(bands, num_bands, wp.clamp(fu - eps, 0.0, 1.0))
    aR = sample_bands_linear(bands, num_bands, wp.clamp(fu + eps, 0.0, 1.0))
    dax = aR - aL

    bottom = smoothstep01(1.0 - y)
    top = smoothstep01(y)
    midz = 1.0 - 2.0 * wp.abs(z - 0.5)

    phase = 0.5 + 0.5 * wp.sin(7.0 * x + 8.0 * y + 6.0 * z)
    amp = a0 * (0.65 + 0.35 * phase)

    ux = 0.0
    vy = 0.0
    wz = 0.0

    # weaker upward injection, mostly near lower half
    vy += strength * 0.45 * amp * bottom * (1.0 + 0.5 * flux)

    # explicit downward return in upper half to avoid ceiling parking
    vy -= strength * 0.38 * amp * top

    # lateral spread from spectral gradient
    ux += strength * 1.35 * dax * (0.6 + 0.4 * bottom)

    # depth circulation
    wz += strength * 0.60 * amp * (z - 0.5) * (0.5 + 0.5 * bottom)

    # slight recentering
    ux += strength * (-0.10) * (x - 0.5) * a0
    wz += strength * (-0.08) * (z - 0.5) * a0

    # highs create finer side/depth shimmer, not extra lift
    hi = sample_bands_linear(bands, num_bands, wp.clamp(0.7 + 0.3 * x, 0.0, 1.0))
    ux += strength * 0.30 * hi * midz * (0.5 - z)
    wz += strength * 0.24 * hi * (x - 0.5)

    u_prev[tid] += ux
    v_prev[tid] += vy
    w_prev[tid] += wz

@wp.kernel
def k_inject_gradient(
    u_prev: wp.array(dtype=float), v_prev: wp.array(dtype=float), w_prev: wp.array(dtype=float),
    grad_x: wp.array(dtype=float), grad_y: wp.array(dtype=float), lum: wp.array(dtype=float),
    strength: float, G: wp.vec3i, mode: int
):
    i = wp.tid() % G.x
    j = (wp.tid() // G.x) % G.y
    k = wp.tid() // (G.x*G.y)

    if i < 1 or i >= G.x-1 or j < 1 or j >= G.y-1 or k < 1 or k >= G.z-1:    
        return
    id2 = i + G.x*j
    idx = i + G.x*j + G.x*G.y*k
    l = lum[id2]
    scale = float(k) / float(G.z)

    if mode == 1:
        dz = 0.0
        if scale < 0.5:
            dx = -grad_y[id2] * strength * (0.5 - scale) * 0.2
            dy = grad_x[id2] * strength * (0.5 - scale) * 0.2
            if l > 0.5:
                dz = strength * l * l
        elif scale < 0.9:
            dx = grad_y[id2] * strength * (0.5 - scale) * 0.4
            dy = -grad_x[id2] * strength * (0.5 - scale) * 0.4
            if l < 0.5:
                dz = -strength * (1.0 - l) * (1.0 - l) * 6.0
        else:
            dz = -strength
        u_prev[idx] += dx
        v_prev[idx] += dy
        w_prev[idx] += dz
        u_prev[idx] = wp.clamp(u_prev[idx], -0.5, 0.5)
        v_prev[idx] = wp.clamp(v_prev[idx], -0.5, 0.5)
        w_prev[idx] = wp.clamp(w_prev[idx], -0.5, 0.5)
    elif mode == 2:
        sign  = float(1) if i % 16 < 8 else float(-1)
        w_prev[idx] = sign * 10.0 * lum[id2] * strength * (0.5 - scale)

        if i % 16 < 8:
            v_prev[idx] -= wp.clamp(strength * l, -0.5, 0.5)
        else:
            if j < G.y//2:
                v_prev[idx] += wp.clamp(strength * l, -0.5, 0.5)
            else:
                u_prev[idx] -= wp.clamp(strength * l, -0.5, 0.5)
        u_prev[idx] = wp.clamp(u_prev[idx], -0.5, 0.5)
        v_prev[idx] = wp.clamp(v_prev[idx], -0.5, 0.5)
        if scale < 0.9:
            w_prev[idx] = wp.clamp(w_prev[idx], -0.5, 0.5)
        else:
            w_prev[idx] = -strength
    elif mode == 3:
        dx = grad_x[id2] * strength * l * (0.5 - scale) * 2.0
        dy = grad_y[id2] * strength * l * (0.5 - scale) * 2.0
        dz = strength * l * (0.5 - scale) * 4.0
        u_prev[idx] += dx
        v_prev[idx] += dy
        w_prev[idx] += dz
        u_prev[idx] = wp.clamp(u_prev[idx], -0.5, 0.5)
        v_prev[idx] = wp.clamp(v_prev[idx], -0.5, 0.5)
        w_prev[idx] = wp.clamp(w_prev[idx], -0.5, 0.5)

@wp.kernel
def k_update_rotation(
    xyz: wp.array(dtype=wp.vec3), xyz_prior: wp.array(dtype=wp.vec3),
    rot: wp.array(dtype=wp.vec3),
    velocity_threshold: float, lerp_factor: float
):
    tid = wp.tid()    
    p = xyz[tid]
    p_prior = xyz_prior[tid]
    dir = rot[tid]   

    d = p - p_prior    
    dist = wp.length(d)
    
    # if dist < velocity_threshold:
    #     n = wp.vec3(0.0, 0.0, 1.0) # Points to camera
    # else:
    #     n = d / dist # Normalize velocity direction

    if dist < velocity_threshold:
        n = dir  # hold current orientation, lerp target == current == no change
    else:
        n = d / dist

    # Smoothly interpolate
    blended = dir + (n - dir) * lerp_factor
    rot[tid] = wp.normalize(blended)

@wp.kernel
def k_constraint_force(
    xyz: wp.array(dtype=wp.vec3), xyz_corr: wp.array(dtype=wp.vec3),
    invmass: wp.array(dtype=float),
    next: wp.array(dtype=int), l0: float, compliance: float, dt: float
):
    id0 = wp.tid()
    w0 = invmass[id0]
    p0 = xyz[id0]

    id1 = next[id0]
    if id1 < 0:
        return
    w1 = invmass[id1]
    w = w0 + w1
    if w == 0.0:
        return
    p1 = xyz[id1]
    d = p1 - p0
    l = wp.sqrt(d.x * d.x + d.y * d.y + d.z * d.z)
    if l <= 0.0:
        return    
    n = d / l
    alpha = compliance / dt / dt
    displace = n * (l - l0) / (w + alpha)
    wp.atomic_add(xyz_corr, id0, w0 * displace)
    wp.atomic_sub(xyz_corr, id1, w1 * displace)

@wp.kernel
def k_add_corrections(
    xyz: wp.array(dtype=wp.vec3), xyz_corr: wp.array(dtype=wp.vec3),
    jacobi: float
):
    tid = wp.tid()    
    xyz[tid] += xyz_corr[tid] * jacobi

@wp.func
def lum(image: wp.array(dtype=wp.uint8), idx: int) -> float:
    r = float(image[idx + 2])
    g = float(image[idx + 1])
    b = float(image[idx])
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0

@wp.kernel
def k_goal_force(
    xyz: wp.array(dtype=wp.vec3),
    xyz_goal: wp.array(dtype=wp.vec3),
    strength: float
):
    tid = wp.tid()

    d = xyz_goal[tid] - xyz[tid]
    if wp.length(d) < 1e-4:
        xyz[tid] = xyz_goal[tid]
    else:
        xyz[tid] += d * strength

@wp.kernel
def k_base_id(
    rgb: wp.array(dtype=wp.vec3), color_id: wp.array(dtype=int)
):
    tid = wp.tid()

    r = int(rgb[tid].x)
    g = int(rgb[tid].y)
    b = int(rgb[tid].z)

    lum = int(0.2126 * float(r) + 0.7152 * float(g) + 0.0722 * float(b))
    color_id[tid] = lum

@wp.kernel
def k_color_id(
    pixels: wp.array(dtype=wp.uint8), IMAGE_W: int, IMAGE_H: int,
    xyz_base: wp.array(dtype=wp.vec3), rgb: wp.array(dtype=wp.vec3),
    color_id: wp.array(dtype=int),
    h: float, G: wp.vec3i
):
    tid = wp.tid()

    p = xyz_base[tid]
    scale = float(IMAGE_W)
    ix = wp.clamp(int(p.x * scale), 0, IMAGE_W - 1)
    iy = wp.clamp(int((h * float(G.y) - p.y) * scale), 0, IMAGE_H - 1)

    idx = 3 * (ix + iy * IMAGE_W)
    r = int(pixels[idx + 2])
    g = int(pixels[idx + 1])
    b = int(pixels[idx])

    rgb[tid] = wp.vec3(float(r), float(g), float(b))

    lum = int(0.2126 * float(r) + 0.7152 * float(g) + 0.0722 * float(b))
    color_id[tid] = lum

@wp.kernel
def k_new_image_sorted(
    pixels: wp.array(dtype=wp.uint8), blurred: wp.array(dtype=wp.uint8), IMAGE_W: int, IMAGE_H: int,
    xyz_base: wp.array(dtype=wp.vec3), xyz_goal: wp.array(dtype=wp.vec3), rgb: wp.array(dtype=wp.vec3), invmass: wp.array(dtype=float), base_id: wp.array(dtype=int), goal_id: wp.array(dtype=int),
    h: float, G: wp.vec3i, depth_factor: float
):
    tid = wp.tid()

    bid = base_id[tid]
    gid = goal_id[tid]
    p = xyz_base[gid]
    scale = float(IMAGE_W)
    ix = wp.clamp(int(p.x * scale), 0, IMAGE_W - 1)
    iy = wp.clamp(int((h * float(G.y) - p.y) * scale), 0, IMAGE_H - 1)
    idc  = 3 * (ix  + iy  * IMAGE_W)
    
    rgb[bid] = wp.vec3(
        float(pixels[idc + 2]),
        float(pixels[idc + 1]),
        float(pixels[idc])
    )
    lumc = lum(blurred, idc)
    z_max = depth_factor * h * float(G.z - 2)
    xyz_goal[bid] = wp.vec3(p.x, p.y, p.z + z_max * lumc)
    invmass[bid] = 0.1 + 0.9 * lumc

    # if ix + 1 > IMAGE_SIZE - 1:
    #     lumr = 1.0
    # else:
    #     idcr = 3 * (ix + 1 + iy  * IMAGE_SIZE)
    #     lumr = lum(blurred, idcr)

    # if iy + 1 > IMAGE_SIZE - 1:
    #     lumu = 1.0
    # else:
    #     idcu = 3 * (ix  + (iy + 1) * IMAGE_SIZE)
    #     lumu = lum(blurred, idcu)

    # dzdx = (lumr - lumc) * z_max * float(IMAGE_SIZE)
    # dzdy = (lumu - lumc) * z_max * float(IMAGE_SIZE)

    # nx = -dzdx
    # ny = dzdy
    # nz = 1.0
    # length = wp.sqrt(nx*nx + ny*ny + nz*nz) + 1e-6
    # rot[tid] = wp.vec3(nx / length, ny / length, nz / length)

@wp.kernel
def k_new_image(
    pixels: wp.array(dtype=wp.uint8), blurred: wp.array(dtype=wp.uint8), IMAGE_W: int, IMAGE_H: int,
    xyz_base: wp.array(dtype=wp.vec3), xyz_goal: wp.array(dtype=wp.vec3), rgb: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.vec3), invmass: wp.array(dtype=float),
    h: float, G: wp.vec3i, depth_factor: float
):
    tid = wp.tid()

    p = xyz_base[tid]
    scale = float(IMAGE_W)
    ix = wp.clamp(int(p.x * scale), 0, IMAGE_W - 1)
    iy = wp.clamp(int((h * float(G.y) - p.y) * scale), 0, IMAGE_H - 1)
    idc  = 3 * (ix  + iy  * IMAGE_W)
    
    rgb[tid] = wp.vec3(
        float(pixels[idc + 2]),
        float(pixels[idc + 1]),
        float(pixels[idc])
    )
    lumc = lum(blurred, idc)
    z_max = depth_factor * h * float(G.z - 2)
    xyz_goal[tid] = wp.vec3(p.x, p.y, p.z + z_max * lumc)
    rot[tid] = wp.vec3(0.0, 0.0, 1.0)
    invmass[tid] = 0.1 + 0.9 * lumc

@wp.kernel
def k_sphere_boundaries(
    xyz: wp.array(dtype=wp.vec3),
    xyz_prior: wp.array(dtype=wp.vec3),
    wc: wp.vec3,
    wr: wp.vec3,
    push: float,       # bounded normal correction per step
    damping: float,    # global displacement damping in [0, 1]
    swirl: float,      # bounded tangential drift near surface
    band: float,       # soft band around surface in scaled-space
    flow_axis: wp.vec3
):
    tid = wp.tid()

    p = xyz[tid]
    p_old = p
    p_prior = xyz_prior[tid]

    v = p - p_prior
    d = p - wc

    q = wp.vec3(
        d[0] / wr[0],
        d[1] / wr[1],
        d[2] / wr[2],
    )

    s = wp.dot(q, q) - 1.0

    g = wp.vec3(
        2.0 * d[0] / (wr[0] * wr[0]),
        2.0 * d[1] / (wr[1] * wr[1]),
        2.0 * d[2] / (wr[2] * wr[2]),
    )

    g_len = wp.length(g)
    if g_len > 1.0e-6:
        n = g / g_len

        alpha = 1.0
        surface_weight = 1.0
        if band > 1.0e-6:
            alpha = wp.min(wp.abs(s) / band, 1.0)
            surface_weight = 1.0 - alpha

        # one symmetric bounded normal push
        if s > 0.0:
            v -= n * (push * alpha)
        else:
            v += n * (push * alpha)

        # tangent direction
        vn = wp.dot(v, n) * n
        vt = v - vn
        vt_len = wp.length(vt)

        t = wp.vec3(0.0, 0.0, 0.0)
        if vt_len > 1.0e-6:
            t = vt / vt_len
        else:
            axis_tan = flow_axis - wp.dot(flow_axis, n) * n
            axis_tan_len = wp.length(axis_tan)

            if axis_tan_len > 1.0e-6:
                t = axis_tan / axis_tan_len
            else:
                ref = wp.vec3(1.0, 0.0, 0.0)
                if wp.abs(n[0]) > 0.9:
                    ref = wp.vec3(0.0, 1.0, 0.0)
                t = wp.normalize(wp.cross(n, ref))

        # one bounded tangential drift near the boundary
        v += t * (swirl * surface_weight)

        # one damping knob
        v *= wp.max(0.0, 1.0 - damping)

    p = p + v

    xyz[tid] = p
    xyz_prior[tid] = p_old

@wp.kernel
def k_torus_boundaries(
    xyz: wp.array(dtype=wp.vec3),
    xyz_prior: wp.array(dtype=wp.vec3),
    wc: wp.vec3,
    torus_axis: wp.vec3,   # should be normalized
    major_radius: float,   # R
    minor_radius: float,   # r
    push: float,           # bounded normal correction per step
    damping: float,        # global displacement damping in [0, 1]
    swirl: float,          # bounded tangential drift near surface
    band: float,           # soft band around torus field
    flow_axis: wp.vec3
):
    tid = wp.tid()

    p = xyz[tid]
    p_old = p
    p_prior = xyz_prior[tid]

    v = p - p_prior
    d = p - wc

    a = torus_axis
    d_parallel = wp.dot(d, a)
    d_perp = d - d_parallel * a
    d_perp_len = wp.length(d_perp)

    # torus implicit field
    ring_dist = d_perp_len - major_radius
    s = ring_dist * ring_dist + d_parallel * d_parallel - minor_radius * minor_radius

    # gradient of torus field
    g = wp.vec3(0.0, 0.0, 0.0)

    if d_perp_len > 1.0e-6:
        g = 2.0 * ring_dist * (d_perp / d_perp_len) + 2.0 * d_parallel * a
    else:
        # on the torus axis, the field is singular for the radial part;
        # fall back to axial direction
        g = 2.0 * d_parallel * a

    g_len = wp.length(g)
    if g_len > 1.0e-6:
        n = g / g_len

        alpha = 1.0
        surface_weight = 1.0
        if band > 1.0e-6:
            alpha = wp.min(wp.abs(s) / band, 1.0)
            surface_weight = 1.0 - alpha

        # symmetric bounded push toward torus surface
        if s > 0.0:
            v -= n * (push * alpha)
        else:
            v += n * (push * alpha)

        # tangent direction
        vn = wp.dot(v, n) * n
        vt = v - vn
        vt_len = wp.length(vt)

        t = wp.vec3(0.0, 0.0, 0.0)
        if vt_len > 1.0e-6:
            t = vt / vt_len
        else:
            axis_tan = flow_axis - wp.dot(flow_axis, n) * n
            axis_tan_len = wp.length(axis_tan)

            if axis_tan_len > 1.0e-6:
                t = axis_tan / axis_tan_len
            else:
                ref = wp.vec3(1.0, 0.0, 0.0)
                if wp.abs(n[0]) > 0.9:
                    ref = wp.vec3(0.0, 1.0, 0.0)
                t = wp.normalize(wp.cross(n, ref))

        v += t * (swirl * surface_weight)
        v *= wp.max(0.0, 1.0 - damping)

    p = p + v

    xyz[tid] = p
    xyz_prior[tid] = p_old

@wp.kernel
def k_world_boundaries(
    xyz: wp.array(dtype=wp.vec3), xyz_prior: wp.array(dtype=wp.vec3),
    r: float, h: float, G: wp.vec3i
):
    tid = wp.tid()
    p = xyz[tid]
    p_prior = xyz_prior[tid]

    x = p.x
    y = p.y
    z = p.z
    x_prior = p_prior.x
    y_prior = p_prior.y
    z_prior = p_prior.z

    margin = h + r
    restitution = 0.9

    if p.x < margin:
        vx = p.x - p_prior.x
        x = margin
        vx = restitution * wp.abs(vx)
        x_prior = x - vx
    if p.x > h * float(G.x) - margin:
        vx = p.x - p_prior.x
        x = h * float(G.x) - margin
        vx = -restitution * wp.abs(vx)
        x_prior = x - vx
    if p.y < margin:
        vy = p.y - p_prior.y
        y = margin
        vy = restitution * wp.abs(vy)
        y_prior = y - vy
    if p.y > h * float(G.y) - margin:
        vy = p.y - p_prior.y
        y = h * float(G.y) - margin
        vy = -restitution * wp.abs(vy)
        y_prior = y - vy
    if p.z < margin:
        vz = p.z - p_prior.z
        z = margin
        vz = restitution * wp.abs(vz)
        z_prior = z - vz
    if p.z > h * float(G.z) - margin:
        vz = p.z - p_prior.z
        z = h * float(G.z) - margin
        vz = -restitution * wp.abs(vz)
        z_prior = z - vz

    xyz[tid] = wp.vec3(x, y, z)
    xyz_prior[tid] = wp.vec3(x_prior, y_prior, z_prior)

@wp.kernel
def k_particles_to_grid(
    xyz: wp.array(dtype=wp.vec3),
    xyz_prior: wp.array(dtype=wp.vec3),
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    weight: wp.array(dtype=float),
    dt: float,
    h: float, G: wp.vec3i
):
    pid = wp.tid()

    # Particle velocity from Verlet positions
    vel = (xyz[pid] - xyz_prior[pid]) / dt

    p = xyz[pid]
    gx = p / h
    x = wp.clamp(gx[0], 0.5, float(G.x - 1) + 0.5)
    y = wp.clamp(gx[1], 0.5, float(G.y - 1) + 0.5)
    z = wp.clamp(gx[2], 0.5, float(G.z - 1) + 0.5)

    i0 = int(x); i1 = wp.min(i0 + 1, G.x - 1)
    j0 = int(y); j1 = wp.min(j0 + 1, G.y - 1)
    k0 = int(z); k1 = wp.min(k0 + 1, G.z - 1)

    s1 = x - float(i0); s0 = 1.0 - s1
    t1 = y - float(j0); t0 = 1.0 - t1
    r1 = z - float(k0); r0 = 1.0 - r1

    idx000 = i0 + G.x*j0 + G.x*G.y*k0
    idx001 = i0 + G.x*j0 + G.x*G.y*k1
    idx010 = i0 + G.x*j1 + G.x*G.y*k0
    idx011 = i0 + G.x*j1 + G.x*G.y*k1
    idx100 = i1 + G.x*j0 + G.x*G.y*k0
    idx101 = i1 + G.x*j0 + G.x*G.y*k1
    idx110 = i1 + G.x*j1 + G.x*G.y*k0
    idx111 = i1 + G.x*j1 + G.x*G.y*k1

    w000=s0*t0*r0; w001=s0*t0*r1; w010=s0*t1*r0; w011=s0*t1*r1
    w100=s1*t0*r0; w101=s1*t0*r1; w110=s1*t1*r0; w111=s1*t1*r1

    wp.atomic_add(u, idx000, vel[0]*w000); wp.atomic_add(u, idx001, vel[0]*w001)
    wp.atomic_add(u, idx010, vel[0]*w010); wp.atomic_add(u, idx011, vel[0]*w011)
    wp.atomic_add(u, idx100, vel[0]*w100); wp.atomic_add(u, idx101, vel[0]*w101)
    wp.atomic_add(u, idx110, vel[0]*w110); wp.atomic_add(u, idx111, vel[0]*w111)

    wp.atomic_add(v, idx000, vel[1]*w000); wp.atomic_add(v, idx001, vel[1]*w001)
    wp.atomic_add(v, idx010, vel[1]*w010); wp.atomic_add(v, idx011, vel[1]*w011)
    wp.atomic_add(v, idx100, vel[1]*w100); wp.atomic_add(v, idx101, vel[1]*w101)
    wp.atomic_add(v, idx110, vel[1]*w110); wp.atomic_add(v, idx111, vel[1]*w111)

    wp.atomic_add(w, idx000, vel[2]*w000); wp.atomic_add(w, idx001, vel[2]*w001)
    wp.atomic_add(w, idx010, vel[2]*w010); wp.atomic_add(w, idx011, vel[2]*w011)
    wp.atomic_add(w, idx100, vel[2]*w100); wp.atomic_add(w, idx101, vel[2]*w101)
    wp.atomic_add(w, idx110, vel[2]*w110); wp.atomic_add(w, idx111, vel[2]*w111)

    wp.atomic_add(weight, idx000, w000); wp.atomic_add(weight, idx001, w001)
    wp.atomic_add(weight, idx010, w010); wp.atomic_add(weight, idx011, w011)
    wp.atomic_add(weight, idx100, w100); wp.atomic_add(weight, idx101, w101)
    wp.atomic_add(weight, idx110, w110); wp.atomic_add(weight, idx111, w111)

@wp.kernel
def k_normalize_grid_velocity(
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    weight: wp.array(dtype=float)
):
    tid = wp.tid()
    wt = weight[tid]
    if wt > 1e-6:
        u[tid] = u[tid] / wt
        v[tid] = v[tid] / wt
        w[tid] = w[tid] / wt

@wp.kernel
def k_init_particle_vel(
    xyz: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    h: float, G: wp.vec3i
):
    pid = wp.tid()
    vel[pid] = sample_velocity(xyz[pid], u, v, w, h, G)

@wp.kernel
def k_flip_update(
    xyz: wp.array(dtype=wp.vec3),
    xyz_prior: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    u_new: wp.array(dtype=float), v_new: wp.array(dtype=float), w_new: wp.array(dtype=float),
    u_old: wp.array(dtype=float), v_old: wp.array(dtype=float), w_old: wp.array(dtype=float),
    alpha: float, dt: float, h: float, G: wp.vec3i
):
    pid = wp.tid()
    p = xyz_prior[pid]

    vel_pic = sample_velocity(p, u_new, v_new, w_new, h, G)
    vel_old_grid = sample_velocity(p, u_old, v_old, w_old, h, G)
    vel_du = vel_pic - vel_old_grid
    vel_flip = vel[pid] + vel_du

    vel_new = (1.0 - alpha) * vel_pic + alpha * vel_flip
    vel[pid] = vel_new
    xyz[pid] = clamp_position(p + vel_new * dt, h, G)

@wp.kernel
def k_particles_to_grid_vel(
    xyz: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    weight: wp.array(dtype=float),
    h: float, G: wp.vec3i
):
    pid = wp.tid()

    p = xyz[pid]
    pv = vel[pid]

    gx = p / h
    x = wp.clamp(gx[0], 0.5, float(G.x - 1) + 0.5)
    y = wp.clamp(gx[1], 0.5, float(G.y - 1) + 0.5)
    z = wp.clamp(gx[2], 0.5, float(G.z - 1) + 0.5)

    i0 = int(x); i1 = wp.min(i0 + 1, G.x - 1)
    j0 = int(y); j1 = wp.min(j0 + 1, G.y - 1)
    k0 = int(z); k1 = wp.min(k0 + 1, G.z - 1)

    s1 = x - float(i0); s0 = 1.0 - s1
    t1 = y - float(j0); t0 = 1.0 - t1
    r1 = z - float(k0); r0 = 1.0 - r1

    idx000 = i0 + G.x*j0 + G.x*G.y*k0
    idx001 = i0 + G.x*j0 + G.x*G.y*k1
    idx010 = i0 + G.x*j1 + G.x*G.y*k0
    idx011 = i0 + G.x*j1 + G.x*G.y*k1
    idx100 = i1 + G.x*j0 + G.x*G.y*k0
    idx101 = i1 + G.x*j0 + G.x*G.y*k1
    idx110 = i1 + G.x*j1 + G.x*G.y*k0
    idx111 = i1 + G.x*j1 + G.x*G.y*k1

    w000=s0*t0*r0; w001=s0*t0*r1; w010=s0*t1*r0; w011=s0*t1*r1
    w100=s1*t0*r0; w101=s1*t0*r1; w110=s1*t1*r0; w111=s1*t1*r1

    wp.atomic_add(u, idx000, pv[0]*w000); wp.atomic_add(u, idx001, pv[0]*w001)
    wp.atomic_add(u, idx010, pv[0]*w010); wp.atomic_add(u, idx011, pv[0]*w011)
    wp.atomic_add(u, idx100, pv[0]*w100); wp.atomic_add(u, idx101, pv[0]*w101)
    wp.atomic_add(u, idx110, pv[0]*w110); wp.atomic_add(u, idx111, pv[0]*w111)

    wp.atomic_add(v, idx000, pv[1]*w000); wp.atomic_add(v, idx001, pv[1]*w001)
    wp.atomic_add(v, idx010, pv[1]*w010); wp.atomic_add(v, idx011, pv[1]*w011)
    wp.atomic_add(v, idx100, pv[1]*w100); wp.atomic_add(v, idx101, pv[1]*w101)
    wp.atomic_add(v, idx110, pv[1]*w110); wp.atomic_add(v, idx111, pv[1]*w111)

    wp.atomic_add(w, idx000, pv[2]*w000); wp.atomic_add(w, idx001, pv[2]*w001)
    wp.atomic_add(w, idx010, pv[2]*w010); wp.atomic_add(w, idx011, pv[2]*w011)
    wp.atomic_add(w, idx100, pv[2]*w100); wp.atomic_add(w, idx101, pv[2]*w101)
    wp.atomic_add(w, idx110, pv[2]*w110); wp.atomic_add(w, idx111, pv[2]*w111)

    wp.atomic_add(weight, idx000, w000); wp.atomic_add(weight, idx001, w001)
    wp.atomic_add(weight, idx010, w010); wp.atomic_add(weight, idx011, w011)
    wp.atomic_add(weight, idx100, w100); wp.atomic_add(weight, idx101, w101)
    wp.atomic_add(weight, idx110, w110); wp.atomic_add(weight, idx111, w111)

@wp.kernel
def k_update_vel_from_positions(
    xyz: wp.array(dtype=wp.vec3),
    xyz_prior: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    dt: float
):
    pid = wp.tid()
    vel[pid] = (xyz[pid] - xyz_prior[pid]) / dt

# Line fit

@wp.kernel
def k_sum_xyz(
    xyz: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=float),          # [count, sx, sy, sz]
):
    tid = wp.tid()
    p = xyz[tid]
    wp.atomic_add(out, 0, 1.0)
    wp.atomic_add(out, 1, p.x)
    wp.atomic_add(out, 2, p.y)
    wp.atomic_add(out, 3, p.z)

@wp.kernel
def k_compute_angle(
    xyz: wp.array(dtype=wp.vec3),
    centroid: wp.vec3,
    ax0: wp.vec3,                        # first  principal axis (unit)
    ax1: wp.vec3,                        # second principal axis (unit)
    angles: wp.array(dtype=float),       # output per-point angle in [0, 2π)
):
    tid = wp.tid()
    d = xyz[tid] - centroid
    u = wp.dot(d, ax0)
    v = wp.dot(d, ax1)
    a = wp.atan2(v, u)
    if a < 0.0:
        a = a + 6.2831853071795864
    angles[tid] = a

@wp.kernel
def k_accumulate_fourier(
    xyz: wp.array(dtype=wp.vec3),
    angles: wp.array(dtype=float),
    K: int,                              # number of harmonics
    out: wp.array(dtype=float),          # length 3*(2*K+1)  row = [X coeffs | Y coeffs | Z coeffs]
):
    tid = wp.tid()
    t = angles[tid]
    p = xyz[tid]
    n = float(xyz.shape[0])

    # stride between axis blocks
    stride = 2 * K + 1

    # a0 term
    wp.atomic_add(out, 0 * stride + 0, p.x / n)
    wp.atomic_add(out, 1 * stride + 0, p.y / n)
    wp.atomic_add(out, 2 * stride + 0, p.z / n)

    for k in range(1, K + 1):
        kf = float(k)
        c = wp.cos(kf * t)
        s = wp.sin(kf * t)
        scale = 2.0 / n

        # cos coefficients at positions 1..K
        wp.atomic_add(out, 0 * stride + k,         p.x * c * scale)
        wp.atomic_add(out, 1 * stride + k,         p.y * c * scale)
        wp.atomic_add(out, 2 * stride + k,         p.z * c * scale)

        # sin coefficients at positions K+1..2K
        wp.atomic_add(out, 0 * stride + K + k,     p.x * s * scale)
        wp.atomic_add(out, 1 * stride + K + k,     p.y * s * scale)
        wp.atomic_add(out, 2 * stride + K + k,     p.z * s * scale)

@wp.kernel
def k_eval_fourier_curve(
    coeffs: wp.array(dtype=float),       # 3*(2K+1)
    K: int,
    M: int,                              # number of output samples
    line_xyz: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    t = 6.2831853071795864 * float(tid) / float(M)
    stride = 2 * K + 1

    x = coeffs[0 * stride + 0]
    y = coeffs[1 * stride + 0]
    z = coeffs[2 * stride + 0]

    for k in range(1, K + 1):
        kf = float(k)
        c = wp.cos(kf * t)
        s = wp.sin(kf * t)

        x = x + coeffs[0 * stride + k] * c + coeffs[0 * stride + K + k] * s
        y = y + coeffs[1 * stride + k] * c + coeffs[1 * stride + K + k] * s
        z = z + coeffs[2 * stride + k] * c + coeffs[2 * stride + K + k] * s

    line_xyz[tid] = wp.vec3(x, y, z)

@wp.kernel
def k_build_line_next_closed(
    line_next: wp.array(dtype=int),
):
    """Closed loop: last point connects back to first."""
    tid = wp.tid()
    n = line_next.shape[0]
    line_next[tid] = (tid + 1) % n

class Simulator:
    def __init__(self,
        IMAGE_W=512, IMAGE_H=512, PIXELS_PER_CELL=4,
        G=[128, 128, 32], L=1, smooth=15,
        gradient_strength=0.1, pressure_steps=5, jacobi=0.1, dt=0.1, flip_alpha=0.95,
        num_bands=16
    ):
        self.G = wp.vec3i(*G)
        self.h = 1.0 / max(self.G.x, self.G.y)
        self.P = self.G * PIXELS_PER_CELL
        self.r = 0.5 * self.h / PIXELS_PER_CELL
        self.L = L

        self.IMAGE_W = IMAGE_W
        self.IMAGE_H = IMAGE_H
        self.smooth = smooth

        self.gradient_strength = gradient_strength
        self.pressure_steps = pressure_steps
        self.jacobi = jacobi
        self.dt = dt
        self.dt0 = dt * float(max(self.G.x, self.G.y, self.G.z))
        self.flip_alpha = flip_alpha

        x0 = self.h + self.r
        y0 = self.h + self.r
        z0 = self.h + self.r
        spacing = 2.0 * self.r
        PX_inner = self.P.x - 2 * PIXELS_PER_CELL
        PY_inner = self.P.y - 2 * PIXELS_PER_CELL
        N_layer = PX_inner * PY_inner

        jj_grid, ii_grid = np.meshgrid(np.arange(PY_inner), np.arange(PX_inner), indexing='ij')
        linear = jj_grid * PX_inner + ii_grid

        all_pos, all_nx, all_ny, all_nxx, all_nyy, all_ndu, all_ndd = [], [], [], [], [], [], []
        for k in range(self.L):
            offset = k * N_layer
            jz = 0.0#self.r * 0.1 * (np.random.rand(PY_inner, PX_inner) - 0.5)
            pos_k = np.stack([
                x0 + ii_grid * spacing,
                y0 + jj_grid * spacing,
                np.full((PY_inner, PX_inner), z0 + k * 2.0 * self.h) + jz,
            ], axis=-1).reshape(-1, 3).astype(np.float32)
            all_pos.append(pos_k)
            lin = linear + offset
            all_nx.append(np.where(ii_grid < PX_inner - 1, lin + 1, -1).ravel())
            all_ny.append(np.where(jj_grid < PY_inner - 1, lin + PX_inner, -1).ravel())
            all_nxx.append(np.where(ii_grid < PX_inner - 2, lin + 2, -1).ravel())
            all_nyy.append(np.where(jj_grid < PY_inner - 2, lin + 2 * PX_inner, -1).ravel())
            all_ndu.append(np.where((ii_grid < PX_inner-1) & (jj_grid < PY_inner-1), lin + 1 + PX_inner, -1).ravel())
            all_ndd.append(np.where((ii_grid < PX_inner-1) & (jj_grid > 0), lin + 1 - PX_inner, -1).ravel())

        pos_np = np.vstack(all_pos)
        n_x  = np.concatenate(all_nx).astype(np.int32)
        n_y  = np.concatenate(all_ny).astype(np.int32)
        n_xx = np.concatenate(all_nxx).astype(np.int32)
        n_yy = np.concatenate(all_nyy).astype(np.int32)
        n_du = np.concatenate(all_ndu).astype(np.int32)
        n_dd = np.concatenate(all_ndd).astype(np.int32)
        self.particles = len(pos_np)
        # Particle arrays
        self.xyz = wp.array(pos_np, dtype=wp.vec3, device="cuda")
        self.xyz_prior = wp.array(pos_np, dtype=wp.vec3, device="cuda")
        self.xyz_goal = wp.array(pos_np, dtype=wp.vec3, device="cuda")
        self.xyz_base = wp.array(pos_np, dtype=wp.vec3, device="cuda")
        self.xyz_corr = wp.zeros(self.particles, dtype=wp.vec3, device="cuda")
        self.vel = wp.zeros(self.particles, dtype=wp.vec3, device="cuda")
        self.rgb = wp.zeros(self.particles, dtype=wp.vec3, device="cuda")
        self.rot = wp.zeros(self.particles, dtype=wp.vec3, device="cuda")
        self.invmass = wp.ones(self.particles, dtype=float, device="cuda")
        self.radii = wp.zeros(self.particles, dtype=float, device="cuda")
        # Particle neighbors
        self.next_x = wp.array(n_x, dtype=int, device="cuda")
        self.next_y = wp.array(n_y, dtype=int, device="cuda")
        self.next_xx = wp.array(n_xx, dtype=int, device="cuda")
        self.next_yy = wp.array(n_yy, dtype=int, device="cuda")
        self.next_du = wp.array(n_du, dtype=int, device="cuda")
        self.next_dd = wp.array(n_dd, dtype=int, device="cuda")
        # Image buffers (GPU) — pre-allocated once; reused in new_image() via wp.copy
        self.pixels  = wp.zeros(self.IMAGE_W * self.IMAGE_H * 3, dtype=wp.uint8, device="cuda")
        self.blurred = wp.zeros(self.IMAGE_W * self.IMAGE_H * 3, dtype=wp.uint8, device="cuda")

        # Grid arrays
        self.G2 = self.G.x * self.G.y
        self.G3 = self.G.x * self.G.y * self.G.z
        self.grad_x = wp.zeros(self.G2, dtype=float, device="cuda")
        self.grad_y = wp.zeros(self.G2, dtype=float, device="cuda")
        self.lum = wp.zeros(self.G2, dtype=float, device="cuda")
        self.u      = wp.zeros(self.G3, dtype=float, device="cuda")
        self.v      = wp.zeros(self.G3, dtype=float, device="cuda")
        self.w      = wp.zeros(self.G3, dtype=float, device="cuda")
        self.u_prior = wp.zeros(self.G3, dtype=float, device="cuda")
        self.v_prior = wp.zeros(self.G3, dtype=float, device="cuda")
        self.w_prior = wp.zeros(self.G3, dtype=float, device="cuda")
        self.u_prev = wp.zeros(self.G3, dtype=float, device="cuda")
        self.v_prev = wp.zeros(self.G3, dtype=float, device="cuda")
        self.w_prev = wp.zeros(self.G3, dtype=float, device="cuda")
        self.div    = wp.zeros(self.G3, dtype=float, device="cuda")
        self.p      = wp.zeros(self.G3, dtype=float, device="cuda")
        self.p2g_weight = wp.zeros(self.G3, dtype=float, device="cuda")

        # CPU staging buffers — pre-allocated once to avoid per-call GPU malloc
        img_flat = self.IMAGE_W * self.IMAGE_H * 3
        self._pixels_np  = np.zeros(img_flat, dtype=np.uint8)
        self._blurred_np = np.zeros(img_flat, dtype=np.uint8)
        self._grad_x_np  = np.zeros(self.G2, dtype=np.float32)
        self._grad_y_np  = np.zeros(self.G2, dtype=np.float32)
        self._lum_np     = np.zeros(self.G2, dtype=np.float32)
        self._bands_np   = np.zeros(num_bands, dtype=np.float32)
        self._bands_gpu  = wp.zeros(num_bands, dtype=float, device="cuda")

        print(f"✓ SIM: simulator ready")
        print(f"    Image({self.IMAGE_W},{self.IMAGE_H})")
        print(f"    {self.particles:,} particles({PX_inner},{PY_inner},{L})")
        print(f"    {self.G3:,} cells in grid({self.G.x},{self.G.y},{self.G.z})")

    def _compute_gradients(self, img):
        img_grid = cv2.resize(img, (self.G.x, self.G.y), interpolation=cv2.INTER_AREA)

        lum = cv2.cvtColor(img_grid, cv2.COLOR_BGR2GRAY)
        luminance = lum.astype(np.float32) / 255.0

        grad_x = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)

        max_mag = np.hypot(grad_x, grad_y).max()
        if max_mag > 0:
            grad_x /= max_mag
            grad_y /= max_mag

        # Flip vertically so row 0 = bottom of image (matches GPU grid Y-up convention)
        np.copyto(self._grad_x_np, np.ascontiguousarray(np.flipud(grad_x)).ravel())
        np.copyto(self._grad_y_np, np.ascontiguousarray(np.flipud(grad_y)).ravel())
        np.copyto(self._lum_np,    np.ascontiguousarray(np.flipud(luminance)).ravel())

        wp.copy(self.grad_x, wp.array(self._grad_x_np, dtype=wp.float32, device="cpu"))
        wp.copy(self.grad_y, wp.array(self._grad_y_np, dtype=wp.float32, device="cpu"))
        wp.copy(self.lum,    wp.array(self._lum_np,    dtype=wp.float32, device="cpu"))

    def new_image(self, img, depth_factor=1.0):
        for arr in [self.u, self.v, self.w,
                    self.u_prev, self.v_prev, self.w_prev,
                    self.div, self.p]:
            arr.zero_()

        np.copyto(self._pixels_np, np.ascontiguousarray(img).ravel())
        wp.copy(self.pixels, wp.array(self._pixels_np, dtype=wp.uint8, device="cpu"))

        blurred = cv2.GaussianBlur(img, (self.smooth, self.smooth), 0)
        np.copyto(self._blurred_np, np.ascontiguousarray(blurred).ravel())
        wp.copy(self.blurred, wp.array(self._blurred_np, dtype=wp.uint8, device="cpu"))

        self._compute_gradients(blurred)

        wp.launch(k_new_image, dim=self.particles, inputs=[
            self.pixels, self.blurred, self.IMAGE_W, self.IMAGE_H,
            self.xyz_base, self.xyz_goal, self.rgb, self.rot, self.invmass,
            self.h, self.G, depth_factor
        ], device="cuda")

    def project(self, steps=None):
        if steps is None:
            steps = self.pressure_steps
        wp.launch(k_divergence, dim=self.G3, inputs=[
            self.div, self.u, self.v, self.w, self.h, self.G
        ], device="cuda")
        self.p.zero_()
        for _ in range(steps):
            wp.launch(k_solve_pressure_rb, dim=self.G3, inputs=[self.p, self.div, self.G, 0], device="cuda")
            wp.launch(k_solve_pressure_rb, dim=self.G3, inputs=[self.p, self.div, self.G, 1], device="cuda")
        wp.launch(k_subtract_gradient, dim=self.G3, inputs=[self.u, self.v, self.w, self.p, self.h, self.G], device="cuda")

    def inject_mouse(self, m, mv):
        wp.launch(k_inject_mouse, dim=self.G3, inputs=[
            self.u_prev, self.v_prev, self.w_prev,
            wp.vec3(*m), wp.vec3(*mv),
            0.1, 1.0, self.G
        ], device="cuda")

    def inject_audio(self, bands, flux):
        np.clip(np.asarray(bands, dtype=np.float32), 0.0, 1.0, out=self._bands_np)
        np.sqrt(self._bands_np, out=self._bands_np)
        wp.copy(self._bands_gpu, wp.array(self._bands_np, dtype=float, device="cpu"))
        wp.launch(k_inject_audio_field, dim=self.G3, inputs=[
            self.u_prev, self.v_prev, self.w_prev,
            self._bands_gpu,
            int(self._bands_np.shape[0]),
            float(flux),
            0.08,
            self.G
        ], device="cuda")

    def _inject_gradient(self, mode=0):
        if mode == 0:
            return
        wp.launch(k_inject_gradient, dim=self.G3, inputs=[
            self.u_prev, self.v_prev, self.w_prev,
            self.grad_x, self.grad_y, self.lum, self.gradient_strength,
            self.G, mode
        ], device="cuda")

    def _apply_constraints(self, mode=0):
        self.xyz_corr.zero_()
        if mode == 0:
            return
        elif mode == 1: # Thread
            for bond, l0 in [
                (self.next_y, 2.0 * self.r),
                (self.next_yy, 4.0 * self.r)
            ]:
                wp.launch(k_constraint_force, dim=self.particles, inputs=[
                    self.xyz, self.xyz_corr,
                    self.invmass, bond, l0, 0.0, self.dt
                ], device="cuda")
        elif mode == 2: # Cloth
            aux = 2.0 * np.sqrt(2.0)
            for bond, l0 in [
                (self.next_y, 2.0 * self.r),
                (self.next_x, 2.0 * self.r),
                (self.next_yy, 4.0 * self.r),
                (self.next_xx, 4.0 * self.r),
                (self.next_du, aux * self.r),
                (self.next_dd, aux * self.r),
            ]:
                wp.launch(k_constraint_force, dim=self.particles, inputs=[
                    self.xyz, self.xyz_corr,
                    self.invmass, bond, l0, 0.0, self.dt
                ], device="cuda")
        wp.launch(k_add_corrections, dim=self.particles, inputs=[
            self.xyz, self.xyz_corr,
            self.jacobi
        ], device="cuda")

    def _apply_boundaries(self, world_mode, world_center, world_radius):
        if world_mode == 1: # spherical
            wp.launch(k_sphere_boundaries, dim=self.particles, inputs=[
                self.xyz, self.xyz_prior,
                wp.vec3(*world_center), wp.vec3(*world_radius),
                4.0 * self.r, 0.02, 0.5 * self.r, # push, damping, swirl     
                0.25 * world_radius[0],           # band
                wp.vec3(0.0, 1.0, 0.0),           # flow axis
            ], device="cuda")
        elif world_mode == 2: # torus
            wp.launch(k_torus_boundaries, dim=self.particles, inputs=[
                self.xyz, self.xyz_prior,
                wp.vec3(*world_center), wp.normalize(wp.vec3(0.0, 0.0, 1.0)),  # torus axis
                0.25, 0.1,                             # major_radius, minor_radius
                4.0 * self.r, 0.02, 0.5 * self.r,      # push, damping, swirl        
                0.15,                                  # band
                wp.vec3(0.0, 1.0, 0.0),                # flow_axis
            ], device="cuda")

        wp.launch(k_world_boundaries, dim=self.particles, inputs=[
            self.xyz, self.xyz_prior,
            self.r, self.h, self.G
        ], device="cuda")

    def fit_fourier_line(self,
        xyz_wp: wp.array,
        samples: int = 128,
        harmonics: int = 8,
        device: str = "cuda",
    ):
        n = len(xyz_wp)

        # ── centroid ──────────────────────────────────────────────────────────
        sum_buf = wp.zeros(4, dtype=float, device=device)
        wp.launch(k_sum_xyz, dim=n, inputs=[xyz_wp, sum_buf], device=device)
        wp.synchronize_device(device)

        s = sum_buf.numpy()
        count = max(float(s[0]), 1.0)
        centroid_np = np.array([s[1], s[2], s[3]], dtype=np.float64) / count

        # ── two ordering axes via covariance (CPU, tiny 3×3) ──────────────────
        # Download xyz once — only for axis extraction, not per-frame bottleneck
        xyz_np = xyz_wp.numpy().view(np.float32).reshape(-1, 3).astype(np.float64)
        d = xyz_np - centroid_np
        C = (d.T @ d) / count
        evals, evecs = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1]     # descending eigenvalue order
        ax0 = evecs[:, order[0]].astype(np.float32)
        ax1 = evecs[:, order[1]].astype(np.float32)

        centroid_wp = wp.vec3(*centroid_np.astype(np.float32).tolist())
        ax0_wp = wp.vec3(*ax0.tolist())
        ax1_wp = wp.vec3(*ax1.tolist())

        # ── angle per point ───────────────────────────────────────────────────
        angles = wp.empty(n, dtype=float, device=device)
        wp.launch(
            k_compute_angle,
            dim=n,
            inputs=[xyz_wp, centroid_wp, ax0_wp, ax1_wp, angles],
            device=device,
        )

        # ── Fourier coefficients ──────────────────────────────────────────────
        K = harmonics
        coeff_len = 3 * (2 * K + 1)
        coeffs = wp.zeros(coeff_len, dtype=float, device=device)
        wp.launch(
            k_accumulate_fourier,
            dim=n,
            inputs=[xyz_wp, angles, K, coeffs],
            device=device,
        )

        # ── evaluate curve ────────────────────────────────────────────────────
        line_xyz = wp.empty(samples, dtype=wp.vec3, device=device)
        line_next = wp.empty(samples, dtype=int, device=device)

        wp.launch(
            k_eval_fourier_curve,
            dim=samples,
            inputs=[coeffs, K, samples, line_xyz],
            device=device,
        )
        wp.launch(
            k_build_line_next_closed,
            dim=samples,
            inputs=[line_next],
            device=device,
        )
        wp.synchronize_device(device)

        return line_xyz, line_next

    def update_fluid(self, go_back_on=True, world_mode=0, world_center=[0.5, 0.5, 0.5], world_radius=[0.4, 0.4, 0.1]):
        # Grid   
        wp.launch(k_add_source, dim=self.G3, inputs=[self.u, self.u_prev, self.dt], device="cuda")
        wp.launch(k_add_source, dim=self.G3, inputs=[self.v, self.v_prev, self.dt], device="cuda")
        wp.launch(k_add_source, dim=self.G3, inputs=[self.w, self.w_prev, self.dt], device="cuda")
        wp.launch(k_advect, dim=self.G3, inputs=[self.u_prev, self.u, self.u, self.v, self.w, self.dt0, self.G], device="cuda")
        wp.launch(k_advect, dim=self.G3, inputs=[self.v_prev, self.v, self.u, self.v, self.w, self.dt0, self.G], device="cuda")
        wp.launch(k_advect, dim=self.G3, inputs=[self.w_prev, self.w, self.u, self.v, self.w, self.dt0, self.G], device="cuda")
        self.u, self.u_prev = self.u_prev, self.u
        self.v, self.v_prev = self.v_prev, self.v
        self.w, self.w_prev = self.w_prev, self.w
        self.project()

        # Particles
        wp.copy(self.xyz_prior, self.xyz)
        wp.launch(k_advect_particles_rk2, dim=self.particles, inputs=[
            self.xyz, self.u, self.v, self.w, self.invmass, self.dt, self.h, self.G
        ], device="cuda")

        if go_back_on:
            wp.launch(k_goal_force, dim=self.particles, inputs=[
                self.xyz, self.xyz_goal, 0.05
            ], device="cuda")

        self._apply_boundaries(world_mode, world_center, world_radius)

        wp.launch(k_update_rotation, dim=self.particles, inputs=[
            self.xyz, self.xyz_prior, self.rot, 0.1 * self.r, 0.1
        ], device="cuda")

        self.u_prev.zero_()
        self.v_prev.zero_()
        self.w_prev.zero_()

    def update_flip(
        self, go_back_on=True, constraints_mode=0, gradient_mode=0,
        world_mode=0, world_center=[0.5, 0.5, 0.5], world_radius=[0.4, 0.4, 0.1]
    ):
        # Grid
        wp.copy(self.u_prior, self.u)
        wp.copy(self.v_prior, self.v)
        wp.copy(self.w_prior, self.w)
    
        self._inject_gradient(mode=gradient_mode)
        wp.launch(k_add_source, dim=self.G3, inputs=[self.u, self.u_prev, self.dt], device="cuda")
        wp.launch(k_add_source, dim=self.G3, inputs=[self.v, self.v_prev, self.dt], device="cuda")
        wp.launch(k_add_source, dim=self.G3, inputs=[self.w, self.w_prev, self.dt], device="cuda")
        wp.launch(k_advect, dim=self.G3, inputs=[self.u_prev, self.u, self.u, self.v, self.w, self.dt0, self.G], device="cuda")
        wp.launch(k_advect, dim=self.G3, inputs=[self.v_prev, self.v, self.u, self.v, self.w, self.dt0, self.G], device="cuda")
        wp.launch(k_advect, dim=self.G3, inputs=[self.w_prev, self.w, self.u, self.v, self.w, self.dt0, self.G], device="cuda")
        self.u, self.u_prev = self.u_prev, self.u
        self.v, self.v_prev = self.v_prev, self.v
        self.w, self.w_prev = self.w_prev, self.w
        self.project()

        # Particles
        wp.copy(self.xyz_prior, self.xyz)
        wp.launch(k_flip_update, dim=self.particles, inputs=[
            self.xyz, self.xyz_prior, self.vel,
            self.u, self.v, self.w,
            self.u_prior, self.v_prior, self.w_prior,
            self.flip_alpha, self.dt, self.h, self.G
        ], device="cuda")

        if go_back_on:
            wp.launch(k_goal_force, dim=self.particles, inputs=[
                self.xyz, self.xyz_goal, 0.005
            ], device="cuda")

        self._apply_constraints(mode=constraints_mode)
        self._apply_boundaries(world_mode, world_center, world_radius)

        # Keep particle velocity consistent with post-correction positions
        wp.launch(k_update_vel_from_positions, dim=self.particles, inputs=[
            self.xyz, self.xyz_prior, self.vel, self.dt
        ], device="cuda")

        # P2G rebuild from particle velocities
        self.u.zero_()
        self.v.zero_()
        self.w.zero_()
        self.p2g_weight.zero_()

        wp.launch(k_particles_to_grid_vel, dim=self.particles, inputs=[
            self.xyz, self.vel,
            self.u, self.v, self.w, self.p2g_weight,
            self.h, self.G
        ], device="cuda")
        wp.launch(k_normalize_grid_velocity, dim=self.G3, inputs=[
            self.u, self.v, self.w, self.p2g_weight
        ], device="cuda")
        self.project(steps=max(2, self.pressure_steps // 2))

        wp.launch(k_update_rotation, dim=self.particles, inputs=[
            self.xyz, self.xyz_prior, self.rot, 0.1 * self.r, 0.1
        ], device="cuda")

        self.u_prev.zero_()
        self.v_prev.zero_()
        self.w_prev.zero_()