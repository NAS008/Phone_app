import cv2
import random
import numpy as np
from scipy import ndimage
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
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    dt: float,
    h: float, G: wp.vec3i
):
    pid = wp.tid()
    p = xyz[pid]
    v1 = sample_velocity(p, u, v, w, h, G)
    v2 = sample_velocity(p + v1*(dt*0.5), u, v, w, h, G)
    xyz[pid] = clamp_position(p + v2*dt, h, G)

@wp.kernel
def k_goal_force(
    xyz: wp.array(dtype=wp.vec3),
    xyz_goal: wp.array(dtype=wp.vec3),
    strength: float, dt: float
):
    tid = wp.tid()

    v = (xyz_goal[tid] - xyz[tid]) * strength * dt
    xyz[tid] += v * dt

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
    dx = wx - m.x;  dy = wy - m.y
    dist2 = dx*dx + dy*dy
    if dist2 < mradius * mradius:
        q  = wp.sqrt(dist2) / mradius
        wt = (1.0 - q) * (1.0 - q)
        u_prev[tid] += mv.x * strength * wt
        v_prev[tid] += mv.y * strength * wt
        w_prev[tid] += mv.z * strength * wt

@wp.kernel
def k_set_velocity_fluid(
    u_prev: wp.array(dtype=float), v_prev: wp.array(dtype=float), w_prev: wp.array(dtype=float),
    grad_x: wp.array(dtype=float), grad_y: wp.array(dtype=float), lum: wp.array(dtype=float),
    strength: float, G: wp.vec3i, depth_factor: float
):
    i = wp.tid() % G.x
    j = (wp.tid() // G.x) % G.y
    k = wp.tid() // (G.x*G.y)
    depth_max = int(depth_factor * float(G.z))
    if i < 1 or i >= G.x-1 or j < 1 or j >= G.y-1 or k < 1 or k >= depth_max:
        return
    id2 = i + G.x*j
    idx = i + G.x*j + G.x*G.y*k
    l = lum[id2]
    scale = float(k) / float(G.z)
    # u_prev[idx] = -grad_y[id2] * strength * l * scale * 2.0
    # v_prev[idx] = grad_x[id2] * strength * l * scale * 2.0
    # #w_prev[idx] = l * strength * (0.5 - scale) * 1.0
    # sign = float(1) if i % 10 < 5 else float(-1)
    # w_prev[idx] = sign * 2.0 * l * strength * float(k) / float(depth_max)
    # if i % 10 < 5:
    #     v_prev[idx] -= (2.5 if j < G.y//2 else 0.5) * l * strength
    # else:
    #     if j < G.y//2:
    #         v_prev[idx] += 0.5 * l * strength
    #     else:
    #         u_prev[idx] -= 0.5 * l * strength

    u_prev[idx] = -grad_y[id2] * strength * l * 8.0
    v_prev[idx] = grad_x[id2] * strength * l * 8.0
    w_prev[idx] = l * strength * (0.5 - scale) * 2.0

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
    
    if dist < velocity_threshold:
        n = wp.vec3(0.0, 0.0, 1.0) # Points to camera
    else:
        n = d / dist # Normalize velocity direction
    # Smoothly interpolate
    dir += (n - dir) * lerp_factor
    rot[tid] = dir

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

    id2 = next[id1]
    if id2 < 0:
        return  
    w2 = invmass[id2]
    w = w0 + w2
    if w == 0.0:
        return
    p2 = xyz[id2]
    d = p2 - p0
    l = wp.sqrt(d.x * d.x + d.y * d.y + d.z * d.z)
    if l <= 0.0:
        return    
    n = d / l
    alpha = compliance / dt / dt
    displace = n * (l - 2.0 * l0) / (w + alpha)
    wp.atomic_add(xyz_corr, id0, w0 * displace)
    wp.atomic_sub(xyz_corr, id2, w2 * displace)

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
    return 1.0 - (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0

@wp.kernel
def k_new_image(
    pixels: wp.array(dtype=wp.uint8), IMAGE_SIZE: int,
    xyz_goal: wp.array(dtype=wp.vec3), rgb: wp.array(dtype=wp.vec3),
    h: float, G: wp.vec3i
):
    tid = wp.tid()
    p = xyz_goal[tid]

    x0 = (1.0 - h * float(G.x)) / 2.0
    y0 = (1.0 - h * float(G.y)) / 2.0

    ix = wp.clamp(int((x0 + p.x) * float(IMAGE_SIZE)), 0, IMAGE_SIZE - 1)
    iy = wp.clamp(int((1.0 - y0 - p.y) * float(IMAGE_SIZE)), 0, IMAGE_SIZE - 1)
    idc  = 3 * (ix  + iy  * IMAGE_SIZE)
    rgb[tid] = wp.vec3(
        float(pixels[idc + 2]),
        float(pixels[idc + 1]),
        float(pixels[idc])
    )
     
@wp.kernel
def k_reset(
    pixels: wp.array(dtype=wp.uint8), blurred: wp.array(dtype=wp.uint8), IMAGE_SIZE: int,
    xyz_goal: wp.array(dtype=wp.vec3), rgb: wp.array(dtype=wp.vec3),
    h: float, G: wp.vec3i, depth_factor: float
):
    tid = wp.tid()
    p = xyz_goal[tid]

    x0 = (1.0 - h * float(G.x)) / 2.0
    y0 = (1.0 - h * float(G.y)) / 2.0

    ix = wp.clamp(int((x0 + p.x) * float(IMAGE_SIZE)), 0, IMAGE_SIZE - 1)
    iy = wp.clamp(int((1.0 - y0 - p.y) * float(IMAGE_SIZE)), 0, IMAGE_SIZE - 1)
    idc  = 3 * (ix  + iy  * IMAGE_SIZE)
    rgb[tid] = wp.vec3(
        float(pixels[idc + 2]),
        float(pixels[idc + 1]),
        float(pixels[idc])
    )
    lumc = lum(blurred, idc)
    z_max = depth_factor * h * float(G.z - 2)
    xyz_goal[tid] = wp.vec3(p.x, p.y, h + z_max * lumc)

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

class Simulator:
    def __init__(self,
        IMAGE_SIZE=512, PIXELS_PER_CELL=4,
        G=[128, 128, 32], L=1, smooth=15,
        gradient_strength=0.1, pressure_steps=10, jacobi=0.1, dt=0.5
    ):
        self.G = wp.vec3i(*G)
        self.h = 1.0 / max(self.G.x, self.G.y)    
        self.P = self.G * PIXELS_PER_CELL
        self.r = 0.5 * self.h / PIXELS_PER_CELL

        self.IMAGE_SIZE = IMAGE_SIZE
        self.smooth = smooth

        self.gradient_strength = gradient_strength
        self.pressure_steps = pressure_steps
        self.jacobi = jacobi
        self.dt = dt
        self.dt0 = dt * float(max(self.G.x, self.G.y, self.G.z))

        pos = []
        n_x, n_y = [], []
        x0 = self.h + self.r
        y0 = self.h + self.r
        z0 = self.h + self.r
        spacing = 2.0 * self.r   
        PX_inner = self.P.x - 2 * PIXELS_PER_CELL
        PY_inner = self.P.y - 2 * PIXELS_PER_CELL
        particles = 0
        for k in range(L):
            for j in range(PY_inner):
                for i in range(PX_inner):
                    jx = 0.05 * spacing * random.random()
                    jy = 0.05 * spacing * random.random()
                    jz = 0.05 * spacing * random.random()
                    pos.append([x0 + i * spacing + jx, y0 + j * spacing + jy, z0 + k * spacing + jz])
                    if i < PX_inner - 1: n_x.append(particles + 1)
                    else: n_x.append(-1)
                    if j < PY_inner - 1: n_y.append(particles + PX_inner)
                    else: n_y.append(-1)
                    particles += 1
        self.particles = len(pos)
        # Particle arrays
        self.xyz = wp.array(pos, dtype=wp.vec3, device="cuda")
        self.xyz_prior = wp.array(pos, dtype=wp.vec3, device="cuda")
        self.xyz_goal = wp.array(pos, dtype=wp.vec3, device="cuda")
        self.xyz_corr = wp.zeros(self.particles, dtype=wp.vec3, device="cuda")
        self.rgb = wp.zeros(self.particles, dtype=wp.vec3, device="cuda")
        self.rot = wp.zeros(self.particles, dtype=wp.vec3, device="cuda")
        self.invmass = wp.ones(self.particles, dtype=float, device="cuda")
        self.radii = wp.zeros(self.particles, dtype=float, device="cuda")
        # Particle neighbors
        self.next_x = wp.array(n_x, dtype=int, device="cuda")
        self.next_y = wp.array(n_y, dtype=int, device="cuda")
        # Image buffers
        self.pixels = wp.zeros(self.IMAGE_SIZE * self.IMAGE_SIZE * 3, dtype=wp.uint8, device="cuda") 
        self.blurred = wp.zeros(self.IMAGE_SIZE * self.IMAGE_SIZE * 3, dtype=wp.uint8, device="cuda")

        # Grid arrays
        self.G2 = self.G.x * self.G.y
        self.G3 = self.G.x * self.G.y * self.G.z
        self.grad_x = wp.zeros(self.G2, dtype=float, device="cuda") 
        self.grad_y = wp.zeros(self.G2, dtype=float, device="cuda") 
        self.lum = wp.zeros(self.G2, dtype=float, device="cuda") 
        self.u      = wp.zeros(self.G3, dtype=float, device="cuda")
        self.v      = wp.zeros(self.G3, dtype=float, device="cuda")
        self.w      = wp.zeros(self.G3, dtype=float, device="cuda")
        self.u_prev = wp.zeros(self.G3, dtype=float, device="cuda")
        self.v_prev = wp.zeros(self.G3, dtype=float, device="cuda")
        self.w_prev = wp.zeros(self.G3, dtype=float, device="cuda")
        self.div    = wp.zeros(self.G3, dtype=float, device="cuda")
        self.p      = wp.zeros(self.G3, dtype=float, device="cuda")

        print(f"✓ SIM: simulator ready")
        print(f"    Image({self.IMAGE_SIZE},{self.IMAGE_SIZE})")
        print(f"    {self.particles:,} particles({PX_inner},{PY_inner},{L})")
        print(f"    {self.G3:,} cells in grid({self.G.x},{self.G.y},{self.G.z})")

    def _compute_gradients(self, img):
        img_grid = cv2.resize(img, (self.G.x, self.G.y), interpolation=cv2.INTER_AREA)

        # Grayscale luminance
        lum = cv2.cvtColor(img_grid, cv2.COLOR_BGR2GRAY)
        luminance = lum.astype(np.float32) / 255.0  # shape: (GY, GX)

        # 2D Sobel gradients
        grad_x = ndimage.sobel(luminance, axis=1)  # X direction (cols)
        grad_y = ndimage.sobel(luminance, axis=0)  # Y direction (rows)

        max_mag = np.hypot(grad_x, grad_y).max()
        if max_mag > 0:
            grad_x /= max_mag
            grad_y /= max_mag

        gx  = wp.array(grad_x.flatten().astype(np.float32), dtype=wp.float32, device="cuda")
        gy  = wp.array(grad_y.flatten().astype(np.float32), dtype=wp.float32, device="cuda")
        lum = wp.array(luminance.flatten().astype(np.float32), dtype=wp.float32, device="cuda")
        return gx, gy, lum   

    def inject_gradient(self, depth_factor=1.0):
        wp.launch(k_set_velocity_fluid, dim=self.G3, inputs=[
            self.u_prev, self.v_prev, self.w_prev,
            self.grad_x, self.grad_y, self.lum, self.gradient_strength,
            self.G, depth_factor
        ], device="cuda")

    def reset(self, img, depth_factor=1.0):
        for arr in [self.u, self.v, self.w,
                    self.u_prev, self.v_prev, self.w_prev,
                    self.div, self.p]:
            arr.zero_()

        self.pixels = wp.array(np.ascontiguousarray(img).flatten(), dtype=wp.uint8, device="cuda")  
        blurred = cv2.GaussianBlur(img, (self.smooth, self.smooth), 0)   
        self.blurred = wp.array(np.ascontiguousarray(blurred).flatten(), dtype=wp.uint8, device="cuda")   
        self.grad_x, self.grad_y, self.lum = self._compute_gradients(blurred)  
        wp.launch(k_reset, dim=self.particles, inputs=[
            self.pixels, self.blurred, self.IMAGE_SIZE,
            self.xyz_goal, self.rgb,
            self.h, self.G, depth_factor
        ], device="cuda")

    def project(self):
        wp.launch(k_divergence, dim=self.G3, inputs=[
            self.div, self.u, self.v, self.w, self.h, self.G
        ], device="cuda")
        self.p.zero_()
        for _ in range(self.pressure_steps):
            wp.launch(k_solve_pressure, dim=self.G3, inputs=[self.p, self.div, self.G], device="cuda")
        wp.launch(k_subtract_gradient, dim=self.G3, inputs=[self.u, self.v, self.w, self.p, self.h, self.G], device="cuda")

    def inject_mouse(self, m, mv):
        wp.launch(k_inject_mouse, dim=self.G3, inputs=[
            self.u_prev, self.v_prev, self.w_prev,
            wp.vec3(*m), wp.vec3(*mv),
            0.05, 1.0, self.G
        ], device="cuda")

    def _apply_constraints(self):
        self.xyz_corr.zero_()
        for bond, l0 in [
            (self.next_y, 2.0 * self.r)
        ]:
            wp.launch(k_constraint_force, dim=self.particles, inputs=[
                self.xyz, self.xyz_corr,
                self.invmass, bond, l0, 0.0, self.dt
            ], device="cuda")
        wp.launch(k_add_corrections, dim=self.particles, inputs=[
            self.xyz, self.xyz_corr,
            self.jacobi
        ], device="cuda")

    def update(self, constraints_on=True, go_back_on=True, ):
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
            self.xyz, self.u, self.v, self.w, self.dt, self.h, self.G
        ], device="cuda")
        
        if constraints_on:
            self._apply_constraints()

        if go_back_on:
            wp.launch(k_goal_force, dim=self.particles, inputs=[
                self.xyz, self.xyz_goal, 0.1, self.dt
            ], device="cuda")

        wp.launch(k_update_rotation, dim=self.particles, inputs=[
            self.xyz, self.xyz_prior,
            self.rot,
            0.25 * self.r, 0.05
        ], device="cuda")

        self.u_prev.zero_()
        self.v_prev.zero_()
        self.w_prev.zero_()
