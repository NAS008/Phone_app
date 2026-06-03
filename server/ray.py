import numpy as np
import warp as wp
wp.init()    

@wp.func
def world_to_grid(
    p: wp.vec3,
    h: float, G: wp.vec3i
):
    cx = wp.clamp(int(p.x / h), 0, G.x - 1)
    cy = wp.clamp(int(p.y / h), 0, G.y - 1)
    cz = wp.clamp(int(p.z / h), 0, G.z - 1)  
    return wp.vec3i(cx, cy, cz)

@wp.func
def flat_cell(
    x: int, y: int, z: int,
    G: wp.vec3i
):
    return x + y * G.x + z * G.x * G.y

@wp.kernel
def k_insert(
    xyz: wp.array(dtype=wp.vec3), r: float,
    h: float, G: wp.vec3i, cell_count: wp.array(dtype=int)
):
    tid = wp.tid()    
    p = xyz[tid]    
    p_min = wp.vec3(p.x - r, p.y - r, p.z - r)
    p_max = wp.vec3(p.x + r, p.y + r, p.z + r)    
    cell_min = world_to_grid(p_min, h, G)
    cell_max = world_to_grid(p_max, h, G)
    for z in range(cell_min.z, cell_max.z + 1):
        for y in range(cell_min.y, cell_max.y + 1):
            for x in range(cell_min.x, cell_max.x + 1):
                cell = flat_cell(x, y, z, G)
                wp.atomic_add(cell_count, cell, 1)

@wp.kernel
def k_fill(
    xyz: wp.array(dtype=wp.vec3), r: float,
    h: float, G: wp.vec3i, cell_start: wp.array(dtype=int), particle_ids: wp.array(dtype=int), cell_offset: wp.array(dtype=int)
):
    tid = wp.tid()     
    p = xyz[tid]    
    p_min = wp.vec3(p.x - r, p.y - r, p.z - r)
    p_max = wp.vec3(p.x + r, p.y + r, p.z + r)    
    cell_min = world_to_grid(p_min, h, G)
    cell_max = world_to_grid(p_max, h, G)
    for z in range(cell_min.z, cell_max.z + 1):
        for y in range(cell_min.y, cell_max.y + 1):
            for x in range(cell_min.x, cell_max.x + 1):
                cell = flat_cell(x, y, z, G)
                start = cell_start[cell]
                offset = wp.atomic_add(cell_offset, cell, 1)
                particle_ids[start + offset] = tid

@wp.func
def mat_to_aabb(
    rx: float, ry: float, rz: float,
    mat: wp.mat33
) -> wp.vec3:
    ex = wp.abs(mat[0, 0])*rx + wp.abs(mat[0, 1])*ry + wp.abs(mat[0, 2])*rz
    ey = wp.abs(mat[1, 0])*rx + wp.abs(mat[1, 1])*ry + wp.abs(mat[1, 2])*rz
    ez = wp.abs(mat[2, 0])*rx + wp.abs(mat[2, 1])*ry + wp.abs(mat[2, 2])*rz
    return wp.vec3(ex, ey, ez)

@wp.kernel
def k_insert_oriented(
    xyz: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.vec3), rx: float, ry: float, rz: float,
    h: float, G: wp.vec3i, cell_count: wp.array(dtype=int)
):
    tid = wp.tid()
    p   = xyz[tid]
    mat = dir_to_mat(rot[tid])
    ext = mat_to_aabb(rx, ry, rz, mat)

    p_min = wp.vec3(p[0] - ext[0], p[1] - ext[1], p[2] - ext[2])
    p_max = wp.vec3(p[0] + ext[0], p[1] + ext[1], p[2] + ext[2])

    cell_min = world_to_grid(p_min, h, G)
    cell_max = world_to_grid(p_max, h, G)

    for z in range(cell_min.z, cell_max.z + 1):
        for y in range(cell_min.y, cell_max.y + 1):
            for x in range(cell_min.x, cell_max.x + 1):
                cell = flat_cell(x, y, z, G)
                wp.atomic_add(cell_count, cell, 1)

@wp.kernel
def k_fill_oriented(
    xyz: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.vec3), rx: float, ry: float, rz: float,
    h: float, G: wp.vec3i, cell_start: wp.array(dtype=int), particle_ids: wp.array(dtype=int), cell_offset: wp.array(dtype=int)
):
    tid = wp.tid()
    p   = xyz[tid]
    mat = dir_to_mat(rot[tid])
    ext = mat_to_aabb(rx, ry, rz, mat)

    p_min = wp.vec3(p[0] - ext[0], p[1] - ext[1], p[2] - ext[2])
    p_max = wp.vec3(p[0] + ext[0], p[1] + ext[1], p[2] + ext[2])

    cell_min = world_to_grid(p_min, h, G)
    cell_max = world_to_grid(p_max, h, G)

    for z in range(cell_min.z, cell_max.z + 1):
        for y in range(cell_min.y, cell_max.y + 1):
            for x in range(cell_min.x, cell_max.x + 1):
                cell = flat_cell(x, y, z, G)
                start  = cell_start[cell]
                offset = wp.atomic_add(cell_offset, cell, 1)
                particle_ids[start + offset] = tid

@wp.kernel
def k_insert_line(
    xyz: wp.array(dtype=wp.vec3), next: wp.array(dtype=int), r: float,
    h: float, G: wp.vec3i, cell_count: wp.array(dtype=int)
):
    tid = wp.tid()   
    p0 = xyz[tid]  
    next_id = next[tid]
    if next_id >= 0:
        p1 = xyz[next_id]
    else:
        p1 = p0
    
    line_min = wp.vec3(
        wp.min(p0.x, p1.x) - r,
        wp.min(p0.y, p1.y) - r,
        wp.min(p0.z, p1.z) - r
    )
    line_max = wp.vec3(
        wp.max(p0.x, p1.x) + r,
        wp.max(p0.y, p1.y) + r,
        wp.max(p0.z, p1.z) + r
    )    
    cell_min = world_to_grid(line_min, h, G)
    cell_max = world_to_grid(line_max, h, G)
    for z in range(cell_min.z, cell_max.z + 1):
        for y in range(cell_min.y, cell_max.y + 1):
            for x in range(cell_min.x, cell_max.x + 1):
                cell = flat_cell(x, y, z, G)
                wp.atomic_add(cell_count, cell, 1)

@wp.kernel
def k_fill_line(
    xyz: wp.array(dtype=wp.vec3), next: wp.array(dtype=int), r: float,
    h: float, G: wp.vec3i, cell_start: wp.array(dtype=int), particle_ids: wp.array(dtype=int), cell_offset: wp.array(dtype=int)
): 
    tid = wp.tid()   
    p0 = xyz[tid]  
    next_id = next[tid]
    if next_id >= 0:
        p1 = xyz[next_id]
    else:
        p1 = p0
    
    line_min = wp.vec3(
        wp.min(p0.x, p1.x) - r,
        wp.min(p0.y, p1.y) - r,
        wp.min(p0.z, p1.z) - r
    )
    line_max = wp.vec3(
        wp.max(p0.x, p1.x) + r,
        wp.max(p0.y, p1.y) + r,
        wp.max(p0.z, p1.z) + r
    )    
    cell_min = world_to_grid(line_min, h, G)
    cell_max = world_to_grid(line_max, h, G)
    for z in range(cell_min.z, cell_max.z + 1):
        for y in range(cell_min.y, cell_max.y + 1):
            for x in range(cell_min.x, cell_max.x + 1):
                cell = flat_cell(x, y, z, G)
                start = cell_start[cell]
                offset = wp.atomic_add(cell_offset, cell, 1)
                particle_ids[start + offset] = tid

class Grid:
    def __init__(self, G):
        self.G = G
        self.h = 1.0 / max(G.x, G.y)
        self.N = G.x*G.y*G.z          
        self.cell_count = wp.zeros(self.N, dtype=int, device='cuda')
        self.cell_start = wp.zeros(self.N, dtype=int, device='cuda')
        self.cell_offset = wp.zeros(self.N, dtype=int, device='cuda')
        self.particle_ids = None

    def _build_prefix_sum(self):
        counts = self.cell_count.numpy()
        starts = np.empty(self.N, dtype=np.int32)
        starts[0] = 0
        np.cumsum(counts[:-1], out=starts[1:])
        total_entries = int(starts[-1] + counts[-1])
        self.cell_start = wp.array(starts, dtype=int, device="cuda")
        if self.particle_ids is None or len(self.particle_ids) < total_entries:
            self.particle_ids = wp.zeros(max(total_entries, 1), dtype=int, device="cuda")

    def build(self, n: int, insert_kernel, fill_kernel, insert_args: list, fill_args: list):
        self.cell_count.zero_()
        self.cell_offset.zero_()
        wp.launch(insert_kernel, dim=n, inputs=insert_args + [self.h, self.G, self.cell_count], device="cuda")
        wp.synchronize()
        self._build_prefix_sum()
        wp.launch(fill_kernel,   dim=n, inputs=fill_args   + [self.h, self.G, self.cell_start, self.particle_ids, self.cell_offset], device="cuda")
        wp.synchronize()

# Sphere
@wp.func
def intersect_sphere(
    ro: wp.vec3, rd: wp.vec3, center: wp.vec3, r: float
) -> float:
    oc   = ro - center
    b    = wp.dot(oc, rd)
    c    = wp.dot(oc, oc) - r * r
    disc = b * b - c
    if disc < 0.0:
        return -1.0
    sq = wp.sqrt(disc)
    t  = -b - sq
    if t < 0.001:
        t = -b + sq
    return t

@wp.func
def normal_sphere(
    hit_pt: wp.vec3, center: wp.vec3
):
    return wp.normalize(hit_pt - center)

@wp.func
def shadow_sphere(
    xyz: wp.array(dtype=wp.vec3), r: float,
    ray_origin: wp.vec3, ray_dir: wp.vec3, shadow_distance: float,
    h: float, G: wp.vec3i, cell_start: wp.array(dtype=int), cell_count: wp.array(dtype=int), particle_ids: wp.array(dtype=int)
) -> bool:
    inv_dir = wp.vec3(
        1.0/ray_dir[0] if wp.abs(ray_dir[0]) > 1e-6 else 1e10,
        1.0/ray_dir[1] if wp.abs(ray_dir[1]) > 1e-6 else 1e10,
        1.0/ray_dir[2] if wp.abs(ray_dir[2]) > 1e-6 else 1e10
    )
    current_cell = world_to_grid(ray_origin, h, G)
    if ray_origin[0] < 0.0 or ray_origin[0] > float(G.x)*h or \
       ray_origin[1] < 0.0 or ray_origin[1] > float(G.y)*h or \
       ray_origin[2] < 0.0 or ray_origin[2] > float(G.z)*h:
        return False

    step_x = 1 if ray_dir[0] > 0.0 else -1
    step_y = 1 if ray_dir[1] > 0.0 else -1
    step_z = 1 if ray_dir[2] > 0.0 else -1

    tmax_x = ((float(current_cell[0]) + (1.0 if ray_dir[0]>0.0 else 0.0))*h - ray_origin[0]) * inv_dir[0]
    tmax_y = ((float(current_cell[1]) + (1.0 if ray_dir[1]>0.0 else 0.0))*h - ray_origin[1]) * inv_dir[1]
    tmax_z = ((float(current_cell[2]) + (1.0 if ray_dir[2]>0.0 else 0.0))*h - ray_origin[2]) * inv_dir[2]
    tdelta_x = h * wp.abs(inv_dir[0])
    tdelta_y = h * wp.abs(inv_dir[1])
    tdelta_z = h * wp.abs(inv_dir[2])

    for step in range(G.x + G.y + G.z):
        if current_cell[0]<0 or current_cell[0]>=G.x or \
           current_cell[1]<0 or current_cell[1]>=G.y or \
           current_cell[2]<0 or current_cell[2]>=G.z:
            break
        cell_id   = flat_cell(current_cell[0], current_cell[1], current_cell[2], G)
        start_idx = cell_start[cell_id]
        count     = cell_count[cell_id]

        t = float(-1.0)
        for i in range(count):
            pid = particle_ids[start_idx + i]
            
            # SHAPE LINE 1
            t = intersect_sphere(ray_origin, ray_dir, xyz[pid], r)
            
            if t > 0.001 and t < shadow_distance:
                return True
        if tmax_x < tmax_y:
            if tmax_x < tmax_z:
                if tmax_x > shadow_distance: break
                current_cell = wp.vec3i(current_cell[0]+step_x, current_cell[1], current_cell[2])
                tmax_x += tdelta_x
            else:
                if tmax_z > shadow_distance: break
                current_cell = wp.vec3i(current_cell[0], current_cell[1], current_cell[2]+step_z)
                tmax_z += tdelta_z
        else:
            if tmax_y < tmax_z:
                if tmax_y > shadow_distance: break
                current_cell = wp.vec3i(current_cell[0], current_cell[1]+step_y, current_cell[2])
                tmax_y += tdelta_y
            else:
                if tmax_z > shadow_distance: break
                current_cell = wp.vec3i(current_cell[0], current_cell[1], current_cell[2]+step_z)
                tmax_z += tdelta_z
    return False

@wp.kernel
def raytrace_sphere(
    xyz: wp.array(dtype=wp.vec3), rgb: wp.array(dtype=wp.vec3), r: float,
    img: wp.array(dtype=wp.uint8, ndim=3), W: int, H: int, samples: int, sqrtN: int, background: wp.vec3, ambient: float, shadow: float,
    h: float, G: wp.vec3i, cell_start: wp.array(dtype=int), cell_count: wp.array(dtype=int), particle_ids: wp.array(dtype=int),
    camera: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3, light: wp.vec3, fov: float
):
    x, y = wp.tid()
    aspect  = float(W) / float(H)
    accum_r = float(0.0); accum_g = float(0.0); accum_b = float(0.0)
    seed    = x + y * W

    for s in range(samples):
        pixel = background

        # Compute grid cell (row, col) for this sample
        si = s % sqrtN   # column index
        sj = s / sqrtN   # row index (integer division)

        jx = float(0.0)
        jy = float(0.0)

        if samples > 1:
            rng = wp.rand_init(seed, s)
            # Stratified jitter: cell center offset + rand[0,1]/2 within cell
            jx = (float(si) + wp.randf(rng) * 0.5) / float(sqrtN) - 0.5
            jy = (float(sj) + wp.randf(rng) * 0.5) / float(sqrtN) - 0.5

        if aspect >= 1.0:
            u = ((float(x) + 0.5 + jx) / float(W) - 0.5) * fov
            v = -((float(y) + 0.5 + jy) / float(H) - 0.5) * fov / aspect
        else:
            u = ((float(x) + 0.5 + jx) / float(W) - 0.5) * fov * aspect
            v = -((float(y) + 0.5 + jy) / float(H) - 0.5) * fov

        ray_dir = wp.normalize(fwd + right * u + up * v)
        ray_origin = camera

        inv_dir = wp.vec3(
            1.0/ray_dir[0] if wp.abs(ray_dir[0]) > 1e-6 else 1e10,
            1.0/ray_dir[1] if wp.abs(ray_dir[1]) > 1e-6 else 1e10,
            1.0/ray_dir[2] if wp.abs(ray_dir[2]) > 1e-6 else 1e10
        )

        if ray_origin[0]<0.0 or ray_origin[0]>float(G.x)*h or \
           ray_origin[1]<0.0 or ray_origin[1]>float(G.y)*h or \
           ray_origin[2]<0.0 or ray_origin[2]>float(G.z)*h:
            t_min = float(-1e10); t_max = float(1e10)
            t1=(0.0-ray_origin[0])*inv_dir[0]; t2=(float(G.x)*h-ray_origin[0])*inv_dir[0]
            t_min=wp.max(t_min,wp.min(t1,t2));  t_max=wp.min(t_max,wp.max(t1,t2))
            t1=(0.0-ray_origin[1])*inv_dir[1]; t2=(float(G.y)*h-ray_origin[1])*inv_dir[1]
            t_min=wp.max(t_min,wp.min(t1,t2));  t_max=wp.min(t_max,wp.max(t1,t2))
            t1=(0.0-ray_origin[2])*inv_dir[2]; t2=(float(G.z)*h-ray_origin[2])*inv_dir[2]
            t_min=wp.max(t_min,wp.min(t1,t2));  t_max=wp.min(t_max,wp.max(t1,t2))
            if t_min > t_max or t_max < 0.0:
                continue
            entry  = ray_origin + ray_dir * (wp.max(0.0, t_min) + 0.0001)
            current_cell = world_to_grid(entry, h, G)

        step_x = 1 if ray_dir[0]>0.0 else -1
        step_y = 1 if ray_dir[1]>0.0 else -1
        step_z = 1 if ray_dir[2]>0.0 else -1
        tmax_x = ((float(current_cell[0])+(1.0 if ray_dir[0]>0.0 else 0.0))*h - ray_origin[0])*inv_dir[0]
        tmax_y = ((float(current_cell[1])+(1.0 if ray_dir[1]>0.0 else 0.0))*h - ray_origin[1])*inv_dir[1]
        tmax_z = ((float(current_cell[2])+(1.0 if ray_dir[2]>0.0 else 0.0))*h - ray_origin[2])*inv_dir[2]
        tdelta_x = h*wp.abs(inv_dir[0])
        tdelta_y = h*wp.abs(inv_dir[1])
        tdelta_z = h*wp.abs(inv_dir[2])

        closest_t = float(1e10); hit_id = int(-1)

        for step in range(G.x + G.y + G.z):
            if current_cell[0]<0 or current_cell[0]>=G.x or \
               current_cell[1]<0 or current_cell[1]>=G.y or \
               current_cell[2]<0 or current_cell[2]>=G.z:
                break
            cell_id   = flat_cell(current_cell[0], current_cell[1], current_cell[2], G)
            start_idx = cell_start[cell_id]
            count     = cell_count[cell_id]

            t = float(-1.0)
            for i in range(count):
                pid = particle_ids[start_idx + i]
                
                # SHAPE LINE 1
                t = intersect_sphere(ray_origin, ray_dir, xyz[pid], r)

                if t > 0.001 and t < closest_t:
                    closest_t = t; hit_id = pid
            if hit_id >= 0 and closest_t < wp.min(tmax_x, wp.min(tmax_y, tmax_z)):
                break
            if tmax_x < tmax_y:
                if tmax_x < tmax_z:
                    current_cell = wp.vec3i(current_cell[0]+step_x, current_cell[1], current_cell[2]); tmax_x+=tdelta_x
                else:
                    current_cell = wp.vec3i(current_cell[0], current_cell[1], current_cell[2]+step_z); tmax_z+=tdelta_z
            else:
                if tmax_y < tmax_z:
                    current_cell = wp.vec3i(current_cell[0], current_cell[1]+step_y, current_cell[2]); tmax_y+=tdelta_y
                else:
                    current_cell = wp.vec3i(current_cell[0], current_cell[1], current_cell[2]+step_z); tmax_z+=tdelta_z

        if hit_id >= 0:
            hit_pt = ray_origin + ray_dir * closest_t

            # SHAPE LINE 2
            normal = normal_sphere(hit_pt, xyz[hit_id])

            if wp.dot(normal, ray_dir) > 0.0:
                normal = -normal
            light_dir = wp.normalize(light - hit_pt)
            diffuse   = wp.max(0.0, wp.dot(normal, light_dir))
            shadow_factor = float(1.0)
            if diffuse > 0.0:

                # SHAPE LINE 3
                if shadow_sphere(
                    xyz, r,
                    hit_pt + normal * h * 0.01, light_dir, wp.length(light - hit_pt),
                    h, G, cell_start, cell_count, particle_ids
                ):
                    shadow_factor = shadow

            pixel = rgb[hit_id] * (ambient + diffuse * shadow_factor)

        accum_r += pixel[0]
        accum_g += pixel[1]
        accum_b += pixel[2]

    inv_n = 1.0 / float(samples)
    img[y, x, 0] = wp.uint8(wp.clamp(accum_b * inv_n, 0.0, 255.0))
    img[y, x, 1] = wp.uint8(wp.clamp(accum_g * inv_n, 0.0, 255.0))
    img[y, x, 2] = wp.uint8(wp.clamp(accum_r * inv_n, 0.0, 255.0))

# Oriented helper
@wp.func
def dir_to_mat(dir: wp.vec3) -> wp.mat33:
    fwd = wp.normalize(dir)
    # Frisvad / Pixar ONB — no branch, numerically stable everywhere
    sign = 1.0 if fwd[2] >= 0.0 else -1.0
    a = -1.0 / (sign + fwd[2])
    b = fwd[0] * fwd[1] * a
    right = wp.vec3(1.0 + sign * fwd[0] * fwd[0] * a, sign * b, -sign * fwd[0])
    up    = wp.vec3(b, sign + fwd[1] * fwd[1] * a, -fwd[1])
    return wp.mat33(
        right[0], up[0], fwd[0],
        right[1], up[1], fwd[1],
        right[2], up[2], fwd[2]
    )

# Oriented quad
@wp.func
def intersect_quad(
    ro: wp.vec3, rd: wp.vec3,
    center: wp.vec3,
    rx: float,
    normal: wp.vec3
) -> float:
    denom = wp.dot(rd, normal)
    if wp.abs(denom) < 1e-6:
        return -1.0
    t = wp.dot(center - ro, normal) / denom
    if t < 0.001:
        return -1.0
    hit = ro + rd * t
    d   = hit - center

    ax = wp.abs(normal[0])
    ay = wp.abs(normal[1])
    az = wp.abs(normal[2])
    if ax <= ay and ax <= az:
        world_up = wp.vec3(1.0, 0.0, 0.0)
    elif ay <= ax and ay <= az:
        world_up = wp.vec3(0.0, 1.0, 0.0)
    else:
        world_up = wp.vec3(0.0, 0.0, 1.0)

    tan   = wp.normalize(wp.cross(world_up, normal))
    bitan = wp.normalize(wp.cross(normal, tan))
    lu = wp.dot(d, tan)
    lv = wp.dot(d, bitan)
    eps = rx * 0.001
    if wp.abs(lu) > rx + eps or wp.abs(lv) > rx + eps:
        return -1.0
    return t

@wp.func
def shadow_quad(
    xyz: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.vec3), r: float,
    ray_origin: wp.vec3, ray_dir: wp.vec3, shadow_distance: float,
    h: float, G: wp.vec3i, cell_start: wp.array(dtype=int), cell_count: wp.array(dtype=int), particle_ids: wp.array(dtype=int)
) -> bool:
    inv_dir = wp.vec3(
        1.0/ray_dir[0] if wp.abs(ray_dir[0]) > 1e-6 else 1e10,
        1.0/ray_dir[1] if wp.abs(ray_dir[1]) > 1e-6 else 1e10,
        1.0/ray_dir[2] if wp.abs(ray_dir[2]) > 1e-6 else 1e10
    )
    current_cell = world_to_grid(ray_origin, h, G)
    if ray_origin[0] < 0.0 or ray_origin[0] > float(G.x)*h or \
       ray_origin[1] < 0.0 or ray_origin[1] > float(G.y)*h or \
       ray_origin[2] < 0.0 or ray_origin[2] > float(G.z)*h:
        return False

    step_x = 1 if ray_dir[0] > 0.0 else -1
    step_y = 1 if ray_dir[1] > 0.0 else -1
    step_z = 1 if ray_dir[2] > 0.0 else -1
    tmax_x = ((float(current_cell[0]) + (1.0 if ray_dir[0]>0.0 else 0.0))*h - ray_origin[0]) * inv_dir[0]
    tmax_y = ((float(current_cell[1]) + (1.0 if ray_dir[1]>0.0 else 0.0))*h - ray_origin[1]) * inv_dir[1]
    tmax_z = ((float(current_cell[2]) + (1.0 if ray_dir[2]>0.0 else 0.0))*h - ray_origin[2]) * inv_dir[2]
    tdelta_x = h * wp.abs(inv_dir[0])
    tdelta_y = h * wp.abs(inv_dir[1])
    tdelta_z = h * wp.abs(inv_dir[2])

    for step in range(G.x + G.y + G.z):
        if current_cell[0]<0 or current_cell[0]>=G.x or current_cell[1]<0 or current_cell[1]>=G.y or current_cell[2]<0 or current_cell[2]>=G.z:
            break
        cell_id   = flat_cell(current_cell[0], current_cell[1], current_cell[2], G)
        start_idx = cell_start[cell_id]
        count     = cell_count[cell_id]
        for i in range(count):
            pid = particle_ids[start_idx + i]
            t = intersect_quad(ray_origin, ray_dir, xyz[pid], r, rot[pid])
            if t > 0.001 and t < shadow_distance:
                return True
        if tmax_x < tmax_y:
            if tmax_x < tmax_z:
                if tmax_x > shadow_distance: break
                current_cell = wp.vec3i(current_cell[0]+step_x, current_cell[1], current_cell[2]); tmax_x += tdelta_x
            else:
                if tmax_z > shadow_distance: break
                current_cell = wp.vec3i(current_cell[0], current_cell[1], current_cell[2]+step_z); tmax_z += tdelta_z
        else:
            if tmax_y < tmax_z:
                if tmax_y > shadow_distance: break
                current_cell = wp.vec3i(current_cell[0], current_cell[1]+step_y, current_cell[2]); tmax_y += tdelta_y
            else:
                if tmax_z > shadow_distance: break
                current_cell = wp.vec3i(current_cell[0], current_cell[1], current_cell[2]+step_z); tmax_z += tdelta_z
    return False

@wp.kernel
def raytrace_quad(
    xyz: wp.array(dtype=wp.vec3), rgb: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.vec3), r: float,
    img: wp.array(dtype=wp.uint8, ndim=3), W: int, H: int, samples: int, sqrtN: int, background: wp.vec3, ambient: float, shadow: float,
    h: float, G: wp.vec3i, cell_start: wp.array(dtype=int), cell_count: wp.array(dtype=int), particle_ids: wp.array(dtype=int),
    camera: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3, light: wp.vec3, fov: float
):
    x, y = wp.tid()
    aspect = float(W) / float(H)
    accum_r = float(0.0); accum_g = float(0.0); accum_b = float(0.0)
    seed    = x + y * W

    for s in range(samples):
        pixel = background

        # Compute grid cell (row, col) for this sample
        si = s % sqrtN   # column index
        sj = s / sqrtN   # row index (integer division)

        jx = float(0.0)
        jy = float(0.0)

        if samples > 1:
            rng = wp.rand_init(seed, s)
            # Stratified jitter: cell center offset + rand[0,1]/2 within cell
            jx = (float(si) + wp.randf(rng) * 0.5) / float(sqrtN) - 0.5
            jy = (float(sj) + wp.randf(rng) * 0.5) / float(sqrtN) - 0.5

        if aspect >= 1.0:
            u = ((float(x) + 0.5 + jx) / float(W) - 0.5) * fov
            v = -((float(y) + 0.5 + jy) / float(H) - 0.5) * fov / aspect
        else:
            u = ((float(x) + 0.5 + jx) / float(W) - 0.5) * fov * aspect
            v = -((float(y) + 0.5 + jy) / float(H) - 0.5) * fov

        ray_dir = wp.normalize(fwd + right * u + up * v)
        ray_origin = camera

        inv_dir = wp.vec3(
            1.0/ray_dir[0] if wp.abs(ray_dir[0]) > 1e-6 else 1e10,
            1.0/ray_dir[1] if wp.abs(ray_dir[1]) > 1e-6 else 1e10,
            1.0/ray_dir[2] if wp.abs(ray_dir[2]) > 1e-6 else 1e10
        )

        current_cell = world_to_grid(ray_origin, h, G)
        if ray_origin[0]<0.0 or ray_origin[0]>float(G.x)*h or ray_origin[1]<0.0 or ray_origin[1]>float(G.y)*h or ray_origin[2]<0.0 or ray_origin[2]>float(G.z)*h:
            t_min = float(-1e10); t_max = float(1e10)
            t1=(0.0-ray_origin[0])*inv_dir[0]; t2=(float(G.x)*h-ray_origin[0])*inv_dir[0]
            t_min=wp.max(t_min,wp.min(t1,t2));  t_max=wp.min(t_max,wp.max(t1,t2))
            t1=(0.0-ray_origin[1])*inv_dir[1]; t2=(float(G.y)*h-ray_origin[1])*inv_dir[1]
            t_min=wp.max(t_min,wp.min(t1,t2));  t_max=wp.min(t_max,wp.max(t1,t2))
            t1=(0.0-ray_origin[2])*inv_dir[2]; t2=(float(G.z)*h-ray_origin[2])*inv_dir[2]
            t_min=wp.max(t_min,wp.min(t1,t2));  t_max=wp.min(t_max,wp.max(t1,t2))
            if t_min > t_max or t_max < 0.0:
                img[y,x,0]=wp.uint8(0); img[y,x,1]=wp.uint8(0); img[y,x,2]=wp.uint8(0)
                return
            entry = ray_origin + ray_dir * (wp.max(0.0, t_min) + 0.0001)
            current_cell = world_to_grid(entry, h, G)

        step_x = 1 if ray_dir[0]>0.0 else -1
        step_y = 1 if ray_dir[1]>0.0 else -1
        step_z = 1 if ray_dir[2]>0.0 else -1
        tmax_x = ((float(current_cell[0])+(1.0 if ray_dir[0]>0.0 else 0.0))*h - ray_origin[0])*inv_dir[0]
        tmax_y = ((float(current_cell[1])+(1.0 if ray_dir[1]>0.0 else 0.0))*h - ray_origin[1])*inv_dir[1]
        tmax_z = ((float(current_cell[2])+(1.0 if ray_dir[2]>0.0 else 0.0))*h - ray_origin[2])*inv_dir[2]
        tdelta_x = h*wp.abs(inv_dir[0])
        tdelta_y = h*wp.abs(inv_dir[1])
        tdelta_z = h*wp.abs(inv_dir[2])

        closest_t = float(1e10); hit_id = int(-1)

        for step in range(G.x + G.y + G.z):
            if current_cell[0]<0 or current_cell[0]>=G.x or \
            current_cell[1]<0 or current_cell[1]>=G.y or \
            current_cell[2]<0 or current_cell[2]>=G.z:
                break
            cell_id   = flat_cell(current_cell[0], current_cell[1], current_cell[2], G)
            start_idx = cell_start[cell_id]
            count = cell_count[cell_id]
            for i in range(count):
                pid = particle_ids[start_idx + i]
                t = intersect_quad(ray_origin, ray_dir, xyz[pid], r, rot[pid])
                if t > 0.001 and t < closest_t:
                    closest_t = t; hit_id = pid
            if hit_id >= 0 and closest_t < wp.min(tmax_x, wp.min(tmax_y, tmax_z)):
                break
            if tmax_x < tmax_y:
                if tmax_x < tmax_z:
                    current_cell = wp.vec3i(current_cell[0]+step_x, current_cell[1], current_cell[2]); tmax_x+=tdelta_x
                else:
                    current_cell = wp.vec3i(current_cell[0], current_cell[1], current_cell[2]+step_z); tmax_z+=tdelta_z
            else:
                if tmax_y < tmax_z:
                    current_cell = wp.vec3i(current_cell[0], current_cell[1]+step_y, current_cell[2]); tmax_y+=tdelta_y
                else:
                    current_cell = wp.vec3i(current_cell[0], current_cell[1], current_cell[2]+step_z); tmax_z+=tdelta_z

        if hit_id >= 0:
            hit_pt = ray_origin + ray_dir * closest_t
            normal = rot[hit_id]

            # flip if facing away from camera — keeps shading consistent
            if wp.dot(normal, ray_dir) > 0.0:
                normal = -normal

            light_dir = wp.normalize(light - hit_pt)
            diffuse   = 0.6 * wp.pow(wp.max(0.0, wp.dot(normal, light_dir)), 4.0)
            shadow_factor = float(1.0)
            if diffuse > 0.0:
                if shadow_quad(
                    xyz, rot, r,
                    hit_pt + normal * h * 0.01, light_dir, wp.length(light - hit_pt),
                    h, G, cell_start, cell_count, particle_ids
                ):
                    shadow_factor = shadow

            pixel = rgb[hit_id] * (ambient + diffuse * shadow_factor)

        accum_r += pixel[0]; accum_g += pixel[1]; accum_b += pixel[2]

    inv_n = 1.0 / float(samples)
    img[y, x, 0] = wp.uint8(wp.clamp(accum_b * inv_n, 0.0, 255.0))
    img[y, x, 1] = wp.uint8(wp.clamp(accum_g * inv_n, 0.0, 255.0))
    img[y, x, 2] = wp.uint8(wp.clamp(accum_r * inv_n, 0.0, 255.0))

# Oriented ellipsoid
@wp.func
def intersect_ellipsoid(
    ro: wp.vec3, rd: wp.vec3,
    center: wp.vec3,
    rx: float, ry: float, rz: float,
    rot: wp.vec3
) -> float:
    mat = dir_to_mat(rot)
    rt   = wp.transpose(mat)
    ro_l = rt @ (ro - center)
    rd_l = rt @ rd

    # Scale into unit-sphere space
    iro  = wp.vec3(1.0/rx, 1.0/ry, 1.0/rz)
    ro_s = wp.cw_mul(ro_l, iro)
    rd_s = wp.cw_mul(rd_l, iro)

    # Full quadratic: a*t^2 + 2*b*t + c = 0
    a    = wp.dot(rd_s, rd_s)
    b    = wp.dot(ro_s, rd_s)
    c    = wp.dot(ro_s, ro_s) - 1.0
    disc = b * b - a * c

    if disc < 1e-7:
        return -1.0
    if a < 1e-12:
        return -1.0

    sq = wp.sqrt(disc)
    t  = (-b - sq) / a
    if t < 0.001:
        t = (-b + sq) / a
    if t < 0.001:
        return -1.0

    return t 

@wp.func
def normal_ellipsoid(
    hit_pt: wp.vec3, center: wp.vec3,
    rx: float, ry: float, rz: float,
    rot: wp.vec3
) -> wp.vec3:
    mat = dir_to_mat(rot)
    rt    = wp.transpose(mat)
    local = rt @ (hit_pt - center)
    grad  = wp.vec3(local[0]/(rx*rx), local[1]/(ry*ry), local[2]/(rz*rz))
    return wp.normalize(mat @ grad)

@wp.func
def shadow_ellipsoid(
    xyz: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.vec3), rx: float, ry: float, rz: float,
    ray_origin: wp.vec3, ray_dir: wp.vec3, shadow_distance: float,
    h: float, G: wp.vec3i, cell_start: wp.array(dtype=int), cell_count: wp.array(dtype=int), particle_ids: wp.array(dtype=int)
) -> bool:
    inv_dir = wp.vec3(
        1.0/ray_dir[0] if wp.abs(ray_dir[0]) > 1e-6 else 1e10,
        1.0/ray_dir[1] if wp.abs(ray_dir[1]) > 1e-6 else 1e10,
        1.0/ray_dir[2] if wp.abs(ray_dir[2]) > 1e-6 else 1e10
    )
    current_cell = world_to_grid(ray_origin, h, G)
    if ray_origin[0] < 0.0 or ray_origin[0] > float(G.x)*h or \
       ray_origin[1] < 0.0 or ray_origin[1] > float(G.y)*h or \
       ray_origin[2] < 0.0 or ray_origin[2] > float(G.z)*h:
        return False

    step_x = 1 if ray_dir[0] > 0.0 else -1
    step_y = 1 if ray_dir[1] > 0.0 else -1
    step_z = 1 if ray_dir[2] > 0.0 else -1

    tmax_x = ((float(current_cell[0]) + (1.0 if ray_dir[0]>0.0 else 0.0))*h - ray_origin[0]) * inv_dir[0]
    tmax_y = ((float(current_cell[1]) + (1.0 if ray_dir[1]>0.0 else 0.0))*h - ray_origin[1]) * inv_dir[1]
    tmax_z = ((float(current_cell[2]) + (1.0 if ray_dir[2]>0.0 else 0.0))*h - ray_origin[2]) * inv_dir[2]
    tdelta_x = h * wp.abs(inv_dir[0])
    tdelta_y = h * wp.abs(inv_dir[1])
    tdelta_z = h * wp.abs(inv_dir[2])

    for step in range(G.x + G.y + G.z):
        if current_cell[0]<0 or current_cell[0]>=G.x or \
           current_cell[1]<0 or current_cell[1]>=G.y or \
           current_cell[2]<0 or current_cell[2]>=G.z:
            break
        cell_id   = flat_cell(current_cell[0], current_cell[1], current_cell[2], G)
        start_idx = cell_start[cell_id]
        count     = cell_count[cell_id]

        t = float(-1.0)
        for i in range(count):
            pid = particle_ids[start_idx + i]
            
            # SHAPE LINE 1
            t = intersect_ellipsoid(ray_origin, ray_dir, xyz[pid], rx, ry, rz, rot[pid])
            
            if t > 0.001 and t < shadow_distance:
                return True
        if tmax_x < tmax_y:
            if tmax_x < tmax_z:
                if tmax_x > shadow_distance: break
                current_cell = wp.vec3i(current_cell[0]+step_x, current_cell[1], current_cell[2])
                tmax_x += tdelta_x
            else:
                if tmax_z > shadow_distance: break
                current_cell = wp.vec3i(current_cell[0], current_cell[1], current_cell[2]+step_z)
                tmax_z += tdelta_z
        else:
            if tmax_y < tmax_z:
                if tmax_y > shadow_distance: break
                current_cell = wp.vec3i(current_cell[0], current_cell[1]+step_y, current_cell[2])
                tmax_y += tdelta_y
            else:
                if tmax_z > shadow_distance: break
                current_cell = wp.vec3i(current_cell[0], current_cell[1], current_cell[2]+step_z)
                tmax_z += tdelta_z
    return False         

@wp.kernel
def raytrace_ellipsoid(
    xyz: wp.array(dtype=wp.vec3), rgb: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.vec3), rx: float, ry: float, rz: float,
    img: wp.array(dtype=wp.uint8, ndim=3), W: int, H: int, samples: int, sqrtN: int, background: wp.vec3, ambient: float, shadow: float,
    h: float, G: wp.vec3i, cell_start: wp.array(dtype=int), cell_count: wp.array(dtype=int), particle_ids: wp.array(dtype=int),
    camera: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3, light: wp.vec3, fov: float
):
    x, y = wp.tid()
    aspect  = float(W) / float(H)
    accum_r = float(0.0); accum_g = float(0.0); accum_b = float(0.0)
    seed    = x + y * W

    for s in range(samples):
        pixel = background

        # Compute grid cell (row, col) for this sample
        si = s % sqrtN   # column index
        sj = s / sqrtN   # row index (integer division)

        jx = float(0.0)
        jy = float(0.0)

        if samples > 1:
            rng = wp.rand_init(seed, s)
            # Stratified jitter: cell center offset + rand[0,1]/2 within cell
            jx = (float(si) + wp.randf(rng) * 0.5) / float(sqrtN) - 0.5
            jy = (float(sj) + wp.randf(rng) * 0.5) / float(sqrtN) - 0.5

        if aspect >= 1.0:
            u = ((float(x) + 0.5 + jx) / float(W) - 0.5) * fov
            v = -((float(y) + 0.5 + jy) / float(H) - 0.5) * fov / aspect
        else:
            u = ((float(x) + 0.5 + jx) / float(W) - 0.5) * fov * aspect
            v = -((float(y) + 0.5 + jy) / float(H) - 0.5) * fov

        ray_dir = wp.normalize(fwd + right * u + up * v)
        ray_origin = camera

        inv_dir = wp.vec3(
            1.0/ray_dir[0] if wp.abs(ray_dir[0]) > 1e-6 else 1e10,
            1.0/ray_dir[1] if wp.abs(ray_dir[1]) > 1e-6 else 1e10,
            1.0/ray_dir[2] if wp.abs(ray_dir[2]) > 1e-6 else 1e10
        )

        if ray_origin[0]<0.0 or ray_origin[0]>float(G.x)*h or \
           ray_origin[1]<0.0 or ray_origin[1]>float(G.y)*h or \
           ray_origin[2]<0.0 or ray_origin[2]>float(G.z)*h:
            t_min = float(-1e10); t_max = float(1e10)
            t1=(0.0-ray_origin[0])*inv_dir[0]; t2=(float(G.x)*h-ray_origin[0])*inv_dir[0]
            t_min=wp.max(t_min,wp.min(t1,t2));  t_max=wp.min(t_max,wp.max(t1,t2))
            t1=(0.0-ray_origin[1])*inv_dir[1]; t2=(float(G.y)*h-ray_origin[1])*inv_dir[1]
            t_min=wp.max(t_min,wp.min(t1,t2));  t_max=wp.min(t_max,wp.max(t1,t2))
            t1=(0.0-ray_origin[2])*inv_dir[2]; t2=(float(G.z)*h-ray_origin[2])*inv_dir[2]
            t_min=wp.max(t_min,wp.min(t1,t2));  t_max=wp.min(t_max,wp.max(t1,t2))
            if t_min > t_max or t_max < 0.0:
                continue
            entry  = ray_origin + ray_dir * (wp.max(0.0, t_min) + 0.0001)
            current_cell = world_to_grid(entry, h, G)

        step_x = 1 if ray_dir[0]>0.0 else -1
        step_y = 1 if ray_dir[1]>0.0 else -1
        step_z = 1 if ray_dir[2]>0.0 else -1
        tmax_x = ((float(current_cell[0])+(1.0 if ray_dir[0]>0.0 else 0.0))*h - ray_origin[0])*inv_dir[0]
        tmax_y = ((float(current_cell[1])+(1.0 if ray_dir[1]>0.0 else 0.0))*h - ray_origin[1])*inv_dir[1]
        tmax_z = ((float(current_cell[2])+(1.0 if ray_dir[2]>0.0 else 0.0))*h - ray_origin[2])*inv_dir[2]
        tdelta_x = h*wp.abs(inv_dir[0])
        tdelta_y = h*wp.abs(inv_dir[1])
        tdelta_z = h*wp.abs(inv_dir[2])

        closest_t = float(1e10); hit_id = int(-1)

        for step in range(G.x + G.y + G.z):
            if current_cell[0]<0 or current_cell[0]>=G.x or \
               current_cell[1]<0 or current_cell[1]>=G.y or \
               current_cell[2]<0 or current_cell[2]>=G.z:
                break
            cell_id   = flat_cell(current_cell[0], current_cell[1], current_cell[2], G)
            start_idx = cell_start[cell_id]
            count     = cell_count[cell_id]
            t = float(-1.0)
            for i in range(count):
                pid = particle_ids[start_idx + i]
                
                # SHAPE LINE 1
                t = intersect_ellipsoid(ray_origin, ray_dir, xyz[pid], rx, ry, rz, rot[pid])

                if t > 0.001 and t < closest_t:
                    closest_t = t; hit_id = pid
            if hit_id >= 0 and closest_t < wp.min(tmax_x, wp.min(tmax_y, tmax_z)):
                break
            if tmax_x < tmax_y:
                if tmax_x < tmax_z:
                    current_cell = wp.vec3i(current_cell[0]+step_x, current_cell[1], current_cell[2]); tmax_x+=tdelta_x
                else:
                    current_cell = wp.vec3i(current_cell[0], current_cell[1], current_cell[2]+step_z); tmax_z+=tdelta_z
            else:
                if tmax_y < tmax_z:
                    current_cell = wp.vec3i(current_cell[0], current_cell[1]+step_y, current_cell[2]); tmax_y+=tdelta_y
                else:
                    current_cell = wp.vec3i(current_cell[0], current_cell[1], current_cell[2]+step_z); tmax_z+=tdelta_z

        if hit_id >= 0:
            hit_pt = ray_origin + ray_dir * closest_t

            # SHAPE LINE 2
            normal = normal_ellipsoid(hit_pt, xyz[hit_id], rx, ry, rz, rot[hit_id])

            if wp.dot(normal, ray_dir) > 0.0:
                normal = -normal
            light_dir = wp.normalize(light - hit_pt)
            diffuse   = wp.max(0.0, wp.dot(normal, light_dir))
            shadow_factor = float(1.0)
            if diffuse > 0.0:

                # SHAPE LINE 4
                shadow_origin = hit_pt + normal * wp.max(rx, wp.max(ry, rz)) * 0.005
                if shadow_ellipsoid(
                    xyz, rot, rx, ry, rz,
                    shadow_origin, light_dir, wp.length(light - hit_pt),
                    h, G, cell_start, cell_count, particle_ids
                ):
                    shadow_factor = shadow

            pixel = rgb[hit_id] * (ambient + diffuse * shadow_factor)

        accum_r += pixel[0]
        accum_g += pixel[1]
        accum_b += pixel[2]

    inv_n = 1.0 / float(samples)
    img[y, x, 0] = wp.uint8(wp.clamp(accum_b * inv_n, 0.0, 255.0))
    img[y, x, 1] = wp.uint8(wp.clamp(accum_g * inv_n, 0.0, 255.0))
    img[y, x, 2] = wp.uint8(wp.clamp(accum_r * inv_n, 0.0, 255.0))

# Oriented prism
@wp.func
def intersect_prism(
    ro: wp.vec3, rd: wp.vec3,
    center: wp.vec3,
    rx: float, ry: float, rz: float,
    rot: wp.vec3
) -> float:
    mat = dir_to_mat(rot)
    rt   = wp.transpose(mat)
    ro_l = rt @ (ro - center)
    rd_l = rt @ rd
    half = wp.vec3(rx, ry, rz)

    ird  = wp.vec3(
        1.0/rd_l[0] if wp.abs(rd_l[0]) > 1e-9 else 1e10,
        1.0/rd_l[1] if wp.abs(rd_l[1]) > 1e-9 else 1e10,
        1.0/rd_l[2] if wp.abs(rd_l[2]) > 1e-9 else 1e10
    )
    t0 = wp.cw_mul((-half - ro_l), ird)
    t1 = wp.cw_mul(( half - ro_l), ird)
    tmin3 = wp.vec3(wp.min(t0[0],t1[0]), wp.min(t0[1],t1[1]), wp.min(t0[2],t1[2]))
    tmax3 = wp.vec3(wp.max(t0[0],t1[0]), wp.max(t0[1],t1[1]), wp.max(t0[2],t1[2]))
    tenter = wp.max(tmin3[0], wp.max(tmin3[1], tmin3[2]))
    texit  = wp.min(tmax3[0], wp.min(tmax3[1], tmax3[2]))
    if texit < 0.001 or tenter > texit:
        return -1.0
    t = tenter if tenter > 0.001 else texit
    return t

@wp.func
def normal_prism(
    hit_pt: wp.vec3, center: wp.vec3,
    rx: float, ry: float, rz: float,
    rot: wp.vec3
) -> wp.vec3:
    mat = dir_to_mat(rot)
    rt    = wp.transpose(mat)
    local = rt @ (hit_pt - center)
    half  = wp.vec3(rx, ry, rz)
    # snap to nearest face
    d     = wp.cw_div(local, half)
    ax    = wp.abs(d[0]); ay = wp.abs(d[1]); az = wp.abs(d[2])
    if ax >= ay and ax >= az:
        n_local = wp.vec3(wp.sign(d[0]), 0.0, 0.0)
    elif ay >= az:
        n_local = wp.vec3(0.0, wp.sign(d[1]), 0.0)
    else:
        n_local = wp.vec3(0.0, 0.0, wp.sign(d[2]))
    return wp.normalize(mat @ n_local)

@wp.func
def shadow_prism(
    xyz: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.vec3), rx: float, ry: float, rz: float,
    ray_origin: wp.vec3, ray_dir: wp.vec3, shadow_distance: float,
    h: float, G: wp.vec3i, cell_start: wp.array(dtype=int), cell_count: wp.array(dtype=int), particle_ids: wp.array(dtype=int)
) -> bool:
    inv_dir = wp.vec3(
        1.0/ray_dir[0] if wp.abs(ray_dir[0]) > 1e-6 else 1e10,
        1.0/ray_dir[1] if wp.abs(ray_dir[1]) > 1e-6 else 1e10,
        1.0/ray_dir[2] if wp.abs(ray_dir[2]) > 1e-6 else 1e10
    )
    current_cell = world_to_grid(ray_origin, h, G)
    if ray_origin[0] < 0.0 or ray_origin[0] > float(G.x)*h or \
       ray_origin[1] < 0.0 or ray_origin[1] > float(G.y)*h or \
       ray_origin[2] < 0.0 or ray_origin[2] > float(G.z)*h:
        return False

    step_x = 1 if ray_dir[0] > 0.0 else -1
    step_y = 1 if ray_dir[1] > 0.0 else -1
    step_z = 1 if ray_dir[2] > 0.0 else -1

    tmax_x = ((float(current_cell[0]) + (1.0 if ray_dir[0]>0.0 else 0.0))*h - ray_origin[0]) * inv_dir[0]
    tmax_y = ((float(current_cell[1]) + (1.0 if ray_dir[1]>0.0 else 0.0))*h - ray_origin[1]) * inv_dir[1]
    tmax_z = ((float(current_cell[2]) + (1.0 if ray_dir[2]>0.0 else 0.0))*h - ray_origin[2]) * inv_dir[2]
    tdelta_x = h * wp.abs(inv_dir[0])
    tdelta_y = h * wp.abs(inv_dir[1])
    tdelta_z = h * wp.abs(inv_dir[2])

    for step in range(G.x + G.y + G.z):
        if current_cell[0]<0 or current_cell[0]>=G.x or \
           current_cell[1]<0 or current_cell[1]>=G.y or \
           current_cell[2]<0 or current_cell[2]>=G.z:
            break
        cell_id   = flat_cell(current_cell[0], current_cell[1], current_cell[2], G)
        start_idx = cell_start[cell_id]
        count     = cell_count[cell_id]
        for i in range(count):
            pid = particle_ids[start_idx + i]

            # SHAPE LINE 1
            t = intersect_prism(ray_origin, ray_dir, xyz[pid], rx, ry, rz, rot[pid])

            if t > 0.001 and t < shadow_distance:
                return True
        if tmax_x < tmax_y:
            if tmax_x < tmax_z:
                if tmax_x > shadow_distance: break
                current_cell = wp.vec3i(current_cell[0]+step_x, current_cell[1], current_cell[2])
                tmax_x += tdelta_x
            else:
                if tmax_z > shadow_distance: break
                current_cell = wp.vec3i(current_cell[0], current_cell[1], current_cell[2]+step_z)
                tmax_z += tdelta_z
        else:
            if tmax_y < tmax_z:
                if tmax_y > shadow_distance: break
                current_cell = wp.vec3i(current_cell[0], current_cell[1]+step_y, current_cell[2])
                tmax_y += tdelta_y
            else:
                if tmax_z > shadow_distance: break
                current_cell = wp.vec3i(current_cell[0], current_cell[1], current_cell[2]+step_z)
                tmax_z += tdelta_z
    return False

@wp.kernel
def raytrace_prism(
    xyz: wp.array(dtype=wp.vec3), rgb: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.vec3), rx: float, ry: float, rz: float,
    img: wp.array(dtype=wp.uint8, ndim=3), W: int, H: int, samples: int, sqrtN: int, background: wp.vec3, ambient: float, shadow: float,
    h: float, G: wp.vec3i, cell_start: wp.array(dtype=int), cell_count: wp.array(dtype=int), particle_ids: wp.array(dtype=int),
    camera: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3, light: wp.vec3, fov: float
):
    x, y = wp.tid()
    aspect  = float(W) / float(H)
    accum_r = float(0.0); accum_g = float(0.0); accum_b = float(0.0)
    seed    = x + y * W

    for s in range(samples):
        pixel = background

        # Compute grid cell (row, col) for this sample
        si = s % sqrtN   # column index
        sj = s / sqrtN   # row index (integer division)

        jx = float(0.0)
        jy = float(0.0)

        if samples > 1:
            rng = wp.rand_init(seed, s)
            # Stratified jitter: cell center offset + rand[0,1]/2 within cell
            jx = (float(si) + wp.randf(rng) * 0.5) / float(sqrtN) - 0.5
            jy = (float(sj) + wp.randf(rng) * 0.5) / float(sqrtN) - 0.5

        if aspect >= 1.0:
            u = ((float(x) + 0.5 + jx) / float(W) - 0.5) * fov
            v = -((float(y) + 0.5 + jy) / float(H) - 0.5) * fov / aspect
        else:
            u = ((float(x) + 0.5 + jx) / float(W) - 0.5) * fov * aspect
            v = -((float(y) + 0.5 + jy) / float(H) - 0.5) * fov

        ray_dir = wp.normalize(fwd + right * u + up * v)
        ray_origin = camera

        inv_dir = wp.vec3(
            1.0/ray_dir[0] if wp.abs(ray_dir[0]) > 1e-6 else 1e10,
            1.0/ray_dir[1] if wp.abs(ray_dir[1]) > 1e-6 else 1e10,
            1.0/ray_dir[2] if wp.abs(ray_dir[2]) > 1e-6 else 1e10
        )

        if ray_origin[0]<0.0 or ray_origin[0]>float(G.x)*h or \
           ray_origin[1]<0.0 or ray_origin[1]>float(G.y)*h or \
           ray_origin[2]<0.0 or ray_origin[2]>float(G.z)*h:
            t_min = float(-1e10); t_max = float(1e10)
            t1=(0.0-ray_origin[0])*inv_dir[0]; t2=(float(G.x)*h-ray_origin[0])*inv_dir[0]
            t_min=wp.max(t_min,wp.min(t1,t2));  t_max=wp.min(t_max,wp.max(t1,t2))
            t1=(0.0-ray_origin[1])*inv_dir[1]; t2=(float(G.y)*h-ray_origin[1])*inv_dir[1]
            t_min=wp.max(t_min,wp.min(t1,t2));  t_max=wp.min(t_max,wp.max(t1,t2))
            t1=(0.0-ray_origin[2])*inv_dir[2]; t2=(float(G.z)*h-ray_origin[2])*inv_dir[2]
            t_min=wp.max(t_min,wp.min(t1,t2));  t_max=wp.min(t_max,wp.max(t1,t2))
            if t_min > t_max or t_max < 0.0:
                continue
            entry  = ray_origin + ray_dir * (wp.max(0.0, t_min) + 0.0001)
            current_cell = world_to_grid(entry, h, G)

        step_x = 1 if ray_dir[0]>0.0 else -1
        step_y = 1 if ray_dir[1]>0.0 else -1
        step_z = 1 if ray_dir[2]>0.0 else -1
        tmax_x = ((float(current_cell[0])+(1.0 if ray_dir[0]>0.0 else 0.0))*h - ray_origin[0])*inv_dir[0]
        tmax_y = ((float(current_cell[1])+(1.0 if ray_dir[1]>0.0 else 0.0))*h - ray_origin[1])*inv_dir[1]
        tmax_z = ((float(current_cell[2])+(1.0 if ray_dir[2]>0.0 else 0.0))*h - ray_origin[2])*inv_dir[2]
        tdelta_x = h*wp.abs(inv_dir[0])
        tdelta_y = h*wp.abs(inv_dir[1])
        tdelta_z = h*wp.abs(inv_dir[2])

        closest_t = float(1e10); hit_id = int(-1)

        for step in range(G.x + G.y + G.z):
            if current_cell[0]<0 or current_cell[0]>=G.x or \
               current_cell[1]<0 or current_cell[1]>=G.y or \
               current_cell[2]<0 or current_cell[2]>=G.z:
                break
            cell_id   = flat_cell(current_cell[0], current_cell[1], current_cell[2], G)
            start_idx = cell_start[cell_id]
            count     = cell_count[cell_id]
            for i in range(count):
                pid = particle_ids[start_idx + i]
                
                # SHAPE LINE 1
                t = intersect_prism(ray_origin, ray_dir, xyz[pid], rx, ry, rz, rot[pid])
                
                if t > 0.001 and t < closest_t:
                    closest_t = t; hit_id = pid
            if hit_id >= 0 and closest_t < wp.min(tmax_x, wp.min(tmax_y, tmax_z)):
                break
            if tmax_x < tmax_y:
                if tmax_x < tmax_z:
                    current_cell = wp.vec3i(current_cell[0]+step_x, current_cell[1], current_cell[2]); tmax_x+=tdelta_x
                else:
                    current_cell = wp.vec3i(current_cell[0], current_cell[1], current_cell[2]+step_z); tmax_z+=tdelta_z
            else:
                if tmax_y < tmax_z:
                    current_cell = wp.vec3i(current_cell[0], current_cell[1]+step_y, current_cell[2]); tmax_y+=tdelta_y
                else:
                    current_cell = wp.vec3i(current_cell[0], current_cell[1], current_cell[2]+step_z); tmax_z+=tdelta_z

        if hit_id >= 0:
            hit_pt = ray_origin + ray_dir * closest_t
            
            # SHAPE LINE 2
            normal = normal_prism(hit_pt, xyz[hit_id], rx, ry, rz, rot[hit_id])
            
            if wp.dot(normal, ray_dir) > 0.0:
                normal = -normal
            light_dir = wp.normalize(light - hit_pt)
            diffuse   = wp.max(0.0, wp.dot(normal, light_dir))
            shadow_factor = float(1.0)
            if diffuse > 0.0:
                light_dist    = wp.length(light - hit_pt)
                
                # SHAPE LINE 3 (shadow func name)
                shadow_origin = hit_pt + normal * wp.max(rx, wp.max(ry, rz)) * 0.005
                if shadow_prism(
                    xyz, rot, rx, ry, rz,
                    shadow_origin, light_dir, light_dist,
                    h, G, cell_start, cell_count, particle_ids
                ):
                    shadow_factor = shadow

            pixel = rgb[hit_id] * (ambient + diffuse * shadow_factor)

        accum_r += pixel[0]; accum_g += pixel[1]; accum_b += pixel[2]

    inv_n = 1.0 / float(samples)
    img[y, x, 0] = wp.uint8(wp.clamp(accum_b * inv_n, 0.0, 255.0))
    img[y, x, 1] = wp.uint8(wp.clamp(accum_g * inv_n, 0.0, 255.0))
    img[y, x, 2] = wp.uint8(wp.clamp(accum_r * inv_n, 0.0, 255.0))

# Cylinder
@wp.func
def intersect_cylinder(
    ray_origin: wp.vec3, ray_dir: wp.vec3,
    p0: wp.vec3, p1: wp.vec3,
    radius: float,
    end: wp.bool
) -> float:
    line_dir = p1 - p0
    line_length = wp.length(line_dir)
    
    if line_length < 1e-6:
        # Degenerate line, treat as sphere
        oc = ray_origin - p0
        b = wp.dot(oc, ray_dir)
        c = wp.dot(oc, oc) - radius * radius
        discriminant = b * b - c
        
        if discriminant >= 0.0:
            t = -b - wp.sqrt(discriminant)
            if t < 0.001:
                t = -b + wp.sqrt(discriminant)
            if t > 0.001:
                return t
        return float(-1.0)
    
    line_axis = line_dir / line_length
    
    # Closest approach on infinite cylinder
    oc = ray_origin - p0
    oc_proj = wp.dot(oc, line_axis)
    ray_proj = wp.dot(ray_dir, line_axis)
    
    # Perpendicular components
    oc_perp = oc - line_axis * oc_proj
    ray_perp = ray_dir - line_axis * ray_proj
    
    # Quadratic coefficients for infinite cylinder
    a = wp.dot(ray_perp, ray_perp)
    b = wp.dot(oc_perp, ray_perp)
    c = wp.dot(oc_perp, oc_perp) - radius * radius
    
    discriminant = b * b - a * c
    
    closest_t = float(-1.0)
    
    # Test cylinder body
    if discriminant >= 0.0 and a > 1e-9:
        sqrt_disc = wp.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / a
        t2 = (-b + sqrt_disc) / a
        
        # Check t1
        if t1 > 0.001:
            hit_point = ray_origin + ray_dir * t1
            proj_on_axis = wp.dot(hit_point - p0, line_axis)
            if proj_on_axis >= 0.0 and proj_on_axis <= line_length:
                if closest_t < 0.0 or t1 < closest_t:
                    closest_t = t1

        # Check t2
        if t2 > 0.001:
            hit_point = ray_origin + ray_dir * t2
            proj_on_axis = wp.dot(hit_point - p0, line_axis)
            if proj_on_axis >= 0.0 and proj_on_axis <= line_length:
                if closest_t < 0.0 or t2 < closest_t:
                    closest_t = t2
    
    # Test spherical cap at p0
    oc_cap0 = ray_origin - p0
    b_cap0 = wp.dot(oc_cap0, ray_dir)
    c_cap0 = wp.dot(oc_cap0, oc_cap0) - radius * radius
    disc_cap0 = b_cap0 * b_cap0 - c_cap0
    
    if disc_cap0 >= 0.0:
        t_cap0 = -b_cap0 - wp.sqrt(disc_cap0)
        if t_cap0 < 0.001:
            t_cap0 = -b_cap0 + wp.sqrt(disc_cap0)
        
        if t_cap0 > 0.001:
            hit_point = ray_origin + ray_dir * t_cap0
            proj = wp.dot(hit_point - p0, line_axis)
            if proj <= 0.0:  # Behind p0
                if closest_t < 0.0 or t_cap0 < closest_t:
                    closest_t = t_cap0
    
    # Test spherical cap at p1 if end
    if end:
        oc_cap1 = ray_origin - p1
        b_cap1 = wp.dot(oc_cap1, ray_dir)
        c_cap1 = wp.dot(oc_cap1, oc_cap1) - radius * radius
        disc_cap1 = b_cap1 * b_cap1 - c_cap1
        
        if disc_cap1 >= 0.0:
            t_cap1 = -b_cap1 - wp.sqrt(disc_cap1)
            if t_cap1 < 0.001:
                t_cap1 = -b_cap1 + wp.sqrt(disc_cap1)
            
            if t_cap1 > 0.001:
                hit_point = ray_origin + ray_dir * t_cap1
                proj = wp.dot(hit_point - p0, line_axis)
                if proj >= line_length:  # Beyond p1
                    if closest_t < 0.0 or t_cap1 < closest_t:
                        closest_t = t_cap1
    
    return closest_t

@wp.func
def normal_cylinder(hit_point: wp.vec3, p0: wp.vec3, p1: wp.vec3) -> wp.vec3:
    line_dir = p1 - p0
    line_length = wp.length(line_dir)
    if line_length < 1e-6:
        return wp.normalize(hit_point - p0)
    line_axis = line_dir / line_length
    proj = wp.dot(hit_point - p0, line_axis)
    eps = line_length * 0.01          # 1% of segment length
    if proj <= eps:
        return wp.normalize(hit_point - p0)
    if proj >= line_length - eps:
        return wp.normalize(hit_point - p1)
    return wp.normalize(hit_point - (p0 + line_axis * proj))

@wp.func
def shadow_cylinder(
    xyz: wp.array(dtype=wp.vec3), next: wp.array(dtype=int), r: float,
    ray_origin: wp.vec3, ray_dir: wp.vec3, shadow_distance: float,
    h: float, G: wp.vec3i, cell_start: wp.array(dtype=int), cell_count: wp.array(dtype=int), particle_ids: wp.array(dtype=int)
) -> bool:
    inv_dir = wp.vec3(
        1.0/ray_dir[0] if wp.abs(ray_dir[0]) > 1e-6 else 1e10,
        1.0/ray_dir[1] if wp.abs(ray_dir[1]) > 1e-6 else 1e10,
        1.0/ray_dir[2] if wp.abs(ray_dir[2]) > 1e-6 else 1e10
    )
    current_cell = world_to_grid(ray_origin, h, G)
    if ray_origin[0] < 0.0 or ray_origin[0] > float(G.x)*h or \
       ray_origin[1] < 0.0 or ray_origin[1] > float(G.y)*h or \
       ray_origin[2] < 0.0 or ray_origin[2] > float(G.z)*h:
        return False

    step_x = 1 if ray_dir[0] > 0.0 else -1
    step_y = 1 if ray_dir[1] > 0.0 else -1
    step_z = 1 if ray_dir[2] > 0.0 else -1

    tmax_x = ((float(current_cell[0]) + (1.0 if ray_dir[0]>0.0 else 0.0))*h - ray_origin[0]) * inv_dir[0]
    tmax_y = ((float(current_cell[1]) + (1.0 if ray_dir[1]>0.0 else 0.0))*h - ray_origin[1]) * inv_dir[1]
    tmax_z = ((float(current_cell[2]) + (1.0 if ray_dir[2]>0.0 else 0.0))*h - ray_origin[2]) * inv_dir[2]
    tdelta_x = h * wp.abs(inv_dir[0])
    tdelta_y = h * wp.abs(inv_dir[1])
    tdelta_z = h * wp.abs(inv_dir[2])

    for step in range(G.x + G.y + G.z):
        if current_cell[0]<0 or current_cell[0]>=G.x or \
           current_cell[1]<0 or current_cell[1]>=G.y or \
           current_cell[2]<0 or current_cell[2]>=G.z:
            break
        cell_id   = flat_cell(current_cell[0], current_cell[1], current_cell[2], G)
        start_idx = cell_start[cell_id]
        count     = cell_count[cell_id]

        t = float(-1.0)
        for i in range(count):
            pid = particle_ids[start_idx + i]
            
            # SHAPE LINE 1
            next_id = next[pid]            
            if next_id < 0:
                continue            
            p0 = xyz[pid]
            p1 = xyz[next_id]
            if next[next_id] < 0:
                end = True
            else:
                end = False          
            t = intersect_cylinder(ray_origin, ray_dir, p0, p1, r, end)
            
            if t > 0.001 and t < shadow_distance:
                return True
        if tmax_x < tmax_y:
            if tmax_x < tmax_z:
                if tmax_x > shadow_distance: break
                current_cell = wp.vec3i(current_cell[0]+step_x, current_cell[1], current_cell[2])
                tmax_x += tdelta_x
            else:
                if tmax_z > shadow_distance: break
                current_cell = wp.vec3i(current_cell[0], current_cell[1], current_cell[2]+step_z)
                tmax_z += tdelta_z
        else:
            if tmax_y < tmax_z:
                if tmax_y > shadow_distance: break
                current_cell = wp.vec3i(current_cell[0], current_cell[1]+step_y, current_cell[2])
                tmax_y += tdelta_y
            else:
                if tmax_z > shadow_distance: break
                current_cell = wp.vec3i(current_cell[0], current_cell[1], current_cell[2]+step_z)
                tmax_z += tdelta_z
    return False

@wp.kernel
def raytrace_cylinder(
    xyz: wp.array(dtype=wp.vec3), rgb: wp.array(dtype=wp.vec3), next: wp.array(dtype=int), r: float,
    img: wp.array(dtype=wp.uint8, ndim=3), W: int, H: int, samples: int, sqrtN: int, background: wp.vec3, ambient: float, shadow: float,
    h: float, G: wp.vec3i, cell_start: wp.array(dtype=int), cell_count: wp.array(dtype=int), particle_ids: wp.array(dtype=int),
    camera: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3, light: wp.vec3, fov: float
):
    x, y = wp.tid()
    aspect  = float(W) / float(H)
    accum_r = float(0.0); accum_g = float(0.0); accum_b = float(0.0)
    seed    = x + y * W

    for s in range(samples):
        pixel = background

        # Compute grid cell (row, col) for this sample
        si = s % sqrtN   # column index
        sj = s / sqrtN   # row index (integer division)

        jx = float(0.0)
        jy = float(0.0)

        if samples > 1:
            rng = wp.rand_init(seed, s)
            # Stratified jitter: cell center offset + rand[0,1]/2 within cell
            jx = (float(si) + wp.randf(rng) * 0.5) / float(sqrtN) - 0.5
            jy = (float(sj) + wp.randf(rng) * 0.5) / float(sqrtN) - 0.5

        if aspect >= 1.0:
            u = ((float(x) + 0.5 + jx) / float(W) - 0.5) * fov
            v = -((float(y) + 0.5 + jy) / float(H) - 0.5) * fov / aspect
        else:
            u = ((float(x) + 0.5 + jx) / float(W) - 0.5) * fov * aspect
            v = -((float(y) + 0.5 + jy) / float(H) - 0.5) * fov

        ray_dir = wp.normalize(fwd + right * u + up * v)
        ray_origin = camera

        inv_dir = wp.vec3(
            1.0/ray_dir[0] if wp.abs(ray_dir[0]) > 1e-6 else 1e10,
            1.0/ray_dir[1] if wp.abs(ray_dir[1]) > 1e-6 else 1e10,
            1.0/ray_dir[2] if wp.abs(ray_dir[2]) > 1e-6 else 1e10
        )

        if ray_origin[0]<0.0 or ray_origin[0]>float(G.x)*h or \
           ray_origin[1]<0.0 or ray_origin[1]>float(G.y)*h or \
           ray_origin[2]<0.0 or ray_origin[2]>float(G.z)*h:
            t_min = float(-1e10); t_max = float(1e10)
            t1=(0.0-ray_origin[0])*inv_dir[0]; t2=(float(G.x)*h-ray_origin[0])*inv_dir[0]
            t_min=wp.max(t_min,wp.min(t1,t2));  t_max=wp.min(t_max,wp.max(t1,t2))
            t1=(0.0-ray_origin[1])*inv_dir[1]; t2=(float(G.y)*h-ray_origin[1])*inv_dir[1]
            t_min=wp.max(t_min,wp.min(t1,t2));  t_max=wp.min(t_max,wp.max(t1,t2))
            t1=(0.0-ray_origin[2])*inv_dir[2]; t2=(float(G.z)*h-ray_origin[2])*inv_dir[2]
            t_min=wp.max(t_min,wp.min(t1,t2));  t_max=wp.min(t_max,wp.max(t1,t2))
            if t_min > t_max or t_max < 0.0:
                continue
            entry  = ray_origin + ray_dir * (wp.max(0.0, t_min) + 0.0001)
            current_cell = world_to_grid(entry, h, G)

        step_x = 1 if ray_dir[0]>0.0 else -1
        step_y = 1 if ray_dir[1]>0.0 else -1
        step_z = 1 if ray_dir[2]>0.0 else -1
        tmax_x = ((float(current_cell[0])+(1.0 if ray_dir[0]>0.0 else 0.0))*h - ray_origin[0])*inv_dir[0]
        tmax_y = ((float(current_cell[1])+(1.0 if ray_dir[1]>0.0 else 0.0))*h - ray_origin[1])*inv_dir[1]
        tmax_z = ((float(current_cell[2])+(1.0 if ray_dir[2]>0.0 else 0.0))*h - ray_origin[2])*inv_dir[2]
        tdelta_x = h*wp.abs(inv_dir[0])
        tdelta_y = h*wp.abs(inv_dir[1])
        tdelta_z = h*wp.abs(inv_dir[2])

        closest_t = float(1e10); hit_id = int(-1)

        for step in range(G.x + G.y + G.z):
            if current_cell[0]<0 or current_cell[0]>=G.x or \
               current_cell[1]<0 or current_cell[1]>=G.y or \
               current_cell[2]<0 or current_cell[2]>=G.z:
                break
            cell_id   = flat_cell(current_cell[0], current_cell[1], current_cell[2], G)
            start_idx = cell_start[cell_id]
            count     = cell_count[cell_id]
            t = float(-1.0)
            for i in range(count):
                pid = particle_ids[start_idx + i]
                
                # SHAPE LINE 1
                next_id = next[pid]            
                if next_id < 0:
                    continue            
                p0 = xyz[pid]
                p1 = xyz[next_id]
                if next[next_id] < 0:
                    end = True
                else:
                    end = False          
                t = intersect_cylinder(ray_origin, ray_dir, p0, p1, r, end)

                if t > 0.001 and t < closest_t:
                    closest_t = t; hit_id = pid
            if hit_id >= 0 and closest_t < wp.min(tmax_x, wp.min(tmax_y, tmax_z)):
                break
            if tmax_x < tmax_y:
                if tmax_x < tmax_z:
                    current_cell = wp.vec3i(current_cell[0]+step_x, current_cell[1], current_cell[2]); tmax_x+=tdelta_x
                else:
                    current_cell = wp.vec3i(current_cell[0], current_cell[1], current_cell[2]+step_z); tmax_z+=tdelta_z
            else:
                if tmax_y < tmax_z:
                    current_cell = wp.vec3i(current_cell[0], current_cell[1]+step_y, current_cell[2]); tmax_y+=tdelta_y
                else:
                    current_cell = wp.vec3i(current_cell[0], current_cell[1], current_cell[2]+step_z); tmax_z+=tdelta_z

        if hit_id >= 0:
            hit_pt = ray_origin + ray_dir * closest_t

            # SHAPE LINE 2
            p0 = xyz[hit_id]
            next_id = next[hit_id]
            if next_id >= 0:          
                p1 = xyz[next_id]   
                normal = normal_cylinder(hit_pt, p0, p1)
            else:
                p1 = p0                        
                normal = wp.normalize(hit_pt - p0)

            if wp.dot(normal, ray_dir) > 0.0:
                normal = -normal
            light_dir = wp.normalize(light - hit_pt)
            diffuse   = wp.max(0.0, wp.dot(normal, light_dir))
            shadow_factor = float(1.0)
            if diffuse > 0.0:

                # SHAPE LINE 3
                shadow_origin = hit_pt + normal * r * 0.005
                if shadow_cylinder(
                    xyz, next, r,
                    shadow_origin, light_dir, wp.length(light - hit_pt),
                    h, G, cell_start, cell_count, particle_ids
                ):
                    shadow_factor = shadow

            pixel = rgb[hit_id] * (ambient + diffuse * shadow_factor)

        accum_r += pixel[0]
        accum_g += pixel[1]
        accum_b += pixel[2]

    inv_n = 1.0 / float(samples)
    img[y, x, 0] = wp.uint8(wp.clamp(accum_b * inv_n, 0.0, 255.0))
    img[y, x, 1] = wp.uint8(wp.clamp(accum_g * inv_n, 0.0, 255.0))
    img[y, x, 2] = wp.uint8(wp.clamp(accum_r * inv_n, 0.0, 255.0))

class RayTracer:
    def __init__(self, W=1024, H=1024, G=[128, 128, 32], camera=[0.5, 0.5, 1.0], target=[0.5, 0.5, 0.0], light=[0.5, 1.0, 0.5], fov=1.0, samples=1, background=[0.0, 0.0, 0.0], ambient=0.4, shadow=0.4):
        self.W = W
        self.H = H
        self.img = wp.zeros((H, W, 3), dtype=wp.uint8, device='cuda')

        self.G = wp.vec3i(*G)
        self.h = 1.0 / max(self.G.x, self.G.y)
        self.grid = Grid(self.G)

        self.camera = wp.vec3(*camera)
        self.target = wp.vec3(*target)
        self.light = wp.vec3(*light)
        self.fov = fov
        self.fwd = wp.normalize(self.target - self.camera)
        self.right = wp.normalize(wp.cross(self.fwd, wp.vec3(0.0, 1.0, 0.0)))
        self.up = wp.cross(self.right, self.fwd)

        self.samples = samples
        self.sqrtN = int(wp.ceil(wp.sqrt(float(samples))))
        self.background = wp.vec3(*background)
        self.ambient = ambient
        self.shadow = shadow
          
    def sphere(self, xyz, rgb, r):
        self.grid.build(len(xyz), k_insert, k_fill, [xyz, r], [xyz, r])
        wp.launch(raytrace_sphere, dim=(self.W, self.H), inputs=[
            xyz, rgb, r,
            self.img, self.W, self.H, self.samples, self.sqrtN, self.background, self.ambient, self.shadow,
            self.h, self.G, self.grid.cell_start, self.grid.cell_count, self.grid.particle_ids,
            self.camera, self.fwd, self.right, self.up, self.light, self.fov
        ], device="cuda")
        wp.synchronize()
        return self.img.numpy()
    
    def quad(self, xyz, rgb, rot, r):
        self.grid.build(len(xyz), k_insert_oriented, k_fill_oriented, [xyz, rot, r, r, 0.0], [xyz, rot, r, r, 0.0])
        wp.launch(raytrace_quad, dim=(self.W, self.H), inputs=[
            xyz, rgb, rot, r,
            self.img, self.W, self.H, self.samples, self.sqrtN, self.background, self.ambient, self.shadow,
            self.h, self.G, self.grid.cell_start, self.grid.cell_count, self.grid.particle_ids,
            self.camera, self.fwd, self.right, self.up, self.light, self.fov
        ], device="cuda")
        wp.synchronize()
        return self.img.numpy()
    
    def prism(self, xyz, rgb, rot, rx, ry, rz):
        self.grid.build(len(xyz), k_insert_oriented, k_fill_oriented, [xyz, rot, rx, ry, rz], [xyz, rot, rx, ry, rz])
        wp.launch(raytrace_prism, dim=(self.W, self.H), inputs=[
            xyz, rgb, rot, rx, ry, rz,
            self.img, self.W, self.H, self.samples, self.sqrtN, self.background, self.ambient, self.shadow,
            self.h, self.G, self.grid.cell_start, self.grid.cell_count, self.grid.particle_ids,
            self.camera, self.fwd, self.right, self.up, self.light, self.fov
        ], device="cuda")
        wp.synchronize()
        return self.img.numpy()
    
    def ellipsoid(self, xyz, rgb, rot, rx, ry, rz):
        self.grid.build(len(xyz), k_insert_oriented, k_fill_oriented, [xyz, rot, rx, ry, rz], [xyz, rot, rx, ry, rz])
        wp.launch(raytrace_ellipsoid, dim=(self.W, self.H), inputs=[
            xyz, rgb, rot, rx, ry, rz,
            self.img, self.W, self.H, self.samples, self.sqrtN, self.background, self.ambient, self.shadow,
            self.h, self.G, self.grid.cell_start, self.grid.cell_count, self.grid.particle_ids,
            self.camera, self.fwd, self.right, self.up, self.light, self.fov
        ], device="cuda")
        wp.synchronize()
        return self.img.numpy()
    
    def cylinder(self, xyz, rgb, next, r):
        self.grid.build(len(xyz), k_insert_line, k_fill_line, [xyz, next, r], [xyz, next, r])
        wp.launch(raytrace_cylinder, dim=(self.W, self.H), inputs=[
            xyz, rgb, next, r,
            self.img, self.W, self.H, self.samples, self.sqrtN, self.background, self.ambient, self.shadow,
            self.h, self.G, self.grid.cell_start, self.grid.cell_count, self.grid.particle_ids,
            self.camera, self.fwd, self.right, self.up, self.light, self.fov
        ], device="cuda")
        wp.synchronize()
        return self.img.numpy()
