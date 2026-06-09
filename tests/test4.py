# Test gradient wind

import cv2
import numpy as np
import time
import asyncio
import ctypes
ctypes.windll.user32.SetProcessDPIAware()

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'shared'))
from config import Config

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'server'))
from ray import RayTracer
from sim import Simulator
from ui import Mouse
from sd35 import Folder
from director import DirectorSimple

async def main():
    config = Config()

    # Simulator setup
    sim = Simulator(
        IMAGE_SIZE=config.IMAGE_SIZE,
        PIXELS_PER_CELL=config.PIXELS_PER_CELL,
        G=config.G, L=config.LAYERS,
        smooth=3,
        dt=1.0 / config.FPS_SIM,
    )
    folder = Folder(config.IMAGE_SIZE, config.INPUT_FOLDER)
    img = folder.load_image()
    sim.new_image(img, depth_factor=0.25)
    sim_goback_on = False
    sim_constraints_mode = 0
    sim_gradient_mode = 0

    # Ray tracer setup
    ray = RayTracer(
        W=config.WINDOW_W,
        H=config.WINDOW_H,
        G=config.G,
        camera=config.camera,
        target=config.target,
        light=config.light,
        fov=config.fov,
        samples=config.samples,
        background=config.background,
        ambient=config.ambient,
        shadow=config.shadow,
    )
    ray_shape = 0

    ms = Mouse(1.0 / config.FPS_SIM, config.WINDOW_W, config.WINDOW_H)

    # Window setup
    cv2.namedWindow(config.APP_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(config.APP_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(config.APP_NAME, config.WINDOW_W, config.WINDOW_H)
    cv2.setMouseCallback(config.APP_NAME, ms.callback)

    ray_period = 1.0 / config.FPS
    sim_period = 1.0 / config.FPS_SIM
    now = time.perf_counter()
    next_ray_tick = now
    next_sim_tick = now
    stats_t0 = now
    ray_frames = 0
    sim_steps = 0

    director = DirectorSimple(ms, ray, sim, folder, config)
    director_enabled = True
    director.ray_fov = config.fov
    director.start() 

    while True:
        # Run sim at a bounded multiple of ray FPS
        now = time.perf_counter()
        
        if director_enabled:
            director.tick(now)

            ms.on  = director.ms_on
            ms.pos = director.ms_pos
            ms.vel = director.ms_vel

            ray.fov += (director.ray_fov - ray.fov) * 0.1

            sim_goback_on        = director.sim_go_back
            sim_constraints_mode = director.sim_constraints_mode
            sim_gradient_mode    = director.sim_gradient_mode
            ray_shape            = director.ray_shape

            if director.new_image_flag:
                img = folder.load_image()
                sim.new_image(img, depth_factor=0.25)

        sim_step_count = 0
        while now >= next_sim_tick and sim_step_count < config.MAX_SIM_STEPS_PER_LOOP:
            if ms.on:
                sim.inject_mouse(ms.pos, ms.vel)
            sim.update_flip(
                go_back_on=sim_goback_on,
                constraints_mode=sim_constraints_mode,
                gradient_mode=sim_gradient_mode
            )
            sim_steps += 1
            sim_step_count += 1
            next_sim_tick += sim_period
        # If we fell too far behind, drop backlog instead of spiraling
        if now > next_sim_tick + sim_period * config.MAX_SIM_STEPS_PER_LOOP:
            next_sim_tick = now + sim_period

        # Raytrace only on visual FPS cadence
        if now < next_ray_tick:
            await asyncio.sleep(min(next_ray_tick - now, sim_period))
            continue

        next_ray_tick += ray_period
        if now > next_ray_tick + ray_period:
            next_ray_tick = now + ray_period

        if ray_shape == 0:
            frame = ray.triangle(sim.xyz, sim.rgb, sim.next_x, sim.next_y)
        elif ray_shape == 1:
            frame = ray.prism(sim.xyz, sim.rgb, sim.rot, 1.0 * sim.r, 1.0 * sim.r, 5.0 * sim.r)
        elif ray_shape == 2:
            frame = ray.pixel(sim.xyz, sim.rgb, 1.0 * sim.r)
        elif ray_shape == 3:
            frame = ray.cylinder(sim.xyz, sim.rgb, sim.next_y, 0.5 * sim.r)
        elif ray_shape == 4:
            frame = ray.sphere(sim.xyz, sim.rgb, 4.0 * sim.r)
        elif ray_shape == 5:
            frame = ray.ellipsoid(sim.xyz, sim.rgb, sim.rot, 3.0 * sim.r, 3.0 * sim.r, 0.5 * sim.r)            
        elif ray_shape == 6:
            frame = ray.mesh(sim.xyz, sim.rgb, sim.rot, 4.0 * sim.r, sections=6)

        ray_frames += 1

        elapsed = max(now - stats_t0, 1e-6)
        avg_ray_fps = ray_frames / elapsed
        avg_sim_fps = sim_steps / elapsed

        cv2.putText(
            frame,
            f"Ray avg FPS: {avg_ray_fps:6.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Sim avg FPS: {avg_sim_fps:6.2f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        # p = (int(ms.pos[0] * config.WINDOW_W), int((1.0 - ms.pos[1]) * config.WINDOW_H))
        # cv2.circle(frame, p, 10, (0, 255, 255), 2)
        # cv2.line(frame, (p[0], p[1]), (p[0] + int(ms.vel[0] * 0.03), p[1] + int(ms.vel[1] * 0.03)), (0, 255, 255), 2)

        cv2.imshow(config.APP_NAME, frame)

        key = cv2.waitKeyEx(1)
        if key in (ord('q'), 27):
            break
        elif key == ord('d'):
            director_enabled = not director_enabled
        elif not director_enabled:
            if key == ord('b'):
                sim_goback_on = not sim_goback_on
                print(f"Go back {sim_goback_on}")
            elif key == ord('c'):
                sim_constraints_mode += 1
                if sim_constraints_mode > 2:
                    sim_constraints_mode = 0
                print(f"Constraints {sim_constraints_mode}")
            elif key == ord('g'):
                sim_gradient_mode += 1
                if sim_gradient_mode > 3:
                    sim_gradient_mode = 0
                print(f"Gradient mode {sim_gradient_mode}")
            elif key == ord('n'):
                img = folder.load_image()
                sim.new_image(img, depth_factor=0.25)
            elif key == ord('r'):
                ray_shape += 1
                if ray_shape > 6:
                    ray_shape = 0
            elif key == ord('z'):
                ray.fov -= 0.1
                if ray.fov < 0.05:
                    ray.fov = 1.1

    cv2.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(main())