import os, sys, cv2, random, threading, queue, time
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from collections import deque
from PIL import Image
import ctypes
ctypes.windll.user32.SetProcessDPIAware()

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'shared'))
from config import Config
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'server'))
from sd35 import AnimateDiff, OpticalFlow

import warp as wp
wp.init()

def build_program(cfg, seed=None):
    rng = random.Random(seed)
    entries = []

    for lora_name, lora_weight, hint, repo in cfg.MOTION_LORAS:
        subj = rng.choice(cfg.SUBJECTS)
        prompt_parts = [subj, hint, cfg.GLOBAL_STYLE]
        prompt = ", ".join(part for part in prompt_parts if part)

        loras = []
        if lora_name is not None and repo is not None and lora_weight is not None:
            loras = [(repo, lora_name, lora_weight)]

        entries.append({
            "prompt": prompt,
            "negative": cfg.GLOBAL_NEGATIVE,
            "loras": loras,
        })

    print("─" * 54)
    for i, e in enumerate(entries):
        lora_str = ", ".join(f"{name}:{weight}" for _, name, weight in e["loras"]) or "no-lora"
        print(f"[{i:02d}] {lora_str}  {e['prompt']}")
    print("─" * 54)

    return entries

def bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def producer(cfg, ad, program, raw_q, stop_event):
    anchor_bgr = np.full((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3), 128, dtype=np.uint8)
    entry_idx = 0
    clip_count = 0

    entry = program[entry_idx]
    lora_str = ", ".join(n for _, n, _ in entry["loras"]) or "no-lora"
    print(f"\n  ▶ [{entry_idx:02d}] [{lora_str}] {entry['prompt']}", flush=True)

    while not stop_event.is_set():
        try:
            pil_frames = ad.generate(entry, bgr_to_pil(anchor_bgr))
        except Exception as e:
            print(f"  [Gen] error: {e}")
            time.sleep(2.0)
            continue

        bgr_frames = [cv2.cvtColor(np.array(f.convert("RGB")), cv2.COLOR_RGB2BGR) for f in pil_frames]
        print(f"  [Gen] {len(bgr_frames)} frames", flush=True)

        anchor_bgr = bgr_frames[-1]
        raw_q.put(bgr_frames)

        clip_count += 1
        if clip_count >= cfg.CLIPS_PER_SCENE:
            clip_count = 0
            entry_idx = (entry_idx + 1) % len(program)
            entry = program[entry_idx]
            lora_str = ", ".join(n for _, n, _ in entry["loras"]) or "no-lora"
            print(f"\n  ▶ [{entry_idx:02d}] [{lora_str}] {entry['prompt']}", flush=True)

def rife_worker(of, raw_q, display_q, display_lock, stop_event, cfg):
    prev_tail = None

    while not stop_event.is_set():
        try:
            bgr_frames = raw_q.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            chain = ([prev_tail] + bgr_frames) if prev_tail is not None else bgr_frames
            batch = []

            for i in range(len(chain) - 1):
                if stop_event.is_set():
                    break
                try:
                    mids = of.interpolate(chain[i], chain[i + 1], cfg.OF_FRAMES)
                    batch.append(chain[i])
                    batch.extend(mids)
                except Exception as e:
                    print(f"  [RIFE] pair error: {e}")
                    batch.append(chain[i])

            batch.append(chain[-1])
            prev_tail = bgr_frames[-1]

            with display_lock:
                display_q.extend(batch)
                while len(display_q) > cfg.DISPLAY_MAXFRAMES:
                    display_q.pop()

        except Exception as e:
            print(f"  [RIFE] clip error: {e}")
        finally:
            raw_q.task_done()

def main():
    cfg = Config()
    os.makedirs(cfg.OUTPUT_FOLDER, exist_ok=True)

    prompts = build_program(cfg)
    print(f"\n{len(prompts)} prompts ready.\n" + "─" * 54)

    ad = AnimateDiff(
        CONTROLNET_ID=cfg.AD_CONTROLNET_ID,
        MOTION_ADAPTER=cfg.AD_MOTION_ADAPTER,
        SD_BASE=cfg.AD_SD_BASE,
        MOTION_LORAS=cfg.MOTION_LORAS,
        IW=cfg.IMAGE_SIZE,
        IH=cfg.IMAGE_SIZE,
        NUM_FRAMES=cfg.AD_NUM_FRAMES,
        INFERENCE_STEPS=cfg.AD_INFERENCE_STEPS,
        GUIDANCE_SCALE=cfg.AD_GUIDANCE_SCALE,
        CONTROLNET_SCALE=cfg.AD_CONTROLNET_SCALE,
        SEED=cfg.AD_SEED,
    )
    of = OpticalFlow()

    display_q = deque()
    display_lock = threading.Lock()
    stop_event = threading.Event()
    raw_q = queue.Queue(maxsize=cfg.RAW_QUEUE_MAXSIZE)

    threading.Thread(
        target=rife_worker,
        args=(of, raw_q, display_q, display_lock, stop_event, cfg),
        daemon=True,
    ).start()

    threading.Thread(
        target=producer,
        args=(cfg, ad, prompts, raw_q, stop_event),
        daemon=True,
    ).start()

    frame = np.zeros((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3), dtype=np.uint8)
    frame_dur = 1.0 / cfg.FPS
    origin = time.perf_counter()
    tick_n = 0

    cv2.namedWindow("AnimateDiff", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AnimateDiff", cfg.WINDOW_W, cfg.WINDOW_H)

    while not stop_event.is_set():
        target = origin + tick_n * frame_dur
        gap = target - time.perf_counter() - 0.0005
        if gap > 0:
            time.sleep(gap)
        while time.perf_counter() < target:
            pass
        tick_n += 1

        with display_lock:
            if display_q:
                frame = display_q.popleft()
                frame = cv2.resize(frame, (cfg.WINDOW_W, cfg.WINDOW_H), interpolation=cv2.INTER_LANCZOS4)

        cv2.imshow("AnimateDiff", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            stop_event.set()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
