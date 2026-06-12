import time
import numpy as np
from scipy.interpolate import CubicSpline
import asyncio
import random
import uuid

class MousePathHelper:
    """
    Cubic-spline interpolator for mouse paths.

    Points are given as normalised (x, y, z) where x/y are in [0,1]
    fractions of the window size.  z is passed through as-is.
    """

    def __init__(self, window_w: int, window_h: int):
        self.window_w = window_w
        self.window_h = window_h
        self._cs_x    = None
        self._cs_y    = None
        self._cs_z    = None
        self._duration = 1.0

    def set_keypoints(self, points: list, duration: float) -> None:
        """Build splines from N >= 2 keypoints spread evenly over `duration` seconds."""
        if len(points) < 2:
            raise ValueError("Need at least 2 keypoints")
        self._duration = max(duration, 1e-6)
        n  = len(points)
        t  = np.linspace(0.0, self._duration, n)
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        zs = np.array([p[2] for p in points])
        bc = "not-a-knot" if n >= 4 else "natural"
        self._cs_x = CubicSpline(t, xs, bc_type=bc)
        self._cs_y = CubicSpline(t, ys, bc_type=bc)
        self._cs_z = CubicSpline(t, zs, bc_type=bc)

    def sample(self, t: float) -> np.ndarray:
        if self._cs_x is None:
            return np.array([0.5, 0.5, 0.0], dtype=np.float32)
        tc = float(np.clip(t, 0.0, self._duration))
        x  = float(np.clip(self._cs_x(tc), 0.0, 1.0))
        y  = float(np.clip(self._cs_y(tc), 0.0, 1.0))
        z  = float(self._cs_z(tc))
        return np.array([x, y, z], dtype=np.float32)

class Director:
    """
    Two modes when active:
      - auto_play: sends empty USER_MESSAGE (server picks from folder) every interval;
                   drives mouse, FOV, shape (0-4,6), constraints, gradient, go_back.
      - auto_gen:  sends themed prompt every interval; forces go_back for 5 s then shape=5.
                   No mouse, no FOV cycling, no constraint/gradient driving.

    All settings changes are published to the bus via publish_settings() so that
    app_pc and app_phone stay in sync without the Director holding sim/ray references.
    Mouse simulation is delegated entirely to mouse_move_fn (Mouse.callback); only ms_on
    is exposed so app_pc knows when to inject mouse pos/vel from the shared Mouse object.
    """

    NICKNAME = "Director"
    _THEMES = [
        "labrador dog",
        "labrador puppy",
        "labrador dog fetching",
        "labrador dog riding a car",
        "labrador dog house",
        "labrador dog dressed as a doctor",
    ]
    # _THEMES = [
    #     "colossal shell",
    #     "tiny abandoned house floating",
    #     "jellyfish beneath a hot air balloon",
    #     "colossal ivory white sculptural on the beach",
    #     "a giant ship stuck in the sand",
    #     "giant wave",
    #     "person with pomegranate diving helmet and scarf on the desert",
    #     "person with thick black goggles and indigo blue scarf on the desert",
    #     "person with thick black goggles and covid mask on orange desert",
    #     "colossal barnacled shell",
    #     "crowd of people rushing on the orange subway",
    #     "crowd of people with shopping carts in crowded supermarket",
    #     "birds eye of crowded city with red bridges",
    #     "colossal red coral sculpture on the beach",
    #     "colossal sea anemone in the desert",
    #     "alien ship in the sea",
    #     "rusty toaster on the beach",
    # ]

    _AUTO_PLAY_SHAPES = [0, 1, 2, 3, 4]  # shape 5 (flat) reserved for auto_gen
    _HUMAN_PATHS = [
        # 1. Lazy drift across centre
        [(0.5, 0.5), (0.22, 0.52),
         (0.15, 0.5), (0.22, 0.52), (0.31, 0.45), (0.40, 0.53),
         (0.50, 0.48), (0.58, 0.56), (0.66, 0.46), (0.74, 0.50),
         (0.82, 0.54), (0.88, 0.46)],
        # 2. Arc from bottom-left to top-right
        [(0.12, 0.80), (0.18, 0.72), (0.27, 0.63), (0.35, 0.55),
         (0.44, 0.47), (0.52, 0.40), (0.61, 0.34), (0.70, 0.28),
         (0.79, 0.23), (0.87, 0.20)],
        # 3. Hovering / small circle around centre
        [(0.20, 0.50), (0.36, 0.76), (0.40, 0.52), (0.5, 0.18),
         (0.70, 0.21), (0.83, 0.58)],
        # 4. Top-right to bottom-left diagonal with overshoot/correction
        [(0.82, 0.18), (0.74, 0.25), (0.65, 0.33), (0.57, 0.41),
         (0.48, 0.50), (0.39, 0.57), (0.30, 0.63), (0.22, 0.70),
         (0.17, 0.76), (0.13, 0.80)],
        # 5. S-curve
        [(0.20, 0.20), (0.35, 0.28), (0.50, 0.25), (0.65, 0.33),
         (0.70, 0.42), (0.60, 0.50), (0.45, 0.55), (0.35, 0.63),
         (0.40, 0.72), (0.55, 0.78)],
        # 6. Slow creep from right edge toward centre
        [(0.90, 0.45), (0.83, 0.46), (0.76, 0.47), (0.68, 0.48),
         (0.61, 0.49), (0.55, 0.50), (0.50, 0.50), (0.46, 0.51),
         (0.43, 0.51), (0.41, 0.52)],
        # 7. Sweep up from bottom-centre, pause near top, drift right
        [(0.50, 0.85), (0.49, 0.75), (0.48, 0.65), (0.50, 0.55),
         (0.51, 0.45), (0.52, 0.38), (0.55, 0.33), (0.60, 0.30),
         (0.67, 0.28), (0.74, 0.27)],
    ]

    def __init__(self, bus, config, mouse_move_fn, session_getter):
        self.bus              = bus
        self.config           = config
        self._mouse_move_fn   = mouse_move_fn   # callable(px: int, py: int) -> None
        self._session_getter  = session_getter

        self._mode    = None   # None | "auto_play" | "auto_gen"
        self._tasks   = []
        self._t0      = 0.0
        self._rng     = np.random.default_rng()
        self._path    = MousePathHelper(config.WINDOW_W, config.WINDOW_H)

        # ms_on is the only mouse state Director owns; pos/vel live in the Mouse object
        self.ms_on = False

        # Settings state — not exposed; changes published via bus.publish_settings()
        self._ray_shape       = 0
        self._ray_fov         = getattr(config, "fov", 1.0)
        self._sim_go_back     = False
        self._sim_constraints = 0
        self._sim_gradient    = 0
        self._sim_world       = 0

        # Internal bookkeeping
        self._mouse_path_end    = 0.0
        self._mouse_path_t0     = 0.0
        self._fov_phase         = "idle"   # idle | zoomed | shape_changed
        self._fov_last_step_t   = 0.0
        self._fov_cycle_last    = -30.0    # first zoom fires at t=30 s
        self._image_gen_t       = None     # perf_counter when last image send was triggered

    # ── Public API ──────────────────────────────────────────────────────────────

    @property
    def mode(self):
        return self._mode

    @property
    def enabled(self):
        return self._mode is not None

    def enable_auto_play(self):
        if self._mode == "auto_play":
            return
        self._cancel_tasks()
        self._mode = "auto_play"
        self._ray_shape   = 0
        self._sim_go_back = False
        asyncio.ensure_future(self.bus.publish_settings(shape=0, go_back_on=False))
        self._tasks = [asyncio.ensure_future(self._auto_play_loop())]
        print("✓ Director: auto-play started")

    def enable_auto_gen(self):
        if self._mode == "auto_gen":
            return
        self._cancel_tasks()
        self._mode = "auto_gen"
        self._ray_shape   = 5
        asyncio.ensure_future(self.bus.publish_settings(shape=5))
        self._tasks = [asyncio.ensure_future(self._auto_gen_loop())]
        print("✓ Director: auto-gen started")

    def disable(self):
        if self._mode is None:
            return
        self._cancel_tasks()
        self._mode = None
        self.ms_on = False
        self._sim_go_back = True
        asyncio.ensure_future(self.bus.publish_settings(go_back_on=True))
        t = asyncio.ensure_future(self._publish_user_mode())
        self._tasks = [t]  # tracked so a quick re-enable cancels the 5 s wait
        print("✓ Director: stopped")

    def _cancel_tasks(self):
        for t in self._tasks:
            t.cancel()
        self._tasks = []

    def sync_from_state(self, ray_shape, sim_go_back, constraints_mode, gradient_mode, world_mode, ray_fov=None):
        """Seed internal state from app before enabling so Director continues from current display state."""
        self._ray_shape       = ray_shape
        self._sim_go_back     = sim_go_back
        self._sim_constraints = constraints_mode
        self._sim_gradient    = gradient_mode
        self._sim_world       = world_mode
        if ray_fov is not None:
            self._ray_fov = ray_fov

    def start(self):
        """Call once before the main loop to initialise timing."""
        self._t0 = time.perf_counter()
        self._generate_mouse_path(self._t0)

    def tick(self, now: float) -> None:
        """Drive display rules and publish any changed settings. Called every main-loop iteration when enabled."""
        prev_shape       = self._ray_shape
        prev_fov         = self._ray_fov
        prev_go_back     = self._sim_go_back
        prev_constraints = self._sim_constraints
        prev_gradient    = self._sim_gradient
        prev_world       = self._sim_world

        if self._mode == "auto_play":
            t = now - self._t0
            self._rule_mouse(now, t)
            self._rule_fov_zoom_auto_play(now, t)
            self._rule_go_back(now, t)
            self._rule_constraints(t)
            self._rule_gradient(t)
        elif self._mode == "auto_gen":
            self.ms_on = False

        changed = {}
        if self._ray_shape != prev_shape:
            changed['shape'] = self._ray_shape
        if abs(self._ray_fov - prev_fov) > 1e-4:
            changed['zoom'] = self._ray_fov
        if self._sim_go_back != prev_go_back:
            changed['go_back_on'] = self._sim_go_back
        if self._sim_constraints != prev_constraints:
            changed['constraints_mode'] = self._sim_constraints
        if self._sim_gradient != prev_gradient:
            changed['gradient_mode'] = self._sim_gradient
        if self._sim_world != prev_world:
            changed['world_mode'] = self._sim_world

        if changed:
            asyncio.ensure_future(self.bus.publish_settings(**changed))

    # ── Background loops ─────────────────────────────────────────────────────────

    async def _auto_play_loop(self):
        while self._mode == "auto_play":
            try:
                await self._pick_from_folder()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"✗ Director: auto-play loop error: {e}")
            await asyncio.sleep(self.config.DIRECTOR_PROMPT_INTERVAL)

    async def _auto_gen_loop(self):
        # One-time startup: go_back for 5 s then lock into flat mode
        self._sim_go_back = True
        await self.bus.publish_settings(go_back_on=True)
        await asyncio.sleep(5.0)
        if self._mode != "auto_gen":
            return
        self._ray_shape   = 5
        self._sim_go_back = False
        self._sim_world   = 0
        await self.bus.publish_settings(shape=5, go_back_on=False, world_mode=0)
        print("✓ Director: flat mode locked for auto-gen")

        while self._mode == "auto_gen":
            try:
                await self._generate_and_send_prompt()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"✗ Director: auto-gen loop error: {e}")
            await asyncio.sleep(self.config.DIRECTOR_PROMPT_INTERVAL)

    async def _pick_from_folder(self):
        session_id = self._session_getter()
        if not session_id:
            return
        await self.bus.publish_user_message(
            session_id=session_id,
            nickname=self.NICKNAME,
            turn_id=uuid.uuid4().hex,
        )
        self._image_gen_t = time.perf_counter()
        print("✓ Director: folder pick triggered")

    async def _generate_and_send_prompt(self):
        session_id = self._session_getter()
        if not session_id:
            return
        prompt_text = random.choice(self._THEMES)
        await self.bus.publish_user_message(
            session_id=session_id,
            nickname=self.NICKNAME,
            text=prompt_text,
            turn_id=uuid.uuid4().hex,
        )
        print(f"✓ Director: auto-gen prompt sent — '{prompt_text}'")

    async def _publish_user_mode(self):
        """On entering user mode: announce director off and go_back for 5 s, then reset to shape=0."""
        await self.bus.publish_settings(
            director_mode='user',
            go_back_on=True,
            constraints_mode=self._sim_constraints,
            gradient_mode=self._sim_gradient,
            zoom=self._ray_fov,
        )
        await asyncio.sleep(5.0)
        self._ray_shape   = 0
        self._sim_go_back = False
        self._sim_world   = 0
        await self.bus.publish_settings(shape=0, go_back_on=False, world_mode=0)

    # ── Display rules ────────────────────────────────────────────────────────────

    def _rule_mouse(self, now: float, t: float) -> None:
        """10 s active, 20 s rest cycle. Delegates to mouse_move_fn so Mouse handles filtering."""
        ACTIVE = 10.0
        REST   = 20.0
        PERIOD = ACTIVE + REST
        phase_t = t % PERIOD

        if phase_t >= ACTIVE:
            self.ms_on = False
            return

        self.ms_on = True
        if now >= self._mouse_path_end:
            self._generate_mouse_path(now, duration=ACTIVE)

        local_t = now - self._mouse_path_t0
        pos = self._path.sample(local_t)
        # Path y=0 is screen-top; pixel y=0 is also screen-top — direct scale, no flip
        self._mouse_move_fn(
            int(pos[0] * self.config.WINDOW_W),
            int(pos[1] * self.config.WINDOW_H),
        )

    def _rule_fov_zoom_auto_play(self, now: float, t: float) -> None:
        """Every 60 s: snap to zoomed FOV, cycle shape (skipping 5), snap back."""
        CYCLE            = 60.0
        HOLD_ZOOM        = 3.0
        HOLD_AFTER_SHAPE = 2.0

        if self._fov_phase == "idle":
            if t - self._fov_cycle_last >= CYCLE:
                self._fov_cycle_last  = t
                self._fov_phase       = "zoomed"
                self._fov_last_step_t = now
                self._ray_fov = float(self._rng.choice(np.array([0.1, 0.2, 0.3])))
            return
        elif self._fov_phase == "zoomed":
            if now - self._fov_last_step_t >= HOLD_ZOOM:
                shapes = self._AUTO_PLAY_SHAPES
                try:
                    idx = shapes.index(self._ray_shape)
                except ValueError:
                    idx = 0
                self._ray_shape       = shapes[(idx + 1) % len(shapes)]
                self._fov_phase       = "shape_changed"
                self._fov_last_step_t = now
            return
        elif self._fov_phase == "shape_changed":
            if now - self._fov_last_step_t >= HOLD_AFTER_SHAPE:
                self._ray_fov   = 1.0
                self._fov_phase = "idle"
            return

    def _rule_go_back(self, now: float, t: float) -> None:
        """5s on when image gen fires; otherwise 40 s off, 2 s on, 5 s off, 3 s on."""
        if self._image_gen_t is not None and now - self._image_gen_t < 5.0:
            self._sim_go_back = True
            self._sim_world = 0
            return
        PERIOD  = 50.0
        phase_t = t % PERIOD
        if phase_t < 25.0:
            self._sim_go_back = False
        elif phase_t < 30.0:
            self._sim_go_back = False
            self._sim_world = 1 + int(t % 2)
        elif phase_t < 35.0:
            self._sim_go_back = True
            self._sim_world = 0
        elif phase_t < 37.0:
            self._sim_go_back = False
            self._sim_world = 1 + int(t % 2)
        elif phase_t < 40.0:
            self._sim_go_back = False
            self._sim_world = 0
        elif phase_t < 42.0:
            self._sim_go_back = True
        elif phase_t < 47.0:
            self._sim_go_back = False
        else:
            self._sim_go_back = True

    def _rule_constraints(self, t: float) -> None:
        """Shape 0 forces mode 2; shape 3 alternates 1/2; others cycle 0-2."""
        if self._ray_shape == 0:
            self._sim_constraints = 2
        elif self._ray_shape == 3:
            PERIOD = 30.0
            OFFSET = 22.5
            self._sim_constraints = 1 + (int((t + OFFSET) / PERIOD) % 2)
        else:
            PERIOD = 30.0
            OFFSET = 22.5
            self._sim_constraints = int((t + OFFSET) / PERIOD) % 3

    def _rule_gradient(self, t: float) -> None:
        """25 s cycle: 15 s mode 0, 7 s mode 1 or 3 (alternating cycles), 3 s mode 2."""
        PERIOD    = 25.0
        phase_t   = t % PERIOD
        cycle_idx = int(t / PERIOD)
        if phase_t < 15.0:
            self._sim_gradient = 0
        elif phase_t < 22.0:
            self._sim_gradient = 1 if (cycle_idx % 2 == 0) else 3
        else:
            self._sim_gradient = 2

    def _generate_mouse_path(self, now: float, duration: float = 4.0) -> None:
        path_idx = int((now - self._t0) / max(duration, 1e-6)) % len(self._HUMAN_PATHS)
        raw = self._HUMAN_PATHS[path_idx]
        pts = [
            (
                np.clip(x + self._rng.uniform(-0.03, 0.03), 0.0, 1.0),
                np.clip(y + self._rng.uniform(-0.03, 0.03), 0.0, 1.0),
                0.0,
            )
            for x, y in raw
        ]
        self._path.set_keypoints(pts, duration=duration)
        self._mouse_path_t0  = now
        self._mouse_path_end = now + duration
