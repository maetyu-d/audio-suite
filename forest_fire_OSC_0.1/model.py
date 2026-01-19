from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Cell states
EMPTY = np.int8(0)
TREE  = np.int8(1)
FIRE  = np.int8(2)
ASH   = np.int8(3)


@dataclass
class ModelParams:
    w: int = 220
    h: int = 160

    # initialization
    p_tree_init: float = 0.62

    # ignition / spread
    lightning_rate: float = 3e-6          # per-tree per-step
    base_spread: float = 0.37             # base probability scale (neighbor influence)
    fuel_burn_rate: float = 0.18          # how fast fuel is consumed while burning
    burnout_fuel: float = 0.05            # below this -> ash

    # embers / spotting
    ember_rate: float = 0.035             # probability a burning cell emits ember per step
    ember_max_dist: int = 18              # maximum travel distance in cells
    spotting_strength: float = 0.9        # ember ignition multiplier

    # regrowth / ecology
    regrow_rate: float = 0.006            # empty -> tree
    ash_regrow_rate: float = 0.003        # ash -> tree (slower)

    # moisture / weather
    moisture_relax: float = 0.01          # relax moisture toward baseline
    rain_chance: float = 0.015            # per step
    rain_strength: float = 0.25           # how much moisture increases on rain

    # wind & terrain
    wind_dir_deg: float = 25.0
    wind_strength: float = 0.75           # 0..~1.5
    slope_strength: float = 0.35          # fire climbs more easily

    # rendering
    show_moisture_overlay: bool = False


class ForestFireModel:
    """Deep-ish forest fire CA:
    - state grid: EMPTY / TREE / FIRE / ASH
    - fuel (0..1): trees carry fuel; burning consumes it
    - moisture (0..1): reduces ignition & spread; changes with rain + relaxation
    - elevation: affects spread uphill (slope term)
    - wind: biases spread and ember landing direction
    """
    def __init__(self, params: ModelParams, seed: int = 1):
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.t = 0

        self.state = np.zeros((params.h, params.w), dtype=np.int8)
        self.fuel = np.zeros((params.h, params.w), dtype=np.float32)
        self.moisture = np.zeros((params.h, params.w), dtype=np.float32)
        self.elev = np.zeros((params.h, params.w), dtype=np.float32)
        self.age = np.zeros((params.h, params.w), dtype=np.uint16)

        self._last_ignitions = 0
        self._last_embers = 0
        self._last_rain = 0

        self.reset()

    def reset(self):
        p = self.params
        self.t = 0

        self.state.fill(EMPTY)
        trees = self.rng.random(self.state.shape) < p.p_tree_init
        self.state[trees] = TREE

        self.fuel.fill(0.0)
        self.fuel[trees] = self.rng.uniform(0.75, 1.0, size=int(trees.sum())).astype(np.float32)

        base = self._smooth_noise(self.state.shape, blur_iters=3)
        base = 0.15 + 0.55 * base
        jitter = self.rng.normal(0.0, 0.06, size=self.state.shape).astype(np.float32)
        self.moisture = np.clip(base + jitter, 0.0, 1.0).astype(np.float32)

        hills = self._smooth_noise(self.state.shape, blur_iters=4)
        self.elev = (hills ** 1.7).astype(np.float32)

        self.age.fill(0)
        self._last_ignitions = 0
        self._last_embers = 0
        self._last_rain = 0

    def randomize(self):
        self.reset()

    def _smooth_noise(self, shape, blur_iters: int = 3) -> np.ndarray:
        x = self.rng.random(shape).astype(np.float32)
        for _ in range(int(blur_iters)):
            x = (x
                 + np.roll(x, 1, 0) + np.roll(x, -1, 0)
                 + np.roll(x, 1, 1) + np.roll(x, -1, 1)
                 + np.roll(np.roll(x, 1, 0), 1, 1)
                 + np.roll(np.roll(x, 1, 0), -1, 1)
                 + np.roll(np.roll(x, -1, 0), 1, 1)
                 + np.roll(np.roll(x, -1, 0), -1, 1)
                 ) / 9.0
        mn, mx = float(x.min()), float(x.max())
        if mx - mn < 1e-6:
            return np.zeros(shape, dtype=np.float32)
        return (x - mn) / (mx - mn)

    def _wind_vec(self) -> tuple[float, float]:
        ang = np.deg2rad(self.params.wind_dir_deg)
        return float(np.cos(ang)), float(np.sin(ang))

    def step(self):
        p = self.params
        self.t += 1
        H, W = self.state.shape

        # --- Rain event ---
        rain = self.rng.random() < p.rain_chance
        self._last_rain = 1 if rain else 0
        if rain:
            self.moisture = np.clip(self.moisture + p.rain_strength, 0.0, 1.0)

        # --- Moisture relax toward baseline influenced by elevation ---
        baseline = np.clip(0.45 - 0.20 * self.elev, 0.05, 0.7).astype(np.float32)
        self.moisture += p.moisture_relax * (baseline - self.moisture)
        self.moisture = np.clip(self.moisture, 0.0, 1.0)

        # --- Burning consumes fuel and becomes ash ---
        burning = (self.state == FIRE)
        if burning.any():
            self.fuel[burning] = np.clip(self.fuel[burning] - p.fuel_burn_rate, 0.0, 1.0)
            burned_out = burning & (self.fuel <= p.burnout_fuel)
            self.state[burned_out] = ASH

        # --- Neighbourhood fire mask ---
        fire = (self.state == FIRE)
        fire_n = (
            np.roll(fire, 1, 0) | np.roll(fire, -1, 0) |
            np.roll(fire, 1, 1) | np.roll(fire, -1, 1) |
            np.roll(np.roll(fire, 1, 0), 1, 1) |
            np.roll(np.roll(fire, 1, 0), -1, 1) |
            np.roll(np.roll(fire, -1, 0), 1, 1) |
            np.roll(np.roll(fire, -1, 0), -1, 1)
        )

        trees = (self.state == TREE)

        # --- Terrain + wind bias ---
        gy, gx = np.gradient(self.elev)
        wx, wy = self._wind_vec()
        uphill = np.clip(-(gx * wx + gy * wy), 0.0, 1.0)
        slope_boost = 1.0 + p.slope_strength * uphill
        wind_boost = 1.0 + p.wind_strength * np.clip((gx * wx + gy * wy) + 0.5, 0.0, 1.0) * 0.6

        fuel_term = np.clip(self.fuel, 0.0, 1.0)
        moist_term = 1.0 - np.clip(self.moisture, 0.0, 1.0)

        spread_p = p.base_spread * moist_term * (0.35 + 0.65 * fuel_term) * slope_boost * wind_boost
        spread_p = np.clip(spread_p, 0.0, 0.99)

        # --- Spread + lightning ---
        will_spread = trees & fire_n & (self.rng.random((H, W)) < spread_p)
        lightning = trees & (self.rng.random((H, W)) < (p.lightning_rate * moist_term))
        ignitions = will_spread | lightning

        # --- Ember spotting ---
        ember_ignitions = np.zeros((H, W), dtype=bool)
        emit = fire & (self.rng.random((H, W)) < p.ember_rate)
        emit_idx = np.argwhere(emit)
        self._last_embers = int(emit_idx.shape[0])

        if emit_idx.shape[0] > 0:
            d = self.rng.integers(3, p.ember_max_dist + 1, size=emit_idx.shape[0])
            jitter = self.rng.normal(0.0, 1.25 + 1.4 * (p.wind_strength), size=(emit_idx.shape[0], 2))
            dx = (wx * d + jitter[:, 0]).astype(np.int32)
            dy = (wy * d + jitter[:, 1]).astype(np.int32)

            yy = (emit_idx[:, 0] + dy) % H
            xx = (emit_idx[:, 1] + dx) % W

            land_moist = self.moisture[yy, xx]
            land_fuel = self.fuel[yy, xx]
            p_ember_ignite = np.clip(p.spotting_strength * (1.0 - land_moist) * (0.25 + 0.75 * land_fuel), 0.0, 0.95)
            ok = (self.state[yy, xx] == TREE) & (self.rng.random(emit_idx.shape[0]) < p_ember_ignite)
            ember_ignitions[yy[ok], xx[ok]] = True

        ignitions |= ember_ignitions
        self._last_ignitions = int(ignitions.sum())

        if ignitions.any():
            self.state[ignitions] = FIRE

        # --- Regrowth ---
        empty = (self.state == EMPTY)
        ash = (self.state == ASH)
        grow_mod = np.clip(0.6 + 0.8 * self.moisture, 0.0, 1.6)

        regrow_empty = empty & (self.rng.random((H, W)) < (p.regrow_rate * grow_mod))
        regrow_ash = ash & (self.rng.random((H, W)) < (p.ash_regrow_rate * grow_mod))

        if regrow_empty.any():
            self.state[regrow_empty] = TREE
            self.fuel[regrow_empty] = self.rng.uniform(0.5, 0.8, size=int(regrow_empty.sum())).astype(np.float32)
            self.age[regrow_empty] = 0

        if regrow_ash.any():
            self.state[regrow_ash] = TREE
            self.fuel[regrow_ash] = self.rng.uniform(0.55, 0.9, size=int(regrow_ash.sum())).astype(np.float32)
            self.age[regrow_ash] = 0

        trees = (self.state == TREE)
        self.age[trees] = np.clip(self.age[trees] + 1, 0, 65535).astype(np.uint16)
        self.fuel[trees] = np.clip(self.fuel[trees] + (0.003 + 0.005 * self.moisture[trees]), 0.0, 1.0)

    def ignite_at(self, x: int, y: int, radius: int = 2):
        H, W = self.state.shape
        rr = max(0, int(radius))
        ys = (np.arange(y - rr, y + rr + 1) % H)
        xs = (np.arange(x - rr, x + rr + 1) % W)
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        mask = (X - x) ** 2 + (Y - y) ** 2 <= rr * rr
        yy = Y[mask]
        xx = X[mask]
        can = (self.state[yy, xx] == TREE)
        self.state[yy[can], xx[can]] = FIRE

    def set_tree_at(self, x: int, y: int, radius: int = 2):
        H, W = self.state.shape
        rr = max(0, int(radius))
        ys = (np.arange(y - rr, y + rr + 1) % H)
        xs = (np.arange(x - rr, x + rr + 1) % W)
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        mask = (X - x) ** 2 + (Y - y) ** 2 <= rr * rr
        yy = Y[mask]
        xx = X[mask]
        self.state[yy, xx] = TREE
        self.fuel[yy, xx] = np.clip(self.fuel[yy, xx] + 0.5, 0.0, 1.0)

    def clear_at(self, x: int, y: int, radius: int = 2):
        H, W = self.state.shape
        rr = max(0, int(radius))
        ys = (np.arange(y - rr, y + rr + 1) % H)
        xs = (np.arange(x - rr, x + rr + 1) % W)
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        mask = (X - x) ** 2 + (Y - y) ** 2 <= rr * rr
        yy = Y[mask]
        xx = X[mask]
        self.state[yy, xx] = EMPTY
        self.fuel[yy, xx] = 0.0

    def get_stats(self) -> dict:
        s = self.state
        return {
            "t": self.t,
            "trees": int((s == TREE).sum()),
            "burning": int((s == FIRE).sum()),
            "ash": int((s == ASH).sum()),
            "empty": int((s == EMPTY).sum()),
            "ignitions": int(self._last_ignitions),
            "embers": int(self._last_embers),
            "rain": int(self._last_rain),
        }

    def render_rgb(self) -> np.ndarray:
        H, W = self.state.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)

        empty = (self.state == EMPTY)
        tree  = (self.state == TREE)
        fire  = (self.state == FIRE)
        ash   = (self.state == ASH)

        rgb[empty] = (18, 16, 16)

        m = self.moisture
        f = self.fuel
        g = (70 + 120 * (0.6 * f + 0.4 * m)).astype(np.uint8)
        r = (20 + 40 * (0.6 * m)).astype(np.uint8)
        b = (18 + 30 * (0.35 * m)).astype(np.uint8)
        rgb[tree, 0] = r[tree]
        rgb[tree, 1] = g[tree]
        rgb[tree, 2] = b[tree]

        inten = np.clip(0.3 + 0.7 * f, 0.0, 1.0)
        rr = (180 + 75 * inten).astype(np.uint8)
        gg = (70 + 120 * inten).astype(np.uint8)
        bb = (15 + 30 * inten).astype(np.uint8)
        rgb[fire, 0] = rr[fire]
        rgb[fire, 1] = gg[fire]
        rgb[fire, 2] = bb[fire]

        e = self.elev
        a = (70 + 80 * e).astype(np.uint8)
        rgb[ash] = np.stack([a, a, a], axis=-1)[ash]

        if self.params.show_moisture_overlay:
            overlay = (m * 255).astype(np.uint8)
            rgb[..., 2] = np.maximum(rgb[..., 2], (overlay // 2))

        return rgb
