from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

import config as cfg

# type aliases
PulsePolarity = Literal["pot", "dep"]
ResetMode = Literal["init", "min", "max", "mid"]


@dataclass
class DeviceState:
    g: float # 현재 conductance 값 [S]
    level_idx: int # 현재 conductance level index (0 to P_MAX-1)


class MemristorDevice:
    def __init__(
        self,
        seed: Optional[int] = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)

        self.g_min_nom = float(cfg.G_MIN)
        self.g_max_nom = float(cfg.G_MAX)
        self.n_levels = int(cfg.P_MAX)
        self.a_pot = float(cfg.A_POT)
        self.a_dep = float(cfg.A_DEP)

        self.enable_d2d = bool(cfg.ENABLE_D2D_VARIATION)
        self.cv_d2d = float(cfg.CV_D2D)
        self.enable_c2c = bool(cfg.ENABLE_C2C_VARIATION)
        self.cv_c2c = float(cfg.CV_C2C)

        self.enable_retention = bool(cfg.ENABLE_RETENTION)
        self.retention_gamma = float(cfg.RETENTION_GAMMA)

        self.g_init_mode = str(cfg.G_INIT_MODE).lower()

        # Device-wise multiplicative window variation.
        d2d_scale = 1.0
        if self.enable_d2d and self.cv_d2d > 0.0:
            d2d_scale = max(0.5, float(self.rng.normal(1.0, self.cv_d2d)))

        self.g_min_eff = float(self.g_min_nom * d2d_scale)
        self.g_max_eff = float(self.g_max_nom * d2d_scale)
        if self.g_max_eff <= self.g_min_eff:
            self.g_max_eff = self.g_min_eff * 1.01

        self.g_rcp = float(np.clip(cfg.G_RCP * d2d_scale, self.g_min_eff, self.g_max_eff))

        self.pot_curve = self._build_curve(direction="pot")
        self.dep_curve = self._build_curve(direction="dep")

        self.state = DeviceState(g=float(self._level_to_g(0)), level_idx=0)
        self.reset(self.g_init_mode)

    # ------------------------------------------------------------------
    # Curve construction
    # ------------------------------------------------------------------
    def _build_curve(self, direction: PulsePolarity) -> np.ndarray:
        n = self.n_levels
        x = np.linspace(0.0, 1.0, n)
        if direction == "pot":
            # Monotonic increase from g_min to g_max.
            alpha = max(1e-6, np.exp(self.a_pot))
            y = (np.exp(alpha * x) - 1.0) / (np.exp(alpha) - 1.0)
        else:
            # Monotonic increase in pulse count when moving from min to max,
            # but used in reverse during dep updates.
            alpha = max(1e-6, np.exp(self.a_dep))
            y = (np.exp(alpha * x) - 1.0) / (np.exp(alpha) - 1.0)
        g = self.g_min_eff + (self.g_max_eff - self.g_min_eff) * y
        return np.asarray(g, dtype=float)

    def _clip_g(self, g: float) -> float:
        return float(np.clip(g, self.g_min_eff, self.g_max_eff))

    def _nearest_level_idx(self, g: float) -> int:
        idx = int(np.argmin(np.abs(self.pot_curve - g)))
        return max(0, min(self.n_levels - 1, idx))

    def _level_to_g(self, level_idx: int) -> float:
        level_idx = max(0, min(self.n_levels - 1, int(level_idx)))
        return float(self.pot_curve[level_idx])

    # ------------------------------------------------------------------
    # Variation-aware pulse stepping
    # ------------------------------------------------------------------
    def _sample_step_count(self, n_pulses: int) -> int:
        n_pulses = int(n_pulses)
        if n_pulses <= 0:
            return 0
        if not self.enable_c2c or self.cv_c2c <= 0.0:
            return n_pulses
        eff = self.rng.normal(loc=float(n_pulses), scale=max(0.2, n_pulses * self.cv_c2c))
        return max(0, int(round(eff)))

    def apply_pot_pulse(self, n_pulses: int = 1) -> None:
        eff_steps = self._sample_step_count(n_pulses)
        if eff_steps <= 0:
            return
        new_idx = min(self.n_levels - 1, self.state.level_idx + eff_steps)
        self.state.level_idx = int(new_idx)
        self.state.g = float(self.pot_curve[self.state.level_idx])

    def apply_dep_pulse(self, n_pulses: int = 1) -> None:
        eff_steps = self._sample_step_count(n_pulses)
        if eff_steps <= 0:
            return
        new_idx = max(0, self.state.level_idx - eff_steps)
        self.state.level_idx = int(new_idx)
        # Map back using the same ordered conductance levels.
        self.state.g = float(self.pot_curve[self.state.level_idx])

    def apply_pulse(self, polarity: PulsePolarity, n_pulses: int = 1) -> None:
        if polarity == "pot":
            self.apply_pot_pulse(n_pulses)
        elif polarity == "dep":
            self.apply_dep_pulse(n_pulses)
        else:
            raise ValueError(f"Unknown polarity: {polarity}")

    # ------------------------------------------------------------------
    # Relaxation / direct set / reset
    # ------------------------------------------------------------------
    def relax(self, dt: float = 1.0) -> None:
        if not self.enable_retention or self.retention_gamma <= 0.0:
            return
        self.state.g += self.retention_gamma * float(dt) * (self.g_rcp - self.state.g)
        self.state.g = self._clip_g(self.state.g)
        self.state.level_idx = self._nearest_level_idx(self.state.g)

    def reset(self, mode: ResetMode = "init") -> None:
        mode = str(mode).lower()
        if mode == "init":
            mode = self.g_init_mode

        if mode == "min":
            idx = 0
        elif mode == "max":
            idx = self.n_levels - 1
        elif mode == "mid":
            idx = (self.n_levels - 1) // 2
        else:
            raise ValueError(f"Unknown reset mode: {mode}")

        self.state.level_idx = int(idx)
        self.state.g = float(self.pot_curve[self.state.level_idx])

    @property
    def g(self) -> float:
        return float(self.state.g)

    def set_g(self, g: float) -> None:
        g = self._clip_g(float(g))
        self.state.g = g
        self.state.level_idx = self._nearest_level_idx(g)

    def snapshot(self) -> DeviceState:
        return DeviceState(g=float(self.state.g), level_idx=int(self.state.level_idx))