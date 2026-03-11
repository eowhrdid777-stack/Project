# 단일 memristor 소자의 내부 conductance 상태 변화 모델링

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import config as cfg


PulsePolarity = Literal["pot", "dep"]

# conductance 변화 개수 저장
@dataclass
class DeviceState:
    g: float
    pulse_count_pot: int = 0
    pulse_count_dep: int = 0

# single memristor device model
class MemristorDevice:
    def __init__(self, seed: Optional[int] = cfg.SEED) -> None:
        self.rng = np.random.default_rng(seed)

        # config 값 load
        self.g_min: float = float(cfg.G_MIN)
        self.g_max: float = float(cfg.G_MAX)
        self.g_init: float = float(cfg.G_INIT)

        # Pulse step parameters (pulse 당 기본 conductance 변화량)
        self.g_pot_step: float = float(cfg.G_POT_STEP)
        self.g_dep_step: float = float(cfg.G_DEP_STEP)

        # soft-bound exponential switching parameters
        self.g_pot_beta: float = float(cfg.G_POT_BETA)
        self.g_dep_beta: float = float(cfg.G_DEP_BETA)

        # Optional asymmetry scaling
        self.pot_scale: float = float(cfg.POT_SCALE)
        self.dep_scale: float = float(cfg.DEP_SCALE)

        # Optional device-to-device variation on step size
        if bool(cfg.ENABLE_D2D_STEP_VARIATION) and float(cfg.CV_D2D_STEP) > 0.0:
            self.step_var_factor_pot = float(self._lognormal_factor(float(cfg.CV_D2D_STEP)))
            self.step_var_factor_dep = float(self._lognormal_factor(float(cfg.CV_D2D_STEP)))
        else:
            self.step_var_factor_pot = 1.0
            self.step_var_factor_dep = 1.0

        # Optional device-to-device variation on initial conductance
        if bool(getattr(cfg, "ENABLE_D2D_INIT_VARIATION", False)) and float(getattr(cfg, "CV_D2D_INIT", 0.0)) > 0.0:
            self.init_var_factor = float(self._lognormal_factor(float(getattr(cfg, "CV_D2D_INIT", 0.0))))
        else:
            self.init_var_factor = 1.0

        self.state = DeviceState(g=self._clip_value(self.g_init))

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def _clip_value(self, g: float) -> float:
        return float(np.clip(g, self.g_min, self.g_max))

    def _clip_state(self) -> None:
        self.state.g = self._clip_value(self.state.g)

    def _norm(self, g: float) -> float:
        """Normalize conductance to [0, 1]."""
        denom = max(self.g_max - self.g_min, 1e-18)
        x = (g - self.g_min) / denom
        return float(np.clip(x, 0.0, 1.0))

    def _lognormal_factor(self, cv: float) -> float:
        sigma = np.sqrt(np.log(cv * cv + 1.0))
        mu = -0.5 * sigma * sigma
        return float(np.exp(mu + sigma * self.rng.standard_normal()))

    # ------------------------------------------------------------------
    # Pulse response model
    # ------------------------------------------------------------------
    def _pot_delta(self, g: float) -> float:
        # soft-bound form: delta * (1 - x)^beta
        x = self._norm(g)
        delta = (
            self.g_pot_step
            * self.pot_scale
            * self.step_var_factor_pot
            * (1.0 - x) ** self.g_pot_beta
        )
        return max(0.0, float(delta))

    def _dep_delta(self, g: float) -> float:
        # soft-bound form: delta * x^beta
        x = self._norm(g)
        delta = (
            self.g_dep_step
            * self.dep_scale
            * self.step_var_factor_dep
            * x ** self.g_dep_beta
        )
        return max(0.0, float(delta))

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------
    def apply_pot_pulse(self, n_pulses: int = 1) -> None:
        """Apply potentiation pulses to the device."""
        n_pulses = int(n_pulses)
        if n_pulses <= 0:
            return

        for _ in range(n_pulses):
            delta = self._pot_delta(self.state.g)

            if bool(cfg.ENABLE_C2C_STEP_NOISE) and float(cfg.CV_C2C_STEP) > 0.0:
                delta *= self._lognormal_factor(float(cfg.CV_C2C_STEP))

            self.state.g += delta
            self._clip_state()
            self.state.pulse_count_pot += 1

    def apply_dep_pulse(self, n_pulses: int = 1) -> None:
        """Apply depression pulses to the device."""
        n_pulses = int(n_pulses)
        if n_pulses <= 0:
            return

        for _ in range(n_pulses):
            delta = self._dep_delta(self.state.g)

            if bool(cfg.ENABLE_C2C_STEP_NOISE) and float(cfg.CV_C2C_STEP) > 0.0:
                delta *= self._lognormal_factor(float(cfg.CV_C2C_STEP))

            self.state.g -= delta
            self._clip_state()
            self.state.pulse_count_dep += 1

    def apply_pulse(self, polarity: PulsePolarity, n_pulses: int = 1) -> None:
        """Generic pulse application API."""
        if polarity == "pot":
            self.apply_pot_pulse(n_pulses=n_pulses)
        elif polarity == "dep":
            self.apply_dep_pulse(n_pulses=n_pulses)
        else:
            raise ValueError(f"Unknown polarity: {polarity}")

    # ------------------------------------------------------------------
    # Retention / relaxation
    # ------------------------------------------------------------------
    def relax(self, dt: float = 1.0) -> None:
        # Simple first-order model: g <- g + gamma * dt * (g_rcp - g)
        if not bool(cfg.ENABLE_RETENTION):
            return

        gamma = float(cfg.RETENTION_GAMMA)
        g_rcp = float(cfg.G_RCP)

        if gamma <= 0.0:
            return

        self.state.g += gamma * float(dt) * (g_rcp - self.state.g)
        self._clip_state()

    # ------------------------------------------------------------------
    # Reset / initialization
    # ------------------------------------------------------------------
    def reset(self, mode: str = "init") -> None:
        mode = str(mode).lower()

        if mode == "init":
            self.state.g = self._clip_value(self.g_init * self.init_var_factor)
        elif mode == "min":
            self.state.g = self.g_min
        elif mode == "max":
            self.state.g = self.g_max
        elif mode == "mid":
            self.state.g = 0.5 * (self.g_min + self.g_max)
        elif mode == "random":
            self.state.g = float(self.rng.uniform(self.g_min, self.g_max))
        else:
            raise ValueError(f"Unknown reset mode: {mode}")

        self.state.pulse_count_pot = 0
        self.state.pulse_count_dep = 0
        self._clip_state()

    # ------------------------------------------------------------------
    # Convenience / inspection (state only, not read circuitry)
    # ------------------------------------------------------------------
    @property
    def g(self) -> float:
        """Raw internal conductance state.

        This is exposed for higher-level modules (crossbar / controller),
        not as a physical read operation.
        """
        return float(self.state.g)

    def set_g(self, g: float) -> None:
        """Force internal state (use carefully, mostly for initialization)."""
        self.state.g = self._clip_value(float(g))

    def snapshot(self) -> DeviceState:
        """Return a copy of current internal state."""
        return DeviceState(
            g=float(self.state.g),
            pulse_count_pot=int(self.state.pulse_count_pot),
            pulse_count_dep=int(self.state.pulse_count_dep),
        )


# ======================================================================
# TEST / DEBUG ONLY
# Delete or comment out this whole section later if not needed.
# ======================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dev = MemristorDevice(seed=cfg.SEED)
    dev.reset("init")

    n_total = 200
    g_hist = []
    pulse_axis = []

    # Example 1: potentiation only
    for t in range(n_total):
        dev.apply_pot_pulse(1)
        g_hist.append(dev.g)
        pulse_axis.append(t + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(pulse_axis, g_hist, linewidth=2)
    plt.title("Potentiation pulse response")
    plt.xlabel("Pulse count")
    plt.ylabel("Internal conductance g")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Example 2: depression after potentiation
    dev.reset("max")
    g_hist2 = []
    pulse_axis2 = []

    for t in range(n_total):
        dev.apply_dep_pulse(1)
        g_hist2.append(dev.g)
        pulse_axis2.append(t + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(pulse_axis2, g_hist2, linewidth=2)
    plt.title("Depression pulse response")
    plt.xlabel("Pulse count")
    plt.ylabel("Internal conductance g")
    plt.grid(True, alpha=0.3)
    plt.show()

    print("Final snapshot:", dev.snapshot())