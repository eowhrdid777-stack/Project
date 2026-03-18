from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import config as cfg


PulsePolarity = Literal["pot", "dep"]


@dataclass
class DeviceState:
    g: float
    level_idx: int = 0  # 0 = min state, n_levels-1 = max state


class MemristorDevice:
    """
    FeTFT paper-based conductance model.

    Key points
    ----------
    1) Potentiation / depression follow paper equations.
    2) D2D is modeled as device-wise conductance-window offset.
    3) C2C is modeled as cycle-wise conductance-window offset.
    4) Within one cycle, curve is fixed. (No pulse-to-pulse random jitter)
    5) Existing config names are kept for compatibility:
       - ENABLE_D2D_INIT_VARIATION / CV_D2D_INIT
       - ENABLE_D2D_STEP_VARIATION / CV_D2D_STEP   (ignored by design)
       - ENABLE_C2C_STEP_NOISE / CV_C2C_STEP       (used as cycle-level variation)
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        device_id: Optional[int] = None,
        cycle_id: int = 0,
    ) -> None:
        if seed is None:
            seed = int(getattr(cfg, "SEED", 0))

        self.seed = int(seed)
        self.device_id = 0 if device_id is None else int(device_id)
        self.cycle_id = int(cycle_id)

        # reproducible RNG per (device, cycle)
        base_seed = self.seed + 10007 * self.device_id + 1000003 * self.cycle_id
        self.rng = np.random.default_rng(base_seed)

        # ------------------------------------------------------------
        # base params from config
        # ------------------------------------------------------------
        self.g_min = float(cfg.G_MIN)
        self.g_max = float(cfg.G_MAX)
        self.g_init = float(cfg.G_INIT)
        self.n_levels = int(cfg.P_MAX)

        self.a_pot = float(cfg.A_POT)
        self.a_dep = float(cfg.A_DEP)

        # compatibility with your current config names
        self.enable_d2d = bool(getattr(cfg, "ENABLE_D2D_VARIATION", True))
        self.cv_d2d = float(getattr(cfg, "CV_D2D", 0.0))

        # current config name says step noise, but here we reinterpret it
        # as cycle-level variation to match paper intent better
        self.enable_c2c = bool(getattr(cfg, "ENABLE_C2C_VARIATION", True))
        self.cv_c2c = float(getattr(cfg, "CV_C2C", 0.0))

        self.enable_retention = bool(getattr(cfg, "ENABLE_RETENTION", False))
        self.retention_gamma = float(getattr(cfg, "RETENTION_GAMMA", 0.0))
        self.g_rcp = float(getattr(cfg, "G_RCP", self.g_init))

        # ------------------------------------------------------------
        # D2D: device-specific offset
        # ------------------------------------------------------------
        if self.enable_d2d and self.cv_d2d > 0.0:
            span = self.g_max - self.g_min
            sigma_d2d = self.cv_d2d * span
            self.g_min_dev = float(self.rng.normal(self.g_min, sigma_d2d))
        else:
            self.g_min_dev = float(self.g_min)

        self.g_min_dev = max(self.g_min_dev, 1e-9)
        self.g_offset_dev = self.g_min_dev - self.g_min
        self.g_max_dev = self.g_max + self.g_offset_dev

        # ------------------------------------------------------------
        # C2C: cycle-specific offset
        # ------------------------------------------------------------
        if self.enable_c2c and self.cv_c2c > 0.0:
            sigma_c2c = self.cv_c2c * (self.g_max - self.g_min)
            self.g_offset_cycle = float(self.rng.normal(0.0, sigma_c2c))
        else:
            self.g_offset_cycle = 0.0

        self.g_min_eff = self.g_min_dev + self.g_offset_cycle
        self.g_max_eff = self.g_max_dev + self.g_offset_cycle

        if self.g_min_eff < 1e-9:
            shift = 1e-9 - self.g_min_eff
            self.g_min_eff += shift
            self.g_max_eff += shift

        if self.g_max_eff <= self.g_min_eff:
            self.g_max_eff = self.g_min_eff + (self.g_max - self.g_min)

        # ------------------------------------------------------------
        # build conductance curves
        # ------------------------------------------------------------
        self.pot_curve = self._build_pot_curve()
        self.dep_curve = self._build_dep_curve()

        # state
        self.state = DeviceState(g=float(self.g_init), level_idx=0)
        self.reset("init")

    # ------------------------------------------------------------
    # paper equations
    # ------------------------------------------------------------
    def _build_pot_curve(self) -> np.ndarray:
        P = np.arange(self.n_levels, dtype=float)
        Pmax = float(self.n_levels - 1)
        x = P / Pmax

        A = float(self.a_pot)
        span = self.g_max_eff - self.g_min_eff
        norm = 1.0 - np.exp(-1.0 / A)

        curve = self.g_min_eff + span * (1.0 - np.exp(-x / A)) / norm
        curve = np.asarray(curve, dtype=float)

        curve[0] = self.g_min_eff
        curve[-1] = self.g_max_eff
        return curve

    def _build_dep_curve(self) -> np.ndarray:
        q = np.arange(self.n_levels, dtype=float)
        Pmax = float(self.n_levels - 1)
        x = q / Pmax

        A = float(self.a_dep)
        span = self.g_max_eff - self.g_min_eff
        norm = 1.0 - np.exp(-1.0 / A)

        curve = self.g_min_eff + span * (1.0 - np.exp(-(1.0 - x) / A)) / norm
        curve = np.asarray(curve, dtype=float)

        curve[0] = self.g_max_eff
        curve[-1] = self.g_min_eff
        return curve
    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    def _clip_value(self, g: float) -> float:
        return float(np.clip(g, self.g_min_eff, self.g_max_eff))

    def _nearest_level_idx(self, g: float) -> int:
        return int(np.argmin(np.abs(self.pot_curve - g)))

    def get_bounds(self) -> tuple[float, float]:
        return float(self.g_min_eff), float(self.g_max_eff)

    # ------------------------------------------------------------
    # pulse application
    # ------------------------------------------------------------
    def apply_pot_pulse(self, n_pulses: int = 1) -> None:
        n_pulses = int(n_pulses)
        if n_pulses <= 0:
            return

        for _ in range(n_pulses):
            if self.state.level_idx >= self.n_levels - 1:
                self.state.level_idx = self.n_levels - 1
                self.state.g = float(self.pot_curve[-1])
                continue

            self.state.level_idx += 1
            self.state.g = float(self.pot_curve[self.state.level_idx])

    def apply_dep_pulse(self, n_pulses: int = 1) -> None:
        n_pulses = int(n_pulses)
        if n_pulses <= 0:
            return

        for _ in range(n_pulses):
            if self.state.level_idx <= 0:
                self.state.level_idx = 0
                self.state.g = float(self.g_min_eff)
                continue

            self.state.level_idx -= 1

            # map level_idx -> depression pulse count from max
            # level_idx = n-1  => q = 0 (Gmax)
            # level_idx = n-2  => q = 1
            q = (self.n_levels - 1) - self.state.level_idx
            self.state.g = float(self.dep_curve[q])

    def apply_pulse(self, polarity: PulsePolarity, n_pulses: int = 1) -> None:
        if polarity == "pot":
            self.apply_pot_pulse(n_pulses)
        elif polarity == "dep":
            self.apply_dep_pulse(n_pulses)
        else:
            raise ValueError(f"Unknown polarity: {polarity}")

    # ------------------------------------------------------------
    # retention
    # ------------------------------------------------------------
    def relax(self, dt: float = 1.0) -> None:
        if not self.enable_retention or self.retention_gamma <= 0.0:
            return

        self.state.g += self.retention_gamma * float(dt) * (self.g_rcp - self.state.g)
        self.state.g = self._clip_value(self.state.g)
        self.state.level_idx = self._nearest_level_idx(self.state.g)

    # ------------------------------------------------------------
    # reset / direct set / state
    # ------------------------------------------------------------
    def reset(self, mode: str = "init") -> None:
        mode = str(mode).lower()

        if mode in ("init", "min"):
            self.state.level_idx = 0
            self.state.g = float(self.pot_curve[0])

        elif mode == "max":
            self.state.level_idx = self.n_levels - 1
            self.state.g = float(self.dep_curve[0])

        elif mode == "mid":
            self.state.level_idx = (self.n_levels - 1) // 2
            self.state.g = float(self.pot_curve[self.state.level_idx])

        else:
            raise ValueError(f"Unknown reset mode: {mode}")

    @property
    def g(self) -> float:
        return float(self.state.g)

    def set_g(self, g: float) -> None:
        self.state.g = self._clip_value(float(g))
        self.state.level_idx = self._nearest_level_idx(self.state.g)

    def snapshot(self) -> DeviceState:
        return DeviceState(
            g=float(self.state.g),
            level_idx=int(self.state.level_idx),
        )

# -------------------------------------------------------
# --------------------- Test code -----------------------
# -------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    def _save_cfg():
        keys = [
            "ENABLE_D2D_VARIATION",
            "CV_D2D",
            "ENABLE_C2C_VARIATION",
            "CV_C2C",
        ]
        saved = {}
        for k in keys:
            saved[k] = getattr(cfg, k)
        return saved

    def _restore_cfg(saved):
        for k, v in saved.items():
            setattr(cfg, k, v)

    def _set_mode(mode: str):
        mode = mode.lower()

        if mode == "none":
            cfg.ENABLE_D2D_VARIATION = False
            cfg.ENABLE_C2C_VARIATION = False

        elif mode == "d2d_only":
            cfg.ENABLE_D2D_VARIATION = True
            cfg.ENABLE_C2C_VARIATION = False

        elif mode == "c2c_only":
            cfg.ENABLE_D2D_VARIATION = False
            cfg.ENABLE_C2C_VARIATION = True

        elif mode == "both":
            cfg.ENABLE_D2D_VARIATION = True
            cfg.ENABLE_C2C_VARIATION = True

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _collect_pot_histories(n_devices: int, n_cycles: int):
        n_pulses = cfg.P_MAX - 1
        all_pot = np.zeros((n_devices, n_cycles, n_pulses + 1))

        for dev_id in range(n_devices):
            for cyc in range(n_cycles):
                dev = MemristorDevice(
                    seed=cfg.SEED,
                    device_id=dev_id,
                    cycle_id=cyc,
                )
                dev.reset("min")

                g_hist = [dev.g]
                for _ in range(n_pulses):
                    dev.apply_pot_pulse(1)
                    g_hist.append(dev.g)

                all_pot[dev_id, cyc, :] = np.array(g_hist)

        return all_pot

    def _collect_dep_histories(n_devices: int, n_cycles: int):
        n_pulses = cfg.P_MAX - 1
        all_dep = np.zeros((n_devices, n_cycles, n_pulses + 1))

        for dev_id in range(n_devices):
            for cyc in range(n_cycles):
                dev = MemristorDevice(
                    seed=cfg.SEED,
                    device_id=dev_id,
                    cycle_id=cyc,
                )
                dev.reset("max")

                g_hist = [dev.g]
                for _ in range(n_pulses):
                    dev.apply_dep_pulse(1)
                    g_hist.append(dev.g)

                all_dep[dev_id, cyc, :] = np.array(g_hist)

        return all_dep

    def _plot_distribution(data, title):
        """
        data shape: (n_runs, n_points)
        """
        x = np.arange(1, data.shape[1] + 1)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        vmin = data.min(axis=0)
        vmax = data.max(axis=0)

        plt.figure(figsize=(8, 5))
        plt.plot(x, mean, label="mean", linewidth=2)
        plt.plot(x, vmin, "--", label="min", linewidth=1.2)
        plt.plot(x, vmax, "--", label="max", linewidth=1.2)
        plt.fill_between(x, mean - std, mean + std, alpha=0.25, label="±1σ")
        plt.title(title)
        plt.xlabel("Pulse number")
        plt.ylabel("Conductance")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def _plot_sample_runs(data, title, n_show=20):
        """
        data shape: (n_runs, n_points)
        """
        x = np.arange(1, data.shape[1] + 1)

        plt.figure(figsize=(8, 5))
        for i in range(min(n_show, data.shape[0])):
            plt.plot(x, data[i], alpha=0.3)
        plt.title(title)
        plt.xlabel("Pulse number")
        plt.ylabel("Conductance")
        plt.grid(True, alpha=0.3)
        plt.show()

    def _print_summary(data, name):
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        print(f"\n[{name}]")
        print("start mean/std =", mean[0], std[0])
        print("mid   mean/std =", mean[len(mean)//2], std[len(std)//2])
        print("end   mean/std =", mean[-1], std[-1])

    saved_cfg = _save_cfg()

    try:
        # =====================================================
        # 1) D2D only
        #    40 devices, cycle fixed
        # =====================================================
        _set_mode("d2d_only")

        pot_d2d = _collect_pot_histories(n_devices=40, n_cycles=1)[:, 0, :]
        dep_d2d = _collect_dep_histories(n_devices=40, n_cycles=1)[:, 0, :]

        _plot_sample_runs(pot_d2d, "Potentiation - D2D only (40 devices)", n_show=40)
        _plot_distribution(pot_d2d, "Potentiation - D2D only distribution")

        _plot_sample_runs(dep_d2d, "Depression - D2D only (40 devices)", n_show=40)
        _plot_distribution(dep_d2d, "Depression - D2D only distribution")

        _print_summary(pot_d2d, "Potentiation D2D only")
        _print_summary(dep_d2d, "Depression D2D only")

        # =====================================================
        # 2) C2C only
        #    1 device, 100 cycles
        # =====================================================
        _set_mode("c2c_only")

        pot_c2c = _collect_pot_histories(n_devices=1, n_cycles=100)[0, :, :]
        dep_c2c = _collect_dep_histories(n_devices=1, n_cycles=100)[0, :, :]

        _plot_sample_runs(pot_c2c, "Potentiation - C2C only (100 cycles)", n_show=100)
        _plot_distribution(pot_c2c, "Potentiation - C2C only distribution")

        _plot_sample_runs(dep_c2c, "Depression - C2C only (100 cycles)", n_show=100)
        _plot_distribution(dep_c2c, "Depression - C2C only distribution")

        _print_summary(pot_c2c, "Potentiation C2C only")
        _print_summary(dep_c2c, "Depression C2C only")

        # =====================================================
        # 3) Both
        #    40 devices x 100 cycles
        # =====================================================
        _set_mode("both")

        all_pot = _collect_pot_histories(n_devices=40, n_cycles=100)
        all_dep = _collect_dep_histories(n_devices=40, n_cycles=100)

        pot_both = all_pot.reshape(-1, all_pot.shape[-1])
        dep_both = all_dep.reshape(-1, all_dep.shape[-1])

        _plot_sample_runs(pot_both, "Potentiation - D2D + C2C (sample runs)", n_show=100)
        _plot_distribution(pot_both, "Potentiation - D2D + C2C distribution")

        _plot_sample_runs(dep_both, "Depression - D2D + C2C (sample runs)", n_show=100)
        _plot_distribution(dep_both, "Depression - D2D + C2C distribution")

        _print_summary(pot_both, "Potentiation D2D + C2C")
        _print_summary(dep_both, "Depression D2D + C2C")

    finally:
        _restore_cfg(saved_cfg)