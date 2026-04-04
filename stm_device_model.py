from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

import config as cfg


def _cfg(name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


@dataclass
class STMDeviceState:
    g: float
    z: float               # fast residual / incubation state [0, 1]
    x: float               # slow structural filament state [0, 1]
    r: float               # available resource / mobile Ag fraction [0, 1]
    t_s: float
    n_applied_pulses: int


class STMDeviceModel:
    """
    Compact diffusive-dynamics STM device model.

    Model intent
    ------------
    - Repeated pulses arriving faster than the relaxation time accumulate conductance.
    - During pulse gaps and after stimulation, conductance decays exponentially.
    - The conductance approaches saturation smoothly near the top of the window.
    - D2D and C2C variability are included.

    States
    ------
    z : fast volatile state built quickly by pulses and erased quickly in gaps.
    x : slower structural state that determines most of the observable conductance.
    r : available mobile resource. Repeated pulses deplete it; gaps recover it.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

        # Conductance window
        self.g_rest_nom = float(_cfg("STM_G_REST", 1.0e-8))
        self.g_peak_nom = float(_cfg("STM_G_PEAK", 8.0e-8))
        self.g_nonlinearity = float(_cfg("STM_G_NONLINEARITY", 1.55))

        # Internal integration step and pulse defaults
        self.dt_internal = float(_cfg("STM_DT_INTERNAL", 2.0e-5))
        self.default_pulse_width_s = float(_cfg("STM_PULSE_WIDTH_S", 1.0e-3))
        self.read_voltage = float(_cfg("STM_READ_VOLTAGE", 0.1))

        # Pulse-to-state coupling
        self.pulse_threshold_v = float(_cfg("STM_PULSE_THRESHOLD_V", 0.18))
        self.pulse_scale_v = float(_cfg("STM_PULSE_SCALE_V", 0.16))
        self.z_pulse_gain = float(_cfg("STM_Z_PULSE_GAIN", 180.0))
        self.x_growth_gain = float(_cfg("STM_X_GROWTH_GAIN", 75.0))
        self.z_to_x_threshold = float(_cfg("STM_Z_TO_X_THRESHOLD", 0.22))
        self.z_to_x_slope = float(_cfg("STM_Z_TO_X_SLOPE", 10.0))
        self.pulse_leak_factor = float(_cfg("STM_PULSE_LEAK_FACTOR", 0.02))

        # Relaxation / recovery time constants
        self.tau_z_s = float(_cfg("STM_TAU_Z_S", 2.0e-3))
        self.tau_x_s = float(_cfg("STM_TAU_X_S", 30.0e-3))
        self.tau_r_s = float(_cfg("STM_TAU_R_S", 90.0e-3))

        # Resource depletion / optional overload
        self.r_depletion_gain = float(_cfg("STM_R_DEPLETION_GAIN", 0.28))
        self.enable_overload_decay = bool(_cfg("STM_ENABLE_OVERLOAD_DECAY", False))
        self.overload_x_threshold = float(_cfg("STM_OVERLOAD_X_THRESHOLD", 0.985))
        self.overload_r_threshold = float(_cfg("STM_OVERLOAD_R_THRESHOLD", 0.10))
        self.overload_decay_gain = float(_cfg("STM_OVERLOAD_DECAY_GAIN", 0.20))

        # Conductance mapping between fast and slow contributions
        self.fast_weight = float(_cfg("STM_FAST_WEIGHT", 0.18))
        self.slow_weight = float(_cfg("STM_SLOW_WEIGHT", 0.82))

        # Variability / readout
        self.enable_d2d_variation = bool(_cfg("STM_ENABLE_D2D_VARIATION", True))
        self.cv_d2d = float(_cfg("STM_CV_D2D", 0.06))
        self.enable_c2c_variation = bool(_cfg("STM_ENABLE_C2C_VARIATION", True))
        self.cv_c2c = float(_cfg("STM_CV_C2C", 0.025))
        self.enable_read_noise = bool(_cfg("STM_ENABLE_READ_NOISE", True))
        self.read_noise_rel_sigma = float(_cfg("STM_READ_NOISE_REL_SIGMA", 0.003))

        # Device-to-device variation
        d2d_window = 1.0
        d2d_tau = 1.0
        d2d_threshold = 1.0
        d2d_gain = 1.0
        if self.enable_d2d_variation and self.cv_d2d > 0.0:
            d2d_window = max(0.65, float(self.rng.normal(1.0, self.cv_d2d)))
            d2d_tau = max(0.60, float(self.rng.normal(1.0, 0.45 * self.cv_d2d)))
            d2d_threshold = max(0.70, float(self.rng.normal(1.0, 0.35 * self.cv_d2d)))
            d2d_gain = max(0.70, float(self.rng.normal(1.0, 0.40 * self.cv_d2d)))

        self.g_rest_eff = self.g_rest_nom * d2d_window
        self.g_peak_eff = max(self.g_rest_eff * 1.05, self.g_peak_nom * d2d_window)

        self.tau_z_eff = self.tau_z_s * d2d_tau
        self.tau_x_eff = self.tau_x_s * d2d_tau
        self.tau_r_eff = self.tau_r_s * d2d_tau

        self.pulse_threshold_eff = self.pulse_threshold_v * d2d_threshold
        self.z_pulse_gain_eff = self.z_pulse_gain * d2d_gain
        self.x_growth_gain_eff = self.x_growth_gain * d2d_gain

        self.state = STMDeviceState(
            g=self.g_rest_eff,
            z=0.0,
            x=0.0,
            r=1.0,
            t_s=0.0,
            n_applied_pulses=0,
        )
        self._update_g()

    @staticmethod
    def _clip01(v: float) -> float:
        return float(np.clip(v, 0.0, 1.0))

    @staticmethod
    def _sigmoid(u: float) -> float:
        return float(1.0 / (1.0 + np.exp(-u)))

    def _sample_c2c_multiplier(self) -> float:
        if not self.enable_c2c_variation or self.cv_c2c <= 0.0:
            return 1.0
        return max(0.0, float(self.rng.normal(1.0, self.cv_c2c)))

    def _drive_from_voltage(self, amplitude_v: float) -> float:
        v_eff = max(0.0, float(amplitude_v) - self.pulse_threshold_eff)
        if v_eff <= 0.0:
            return 0.0
        return float(1.0 - np.exp(-v_eff / max(self.pulse_scale_v, 1e-12)))

    def _observable_activation(self) -> float:
        # slow state dominates, fast state adds short-lived enhancement
        a = self.slow_weight * self.state.x + self.fast_weight * self.state.z
        return self._clip01(a)

    def _update_g(self) -> None:
        a = self._observable_activation()
        self.state.g = float(
            self.g_rest_eff + (self.g_peak_eff - self.g_rest_eff) * (a ** self.g_nonlinearity)
        )

    def reset(self, mode: str = "rest") -> None:
        mode = str(mode).lower()
        if mode == "rest":
            z, x, r = 0.0, 0.0, 1.0
        elif mode == "mid":
            z, x, r = 0.10, 0.25, 0.95
        elif mode == "peak":
            z, x, r = 1.0, 1.0, 0.65
        else:
            raise ValueError(f"Unknown reset mode: {mode}")

        self.state = STMDeviceState(
            g=self.g_rest_eff,
            z=float(z),
            x=float(x),
            r=float(r),
            t_s=0.0,
            n_applied_pulses=0,
        )
        self._update_g()

    def _append_history(self, hist: dict[str, list[float]], event: str) -> None:
        hist["time_s"].append(self.state.t_s)
        hist["conductance_s"].append(self.state.g)
        hist["current_a"].append(self.state.g * self.read_voltage)
        hist["z"].append(self.state.z)
        hist["x"].append(self.state.x)
        hist["r"].append(self.state.r)
        hist["event"].append(event)

    def _step_relax(self, dt_s: float) -> None:
        ez = np.exp(-dt_s / max(self.tau_z_eff, 1e-12))
        ex = np.exp(-dt_s / max(self.tau_x_eff, 1e-12))
        er = np.exp(-dt_s / max(self.tau_r_eff, 1e-12))

        self.state.z *= float(ez)
        self.state.x *= float(ex)
        self.state.r = float(1.0 - (1.0 - self.state.r) * er)

        self.state.z = self._clip01(self.state.z)
        self.state.x = self._clip01(self.state.x)
        self.state.r = self._clip01(self.state.r)
        self.state.t_s += dt_s
        self._update_g()

    def relax(self, dt_s: float, *, record_history: bool = False) -> Optional[dict[str, np.ndarray]]:
        dt_s = float(dt_s)
        if dt_s <= 0.0:
            return None if not record_history else self._empty_history()

        n_steps = max(1, int(np.ceil(dt_s / max(self.dt_internal, 1e-12))))
        h = dt_s / n_steps

        hist = self._new_history() if record_history else None
        if hist is not None:
            self._append_history(hist, "relax_start")

        for _ in range(n_steps):
            self._step_relax(h)
            if hist is not None:
                self._append_history(hist, "relax")

        return None if hist is None else self._history_to_arrays(hist)

    def _step_pulse(self, drive: float, dt_s: float) -> None:
        # fast state buildup with simultaneous volatile decay
        dz = (
            self.z_pulse_gain_eff * drive * (1.0 - self.state.z)
            - self.state.z / max(self.tau_z_eff, 1e-12)
        )
        self.state.z += dt_s * dz
        self.state.z = self._clip01(self.state.z)

        # slow growth triggered once the fast residual state is sufficiently high
        gate = self._sigmoid(self.z_to_x_slope * (self.state.z - self.z_to_x_threshold))
        dx_growth = self.x_growth_gain_eff * drive * gate * self.state.r * ((1.0 - self.state.x) ** 1.35)
        dx_leak = self.pulse_leak_factor * self.state.x / max(self.tau_x_eff, 1e-12)
        dx_overload = 0.0
        if self.enable_overload_decay and self.state.x > self.overload_x_threshold and self.state.r < self.overload_r_threshold:
            dx_overload = self.overload_decay_gain * drive * (self.state.x - self.overload_x_threshold)

        self.state.x += dt_s * (dx_growth - dx_leak - dx_overload)
        self.state.x = self._clip01(self.state.x)

        # resource depletion during pulse, with weak simultaneous recovery
        dr = -self.r_depletion_gain * drive * (0.25 + 0.75 * self.state.x) * self.state.r
        dr += 0.12 * (1.0 - self.state.r) / max(self.tau_r_eff, 1e-12)
        self.state.r += dt_s * dr
        self.state.r = self._clip01(self.state.r)

        self.state.t_s += dt_s
        self._update_g()

    def apply_pulse(
        self,
        amplitude_v: float,
        width_s: Optional[float] = None,
        *,
        record_history: bool = False,
    ) -> Optional[dict[str, np.ndarray]]:
        width_s = self.default_pulse_width_s if width_s is None else float(width_s)
        if width_s <= 0.0:
            return None if not record_history else self._empty_history()

        drive = self._drive_from_voltage(amplitude_v) * self._sample_c2c_multiplier()
        drive = max(0.0, float(drive))

        hist = self._new_history() if record_history else None
        if hist is not None:
            self._append_history(hist, "pulse_start")

        if drive <= 0.0:
            # sub-threshold pulse just behaves like elapsed time with no stimulation
            self.relax(width_s, record_history=False)
            if hist is not None:
                self._append_history(hist, "subthreshold")
            return None if hist is None else self._history_to_arrays(hist)

        n_steps = max(1, int(np.ceil(width_s / max(self.dt_internal, 1e-12))))
        h = width_s / n_steps
        for _ in range(n_steps):
            self._step_pulse(drive, h)
            if hist is not None:
                self._append_history(hist, "pulse")

        self.state.n_applied_pulses += 1
        return None if hist is None else self._history_to_arrays(hist)

    def apply_pulse_then_relax(
        self,
        amplitude_v: float,
        width_s: Optional[float] = None,
        gap_after_s: float = 0.0,
        *,
        record_history: bool = False,
    ) -> Optional[dict[str, np.ndarray]]:
        if not record_history:
            self.apply_pulse(amplitude_v=amplitude_v, width_s=width_s, record_history=False)
            if gap_after_s > 0.0:
                self.relax(gap_after_s, record_history=False)
            return None

        hist = self._new_history()
        p = self.apply_pulse(amplitude_v=amplitude_v, width_s=width_s, record_history=True)
        self._extend_history(hist, p)
        if gap_after_s > 0.0:
            g = self.relax(gap_after_s, record_history=True)
            self._extend_history(hist, g, skip_first=True)
        return self._history_to_arrays(hist)

    def simulate_pulse_train(
        self,
        n_pulses: int,
        amplitude_v: float,
        pulse_width_s: Optional[float] = None,
        interval_s: float = 0.0,
        tail_relax_s: float = 0.0,
    ) -> dict[str, np.ndarray]:
        pulse_width_s = self.default_pulse_width_s if pulse_width_s is None else float(pulse_width_s)
        gap_s = max(0.0, float(interval_s))

        hist = self._new_history()
        self._append_history(hist, "init")

        for _ in range(int(n_pulses)):
            ph = self.apply_pulse(amplitude_v=amplitude_v, width_s=pulse_width_s, record_history=True)
            self._extend_history(hist, ph, skip_first=True)
            if gap_s > 0.0:
                gh = self.relax(gap_s, record_history=True)
                self._extend_history(hist, gh, skip_first=True)

        if tail_relax_s > 0.0:
            th = self.relax(float(tail_relax_s), record_history=True)
            self._extend_history(hist, th, skip_first=True)

        return self._history_to_arrays(hist)

    def read_conductance(self, read_voltage: Optional[float] = None) -> float:
        _ = self.read_voltage if read_voltage is None else float(read_voltage)
        g = self.state.g
        if self.enable_read_noise and self.read_noise_rel_sigma > 0.0:
            g *= float(1.0 + self.rng.normal(0.0, self.read_noise_rel_sigma))
        return max(0.0, float(g))

    @property
    def g(self) -> float:
        return float(self.state.g)

    def snapshot(self) -> STMDeviceState:
        return STMDeviceState(
            g=float(self.state.g),
            z=float(self.state.z),
            x=float(self.state.x),
            r=float(self.state.r),
            t_s=float(self.state.t_s),
            n_applied_pulses=int(self.state.n_applied_pulses),
        )

    @staticmethod
    def _new_history() -> dict[str, list[float]]:
        return {
            "time_s": [],
            "conductance_s": [],
            "current_a": [],
            "z": [],
            "x": [],
            "r": [],
            "event": [],
        }

    @staticmethod
    def _empty_history() -> dict[str, np.ndarray]:
        return {
            "time_s": np.asarray([], dtype=float),
            "conductance_s": np.asarray([], dtype=float),
            "current_a": np.asarray([], dtype=float),
            "z": np.asarray([], dtype=float),
            "x": np.asarray([], dtype=float),
            "r": np.asarray([], dtype=float),
            "event": np.asarray([], dtype=object),
        }

    @staticmethod
    def _extend_history(
        dst: dict[str, list[float]],
        src: Optional[dict[str, np.ndarray]],
        *,
        skip_first: bool = False,
    ) -> None:
        if src is None:
            return
        start = 1 if skip_first else 0
        for key in dst:
            dst[key].extend(src[key][start:].tolist())

    @staticmethod
    def _history_to_arrays(hist: dict[str, list[float]]) -> dict[str, np.ndarray]:
        return {
            "time_s": np.asarray(hist["time_s"], dtype=float),
            "conductance_s": np.asarray(hist["conductance_s"], dtype=float),
            "current_a": np.asarray(hist["current_a"], dtype=float),
            "z": np.asarray(hist["z"], dtype=float),
            "x": np.asarray(hist["x"], dtype=float),
            "r": np.asarray(hist["r"], dtype=float),
            "event": np.asarray(hist["event"], dtype=object),
        }
