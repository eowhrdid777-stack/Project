from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

import config as cfg


def _cfg(name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


@dataclass
class STMDeviceState:
    """Internal state of one short-term-memory synaptic device."""

    g: float                  # observable conductance [S]
    z: float                  # fast volatile residual / ionic accumulation [0, 1]
    x: float                  # slower structural state [0, 1]
    r: float                  # available mobile resource [0, 1]
    t_s: float                # local elapsed time [s]
    n_applied_pulses: int     # number of applied supra-threshold pulses


class STMDeviceModel:
    """
    Compact short-term-memory device model for recurrent feedback synapses.

    Design intent
    -------------
    This model is meant to be used inside a differential-pair STM crossbar:

        logical synapse = G_plus - G_minus

    Therefore one physical STM cell only has non-negative conductance, but a
    signed weight is obtained by pairing two cells in stm_crossbar.py.

    The model is intentionally LTM-compatible at the interface level while
    remaining STM-like at the device-dynamics level:

    - Positive programming pulses increase conductance through fast volatile
      state z and slower structural state x.
    - Negative programming pulses actively depress/reset the conductance.
      This represents a physically plausible opposite-polarity erase/reset
      pulse. If the experimental device only relaxes passively, set the active
      reset gains small and rely mainly on relax().
    - Between pulses, z and x decay toward the rest state, while r recovers.
    - No artificial clipping of observable current or forced spike behavior is
      included here. Only internal state variables constrained by definition
      to [0, 1] are clipped.

    Public API preserved for stm_crossbar.py
    ----------------------------------------
    - reset(mode)
    - relax(dt_s, record_history=False)
    - apply_pulse(amplitude_v, width_s=None, record_history=False)
    - apply_pulse_then_relax(...)
    - simulate_pulse_train(...)
    - read_conductance(read_voltage=None)
    - g property
    - snapshot()
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

        # ------------------------------------------------------------
        # Conductance window
        # ------------------------------------------------------------
        # Keep the STM conductance scale close to the LTM device window so that
        # neuron thresholds/input gains do not require a completely different
        # tuning regime. The default peak is still lower than LTM G_MAX because
        # this cell is intended as a transient feedback device.
        conductance_scale = float(_cfg("STM_CONDUCTANCE_SCALE", 1.0))
        self.g_rest_nom = float(_cfg("STM_G_REST", 1.25e-8)) * conductance_scale
        self.g_peak_nom = float(_cfg("STM_G_PEAK", 1.20e-7)) * conductance_scale
        self.g_nonlinearity = float(_cfg("STM_G_NONLINEARITY", 1.35))

        # ------------------------------------------------------------
        # Time / bias defaults
        # ------------------------------------------------------------
        self.dt_internal = float(_cfg("STM_DT_INTERNAL", 2.0e-5))
        self.default_pulse_width_s = float(_cfg("STM_PULSE_WIDTH_S", 1.0e-3))
        self.read_voltage = float(_cfg("STM_READ_VOLTAGE", 0.1))

        # Bipolar threshold and voltage-to-drive scaling.
        self.pot_threshold_v = float(_cfg("STM_POT_THRESHOLD_V", _cfg("STM_PULSE_THRESHOLD_V", 0.20)))
        self.dep_threshold_v = float(_cfg("STM_DEP_THRESHOLD_V", self.pot_threshold_v))
        self.pot_scale_v = float(_cfg("STM_POT_SCALE_V", _cfg("STM_PULSE_SCALE_V", 0.18)))
        self.dep_scale_v = float(_cfg("STM_DEP_SCALE_V", self.pot_scale_v))

        # Backward-compatible aliases used by old scripts.
        self.pulse_threshold_v = self.pot_threshold_v
        self.pulse_scale_v = self.pot_scale_v

        # ------------------------------------------------------------
        # Pulse-to-state coupling
        # ------------------------------------------------------------
        # z responds quickly; x grows only when z is sufficiently high.
        self.z_pot_gain = float(_cfg("STM_Z_POT_GAIN", _cfg("STM_Z_PULSE_GAIN", 150.0)))
        self.x_pot_gain = float(_cfg("STM_X_POT_GAIN", _cfg("STM_X_GROWTH_GAIN", 55.0)))
        self.z_to_x_threshold = float(_cfg("STM_Z_TO_X_THRESHOLD", 0.25))
        self.z_to_x_slope = float(_cfg("STM_Z_TO_X_SLOPE", 10.0))

        # Active opposite-polarity reset/depression.
        self.z_dep_gain = float(_cfg("STM_Z_DEP_GAIN", 210.0))
        self.x_dep_gain = float(_cfg("STM_X_DEP_GAIN", 35.0))
        self.r_recovery_gain_during_dep = float(_cfg("STM_R_RECOVERY_GAIN_DURING_DEP", 2.0))

        # Small leak during a programming pulse; not an artificial stabilizer,
        # just concurrent relaxation during finite pulse width.
        self.pulse_leak_factor = float(_cfg("STM_PULSE_LEAK_FACTOR", 0.02))

        # ------------------------------------------------------------
        # Relaxation / recovery time constants
        # ------------------------------------------------------------
        # z should decay within a few decision windows; x lasts longer but still
        # relaxes, so this remains STM rather than true nonvolatile LTM.
        self.tau_z_s = float(_cfg("STM_TAU_Z_S", 5.0e-3))
        self.tau_x_s = float(_cfg("STM_TAU_X_S", 0.20))
        self.tau_r_s = float(_cfg("STM_TAU_R_S", 0.50))

        # Resource depletion while potentiating.
        self.r_depletion_gain = float(_cfg("STM_R_DEPLETION_GAIN", 0.20))

        # Optional overload decay is physically interpretable as filament
        # instability at excessive excitation, but keep it disabled by default.
        self.enable_overload_decay = bool(_cfg("STM_ENABLE_OVERLOAD_DECAY", False))
        self.overload_x_threshold = float(_cfg("STM_OVERLOAD_X_THRESHOLD", 0.985))
        self.overload_r_threshold = float(_cfg("STM_OVERLOAD_R_THRESHOLD", 0.10))
        self.overload_decay_gain = float(_cfg("STM_OVERLOAD_DECAY_GAIN", 0.20))

        # Conductance mapping between fast and slow contributions.
        self.fast_weight = float(_cfg("STM_FAST_WEIGHT", 0.35))
        self.slow_weight = float(_cfg("STM_SLOW_WEIGHT", 0.65))
        s = max(1e-12, self.fast_weight + self.slow_weight)
        self.fast_weight /= s
        self.slow_weight /= s

        # ------------------------------------------------------------
        # Variability / readout
        # ------------------------------------------------------------
        self.enable_d2d_variation = bool(_cfg("STM_ENABLE_D2D_VARIATION", True))
        self.cv_d2d = float(_cfg("STM_CV_D2D", 0.03))
        self.enable_c2c_variation = bool(_cfg("STM_ENABLE_C2C_VARIATION", True))
        self.cv_c2c = float(_cfg("STM_CV_C2C", 0.025))
        self.enable_read_noise = bool(_cfg("STM_ENABLE_READ_NOISE", True))
        self.read_noise_rel_sigma = float(_cfg("STM_READ_NOISE_REL_SIGMA", 0.003))

        # Device-to-device variation.
        d2d_window = 1.0
        d2d_tau = 1.0
        d2d_pot_threshold = 1.0
        d2d_dep_threshold = 1.0
        d2d_gain = 1.0
        if self.enable_d2d_variation and self.cv_d2d > 0.0:
            d2d_window = max(0.60, float(self.rng.normal(1.0, self.cv_d2d)))
            d2d_tau = max(0.55, float(self.rng.normal(1.0, 0.50 * self.cv_d2d)))
            d2d_pot_threshold = max(0.65, float(self.rng.normal(1.0, 0.40 * self.cv_d2d)))
            d2d_dep_threshold = max(0.65, float(self.rng.normal(1.0, 0.40 * self.cv_d2d)))
            d2d_gain = max(0.65, float(self.rng.normal(1.0, 0.45 * self.cv_d2d)))

        self.g_rest_eff = float(self.g_rest_nom * d2d_window)
        self.g_peak_eff = float(max(self.g_rest_eff * 1.05, self.g_peak_nom * d2d_window))

        self.tau_z_eff = float(self.tau_z_s * d2d_tau)
        self.tau_x_eff = float(self.tau_x_s * d2d_tau)
        self.tau_r_eff = float(self.tau_r_s * d2d_tau)

        self.pot_threshold_eff = float(self.pot_threshold_v * d2d_pot_threshold)
        self.dep_threshold_eff = float(self.dep_threshold_v * d2d_dep_threshold)
        self.z_pot_gain_eff = float(self.z_pot_gain * d2d_gain)
        self.x_pot_gain_eff = float(self.x_pot_gain * d2d_gain)
        self.z_dep_gain_eff = float(self.z_dep_gain * d2d_gain)
        self.x_dep_gain_eff = float(self.x_dep_gain * d2d_gain)

        # Backward-compatible aliases.
        self.pulse_threshold_eff = self.pot_threshold_eff
        self.z_pulse_gain_eff = self.z_pot_gain_eff
        self.x_growth_gain_eff = self.x_pot_gain_eff

        self.state = STMDeviceState(
            g=self.g_rest_eff,
            z=0.0,
            x=0.0,
            r=1.0,
            t_s=0.0,
            n_applied_pulses=0,
        )
        self._update_g()

    # ------------------------------------------------------------------
    # Scalar helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clip01(v: float) -> float:
        return float(np.clip(v, 0.0, 1.0))

    @staticmethod
    def _sigmoid(u: float) -> float:
        # Avoid overflow for very large |u|.
        u = float(np.clip(u, -80.0, 80.0))
        return float(1.0 / (1.0 + np.exp(-u)))

    def _sample_c2c_multiplier(self) -> float:
        if not self.enable_c2c_variation or self.cv_c2c <= 0.0:
            return 1.0
        return max(0.0, float(self.rng.normal(1.0, self.cv_c2c)))

    @staticmethod
    def _voltage_drive(amplitude_v: float, threshold_v: float, scale_v: float) -> float:
        v_eff = max(0.0, abs(float(amplitude_v)) - float(threshold_v))
        if v_eff <= 0.0:
            return 0.0
        return float(1.0 - np.exp(-v_eff / max(float(scale_v), 1e-12)))

    def _drive_from_voltage(self, amplitude_v: float) -> tuple[str, float]:
        """Return (polarity, normalized_drive)."""
        amplitude_v = float(amplitude_v)
        if amplitude_v >= 0.0:
            drive = self._voltage_drive(amplitude_v, self.pot_threshold_eff, self.pot_scale_v)
            return "pot", drive
        drive = self._voltage_drive(amplitude_v, self.dep_threshold_eff, self.dep_scale_v)
        return "dep", drive

    def _observable_activation(self) -> float:
        a = self.slow_weight * self.state.x + self.fast_weight * self.state.z
        return self._clip01(a)

    def _update_g(self) -> None:
        a = self._observable_activation()
        self.state.g = float(
            self.g_rest_eff + (self.g_peak_eff - self.g_rest_eff) * (a ** self.g_nonlinearity)
        )

    # ------------------------------------------------------------------
    # State reset / diagnostics
    # ------------------------------------------------------------------
    def reset(self, mode: str = "rest") -> None:
        mode = str(mode).lower()
        if mode in ("rest", "min", "init"):
            z, x, r = 0.0, 0.0, 1.0
        elif mode == "mid":
            z, x, r = 0.12, 0.30, 0.92
        elif mode in ("peak", "max"):
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

    def describe(self) -> dict[str, float | bool]:
        return {
            "g_rest_eff": self.g_rest_eff,
            "g_peak_eff": self.g_peak_eff,
            "tau_z_eff": self.tau_z_eff,
            "tau_x_eff": self.tau_x_eff,
            "tau_r_eff": self.tau_r_eff,
            "pot_threshold_eff": self.pot_threshold_eff,
            "dep_threshold_eff": self.dep_threshold_eff,
            "fast_weight": self.fast_weight,
            "slow_weight": self.slow_weight,
            "enable_d2d_variation": self.enable_d2d_variation,
            "enable_c2c_variation": self.enable_c2c_variation,
            "enable_read_noise": self.enable_read_noise,
        }

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------
    def _append_history(self, hist: dict[str, list], event: str) -> None:
        hist["time_s"].append(float(self.state.t_s))
        hist["conductance_s"].append(float(self.state.g))
        hist["current_a"].append(float(self.state.g * self.read_voltage))
        hist["z"].append(float(self.state.z))
        hist["x"].append(float(self.state.x))
        hist["r"].append(float(self.state.r))
        hist["event"].append(str(event))

    # ------------------------------------------------------------------
    # Relaxation dynamics
    # ------------------------------------------------------------------
    def _step_relax(self, dt_s: float) -> None:
        dt_s = max(0.0, float(dt_s))
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

    # ------------------------------------------------------------------
    # Pulse dynamics
    # ------------------------------------------------------------------
    def _step_pot_pulse(self, drive: float, dt_s: float) -> None:
        # Fast volatile buildup with simultaneous finite-width decay.
        dz = (
            self.z_pot_gain_eff * drive * self.state.r * (1.0 - self.state.z)
            - self.pulse_leak_factor * self.state.z / max(self.tau_z_eff, 1e-12)
        )
        self.state.z = self._clip01(self.state.z + dt_s * dz)

        # Slow structural growth occurs only when the fast state is sufficiently high.
        gate = self._sigmoid(self.z_to_x_slope * (self.state.z - self.z_to_x_threshold))
        dx_growth = self.x_pot_gain_eff * drive * gate * self.state.r * ((1.0 - self.state.x) ** 1.30)
        dx_leak = self.pulse_leak_factor * self.state.x / max(self.tau_x_eff, 1e-12)
        dx_overload = 0.0
        if self.enable_overload_decay and self.state.x > self.overload_x_threshold and self.state.r < self.overload_r_threshold:
            dx_overload = self.overload_decay_gain * drive * (self.state.x - self.overload_x_threshold)
        self.state.x = self._clip01(self.state.x + dt_s * (dx_growth - dx_leak - dx_overload))

        # Potentiation consumes available mobile resource.
        dr = -self.r_depletion_gain * drive * (0.20 + 0.80 * self.state.x) * self.state.r
        dr += 0.05 * (1.0 - self.state.r) / max(self.tau_r_eff, 1e-12)
        self.state.r = self._clip01(self.state.r + dt_s * dr)

        self.state.t_s += dt_s
        self._update_g()

    def _step_dep_pulse(self, drive: float, dt_s: float) -> None:
        # Opposite-polarity reset accelerates removal of the volatile residual and
        # partially erases the slower state. This is an explicit hardware assumption
        # of bipolar STM programming, not a numerical shortcut.
        dz = -self.z_dep_gain_eff * drive * self.state.z
        dz -= self.pulse_leak_factor * self.state.z / max(self.tau_z_eff, 1e-12)
        self.state.z = self._clip01(self.state.z + dt_s * dz)

        dx = -self.x_dep_gain_eff * drive * self.state.x * (0.25 + 0.75 * self.state.z)
        dx -= self.pulse_leak_factor * self.state.x / max(self.tau_x_eff, 1e-12)
        self.state.x = self._clip01(self.state.x + dt_s * dx)

        # Reset pulse lets mobile resource recover faster.
        dr = self.r_recovery_gain_during_dep * drive * (1.0 - self.state.r)
        dr += 0.20 * (1.0 - self.state.r) / max(self.tau_r_eff, 1e-12)
        self.state.r = self._clip01(self.state.r + dt_s * dr)

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

        polarity, drive = self._drive_from_voltage(amplitude_v)
        drive *= self._sample_c2c_multiplier()
        drive = max(0.0, float(drive))

        hist = self._new_history() if record_history else None
        if hist is not None:
            self._append_history(hist, f"{polarity}_pulse_start")

        if drive <= 0.0:
            # Sub-threshold programming pulse still consumes elapsed physical time.
            self.relax(width_s, record_history=False)
            if hist is not None:
                self._append_history(hist, "subthreshold_relax")
            return None if hist is None else self._history_to_arrays(hist)

        n_steps = max(1, int(np.ceil(width_s / max(self.dt_internal, 1e-12))))
        h = width_s / n_steps
        for _ in range(n_steps):
            if polarity == "pot":
                self._step_pot_pulse(drive, h)
            else:
                self._step_dep_pulse(drive, h)
            if hist is not None:
                self._append_history(hist, f"{polarity}_pulse")

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

        for _ in range(max(0, int(n_pulses))):
            ph = self.apply_pulse(amplitude_v=amplitude_v, width_s=pulse_width_s, record_history=True)
            self._extend_history(hist, ph, skip_first=True)
            if gap_s > 0.0:
                gh = self.relax(gap_s, record_history=True)
                self._extend_history(hist, gh, skip_first=True)

        if tail_relax_s > 0.0:
            th = self.relax(float(tail_relax_s), record_history=True)
            self._extend_history(hist, th, skip_first=True)

        return self._history_to_arrays(hist)

    # ------------------------------------------------------------------
    # Readout
    # ------------------------------------------------------------------
    def read_conductance(self, read_voltage: Optional[float] = None) -> float:
        _ = self.read_voltage if read_voltage is None else float(read_voltage)
        g = float(self.state.g)
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

    # ------------------------------------------------------------------
    # History container helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _new_history() -> dict[str, list]:
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
        dst: dict[str, list],
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
    def _history_to_arrays(hist: dict[str, list]) -> dict[str, np.ndarray]:
        return {
            "time_s": np.asarray(hist["time_s"], dtype=float),
            "conductance_s": np.asarray(hist["conductance_s"], dtype=float),
            "current_a": np.asarray(hist["current_a"], dtype=float),
            "z": np.asarray(hist["z"], dtype=float),
            "x": np.asarray(hist["x"], dtype=float),
            "r": np.asarray(hist["r"], dtype=float),
            "event": np.asarray(hist["event"], dtype=object),
        }
