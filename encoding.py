from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Union

import numpy as np

import config as cfg


ArrayLike1D = Union[Sequence[float], np.ndarray]
ObsType = Union[Dict[str, float], ArrayLike1D]
EncodingMode = Literal["rate", "population_rate", "population_latency"]


@dataclass
class EncoderOutput:
    """Container for encoded spike information.

    Attributes
    ----------
    spikes:
        Binary spike vector for the current step. For latency mode this is the
        event mask at the queried simulation step.
    firing_rates:
        Per-input normalized firing rate in [0, 1]. For latency mode this is
        the receptive-field activation strength.
    spike_times:
        Scheduled spike time of each input channel. ``np.inf`` means no spike
        within the current encoding window.
    analog_values:
        Normalized analog value associated with each input channel.
    feature_names:
        Human-readable channel names.
    mode:
        Encoding mode used to produce the output.
    """

    spikes: np.ndarray
    firing_rates: np.ndarray
    spike_times: np.ndarray
    analog_values: np.ndarray
    feature_names: List[str]
    mode: str


class SensorSpikeEncoder:
    """Flexible sensor-to-spike encoder for the user's SNN pipeline.

    This file is intentionally self-contained so it works with the uploaded
    project structure without requiring edits to ``config.py``. Configuration
    values are pulled from ``config`` when available and otherwise safe defaults
    are used.

    Supported modes
    ----------------
    - ``rate``:
        One spike channel per sensor dimension. The sensor value controls the
        firing probability at each simulation step.
    - ``population_rate``:
        Each sensor dimension is expanded into overlapping Gaussian receptive
        fields. Field activation controls spike probability.
    - ``population_latency``:
        Each sensor dimension is expanded into overlapping receptive fields and
        each active field emits at most one spike in a latency window. Stronger
        activation means earlier spike timing.
    """

    def __init__(
        self,
        obs_dim: Optional[int] = None,
        feature_names: Optional[Sequence[str]] = None,
        mode: EncodingMode = "population_latency",
        seed: Optional[int] = None,
        value_ranges: Optional[Dict[str, tuple[float, float]]] = None,
        neurons_per_feature: Optional[int] = None,
        latency_steps: Optional[int] = None,
        max_rate_hz: Optional[float] = None,
        dt: Optional[float] = None,
        activation_threshold: Optional[float] = None,
        sigma_scale: Optional[float] = None,
    ) -> None:
        self.mode = str(mode)
        self.rng = np.random.default_rng(getattr(cfg, "SEED", 42) if seed is None else seed)

        self.dt = float(getattr(cfg, "ENCODER_DT", 1.0 if dt is None else dt)) if dt is None else float(dt)
        self.max_rate_hz = float(getattr(cfg, "ENCODER_MAX_RATE_HZ", 200.0 if max_rate_hz is None else max_rate_hz)) if max_rate_hz is None else float(max_rate_hz)
        self.neurons_per_feature = int(getattr(cfg, "ENCODER_NEURONS_PER_FEATURE", 5 if neurons_per_feature is None else neurons_per_feature)) if neurons_per_feature is None else int(neurons_per_feature)
        self.latency_steps = int(getattr(cfg, "ENCODER_LATENCY_STEPS", 8 if latency_steps is None else latency_steps)) if latency_steps is None else int(latency_steps)
        self.activation_threshold = float(getattr(cfg, "ENCODER_ACTIVATION_THRESHOLD", 0.05 if activation_threshold is None else activation_threshold)) if activation_threshold is None else float(activation_threshold)
        self.sigma_scale = float(getattr(cfg, "ENCODER_SIGMA_SCALE", 0.55 if sigma_scale is None else sigma_scale)) if sigma_scale is None else float(sigma_scale)

        if feature_names is not None:
            self.feature_names = [str(x) for x in feature_names]
            self.obs_dim = len(self.feature_names)
        elif obs_dim is not None:
            self.obs_dim = int(obs_dim)
            self.feature_names = [f"x{i}" for i in range(self.obs_dim)]
        else:
            raise ValueError("Either obs_dim or feature_names must be provided.")

        self.value_ranges = self._build_value_ranges(value_ranges)
        self._rf_centers, self._rf_sigma = self._build_receptive_fields()
        self.output_dim = self._infer_output_dim()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode(self, obs: ObsType, sim_step: int = 0) -> EncoderOutput:
        """Encode an observation for a given simulation step.

        Parameters
        ----------
        obs:
            Observation vector or feature dictionary.
        sim_step:
            Current simulation step within the encoding window. For latency mode,
            this determines which scheduled spikes are emitted now.
        """
        values = self._coerce_obs(obs)
        normalized = self._normalize(values)

        if self.mode == "rate":
            return self._encode_rate(normalized)
        if self.mode == "population_rate":
            return self._encode_population_rate(normalized)
        if self.mode == "population_latency":
            return self._encode_population_latency(normalized, sim_step=sim_step)
        raise ValueError(f"Unsupported encoding mode: {self.mode}")

    def encode_window(self, obs: ObsType) -> List[EncoderOutput]:
        """Encode a whole latency window.

        For ``population_latency``, this returns one ``EncoderOutput`` per step
        across ``latency_steps``. For other modes, a single-step list is
        returned.
        """
        if self.mode != "population_latency":
            return [self.encode(obs, sim_step=0)]
        return [self.encode(obs, sim_step=t) for t in range(self.latency_steps)]

    # ------------------------------------------------------------------
    # Observation handling
    # ------------------------------------------------------------------
    def _build_value_ranges(
        self,
        value_ranges: Optional[Dict[str, tuple[float, float]]],
    ) -> Dict[str, tuple[float, float]]:
        if value_ranges is None:
            default = getattr(cfg, "ENCODER_VALUE_RANGES", None)
            if isinstance(default, dict) and default:
                value_ranges = {
                    str(k): (float(v[0]), float(v[1]))
                    for k, v in default.items()
                }
            else:
                value_ranges = {name: (0.0, 1.0) for name in self.feature_names}

        out: Dict[str, tuple[float, float]] = {}
        for name in self.feature_names:
            lo, hi = value_ranges.get(name, (0.0, 1.0))
            lo = float(lo)
            hi = float(hi)
            if hi <= lo:
                hi = lo + 1.0
            out[name] = (lo, hi)
        return out

    def _coerce_obs(self, obs: ObsType) -> np.ndarray:
        if isinstance(obs, dict):
            vals = [float(obs[name]) for name in self.feature_names]
            return np.asarray(vals, dtype=float)

        arr = np.asarray(obs, dtype=float).reshape(-1)
        if arr.size != self.obs_dim:
            raise ValueError(f"Expected observation of length {self.obs_dim}, got {arr.size}")
        return arr

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        out = np.zeros(self.obs_dim, dtype=float)
        for i, name in enumerate(self.feature_names):
            lo, hi = self.value_ranges[name]
            out[i] = np.clip((float(values[i]) - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
        return out

    # ------------------------------------------------------------------
    # Receptive fields
    # ------------------------------------------------------------------
    def _build_receptive_fields(self) -> tuple[np.ndarray, float]:
        if self.neurons_per_feature <= 1:
            centers = np.array([0.5], dtype=float)
            sigma = 0.5
            return centers, sigma

        centers = np.linspace(0.0, 1.0, self.neurons_per_feature, dtype=float)
        spacing = float(centers[1] - centers[0])
        sigma = max(1e-6, self.sigma_scale * spacing)
        return centers, sigma

    def _population_activation(self, normalized: np.ndarray) -> np.ndarray:
        acts = []
        for x in normalized:
            a = np.exp(-0.5 * ((x - self._rf_centers) / self._rf_sigma) ** 2)
            a /= max(np.max(a), 1e-12)
            acts.append(a)
        return np.concatenate(acts, axis=0).astype(float)

    def _population_feature_names(self) -> List[str]:
        names: List[str] = []
        for feat in self.feature_names:
            for k in range(self.neurons_per_feature):
                names.append(f"{feat}_rf{k}")
        return names

    def _infer_output_dim(self) -> int:
        if self.mode == "rate":
            return int(self.obs_dim)
        return int(self.obs_dim * self.neurons_per_feature)

    # ------------------------------------------------------------------
    # Mode-specific implementations
    # ------------------------------------------------------------------
    def _encode_rate(self, normalized: np.ndarray) -> EncoderOutput:
        firing_rates = normalized.copy()
        p_fire = np.clip(self.max_rate_hz * self.dt * firing_rates, 0.0, 1.0)
        spikes = (self.rng.random(self.obs_dim) < p_fire).astype(np.int8)
        spike_times = np.where(spikes > 0, 0.0, np.inf)
        return EncoderOutput(
            spikes=spikes,
            firing_rates=firing_rates,
            spike_times=spike_times,
            analog_values=normalized,
            feature_names=list(self.feature_names),
            mode=self.mode,
        )

    def _encode_population_rate(self, normalized: np.ndarray) -> EncoderOutput:
        activations = self._population_activation(normalized)
        p_fire = np.clip(self.max_rate_hz * self.dt * activations, 0.0, 1.0)
        spikes = (self.rng.random(activations.size) < p_fire).astype(np.int8)
        spike_times = np.where(spikes > 0, 0.0, np.inf)
        analog_values = np.repeat(normalized, self.neurons_per_feature)
        return EncoderOutput(
            spikes=spikes,
            firing_rates=activations,
            spike_times=spike_times,
            analog_values=analog_values,
            feature_names=self._population_feature_names(),
            mode=self.mode,
        )

    def _encode_population_latency(self, normalized: np.ndarray, sim_step: int) -> EncoderOutput:
        activations = self._population_activation(normalized)
        analog_values = np.repeat(normalized, self.neurons_per_feature)

        spike_times = np.full(activations.size, np.inf, dtype=float)
        active = activations >= self.activation_threshold
        if np.any(active):
            times = (self.latency_steps - 1) * (1.0 - activations[active])
            spike_times[active] = np.round(times).astype(int)

        spikes = np.zeros(activations.size, dtype=np.int8)
        spikes[np.isfinite(spike_times) & (spike_times == int(sim_step))] = 1

        return EncoderOutput(
            spikes=spikes,
            firing_rates=activations,
            spike_times=spike_times,
            analog_values=analog_values,
            feature_names=self._population_feature_names(),
            mode=self.mode,
        )