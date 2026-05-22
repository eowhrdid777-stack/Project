from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

import config as cfg

ArrayLike1D = Union[Sequence[float], np.ndarray]
ObsType = Union[Dict[str, float], ArrayLike1D]


@dataclass
class EncoderOutput:

    # 현재 timestep에서 실제로 발생한 입력 spike 벡터, shape = (output_dim,)
    spikes: np.ndarray

    # population receptive field activation.
    # latency encoding에서는 firing rate라기보다 각 입력 채널의 반응 세기이다.
    firing_rates: np.ndarray

    # 각 채널이 spike를 내도록 예약된 timestep.
    # np.inf이면 해당 latency window 안에서는 spike가 발생하지 않는다.
    spike_times: np.ndarray

    # 정규화된 원래 feature 값.
    # population encoding에서는 feature 값을 neurons_per_feature만큼 반복한다.
    analog_values: np.ndarray

    # 각 population channel 이름.
    # 예: front_clearance_rf0, front_clearance_rf1, ...
    feature_names: List[str]

    # 현재 인코딩 모드. 현재 파일에서는 항상 "population_latency"이다.
    mode: str = "population_latency"


class SensorSpikeEncoder:
    """
    Population latency encoder.

    처리 흐름:
        observation
        -> feature_names 순서대로 배열화
        -> feature별 value_ranges 기준 0~1 정규화
        -> Gaussian receptive field population activation 계산
        -> activation이 큰 채널일수록 빠른 timestep에 spike 예약
        -> encode_window()가 latency_steps 길이의 spike window 반환

    현재 simulation 코드와의 연결:
        network.py는 encoder.output_dim을 보고 crossbar row 수를 정한다.
        따라서 neurons_per_feature 또는 feature_names를 바꾸면
        network/crossbar를 새로 생성해야 한다.

    실제 로봇 사용 시 주의:
        이 파일은 실제 센서를 읽지 않는다.
        실제 로봇에서는 real_robot_env.py 또는 robot_interface.py에서
        Arduino로부터 raw sensor를 읽은 뒤 아래처럼 observation을 구성한다.

        예시:
            obs = {
                "front_clearance": normalized_or_scaled_distance,
                "left_clearance": ...,
                "right_clearance": ...,
                "victim_signal": ...,
            }

        만약 실제 로봇에서 raw 값(DIST_MM, R, G, B, C)을 그대로 넣고 싶다면,
        main.py의 build_encoder()에서 feature_names와 value_ranges를
        실제 센서 이름/범위에 맞게 바꾸면 된다.
    """

    SUPPORTED_MODE = "population_latency"

    def __init__(
        self,
        obs_dim: Optional[int] = None,
        feature_names: Optional[Sequence[str]] = None,
        mode: str = "population_latency",
        seed: Optional[int] = None,
        value_ranges: Optional[Dict[str, tuple[float, float]]] = None,
        neurons_per_feature: Optional[int] = None,
        latency_steps: Optional[int] = None,
        activation_threshold: Optional[float] = None,
        sigma_scale: Optional[float] = None,
        ensure_one_spike_per_feature: Optional[bool] = None,
        use_rank_order_tie_break: Optional[bool] = None,
    ) -> None:
        # ------------------------------------------------------------
        # Mode validation
        # ------------------------------------------------------------
        self.mode = str(mode)
        if self.mode != self.SUPPORTED_MODE:
            raise ValueError(
                "This encoding.py supports only 'population_latency'. "
                f"Got mode={self.mode!r}."
            )

        # seed는 현재 deterministic latency encoding에서는 거의 쓰지 않지만,
        # 추후 tie-break나 noise 실험에 대비해 보존한다.
        self.seed = int(getattr(cfg, "SEED", 42) if seed is None else seed)
        self.rng = np.random.default_rng(self.seed)

        # ------------------------------------------------------------
        # Feature definition
        # ------------------------------------------------------------
        if feature_names is not None:
            self.feature_names = [str(name) for name in feature_names]
            self.obs_dim = len(self.feature_names)
        elif obs_dim is not None:
            self.obs_dim = int(obs_dim)
            if self.obs_dim <= 0:
                raise ValueError("obs_dim must be >= 1.")
            self.feature_names = [f"x{i}" for i in range(self.obs_dim)]
        else:
            raise ValueError("Either feature_names or obs_dim must be provided.")

        if self.obs_dim <= 0:
            raise ValueError("At least one input feature is required.")

        # ------------------------------------------------------------
        # Encoder constants
        # ------------------------------------------------------------
        self.neurons_per_feature = int(
            getattr(cfg, "ENCODER_NEURONS_PER_FEATURE", 7)
            if neurons_per_feature is None
            else neurons_per_feature
        )
        self.latency_steps = int(
            getattr(cfg, "ENCODER_LATENCY_STEPS", 8)
            if latency_steps is None
            else latency_steps
        )
        self.activation_threshold = float(
            getattr(cfg, "ENCODER_ACTIVATION_THRESHOLD", 0.08)
            if activation_threshold is None
            else activation_threshold
        )
        self.sigma_scale = float(
            getattr(cfg, "ENCODER_SIGMA_SCALE", 0.65)
            if sigma_scale is None
            else sigma_scale
        )
        self.ensure_one_spike_per_feature = bool(
            getattr(cfg, "ENCODER_ENSURE_ONE_SPIKE_PER_FEATURE", False)
        )
        self.use_rank_order_tie_break = bool(
            getattr(cfg, "ENCODER_USE_RANK_ORDER_TIE_BREAK", False)
        )

        # # ensure_one_spike_per_feature=True이면,
        # # 모든 receptive field activation이 threshold보다 낮더라도
        # # feature별 가장 강한 RF 하나는 spike를 만들게 한다.
        # # 이 옵션은 너무 sparse해서 hidden/output spike가 거의 안 나올 때 유용하다.
        # self.ensure_one_spike_per_feature = bool(
        #     getattr(cfg, "ENCODER_ENSURE_ONE_SPIKE_PER_FEATURE", False)
        #     if ensure_one_spike_per_feature is None
        #     else ensure_one_spike_per_feature
        # )

        # # 같은 timestep에 너무 많은 spike가 몰리는 것을 줄이고 싶을 때 쓴다.
        # # True이면 같은 feature 내부 RF index에 따라 아주 작은 정렬용 offset을 둔 뒤
        # # floor로 latency를 정한다. 기본은 False로 두어 해석을 단순하게 유지한다.
        # self.use_rank_order_tie_break = bool(
        #     getattr(cfg, "ENCODER_USE_RANK_ORDER_TIE_BREAK", False)
        #     if use_rank_order_tie_break is None
        #     else use_rank_order_tie_break
        # )

        self._validate_parameters()

        # feature별 raw 값 범위.
        # simulation 현재 main.py에서는 모두 (0, 1)로 들어온다.
        # 실제 로봇 raw 센서값을 직접 넣을 경우 여기 범위를 반드시 바꿔야 한다.
        self.value_ranges = self._build_value_ranges(value_ranges)

        # RF 중심 및 폭.
        self._rf_centers, self._rf_sigma = self._build_receptive_fields()

        # network.py가 사용하는 최종 입력 차원.
        self.output_dim = int(self.obs_dim * self.neurons_per_feature)

        # channel 이름은 매번 만들지 않고 고정해 둔다.
        self.output_feature_names = self._population_feature_names()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode(self, obs: ObsType, sim_step: int = 0) -> EncoderOutput:
        """
        observation 하나를 특정 timestep의 spike 벡터로 변환한다.

        sim_step:
            0 <= sim_step < latency_steps일 때 해당 timestep spike를 반환한다.
            범위 밖이면 spike가 모두 0인 결과가 반환된다.
        """
        values = self._coerce_obs(obs)
        normalized = self._normalize(values)
        activations = self._population_activation(normalized)
        analog_values = np.repeat(normalized, self.neurons_per_feature)

        spike_times = self._activation_to_spike_times(activations)
        spikes = np.zeros(self.output_dim, dtype=np.int8)

        t = int(sim_step)
        if 0 <= t < self.latency_steps:
            spikes[np.isfinite(spike_times) & (spike_times == t)] = 1

        return EncoderOutput(
            spikes=spikes,
            firing_rates=activations.astype(float, copy=False),
            spike_times=spike_times,
            analog_values=analog_values.astype(float, copy=False),
            feature_names=list(self.output_feature_names),
            mode=self.mode,
        )

    def encode_window(self, obs: ObsType) -> List[EncoderOutput]:
        """
        observation 하나를 latency_steps 길이의 spike window로 변환한다.

        network.py의 MemristiveSNNNetwork._prepare_window()가 이 함수를 호출한다.
        반환 길이가 decision window 길이가 되므로, latency_steps는 너무 크게 잡지 않는다.
        """
        return [self.encode(obs, sim_step=t) for t in range(self.latency_steps)]

    def transform(self, obs: ObsType) -> np.ndarray:
        """
        디버깅용 helper.
        전체 latency window를 np.ndarray로 반환한다.
        shape = (latency_steps, output_dim)
        """
        return np.asarray([out.spikes for out in self.encode_window(obs)], dtype=np.int8)

    def describe(self) -> Dict[str, object]:
        """
        현재 encoder 설정을 dict로 반환한다.
        main.py에서 print하거나 실험 로그에 저장하기 좋다.
        """
        return {
            "mode": self.mode,
            "obs_dim": self.obs_dim,
            "input_feature_names": list(self.feature_names),
            "output_dim": self.output_dim,
            "neurons_per_feature": self.neurons_per_feature,
            "latency_steps": self.latency_steps,
            "activation_threshold": self.activation_threshold,
            "sigma_scale": self.sigma_scale,
            "rf_centers": self._rf_centers.copy(),
            "rf_sigma": self._rf_sigma,
            "value_ranges": dict(self.value_ranges),
            "ensure_one_spike_per_feature": self.ensure_one_spike_per_feature,
        }

    # ------------------------------------------------------------------
    # Validation / construction
    # ------------------------------------------------------------------
    def _validate_parameters(self) -> None:
        if self.neurons_per_feature < 1:
            raise ValueError("neurons_per_feature must be >= 1.")
        if self.latency_steps < 1:
            raise ValueError("latency_steps must be >= 1.")
        if not (0.0 <= self.activation_threshold <= 1.0):
            raise ValueError("activation_threshold must be in [0, 1].")
        if self.sigma_scale <= 0.0:
            raise ValueError("sigma_scale must be > 0.")

    def _build_value_ranges(
        self,
        value_ranges: Optional[Dict[str, tuple[float, float]]],
    ) -> Dict[str, tuple[float, float]]:
        if value_ranges is None:
            cfg_ranges = getattr(cfg, "ENCODER_VALUE_RANGES", None)
            if isinstance(cfg_ranges, dict) and len(cfg_ranges) > 0:
                value_ranges = {
                    str(k): (float(v[0]), float(v[1]))
                    for k, v in cfg_ranges.items()
                }
            else:
                value_ranges = {name: (0.0, 1.0) for name in self.feature_names}

        ranges: Dict[str, tuple[float, float]] = {}
        for name in self.feature_names:
            lo, hi = value_ranges.get(name, (0.0, 1.0))
            lo = float(lo)
            hi = float(hi)
            if not np.isfinite(lo) or not np.isfinite(hi):
                raise ValueError(f"value range for {name!r} must be finite.")
            if hi <= lo:
                raise ValueError(
                    f"value range for {name!r} must satisfy hi > lo. "
                    f"Got ({lo}, {hi})."
                )
            ranges[name] = (lo, hi)
        return ranges

    def _build_receptive_fields(self) -> tuple[np.ndarray, float]:
        if self.neurons_per_feature == 1:
            centers = np.array([0.5], dtype=float)
            sigma = 0.5
            return centers, sigma

        centers = np.linspace(0.0, 1.0, self.neurons_per_feature, dtype=float)
        spacing = float(centers[1] - centers[0])

        # sigma_scale이 너무 작으면 입력이 지나치게 sparse해져서
        # output spike가 거의 안 나올 수 있다.
        # 현재 기본값 0.65는 이웃 RF가 어느 정도 겹치도록 잡은 값이다.
        sigma = max(1e-6, self.sigma_scale * spacing)
        return centers, sigma

    def _population_feature_names(self) -> List[str]:
        names: List[str] = []
        for feat in self.feature_names:
            for k in range(self.neurons_per_feature):
                names.append(f"{feat}_rf{k}")
        return names

    # ------------------------------------------------------------------
    # Observation handling
    # ------------------------------------------------------------------
    def _coerce_obs(self, obs: ObsType) -> np.ndarray:
        if isinstance(obs, dict):
            missing = [name for name in self.feature_names if name not in obs]
            if missing:
                raise KeyError(
                    "Observation dict is missing required features: "
                    + ", ".join(missing)
                )
            values = [float(obs[name]) for name in self.feature_names]
            arr = np.asarray(values, dtype=float)
        else:
            arr = np.asarray(obs, dtype=float).reshape(-1)
            if arr.size != self.obs_dim:
                raise ValueError(
                    f"Expected observation length {self.obs_dim}, got {arr.size}."
                )

        if not np.all(np.isfinite(arr)):
            raise ValueError("Observation contains NaN or inf.")
        return arr

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        normalized = np.zeros(self.obs_dim, dtype=float)
        for i, name in enumerate(self.feature_names):
            lo, hi = self.value_ranges[name]
            normalized[i] = (float(values[i]) - lo) / (hi - lo)

        # simulation/robot 센서가 범위를 살짝 벗어나는 것은 학습을 깨지 않도록 clip한다.
        return np.clip(normalized, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Population latency encoding
    # ------------------------------------------------------------------
    def _population_activation(self, normalized: np.ndarray) -> np.ndarray:
        """
        각 feature를 Gaussian receptive field population으로 펼친다.

        output ordering:
            [feature0_rf0, feature0_rf1, ...,
             feature1_rf0, feature1_rf1, ...]
        """
        blocks: List[np.ndarray] = []

        for x in normalized:
            x = float(x)
            activation = np.exp(
                -0.5 * ((x - self._rf_centers) / self._rf_sigma) ** 2
            )

            # feature별 최대값을 1로 맞춰 값 크기보다는 위치 정보를 안정적으로 표현한다.
            max_a = float(np.max(activation))
            if max_a > 0.0:
                activation = activation / max_a

            blocks.append(activation.astype(float, copy=False))

        return np.concatenate(blocks, axis=0)

    def _activation_to_spike_times(self, activations: np.ndarray) -> np.ndarray:
        """
        activation vector를 latency spike time vector로 바꾼다.

        activation = 1.0 -> t = 0
        activation이 작을수록 -> t가 뒤로 밀림
        activation < threshold -> spike 없음(np.inf)
        """
        spike_times = np.full(self.output_dim, np.inf, dtype=float)

        for feat_idx in range(self.obs_dim):
            start = feat_idx * self.neurons_per_feature
            end = start + self.neurons_per_feature
            block = activations[start:end]

            active_mask = block >= self.activation_threshold

            if self.ensure_one_spike_per_feature and not np.any(active_mask):
                # threshold가 너무 높거나 RF 폭이 너무 좁아도
                # feature 정보가 완전히 사라지지 않도록 가장 큰 RF 하나는 살린다.
                active_mask[int(np.argmax(block))] = True

            if not np.any(active_mask):
                continue

            # 강한 activation일수록 빠른 spike.
            # floor를 쓰면 강한 반응이 더 확실히 앞 timestep에 배치된다.
            raw_times = (self.latency_steps - 1) * (1.0 - block[active_mask])

            if self.use_rank_order_tie_break and raw_times.size > 1:
                # 같은 feature 내부에서 완전히 같은 시간이 나오는 경우를 줄이기 위한 선택 옵션.
                # 아주 작은 offset만 주므로 latency 순서를 크게 바꾸지는 않는다.
                local_indices = np.flatnonzero(active_mask).astype(float)
                raw_times = raw_times + 1e-3 * local_indices

            times = np.floor(raw_times + 1e-12).astype(int)
            times = np.clip(times, 0, self.latency_steps - 1)

            block_times = spike_times[start:end]
            block_times[active_mask] = times.astype(float)
            spike_times[start:end] = block_times

        return spike_times
