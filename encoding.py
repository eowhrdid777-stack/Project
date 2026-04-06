from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Union

import numpy as np

import config as cfg


# 1차원 입력은 list/tuple/numpy array 모두 허용
ArrayLike1D = Union[Sequence[float], np.ndarray]

# observation은 dict 형태도 되고, 그냥 1차원 배열 형태도 가능
ObsType = Union[Dict[str, float], ArrayLike1D]

# 지원하는 인코딩 방식 3가지
# - rate: feature 1개당 뉴런 1개, 값이 클수록 spike 확률 증가
# - population_rate: feature 1개를 여러 receptive field 뉴런으로 펼친 뒤 확률 발화
# - population_latency: feature 1개를 여러 receptive field 뉴런으로 펼친 뒤 강할수록 더 빨리 spike
EncodingMode = Literal["rate", "population_rate", "population_latency"]


@dataclass
class EncoderOutput:
    """
    인코더가 최종적으로 반환하는 결과 묶음.
    이후 neuron/network 쪽에서 이 출력을 받아 사용한다.
    """

    # 현재 sim_step에서 실제로 발생한 spike 벡터 (0 또는 1)
    spikes: np.ndarray

    # 각 입력 채널의 활성도
    # rate mode에서는 firing rate 의미에 가깝고,
    # latency mode에서는 receptive field activation strength 의미
    firing_rates: np.ndarray

    # 각 채널이 spike를 내도록 예약된 시간
    # np.inf 이면 이번 latency window 안에서는 spike 없음
    spike_times: np.ndarray

    # 정규화된 입력값
    # population mode에서는 feature 값이 뉴런 수만큼 반복됨
    analog_values: np.ndarray

    # 사람이 보기 쉬운 채널 이름
    feature_names: List[str]

    # 현재 사용한 encoding mode
    mode: str


class SensorSpikeEncoder:
    """
    센서/환경 observation을 spike 형태로 바꾸는 인코더.

    전체 흐름:
    observation -> 배열화 -> 정규화 -> receptive field 전개(필요시) -> mode별 spike 생성

    이 파일은 가능한 한 self-contained 하게 짜여 있어서,
    config에 값이 있으면 가져오고, 없으면 안전한 기본값을 사용한다.
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
        # 현재 어떤 mode를 쓸지 저장
        self.mode = str(mode)

        # rate 계열 모드에서는 확률적으로 spike를 만들기 때문에 난수 생성기가 필요함
        self.rng = np.random.default_rng(getattr(cfg, "SEED", 42) if seed is None else seed)

        # dt: 시뮬레이션 시간 간격
        # max_rate_hz: rate mode에서 최대 발화율
        # neurons_per_feature: feature 하나를 몇 개의 receptive field 뉴런으로 펼칠지
        # latency_steps: latency mode에서 time window 길이
        # activation_threshold: 너무 약한 활성은 spike 안 내도록 자르는 기준
        # sigma_scale: receptive field 폭 조절
        self.dt = float(getattr(cfg, "ENCODER_DT", 1.0 if dt is None else dt)) if dt is None else float(dt)
        self.max_rate_hz = float(getattr(cfg, "ENCODER_MAX_RATE_HZ", 200.0 if max_rate_hz is None else max_rate_hz)) if max_rate_hz is None else float(max_rate_hz)
        self.neurons_per_feature = int(getattr(cfg, "ENCODER_NEURONS_PER_FEATURE", 5 if neurons_per_feature is None else neurons_per_feature)) if neurons_per_feature is None else int(neurons_per_feature)
        self.latency_steps = int(getattr(cfg, "ENCODER_LATENCY_STEPS", 8 if latency_steps is None else latency_steps)) if latency_steps is None else int(latency_steps)
        self.activation_threshold = float(getattr(cfg, "ENCODER_ACTIVATION_THRESHOLD", 0.05 if activation_threshold is None else activation_threshold)) if activation_threshold is None else float(activation_threshold)
        self.sigma_scale = float(getattr(cfg, "ENCODER_SIGMA_SCALE", 0.55 if sigma_scale is None else sigma_scale)) if sigma_scale is None else float(sigma_scale)

        # 입력 feature 이름이 주어지면 그걸 사용
        # 예: ["front_clearance", "left_clearance", ...]
        if feature_names is not None:
            self.feature_names = [str(x) for x in feature_names]
            self.obs_dim = len(self.feature_names)

        # 이름은 없고 차원만 주어지면 x0, x1, x2 ... 형식으로 자동 생성
        elif obs_dim is not None:
            self.obs_dim = int(obs_dim)
            self.feature_names = [f"x{i}" for i in range(self.obs_dim)]

        # 이름도 차원도 없으면 인코더가 어떤 입력을 처리해야 하는지 모르므로 에러
        else:
            raise ValueError("Either obs_dim or feature_names must be provided.")

        # 각 feature의 최소/최대 범위 설정
        self.value_ranges = self._build_value_ranges(value_ranges)

        # receptive field 중심과 폭 미리 생성
        self._rf_centers, self._rf_sigma = self._build_receptive_fields()

        # 최종 출력 채널 수 계산
        # rate면 obs_dim 그대로, population 계열이면 obs_dim * neurons_per_feature
        self.output_dim = self._infer_output_dim()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode(self, obs: ObsType, sim_step: int = 0) -> EncoderOutput:
        """
        observation 하나를 현재 sim_step 기준으로 spike로 변환.

        핵심 흐름:
        obs -> 배열화 -> 정규화 -> mode별 인코딩
        """

        # dict/list/array 형태 입력을 모두 numpy array로 통일
        values = self._coerce_obs(obs)

        # 각 feature 값을 0~1 범위로 정규화
        normalized = self._normalize(values)

        # 현재 mode에 따라 서로 다른 인코딩 함수를 호출
        if self.mode == "rate":
            return self._encode_rate(normalized)
        if self.mode == "population_rate":
            return self._encode_population_rate(normalized)
        if self.mode == "population_latency":
            return self._encode_population_latency(normalized, sim_step=sim_step)

        raise ValueError(f"Unsupported encoding mode: {self.mode}")

    def encode_window(self, obs: ObsType) -> List[EncoderOutput]:
        """
        observation 하나를 전체 시간 window에 대해 인코딩.

        - population_latency:
          latency_steps 길이만큼 step별 EncoderOutput을 생성
        - 그 외 mode:
          한 step 결과만 반환
        """
        if self.mode != "population_latency":
            return [self.encode(obs, sim_step=0)]

        # latency mode는 같은 observation을 여러 timestep에 걸친 spike 패턴으로 펼침
        return [self.encode(obs, sim_step=t) for t in range(self.latency_steps)]

    # ------------------------------------------------------------------
    # Observation handling
    # ------------------------------------------------------------------
    def _build_value_ranges(
        self,
        value_ranges: Optional[Dict[str, tuple[float, float]]],
    ) -> Dict[str, tuple[float, float]]:
        """
        각 feature의 값 범위를 구성.
        예: victim_signal은 (0,1), distance는 (0,200) 같은 식
        """
        if value_ranges is None:
            default = getattr(cfg, "ENCODER_VALUE_RANGES", None)

            # config에 범위 정보가 있으면 그걸 사용
            if isinstance(default, dict) and default:
                value_ranges = {
                    str(k): (float(v[0]), float(v[1]))
                    for k, v in default.items()
                }

            # 없으면 모든 feature를 기본적으로 (0,1) 범위로 간주
            else:
                value_ranges = {name: (0.0, 1.0) for name in self.feature_names}

        out: Dict[str, tuple[float, float]] = {}
        for name in self.feature_names:
            lo, hi = value_ranges.get(name, (0.0, 1.0))
            lo = float(lo)
            hi = float(hi)

            # 잘못된 범위(hi <= lo)가 들어오면 최소한 1.0 폭은 가지도록 보정
            if hi <= lo:
                hi = lo + 1.0

            out[name] = (lo, hi)
        return out

    def _coerce_obs(self, obs: ObsType) -> np.ndarray:
        """
        observation을 numpy 1차원 배열로 통일.
        """
        if isinstance(obs, dict):
            # feature_names 순서대로 값을 꺼내므로,
            # 입력 dict를 넣을 때 이름/순서가 중요함
            vals = [float(obs[name]) for name in self.feature_names]
            return np.asarray(vals, dtype=float)

        arr = np.asarray(obs, dtype=float).reshape(-1)

        # 입력 길이가 기대한 obs_dim과 다르면 에러
        if arr.size != self.obs_dim:
            raise ValueError(f"Expected observation of length {self.obs_dim}, got {arr.size}")

        return arr

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """
        입력값을 feature별 min-max 범위를 기준으로 0~1 정규화.
        """
        out = np.zeros(self.obs_dim, dtype=float)

        for i, name in enumerate(self.feature_names):
            lo, hi = self.value_ranges[name]

            # (x - lo) / (hi - lo) 후 0~1로 clip
            out[i] = np.clip((float(values[i]) - lo) / max(hi - lo, 1e-12), 0.0, 1.0)

        return out

    # ------------------------------------------------------------------
    # Receptive fields
    # ------------------------------------------------------------------
    def _build_receptive_fields(self) -> tuple[np.ndarray, float]:
        """
        population mode에서 사용할 receptive field 중심과 폭 생성.
        """
        # 뉴런 수가 1개면 중앙 0.5에 receptive field 하나만 둠
        if self.neurons_per_feature <= 1:
            centers = np.array([0.5], dtype=float)
            sigma = 0.5
            return centers, sigma

        # 0~1 구간에 receptive field 중심을 균등 배치
        # 예: 5개면 [0.0, 0.25, 0.5, 0.75, 1.0]
        centers = np.linspace(0.0, 1.0, self.neurons_per_feature, dtype=float)

        # 인접 중심 간 간격
        spacing = float(centers[1] - centers[0])

        # receptive field 폭 sigma 계산
        sigma = max(1e-6, self.sigma_scale * spacing)
        return centers, sigma

    def _population_activation(self, normalized: np.ndarray) -> np.ndarray:
        """
        정규화된 feature 값을 receptive field activation으로 변환.

        핵심 아이디어:
        feature 값 하나를 뉴런 1개로 표현하지 않고,
        여러 receptive field 뉴런의 activation 패턴으로 표현함.
        """
        acts = []

        for x in normalized:
            # Gaussian receptive field
            # x가 어떤 center에 가까울수록 activation이 커짐
            a = np.exp(-0.5 * ((x - self._rf_centers) / self._rf_sigma) ** 2)

            # 각 feature 내부에서 최대 activation을 1로 정규화
            a /= max(np.max(a), 1e-12)

            acts.append(a)

        # feature별 receptive field activation을 한 줄로 이어 붙임
        return np.concatenate(acts, axis=0).astype(float)

    def _population_feature_names(self) -> List[str]:
        """
        population mode에서 채널 이름 생성.
        예: front_clearance_rf0, front_clearance_rf1, ...
        """
        names: List[str] = []
        for feat in self.feature_names:
            for k in range(self.neurons_per_feature):
                names.append(f"{feat}_rf{k}")
        return names

    def _infer_output_dim(self) -> int:
        """
        최종 출력 spike 채널 수 계산.
        """
        if self.mode == "rate":
            return int(self.obs_dim)

        return int(self.obs_dim * self.neurons_per_feature)

    # ------------------------------------------------------------------
    # Mode-specific implementations
    # ------------------------------------------------------------------
    def _encode_rate(self, normalized: np.ndarray) -> EncoderOutput:
        """
        가장 단순한 mode.
        feature 1개당 뉴런 1개, 값이 클수록 발화 확률 증가.
        """
        # 정규화된 값 자체를 firing rate처럼 사용
        firing_rates = normalized.copy()

        # 현재 step에서 spike할 확률 계산
        p_fire = np.clip(self.max_rate_hz * self.dt * firing_rates, 0.0, 1.0)

        # 베르누이 확률 발화: 랜덤값 < p_fire 이면 spike
        spikes = (self.rng.random(self.obs_dim) < p_fire).astype(np.int8)

        # rate mode에서는 현재 step에 쏘면 time=0, 아니면 무한대
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
        """
        feature 하나를 여러 receptive field 뉴런으로 펼친 뒤,
        각 activation 크기에 따라 확률적으로 spike 발생.
        """
        # receptive field activation 계산
        activations = self._population_activation(normalized)

        # 각 receptive field 채널의 발화 확률
        p_fire = np.clip(self.max_rate_hz * self.dt * activations, 0.0, 1.0)

        # 확률적 발화
        spikes = (self.rng.random(activations.size) < p_fire).astype(np.int8)

        # 현재 step에 쐈으면 0, 아니면 inf
        spike_times = np.where(spikes > 0, 0.0, np.inf)

        # 원래 feature 값을 receptive field 개수만큼 반복 저장
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
        """
        현재 프로젝트의 주력 mode

        핵심:
        - receptive field activation이 강할수록 더 이른 step에 spike
        - 약하면 늦게 spike
        - threshold보다 작으면 아예 spike 없음
        """
        activations = self._population_activation(normalized)

        # 원래 feature 값을 receptive field 개수만큼 반복
        analog_values = np.repeat(normalized, self.neurons_per_feature)

        # 기본적으로는 모든 채널이 spike 없음(inf)으로 시작
        spike_times = np.full(activations.size, np.inf, dtype=float)

        # 충분히 강한 activation만 spike 후보로 인정
        active = activations >= self.activation_threshold

        if np.any(active):
            # activation이 1에 가까울수록 time=0에 가까워짐
            # 즉 강한 입력일수록 더 빨리 spike
            times = (self.latency_steps - 1) * (1.0 - activations[active])

            # 시간을 정수 step으로 반올림
            spike_times[active] = np.round(times).astype(int)

        # 현재 sim_step에 해당하는 채널만 실제 spike=1로 출력
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
