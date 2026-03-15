# encoding

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import config as cfg


ArrayLike1D = Union[Sequence[float], np.ndarray]
ObsType = Union[Dict[str, float], ArrayLike1D]


@dataclass
class EncoderOutput:

    values: np.ndarray
    bipolar_values: np.ndarray
    level_spikes: np.ndarray
    latency_spikes: np.ndarray
    analog_current: np.ndarray


class SpikeEncoder:

    def __init__(
        self,
        n_inputs: int,
        sensor_names: Optional[List[str]] = None,
        sensor_min: Optional[ArrayLike1D] = None,
        sensor_max: Optional[ArrayLike1D] = None,
        dt: float = 1.0,
        rate_max_hz: float = 100.0,
        default_steps: int = 20,
        analog_scale: float = 1.0,
        seed: Optional[int] = getattr(cfg, "SEED", 42),
    ) -> None:
        self.n_inputs = int(n_inputs)
        if self.n_inputs <= 0:
            raise ValueError("n_inputs must be positive.")

        self.sensor_names = (
            sensor_names[:] if sensor_names is not None
            else [f"sensor_{i}" for i in range(self.n_inputs)]
        )

        if len(self.sensor_names) != self.n_inputs:
            raise ValueError(
                f"len(sensor_names) must equal n_inputs ({self.n_inputs}), "
                f"got {len(self.sensor_names)}"
            )

        if sensor_min is None:
            self.sensor_min = np.zeros(self.n_inputs, dtype=np.float64)
        else:
            self.sensor_min = self._as_vector(sensor_min, name="sensor_min")

        if sensor_max is None:
            self.sensor_max = np.ones(self.n_inputs, dtype=np.float64)
        else:
            self.sensor_max = self._as_vector(sensor_max, name="sensor_max")

        if np.any(self.sensor_max <= self.sensor_min):
            raise ValueError("Each sensor_max must be greater than sensor_min.")

        self.dt = float(dt)
        self.rate_max_hz = float(rate_max_hz)
        self.default_steps = int(default_steps)
        self.analog_scale = float(analog_scale)

        if self.default_steps <= 0:
            raise ValueError("default_steps must be positive.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if self.rate_max_hz < 0:
            raise ValueError("rate_max_hz must be non-negative.")

        self.rng = np.random.default_rng(seed)

    # 입력을 넘파이 벡터로 통일
    def _as_vector(self, x: ArrayLike1D, name: str = "input") -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if arr.shape[0] != self.n_inputs:
            raise ValueError(
                f"{name} must have length {self.n_inputs}, got shape {arr.shape}"
            )
        return arr
    
    # dictionary 혹은 array로 들어온 입력을 최종 숫자 벡터로 생성
    def _obs_to_vector(self, obs: ObsType) -> np.ndarray: 
        
        if isinstance(obs, dict):
            vec = np.zeros(self.n_inputs, dtype=np.float64) #길이 n_input짜리 0 벡터 생성
            for i, name in enumerate(self.sensor_names):    #센서 이름을 순서대로 하나씩 꺼내면서 번호도 같이 가져옴
                if name not in obs:
                    raise KeyError(
                        f"Observation dict is missing sensor key '{name}'. "
                        f"Expected keys: {self.sensor_names}"
                    )
                vec[i] = float(obs[name]) 
            return vec

        return self._as_vector(obs, name="obs") #딕셔너리가 아니면 그냥 array/list라고 보고 _as_vector로 변환

    #센서 값 0~1로 nomalize
    def normalize(self, obs: ObsType) -> np.ndarray:
        raw = self._obs_to_vector(obs) #입력을 숫자 벡터로 변환
        denom = np.maximum(self.sensor_max - self.sensor_min, 1e-12)
        norm = (raw - self.sensor_min) / denom #정규화 식
        return np.clip(norm, 0.0, 1.0) #정규화 값이 범위 벗어나면 clip
    
    #원래 scale로 값 복원
    def denormalize(self, x_norm: ArrayLike1D) -> np.ndarray: 
        x_norm = self._as_vector(x_norm, name="x_norm")
        return self.sensor_min + x_norm * (self.sensor_max - self.sensor_min)

    #범위를 [0,1]->[-1,1]
    def to_bipolar(self, x_norm: ArrayLike1D) -> np.ndarray:
        x_norm = self._as_vector(x_norm, name="x_norm")
        return 2.0 * x_norm - 1.0

    #crossbar에 넣을 입력 생성
    def to_analog_current(self, x_norm: ArrayLike1D, bipolar: bool = False) -> np.ndarray:
        x_norm = self._as_vector(x_norm, name="x_norm")
        if bipolar:
            return self.to_bipolar(x_norm) * self.analog_scale #입력 강도 조절하기
        return x_norm * self.analog_scale

    # ------------------------------------------------------------------
    # Spike encoders
    # ------------------------------------------------------------------
    
     #값 크기를 spike 개수로 표현하는 encode 방법
    def level_encode(
    self,
    obs: ObsType,
    n_steps: Optional[int] = None,
) -> np.ndarray:
        x_norm = self.normalize(obs)
        n_steps = self.default_steps if n_steps is None else int(n_steps)

        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        
        spikes = np.zeros((n_steps, self.n_inputs), dtype=np.float64)
        
        for i, x in enumerate(x_norm):
            n_active = int(round(x * n_steps))
            n_active = int(np.clip(n_active, 0, n_steps))
            spikes[:n_active, i] = 1.0
        
        return spikes


    #값을 spike 발생 시간으로 표현하는 encode 방법(값이 크면 빨리 spike)
    def latency_encode( 
        self,
        obs: ObsType,
        n_steps: Optional[int] = None,
        allow_zero_spike: bool = True,
    ) -> np.ndarray:
        x_norm = self.normalize(obs)
        n_steps = self.default_steps if n_steps is None else int(n_steps)

        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")

        spikes = np.zeros((n_steps, self.n_inputs), dtype=np.float64)

        for i, x in enumerate(x_norm):
            if allow_zero_spike and x <= 0.0:
                continue

            # x=1 -> step 0, x small -> later step
            t_idx = int(round((1.0 - x) * max(n_steps - 1, 0)))
            t_idx = int(np.clip(t_idx, 0, n_steps - 1))
            spikes[t_idx, i] = 1.0

        return spikes

    #threshold값 넘으면 spike하는 encode 방법
    def threshold_encode( 
        self,
        obs: ObsType,
        threshold: float = 0.5,
        bipolar: bool = False,
    ) -> np.ndarray:
        x_norm = self.normalize(obs)
        binary = (x_norm >= float(threshold)).astype(np.float64)

        if bipolar:
            return 2.0 * binary - 1.0
        return binary
    
    #encoder 종류 select
    def select_spike_encoding(
    self,
    obs: ObsType,
    mode: str = "latency",
    n_steps: Optional[int] = None,
    threshold: float = 0.5,
) -> np.ndarray:
        mode = mode.lower()

        if mode == "level":
            return self.level_encode(obs, n_steps=n_steps)

        if mode == "latency":
            return self.latency_encode(obs, n_steps=n_steps)

        if mode == "threshold":
            x = self.threshold_encode(obs, threshold=threshold, bipolar=False)
            # threshold는 시간축이 없는 1D 벡터라서, VMM 시간 루프와 맞추려면 (1, n_inputs)로 변환
            return x.reshape(1, -1)
        
        raise ValueError(f"Unknown encoding mode: {mode}")

    # ------------------------------------------------------------------
    # High-level bundle API
    # ------------------------------------------------------------------
    def encode(
        self,
        obs: ObsType,
        n_steps: Optional[int] = None,
        bipolar_analog: bool = False,
    ) -> EncoderOutput:
        x_norm = self.normalize(obs)
        n_steps = self.default_steps if n_steps is None else int(n_steps)

        return EncoderOutput(
            values=x_norm,
            bipolar_values=self.to_bipolar(x_norm),
            level_spikes=self.level_encode(obs, n_steps=n_steps),
            latency_spikes=self.latency_encode(obs, n_steps=n_steps),
            analog_current=self.to_analog_current(x_norm, bipolar=bipolar_analog),
        )

    # ------------------------------------------------------------------
    # Rescue-robot friendly helper
    # ------------------------------------------------------------------
    @classmethod
    def build_rescue_encoder(
        cls,
        n_inputs: int = 4,
        seed: Optional[int] = getattr(cfg, "SEED", 42),
    ) -> "SpikeEncoder":

        names = ["front_dist", "left_dist", "right_dist", "survivor_signal"][:n_inputs]
        while len(names) < n_inputs:
            names.append(f"sensor_{len(names)}")

        return cls(
            n_inputs=n_inputs,
            sensor_names=names,
            sensor_min=np.zeros(n_inputs),
            sensor_max=np.ones(n_inputs),
            dt=0.01,              # 10 ms per step
            rate_max_hz=100.0,    # max firing rate
            default_steps=20,
            analog_scale=1.0,
            seed=seed,
        )


# ----------------------------------------------------------------------
# TEST / DEBUG
# ----------------------------------------------------------------------
if __name__ == "__main__":
    enc = SpikeEncoder.build_rescue_encoder(n_inputs=4, seed=getattr(cfg, "SEED", 42))

    obs = {
        "front_dist": 0.8,
        "left_dist": 0.2,
        "right_dist": 0.6,
        "survivor_signal": 0.9,
    }

    out = enc.encode(obs, n_steps=12, bipolar_analog=False)

    print("normalized      :", out.values)
    print("bipolar         :", out.bipolar_values)
    print("analog_current  :", out.analog_current)
    print("level spikes\n", out.level_spikes)
    print("latency spikes\n", out.latency_spikes)