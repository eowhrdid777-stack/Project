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
        self.rng = np.random.default_rng(seed) # random number generator for variation sampling

        self.g_min_nom = float(cfg.G_MIN) # import config as cfg이므로 cfg.G_MIN을 통해 config.py에서 G_MIN 값을 가져옴
        self.g_max_nom = float(cfg.G_MAX) # nom은 nominal의 약자로, 소자의 이론적 최소/최대 conductance 값. 실제로는 d2d variation에 의해 g_min_eff/g_max_eff로 변동될 수 있음.
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

        # 소자별로 곱셈 형태로 적용되는 conductance window(Gmin ~ Gmax) 변동
        d2d_scale = 1.0
        if self.enable_d2d and self.cv_d2d > 0.0:
            d2d_scale = max(0.5, float(self.rng.normal(1.0, self.cv_d2d))) 
            # 표준정규분포에서 평균이 1.0이고 표준편차가 cv_d2d인 값을 샘플링하여 d2d_scale로 사용. 최소값은 0.5로 제한 (낮은 확률로 너무 낮은 값이 나오지 않도록 제한).

        # nom 에서 eff로 변환, min_eff가 max_eff보다 크거나 같아지는 경우를 방지하기 위해 max_eff는 min_eff보다 약간 크게 설정
        self.g_min_eff = float(self.g_min_nom * d2d_scale)
        self.g_max_eff = float(self.g_max_nom * d2d_scale)
        if self.g_max_eff <= self.g_min_eff:
            self.g_max_eff = self.g_min_eff * 1.01

        # rcp는 retention convergence point의 약자로, retention이 활성화된 경우 시간이 충분히 지난 후 소자가 수렴하는 conductance 값
        self.g_rcp = float(np.clip(cfg.G_RCP * d2d_scale, self.g_min_eff, self.g_max_eff))

        # conductance table을 저장
        self.pot_curve = self._build_curve(direction="pot")
        self.dep_curve = self._build_curve(direction="dep")

        # 소자의 실제 상태를 저장
        self.state = DeviceState(g=float(self._level_to_g(0, direction="pot")), level_idx=0)
        
        # 초기화 시 reset 모드에 따라 상태를 설정
        self.reset(self.g_init_mode)

    # ------------------------------------------------------------------
    # Curve construction
    # ------------------------------------------------------------------
    def _build_curve(self, direction: PulsePolarity) -> np.ndarray:
        Pmax = self.n_levels - 1
        P = np.arange(self.n_levels, dtype=float)

        x = P / Pmax      # 0 ~ 1 normalized P
        x_max = 1.0       # normalized Pmax

        if direction == "pot":
            Ap = float(self.a_pot)
            B = (self.g_max_eff - self.g_min_eff) / (1.0 - np.exp(-x_max / Ap))
            g = B * (1.0 - np.exp(-x / Ap)) + self.g_min_eff

        else:  # dep
            Ad = float(self.a_dep)
            B = (self.g_max_eff - self.g_min_eff) / (1.0 - np.exp(-x_max / Ad))
            g = B * (1.0 - np.exp(-x / Ad)) + self.g_min_eff
        return np.asarray(g, dtype=float)

    def _clip_g(self, g: float) -> float:
        return float(np.clip(g, self.g_min_eff, self.g_max_eff))

    def _nearest_level_idx(self, g: float) -> int:
        ref_curve = 0.5 * (self.pot_curve + self.dep_curve)
        idx = int(np.argmin(np.abs(ref_curve - g)))
        return max(0, min(self.n_levels - 1, idx))

    def _level_to_g(self, level_idx: int, direction: PulsePolarity = "pot") -> float:
        level_idx = max(0, min(self.n_levels - 1, int(level_idx)))
        curve = self.pot_curve if direction == "pot" else self.dep_curve
        return float(curve[level_idx])

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
        self.state.g = float(self.dep_curve[self.state.level_idx])

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
        self.state.g = float(0.5 * (self.pot_curve[idx] + self.dep_curve[idx]))

    @property
    def g(self) -> float:
        return float(self.state.g)

    def set_g(self, g: float) -> None:
        g = self._clip_g(float(g))
        idx = self._nearest_level_idx(g)
        self.state.level_idx = idx
        ref_curve = 0.5 * (self.pot_curve + self.dep_curve)
        self.state.g = float(ref_curve[idx])

    def snapshot(self) -> DeviceState:
        return DeviceState(g=float(self.state.g), level_idx=int(self.state.level_idx))
    
import numpy as np
import matplotlib.pyplot as plt

from device_model import MemristorDevice


# --------------------------------------------------
# 1) Potentiation + Depression in one curve
#    0~63 : potentiation
#    64~127 : depression
# --------------------------------------------------
def plot_pot_dep_single_cycle(seed=0, n_pulses_per_step=1):
    dev = MemristorDevice(seed=seed)
    dev.reset("min")

    x_trace = [0]
    g_trace = [dev.g * 1e6]   # uS

    # Potentiation: 0 -> 63
    for step in range(dev.n_levels - 1):
        dev.apply_pot_pulse(n_pulses=n_pulses_per_step)
        x_trace.append(step + 1)
        g_trace.append(dev.g * 1e6)

    # Depression: 64 -> 127
    for step in range(dev.n_levels - 1):
        dev.apply_dep_pulse(n_pulses=n_pulses_per_step)
        x_trace.append(dev.n_levels + step)
        g_trace.append(dev.g * 1e6)

    plt.figure(figsize=(7, 4))
    plt.plot(x_trace, g_trace, marker='o', markersize=3, linewidth=1.5)
    plt.axvline(dev.n_levels - 1, linestyle='--', alpha=0.7, label='Pot -> Dep boundary')
    plt.xlabel('Pulse step')
    plt.ylabel('Conductance (μS)')
    plt.title('Potentiation (0~63) + Depression (64~127)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 2) D2D variation
#    여러 seed = 여러 device
#    각 device에 대해 pot -> dep 한 cycle을 한 그래프에
# --------------------------------------------------
def plot_d2d_variation(num_devices=20, n_pulses_per_step=1):
    plt.figure(figsize=(7, 4))

    for seed in range(num_devices):
        dev = MemristorDevice(seed=seed)
        dev.reset("min")

        x_trace = [0]
        g_trace = [dev.g * 1e6]

        # Potentiation
        for step in range(dev.n_levels - 1):
            dev.apply_pot_pulse(n_pulses=n_pulses_per_step)
            x_trace.append(step + 1)
            g_trace.append(dev.g * 1e6)

        # Depression
        for step in range(dev.n_levels - 1):
            dev.apply_dep_pulse(n_pulses=n_pulses_per_step)
            x_trace.append(dev.n_levels + step)
            g_trace.append(dev.g * 1e6)

        plt.plot(x_trace, g_trace, alpha=0.6)

    plt.axvline(dev.n_levels - 1, linestyle='--', alpha=0.7)
    plt.xlabel('Pulse step')
    plt.ylabel('Conductance (μS)')
    plt.title(f'D2D variation across {num_devices} devices')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 3) C2C variation
#    같은 소자(seed 고정)에서 여러 번 반복
#    pot -> dep 한 cycle을 반복해서 분산 확인
# --------------------------------------------------
def plot_c2c_variation(num_cycles=20, seed=0, n_pulses_per_step=1):
    plt.figure(figsize=(7, 4))

    for trial in range(num_cycles):
        dev = MemristorDevice(seed=seed)   # 같은 소자
        dev.reset("min")

        x_trace = [0]
        g_trace = [dev.g * 1e6]

        # Potentiation
        for step in range(dev.n_levels - 1):
            dev.apply_pot_pulse(n_pulses=n_pulses_per_step)
            x_trace.append(step + 1)
            g_trace.append(dev.g * 1e6)

        # Depression
        for step in range(dev.n_levels - 1):
            dev.apply_dep_pulse(n_pulses=n_pulses_per_step)
            x_trace.append(dev.n_levels + step)
            g_trace.append(dev.g * 1e6)

        plt.plot(x_trace, g_trace, alpha=0.6)

    plt.axvline(dev.n_levels - 1, linestyle='--', alpha=0.7)
    plt.xlabel('Pulse step')
    plt.ylabel('Conductance (μS)')
    plt.title(f'C2C variation over {num_cycles} repeated cycles (same device)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# 4) mean ± std plot for C2C (optional)
#    같은 소자 반복 결과를 평균/표준편차로 보기
# --------------------------------------------------
def plot_c2c_mean_std(num_cycles=50, seed=0, n_pulses_per_step=1):
    all_traces = []

    for trial in range(num_cycles):
        dev = MemristorDevice(seed=seed)
        dev.reset("min")

        g_trace = [dev.g * 1e6]

        for _ in range(dev.n_levels - 1):
            dev.apply_pot_pulse(n_pulses=n_pulses_per_step)
            g_trace.append(dev.g * 1e6)

        for _ in range(dev.n_levels - 1):
            dev.apply_dep_pulse(n_pulses=n_pulses_per_step)
            g_trace.append(dev.g * 1e6)

        all_traces.append(g_trace)

    all_traces = np.array(all_traces)
    mean_trace = np.mean(all_traces, axis=0)
    std_trace = np.std(all_traces, axis=0)

    x = np.arange(all_traces.shape[1])

    plt.figure(figsize=(7, 4))
    plt.plot(x, mean_trace, linewidth=2, label='Mean')
    plt.fill_between(x, mean_trace - std_trace, mean_trace + std_trace, alpha=0.3, label='±1σ')
    plt.axvline(dev.n_levels - 1, linestyle='--', alpha=0.7)
    plt.xlabel('Pulse step')
    plt.ylabel('Conductance (μS)')
    plt.title(f'C2C mean ± std over {num_cycles} repeated cycles')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. 단일 소자의 pot+dep shape 확인
    plot_pot_dep_single_cycle(seed=0, n_pulses_per_step=1)

    # 2. D2D 확인
    plot_d2d_variation(num_devices=20, n_pulses_per_step=1)

    # 3. C2C 확인
    plot_c2c_variation(num_cycles=20, seed=0, n_pulses_per_step=1)

    # 4. C2C 평균±표준편차
    plot_c2c_mean_std(num_cycles=50, seed=0, n_pulses_per_step=1)

