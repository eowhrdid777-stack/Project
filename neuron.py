from __future__ import annotations

# dataclass: 여러 값을 한 번에 묶어서 저장하기 위한 도구
from dataclasses import dataclass
# 타입 힌트를 위한 모듈
from typing import List, Optional, Sequence

# 수치 계산용 라이브러리
import numpy as np

# 프로젝트 전체 설정값(config) 불러오기
import config as cfg
# 실제 conductance 업데이트를 수행하는 controller와 그 결과 구조체
from conductance_modulation import ConductanceModulationController, ProgrammingResult
# differential pair 기반 crossbar 구조
from crossbar import DifferentialCrossbar
# threshold 적응용 memristor device
from device_model import MemristorDevice


@dataclass
class NeuronStepResult:
    """한 simulation step에서 neuron layer 상태를 저장하는 구조체."""

    synaptic_currents: np.ndarray         # 각 뉴런에 들어온 synaptic current
    membrane_potentials: np.ndarray       # 각 뉴런의 현재 막전위(vmem)
    thresholds: np.ndarray                # 각 뉴런의 현재 threshold
    spikes: np.ndarray                    # 이번 step에서 spike 발생 여부
    spike_trace: np.ndarray               # 최근 spike 활동 흔적(trace)
    refractory_counters: np.ndarray       # 각 뉴런의 refractory 카운터
    winner: int                           # 최종 winner 뉴런 index (-1이면 없음)


@dataclass
class LearningEvent:
    """실제 synapse programming 결과를 요약해서 저장하는 구조체."""

    updated_pairs: List[tuple[int, int]]  # 업데이트된 (row, col) 목록
    directions: List[int]                 # 각 업데이트 방향 (+1 강화, -1 약화)
    actions: List[str]                    # controller가 실제로 선택한 action 이름
    n_pulses_plus: int                    # plus 쪽 pulse 개수
    n_pulses_minus: int                   # minus 쪽 pulse 개수
    n_refresh: int                        # refresh 수행 횟수
    reward: float                         # 이번 update에 사용된 reward
    winner: int                           # layer가 선택한 winner neuron
    target: Optional[int]                 # 정답/목표 행동(있는 경우)
    message: str                          # 설명 메시지


class MemristiveLIFOutputLayer:
    """Measured-read 기반의 hardware-aware LIF output layer.

    핵심 설계 철학
    --------------
    1. inference 시 ideal weight를 직접 쓰지 않고,
       crossbar의 differential pair를 실제로 읽어서 synaptic current를 계산한다.
    2. learning 시에도 단순한 w += delta가 아니라,
       ConductanceModulationController를 통해 pulse 기반 programming을 수행한다.
    3. threshold adaptation도 숫자를 직접 더하는 대신,
       작은 memristive device 상태로 표현한다.
    """

    def __init__(
        self,
        crossbar: DifferentialCrossbar,
        seed: Optional[int] = None,
        membrane_decay: Optional[float] = None,
        input_gain: Optional[float] = None,
        base_threshold: Optional[float] = None,
        reset_voltage: Optional[float] = None,
        refractory_steps: Optional[int] = None,
        trace_decay: Optional[float] = None,
        inhibit_on_spike: Optional[bool] = None,
        lateral_inhibition_strength: Optional[float] = None,
        enable_threshold_adaptation: Optional[bool] = None,
        threshold_scale: Optional[float] = None,
        threshold_pot_pulses_on_spike: Optional[int] = None,
        threshold_dep_pulses_recovery: Optional[int] = None,
        threshold_recovery_period: Optional[int] = None,
    ) -> None:
        # 이미 생성된 crossbar를 전달받아 저장
        self.crossbar = crossbar
        # 실제 conductance update는 controller가 담당
        self.controller = ConductanceModulationController(crossbar)
        # 재현 가능한 난수 생성기
        self.rng = np.random.default_rng(getattr(cfg, "SEED", 42) if seed is None else seed)

        # 입력 수 = crossbar row 수
        self.n_inputs = int(crossbar.n_rows)
        # 출력 뉴런 수 = logical column 수
        self.n_neurons = int(crossbar.n_logical_cols)

        # -----------------------------
        # LIF neuron 관련 주요 파라미터
        # -----------------------------
        # 이전 막전위를 얼마나 남길지 (leaky 정도)
        self.membrane_decay = float(getattr(cfg, "NEURON_MEMBRANE_DECAY", 0.90 if membrane_decay is None else membrane_decay)) if membrane_decay is None else float(membrane_decay)
        # 입력 전류를 얼마나 크게 반영할지
        self.input_gain = float(getattr(cfg, "NEURON_INPUT_GAIN", 1.0 if input_gain is None else input_gain)) if input_gain is None else float(input_gain)
        # 기본 threshold
        self.base_threshold = float(getattr(cfg, "NEURON_BASE_THRESHOLD", 8.0e-6 if base_threshold is None else base_threshold)) if base_threshold is None else float(base_threshold)
        # spike 후 막전위를 어디로 reset할지
        self.reset_voltage = float(getattr(cfg, "NEURON_RESET_VOLTAGE", 0.0 if reset_voltage is None else reset_voltage)) if reset_voltage is None else float(reset_voltage)
        # spike 후 몇 step 동안 쉬게 할지
        self.refractory_steps = int(getattr(cfg, "NEURON_REFRACTORY_STEPS", 1 if refractory_steps is None else refractory_steps)) if refractory_steps is None else int(refractory_steps)
        # spike trace 감소율
        self.trace_decay = float(getattr(cfg, "NEURON_TRACE_DECAY", 0.85 if trace_decay is None else trace_decay)) if trace_decay is None else float(trace_decay)
        # WTA 사용 여부
        self.inhibit_on_spike = bool(getattr(cfg, "NEURON_ENABLE_WTA", True if inhibit_on_spike is None else inhibit_on_spike)) if inhibit_on_spike is None else bool(inhibit_on_spike)
        # lateral inhibition 강도
        self.lateral_inhibition_strength = float(getattr(cfg, "NEURON_LATERAL_INHIBITION", 0.5 if lateral_inhibition_strength is None else lateral_inhibition_strength)) if lateral_inhibition_strength is None else float(lateral_inhibition_strength)

        # -----------------------------
        # threshold adaptation 관련 파라미터
        # -----------------------------
        # threshold adaptation 사용 여부
        self.enable_threshold_adaptation = bool(getattr(cfg, "NEURON_ENABLE_THRESHOLD_ADAPTATION", True if enable_threshold_adaptation is None else enable_threshold_adaptation)) if enable_threshold_adaptation is None else bool(enable_threshold_adaptation)
        # threshold device 상태를 실제 threshold offset으로 바꾸는 스케일
        self.threshold_scale = float(getattr(cfg, "NEURON_THRESHOLD_SCALE", 2.0e-6 if threshold_scale is None else threshold_scale)) if threshold_scale is None else float(threshold_scale)
        # spike 시 threshold device에 줄 pot pulse 수
        self.threshold_pot_pulses_on_spike = int(getattr(cfg, "NEURON_THRESHOLD_POT_PULSES_ON_SPIKE", 1 if threshold_pot_pulses_on_spike is None else threshold_pot_pulses_on_spike)) if threshold_pot_pulses_on_spike is None else int(threshold_pot_pulses_on_spike)
        # recovery 시 threshold device에 줄 dep pulse 수
        self.threshold_dep_pulses_recovery = int(getattr(cfg, "NEURON_THRESHOLD_DEP_PULSES_RECOVERY", 1 if threshold_dep_pulses_recovery is None else threshold_dep_pulses_recovery)) if threshold_dep_pulses_recovery is None else int(threshold_dep_pulses_recovery)
        # 몇 step마다 threshold recovery를 수행할지
        self.threshold_recovery_period = int(getattr(cfg, "NEURON_THRESHOLD_RECOVERY_PERIOD", 4 if threshold_recovery_period is None else threshold_recovery_period)) if threshold_recovery_period is None else int(threshold_recovery_period)

        # -----------------------------
        # 뉴런 layer의 내부 상태 변수
        # -----------------------------
        # 각 뉴런의 막전위
        self.vmem = np.full(self.n_neurons, self.reset_voltage, dtype=float)
        # 최근 spike 흔적을 저장하는 trace
        self.spike_trace = np.zeros(self.n_neurons, dtype=float)
        # 현재 refractory 상태(남은 step 수)
        self.refractory = np.zeros(self.n_neurons, dtype=int)
        # 마지막 spike 발생 시점 기록
        self.last_spike_step = np.full(self.n_neurons, -10**9, dtype=int)

        # threshold adaptation을 쓸 경우, 뉴런마다 threshold용 memristor device 생성
        self.threshold_devices: List[MemristorDevice] = []
        if self.enable_threshold_adaptation:
            base_seed = getattr(cfg, "SEED", 42) if seed is None else int(seed)
            for j in range(self.n_neurons):
                dev = MemristorDevice(seed=base_seed + 50000 + 97 * j)
                # 중간 상태에서 시작
                dev.reset("mid")
                self.threshold_devices.append(dev)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        """에피소드 시작 전 neuron 내부 상태를 초기화."""
        self.vmem.fill(self.reset_voltage)
        self.spike_trace.fill(0.0)
        self.refractory.fill(0)
        self.last_spike_step.fill(-10**9)
        for dev in self.threshold_devices:
            dev.reset("mid")

    def _measured_vmm(self, x: Sequence[float]) -> np.ndarray:
        """Measured-only synaptic accumulation.

        ideal matrix multiply를 직접 하지 않고,
        crossbar의 differential pair를 실제로 읽어서
        synaptic current를 계산한다.
        """
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        if x_arr.size != self.n_inputs:
            raise ValueError(f"Expected input length {self.n_inputs}, got {x_arr.size}")

        # 출력 뉴런별 synaptic current 저장용 배열
        out = np.zeros(self.n_neurons, dtype=float)
        for j in range(self.n_neurons):
            acc = 0.0
            for i in range(self.n_inputs):
                # (i, j) 위치의 differential pair conductance 읽기
                gp, gm = self.crossbar.read_pair((i, j))
                # differential weight = gp - gm
                acc += float(x_arr[i]) * (float(gp) - float(gm))
            out[j] = acc
        return out

    def _threshold_offsets(self) -> np.ndarray:
        """threshold device 상태를 threshold offset으로 변환."""
        if not self.enable_threshold_adaptation:
            return np.zeros(self.n_neurons, dtype=float)

        offsets = np.zeros(self.n_neurons, dtype=float)
        for j, dev in enumerate(self.threshold_devices):
            # conductance를 0~1 범위로 정규화해서 offset 계산
            span = max(dev.g_max_eff - dev.g_min_eff, 1e-18)
            norm = (float(dev.g) - dev.g_min_eff) / span
            offsets[j] = self.threshold_scale * norm
        return offsets

    def get_thresholds(self) -> np.ndarray:
        """현재 각 뉴런의 실제 threshold = base + adaptation offset."""
        return self.base_threshold + self._threshold_offsets()

    def _recover_threshold_devices(self, step_idx: int) -> None:
        """일정 주기마다 threshold device를 조금씩 회복시켜 threshold를 낮춤."""
        if not self.enable_threshold_adaptation:
            return
        if self.threshold_dep_pulses_recovery <= 0:
            return
        if step_idx % max(1, self.threshold_recovery_period) != 0:
            return

        for j, dev in enumerate(self.threshold_devices):
            # 현재 refractory 중인 뉴런은 건너뜀
            if self.refractory[j] > 0:
                continue
            # level이 0보다 크면 dep pulse를 줘서 threshold를 천천히 회복
            if dev.state.level_idx > 0:
                dev.apply_dep_pulse(self.threshold_dep_pulses_recovery)

    # ------------------------------------------------------------------
    # Inference dynamics
    # ------------------------------------------------------------------
    def step(self, pre_spikes: Sequence[float], step_idx: int) -> NeuronStepResult:
        """한 simulation step에서 neuron layer를 실행."""
        # 입력 spike로부터 synaptic current 계산
        syn = self._measured_vmm(pre_spikes)
        # 현재 threshold 읽기
        thresholds = self.get_thresholds()

        # spike trace는 시간이 지나면 감소
        self.spike_trace *= self.trace_decay

        # 이번 step의 spike 여부를 저장할 배열
        spikes = np.zeros(self.n_neurons, dtype=np.int8)

        # 각 뉴런을 순회하며 막전위 업데이트
        for j in range(self.n_neurons):
            # refractory 상태면 이번 step에서는 계산하지 않고 쉼
            if self.refractory[j] > 0:
                self.refractory[j] -= 1
                self.vmem[j] = self.reset_voltage
                continue

            # 핵심 LIF 공식:
            # 새 막전위 = 이전 막전위 일부 + 현재 입력 전류
            self.vmem[j] = self.membrane_decay * self.vmem[j] + self.input_gain * syn[j]

            # threshold를 넘으면 spike 후보
            if self.vmem[j] >= thresholds[j]:
                spikes[j] = 1

        # Winner-Take-All (WTA) 처리
        # spike한 뉴런이 여러 개면 가장 막전위가 큰 뉴런 하나만 남김
        winner = -1
        spiking_idx = np.flatnonzero(spikes > 0)
        if spiking_idx.size > 0:
            if self.inhibit_on_spike:
                winner = int(spiking_idx[np.argmax(self.vmem[spiking_idx])])
                spikes[:] = 0
                spikes[winner] = 1
            else:
                winner = int(spiking_idx[np.argmax(self.vmem[spiking_idx])])

        if winner >= 0:
            # winner의 spike 흔적 증가
            self.spike_trace[winner] += 1.0
            # 일정 step 동안 다시 spike 못 하도록 refractory 설정
            self.refractory[winner] = self.refractory_steps
            # 마지막 spike 시점 기록
            self.last_spike_step[winner] = int(step_idx)

            # spike한 뉴런은 막전위 reset
            self.vmem[winner] = self.reset_voltage

            # winner 외 경쟁 뉴런들에 lateral inhibition 적용
            if self.inhibit_on_spike and self.lateral_inhibition_strength > 0.0:
                mask = np.ones(self.n_neurons, dtype=bool)
                mask[winner] = False
                self.vmem[mask] -= self.lateral_inhibition_strength * np.maximum(syn[mask], 0.0)
                # reset_voltage 아래로 내려가지 않게 clip
                self.vmem = np.maximum(self.vmem, self.reset_voltage)

            # threshold adaptation 사용 시, winner 뉴런의 threshold device에 pot pulse 부여
            if self.enable_threshold_adaptation:
                self.threshold_devices[winner].apply_pot_pulse(self.threshold_pot_pulses_on_spike)

        # threshold recovery 주기적으로 수행
        self._recover_threshold_devices(step_idx)

        # 이번 step 결과를 구조체로 반환
        return NeuronStepResult(
            synaptic_currents=syn.copy(),
            membrane_potentials=self.vmem.copy(),
            thresholds=self.get_thresholds(),
            spikes=spikes,
            spike_trace=self.spike_trace.copy(),
            refractory_counters=self.refractory.copy(),
            winner=winner,
        )

    # ------------------------------------------------------------------
    # Learning / programming
    # ------------------------------------------------------------------
    def apply_reward_modulated_update(
        self,
        pre_spikes: Sequence[float],
        winner: int,
        reward: float,
        step_idx: int,
        target: Optional[int] = None,
        update_all_active_to_target: bool = True,
        punish_wrong_winner: bool = True,
    ) -> LearningEvent:
        """Reward를 바탕으로 실제 pulse 기반 synapse update 수행.

        핵심 개념
        ---------
        - active presynaptic row만 update 후보가 된다.
        - reward 부호로 강화(+1)/약화(-1) 방향을 정한다.
        - target이 있으면 target column을 강화할 수도 있고,
          winner가 틀렸다면 wrong winner를 약화할 수도 있다.
        - 실제 conductance 변화는 controller가 수행한다.
        """
        pre = np.asarray(pre_spikes, dtype=float).reshape(-1)
        if pre.size != self.n_inputs:
            raise ValueError(f"Expected input length {self.n_inputs}, got {pre.size}")

        # 현재 실제로 활성화된 input(row)만 선택
        active_rows = [int(i) for i, x in enumerate(pre) if x > 0.0]
        if not active_rows:
            return LearningEvent([], [], [], 0, 0, 0, float(reward), int(winner), target, "No active presynaptic rows.")

        # 최종적으로 update할 목록: (row, col, direction)
        updates: List[tuple[int, int, int]] = []

        # target이 주어졌다면 유효한 범위인지 검사
        if target is not None and not (0 <= int(target) < self.n_neurons):
            raise ValueError(f"target must be in [0, {self.n_neurons - 1}] or None")

        # reward 부호만 사용해서 강화/약화 방향 결정
        sign = +1 if reward >= 0.0 else -1

        # target이 있고, 활성 row 전체를 target column으로 보내고 싶다면
        if target is not None and update_all_active_to_target:
            for i in active_rows:
                updates.append((i, int(target), sign))

        # target이 없거나 target 방식 안 쓸 경우, winner column을 update
        elif winner >= 0:
            for i in active_rows:
                updates.append((i, int(winner), sign))

        # target이 있는데 winner가 틀렸다면, wrong winner를 약화시킬 수 있음
        if target is not None and punish_wrong_winner and winner >= 0 and int(winner) != int(target):
            for i in active_rows:
                updates.append((i, int(winner), -1))

        if not updates:
            return LearningEvent([], [], [], 0, 0, 0, float(reward), int(winner), target, "No eligible synaptic updates.")

        # 실제 programming 결과를 기록할 변수들
        updated_pairs: List[tuple[int, int]] = []
        directions: List[int] = []
        actions: List[str] = []
        n_pulses_plus = 0
        n_pulses_minus = 0
        n_refresh = 0

        # controller를 호출해서 실제 conductance update 수행
        for row, col, direction in updates:
            result: ProgrammingResult = self.controller.update_weight((row, col), int(direction), int(step_idx))
            updated_pairs.append((int(row), int(col)))
            directions.append(int(direction))
            actions.append(str(result.chosen_action))
            n_pulses_plus += int(result.n_pulses_plus)
            n_pulses_minus += int(result.n_pulses_minus)
            if result.did_refresh:
                n_refresh += 1

        # 이번 learning event를 구조체로 반환
        return LearningEvent(
            updated_pairs=updated_pairs,
            directions=directions,
            actions=actions,
            n_pulses_plus=n_pulses_plus,
            n_pulses_minus=n_pulses_minus,
            n_refresh=n_refresh,
            reward=float(reward),
            winner=int(winner),
            target=None if target is None else int(target),
            message="Measured pulse-based synaptic update executed.",
        )
