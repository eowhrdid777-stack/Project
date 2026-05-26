from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ============================================================
# Step 단위 metrics
# ============================================================
@dataclass
class StepMetrics:
    """한 env step / 한 network decision 기록"""

    # fallback 사용 여부
    used_fallback: bool

    # 몇 번째 timestep에서 결정됐는지
    selected_step: int

    # 선택한 action
    action: int

    # spike 개수
    output_spike_count: int
    hidden_spike_count: int

    # spike silence 여부
    output_silent: bool
    hidden_silent: bool

    # pulse 관련
    pulses_plus: int
    pulses_minus: int
    n_refresh: int

    # learning 방향 분석
    updated_pair_count: int
    potentiation_count: int
    depression_count: int

    # winner / target neuron
    winner: int
    target: int

    # reward
    reward: float


# ============================================================
# 전체 summary
# ============================================================
@dataclass
class MetricsSummary:
    """전체 step 평균 통계"""

    # 총 step 수
    num_steps: int

    # fallback 비율
    fallback_rate: float

    # output / hidden silence 비율
    output_silent_rate: float
    hidden_silent_rate: float

    # decision timing 통계
    mean_decision_step: float
    std_decision_step: float

    # 평균 spike 수
    mean_output_spikes: float
    mean_hidden_spikes: float

    # reward 통계
    mean_reward: float
    total_reward: float

    positive_reward_count: int
    negative_reward_count: int
    zero_reward_count: int

    # pulse 통계
    mean_pulses_plus: float
    mean_pulses_minus: float
    mean_refreshes: float

    # learning 통계
    mean_updated_pairs: float
    mean_potentiation_count: float
    mean_depression_count: float

    total_updated_pairs: int
    total_potentiation_count: int
    total_depression_count: int

    # action 선택 분포
    action_histogram: Dict[int, int] = field(default_factory=dict)


# ============================================================
# Metrics collector
# ============================================================
class SNNMetrics:
    """
    SNN 실험용 metrics 수집기

    현재 main.py에서는
    env step마다 add_episode() 호출 중

    실제 의미는 "step metrics"
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """metrics 초기화"""
        self.steps: List[StepMetrics] = []

    @property
    def episodes(self) -> List[StepMetrics]:
        """
        이전 코드 호환용 alias

        실제로는 episode가 아니라
        step 단위 기록임
        """
        return self.steps

    # ========================================================
    # spike 개수 계산
    # ========================================================
    @staticmethod
    def _count_spikes(spikes: Any) -> int:
        if spikes is None:
            return 0

        arr = np.asarray(spikes, dtype=int)

        if arr.size == 0:
            return 0

        return int(arr.sum())

    # ========================================================
    # 안전한 int 변환
    # ========================================================
    @staticmethod
    def _safe_int(value: Any, default: int = -1) -> int:
        if value is None:
            return int(default)

        try:
            return int(value)
        except Exception:
            return int(default)

    # ========================================================
    # 안전한 float 변환
    # ========================================================
    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return float(default)

        try:
            return float(value)
        except Exception:
            return float(default)

    # ========================================================
    # learning 방향 분석
    # ========================================================
    @staticmethod
    def _count_learning_directions(
        learning_event: Optional[Any],
    ) -> tuple[int, int, int]:
        """
        반환값:
            updated_pair_count
            potentiation_count
            depression_count
        """

        if learning_event is None:
            return 0, 0, 0

        updated_pairs = getattr(learning_event, "updated_pairs", [])
        directions = getattr(learning_event, "directions", [])

        try:
            updated_pair_count = len(updated_pairs)
        except Exception:
            updated_pair_count = 0

        potentiation_count = 0
        depression_count = 0

        try:
            for d in directions:
                d_int = int(d)

                if d_int > 0:
                    potentiation_count += 1

                elif d_int < 0:
                    depression_count += 1

        except Exception:
            potentiation_count = 0
            depression_count = 0

        return (
            int(updated_pair_count),
            int(potentiation_count),
            int(depression_count),
        )

    # ========================================================
    # step metrics 추가
    # ========================================================
    def add_step(
        self,
        rollout_info: Dict[str, Any],
        learning_event: Optional[Any] = None,
    ) -> StepMetrics:
        """
        step metrics 저장
        """

        # rollout 정보
        used_fallback = bool(rollout_info.get("used_fallback", False))

        selected_step = self._safe_int(
            rollout_info.get("selected_step", -1),
            -1,
        )

        action = self._safe_int(
            rollout_info.get("action", -1),
            -1,
        )

        # spike 개수 계산
        hidden_spike_count = self._count_spikes(
            rollout_info.get("hidden_spikes")
        )

        output_spike_count = self._count_spikes(
            rollout_info.get("output_spikes")
        )

        # silent 여부
        hidden_silent = hidden_spike_count == 0
        output_silent = output_spike_count == 0

        # learning 없는 경우
        if learning_event is None:
            pulses_plus = 0
            pulses_minus = 0
            n_refresh = 0

            reward = 0.0

            winner = -1
            target = -1

            updated_pair_count = 0
            potentiation_count = 0
            depression_count = 0

        # learning 있는 경우
        else:
            pulses_plus = self._safe_int(
                getattr(learning_event, "n_pulses_plus", 0),
                0,
            )

            pulses_minus = self._safe_int(
                getattr(learning_event, "n_pulses_minus", 0),
                0,
            )

            n_refresh = self._safe_int(
                getattr(learning_event, "n_refresh", 0),
                0,
            )

            reward = self._safe_float(
                getattr(learning_event, "reward", 0.0),
                0.0,
            )

            winner = self._safe_int(
                getattr(learning_event, "winner", -1),
                -1,
            )

            target = self._safe_int(
                getattr(learning_event, "target", -1),
                -1,
            )

            (
                updated_pair_count,
                potentiation_count,
                depression_count,
            ) = self._count_learning_directions(
                learning_event
            )

        # metrics 객체 생성
        step = StepMetrics(
            used_fallback=used_fallback,
            selected_step=selected_step,
            action=action,

            output_spike_count=output_spike_count,
            hidden_spike_count=hidden_spike_count,

            output_silent=output_silent,
            hidden_silent=hidden_silent,

            pulses_plus=pulses_plus,
            pulses_minus=pulses_minus,
            n_refresh=n_refresh,

            updated_pair_count=updated_pair_count,
            potentiation_count=potentiation_count,
            depression_count=depression_count,

            winner=winner,
            target=target,

            reward=reward,
        )

        # 저장
        self.steps.append(step)

        return step

    # ========================================================
    # 이전 코드 호환용
    # ========================================================
    def add_episode(
        self,
        rollout_info: Dict[str, Any],
        learning_event: Optional[Any] = None,
    ) -> StepMetrics:
        """
        이전 코드 호환용 alias
        """
        return self.add_step(
            rollout_info=rollout_info,
            learning_event=learning_event,
        )

    # ========================================================
    # 전체 summary 계산
    # ========================================================
    def summary(self) -> MetricsSummary:

        # 비어있는 경우
        if not self.steps:
            return MetricsSummary(
                num_steps=0,

                fallback_rate=0.0,

                output_silent_rate=0.0,
                hidden_silent_rate=0.0,

                mean_decision_step=0.0,
                std_decision_step=0.0,

                mean_output_spikes=0.0,
                mean_hidden_spikes=0.0,

                mean_reward=0.0,
                total_reward=0.0,

                positive_reward_count=0,
                negative_reward_count=0,
                zero_reward_count=0,

                mean_pulses_plus=0.0,
                mean_pulses_minus=0.0,
                mean_refreshes=0.0,

                mean_updated_pairs=0.0,
                mean_potentiation_count=0.0,
                mean_depression_count=0.0,

                total_updated_pairs=0,
                total_potentiation_count=0,
                total_depression_count=0,

                action_histogram={},
            )

        # 배열 변환
        decision_steps = np.asarray(
            [s.selected_step for s in self.steps],
            dtype=float,
        )

        fallback = np.asarray(
            [s.used_fallback for s in self.steps],
            dtype=float,
        )

        output_spikes = np.asarray(
            [s.output_spike_count for s in self.steps],
            dtype=float,
        )

        hidden_spikes = np.asarray(
            [s.hidden_spike_count for s in self.steps],
            dtype=float,
        )

        output_silent = np.asarray(
            [s.output_silent for s in self.steps],
            dtype=float,
        )

        hidden_silent = np.asarray(
            [s.hidden_silent for s in self.steps],
            dtype=float,
        )

        rewards = np.asarray(
            [s.reward for s in self.steps],
            dtype=float,
        )

        pulses_plus = np.asarray(
            [s.pulses_plus for s in self.steps],
            dtype=float,
        )

        pulses_minus = np.asarray(
            [s.pulses_minus for s in self.steps],
            dtype=float,
        )

        refreshes = np.asarray(
            [s.n_refresh for s in self.steps],
            dtype=float,
        )

        updated_pairs = np.asarray(
            [s.updated_pair_count for s in self.steps],
            dtype=float,
        )

        pot_counts = np.asarray(
            [s.potentiation_count for s in self.steps],
            dtype=float,
        )

        dep_counts = np.asarray(
            [s.depression_count for s in self.steps],
            dtype=float,
        )

        # action 분포 계산
        action_histogram: Dict[int, int] = {}

        for s in self.steps:
            action_histogram[s.action] = (
                action_histogram.get(s.action, 0) + 1
            )

        # reward 부호 통계
        positive_reward_count = int(
            np.sum(rewards > 0.0)
        )

        negative_reward_count = int(
            np.sum(rewards < 0.0)
        )

        zero_reward_count = int(
            np.sum(rewards == 0.0)
        )

        # summary 반환
        return MetricsSummary(
            num_steps=len(self.steps),

            fallback_rate=float(fallback.mean()),

            output_silent_rate=float(output_silent.mean()),
            hidden_silent_rate=float(hidden_silent.mean()),

            mean_decision_step=float(decision_steps.mean()),
            std_decision_step=float(decision_steps.std()),

            mean_output_spikes=float(output_spikes.mean()),
            mean_hidden_spikes=float(hidden_spikes.mean()),

            mean_reward=float(rewards.mean()),
            total_reward=float(rewards.sum()),

            positive_reward_count=positive_reward_count,
            negative_reward_count=negative_reward_count,
            zero_reward_count=zero_reward_count,

            mean_pulses_plus=float(pulses_plus.mean()),
            mean_pulses_minus=float(pulses_minus.mean()),
            mean_refreshes=float(refreshes.mean()),

            mean_updated_pairs=float(updated_pairs.mean()),
            mean_potentiation_count=float(pot_counts.mean()),
            mean_depression_count=float(dep_counts.mean()),

            total_updated_pairs=int(updated_pairs.sum()),
            total_potentiation_count=int(pot_counts.sum()),
            total_depression_count=int(dep_counts.sum()),

            action_histogram=action_histogram,
        )

    # ========================================================
    # dict 형태 반환
    # ========================================================
    def summary_dict(self) -> Dict[str, Any]:

        s = self.summary()

        return {
            "num_steps": s.num_steps,

            "fallback_rate": s.fallback_rate,

            "output_silent_rate": s.output_silent_rate,
            "hidden_silent_rate": s.hidden_silent_rate,

            "mean_decision_step": s.mean_decision_step,
            "std_decision_step": s.std_decision_step,

            "mean_output_spikes": s.mean_output_spikes,
            "mean_hidden_spikes": s.mean_hidden_spikes,

            "mean_reward": s.mean_reward,
            "total_reward": s.total_reward,

            "positive_reward_count": s.positive_reward_count,
            "negative_reward_count": s.negative_reward_count,
            "zero_reward_count": s.zero_reward_count,

            "mean_pulses_plus": s.mean_pulses_plus,
            "mean_pulses_minus": s.mean_pulses_minus,
            "mean_refreshes": s.mean_refreshes,

            "mean_updated_pairs": s.mean_updated_pairs,
            "mean_potentiation_count": s.mean_potentiation_count,
            "mean_depression_count": s.mean_depression_count,

            "total_updated_pairs": s.total_updated_pairs,
            "total_potentiation_count": s.total_potentiation_count,
            "total_depression_count": s.total_depression_count,

            "action_histogram": s.action_histogram,
        }


# ============================================================
# 간단한 test
# ============================================================
if __name__ == "__main__":

    metrics = SNNMetrics()

    rollout_info = {
        "used_fallback": False,
        "selected_step": 2,
        "action": 1,

        "hidden_spikes": [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ],

        "output_spikes": [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
        ],
    }

    class DummyLearning:
        n_pulses_plus = 2
        n_pulses_minus = 1
        n_refresh = 0

        reward = 1.0

        winner = 1
        target = 1

        updated_pairs = [
            (0, 1),
            (1, 1),
            (2, 1),
        ]

        directions = [
            +1,
            +1,
            -1,
        ]

    metrics.add_episode(
        rollout_info,
        DummyLearning(),
    )

    print(metrics.summary_dict())
