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
    selected_by_spike: bool
    fallback_action: int
    fallback_scores: List[float]
    output_scores: List[float]

    # 몇 번째 timestep에서 결정됐는지
    selected_step: int
    first_spike_step: List[int]

    # 선택한 action
    action: int

    # spike 개수
    output_spike_count: int
    hidden_spike_count: int
    input_hidden_current_abs_mean: float
    recurrent_hidden_current_abs_mean: float
    recurrent_to_input_current_ratio: float
    input_hidden_current_norm: float
    recurrent_hidden_current_norm: float
    total_hidden_current_norm: float
    prev_hidden_spike_count: int
    hidden_trace_mean: float
    hidden_trace_nonzero_count: int
    output_spike_histogram: List[int]
    output_winners: List[int]
    output_threshold_mean: List[float]
    output_current_mean: List[float]

    # spike silence 여부
    output_silent: bool
    hidden_silent: bool
    no_action_due_to_no_spike: bool

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

<<<<<<< HEAD
    # env.step(action).info diagnostics
    found_victim: bool
    collision: bool
    moved: bool
    distance_to_nearest_victim: float
    rescued_count: int
    remaining_victims: int
    snn_action: int
    executed_action: int
    delayed_credit_update_count: int
    delayed_credit_action_histogram: Dict[int, int] = field(default_factory=dict)
    direct_credit_action_histogram: Dict[int, int] = field(default_factory=dict)
    previous_turn_credit_action_histogram: Dict[int, int] = field(default_factory=dict)
    learning_reward: float = 0.0
    learning_reason: str = "none"
    turn_sensor_reward_applied: bool = False
    turn_sensor_penalty_applied: bool = False
    pre_step_front_clearance: float = 1.0
    is_wall_avoid_phase: bool = False
    wall_avoid_forward_collision: bool = False
    wall_avoid_forward_after_clearance: bool = False
    wall_avoid_forward_positive_applied: bool = False
    wall_avoid_forward_positive_skipped: bool = False
    wall_avoid_turn_reward_applied: bool = False
    wall_avoid_repeated_turn_spin: bool = False
    env_info: Dict[str, Any] = field(default_factory=dict)
=======
    collision: bool
    moved: bool
    danger_zone: bool
    found_victim: bool
>>>>>>> 791ac2528253af560762704ec2a507d4af97e674


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
    mean_abs_input_hidden_current: float
    mean_abs_recurrent_hidden_current: float
    mean_recurrent_to_input_current_ratio: float
    median_recurrent_to_input_current_ratio: float
    max_recurrent_to_input_current_ratio: float
    mean_input_hidden_current_norm: float
    mean_recurrent_hidden_current_norm: float
    mean_total_hidden_current_norm: float
    mean_prev_hidden_spike_count: float
    mean_hidden_spike_count: float
    hidden_trace_mean: float
    hidden_trace_nonzero_count: float

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
<<<<<<< HEAD
    # env / action diagnostics
    found_victim_count: int
    collision_count: int
    moved_count: int
    mean_distance_to_victim: float
    min_distance_to_victim: float
    final_distance_to_victim: float
    final_rescued_count: int
    final_remaining_victims: int

=======
    collision_rate: float
    moved_rate: float
    danger_zone_rate: float
    found_victim_count: int
>>>>>>> 791ac2528253af560762704ec2a507d4af97e674
    action_histogram: Dict[int, int] = field(default_factory=dict)
    snn_action_histogram: Dict[int, int] = field(default_factory=dict)
    executed_action_histogram: Dict[int, int] = field(default_factory=dict)
    output_neuron_spike_histogram: Dict[int, int] = field(default_factory=dict)
    output_winner_histogram: Dict[int, int] = field(default_factory=dict)
    spike_selected_action_histogram: Dict[int, int] = field(default_factory=dict)
    fallback_action_histogram: Dict[int, int] = field(default_factory=dict)
    fallback_count_by_action: Dict[int, int] = field(default_factory=dict)
    fallback_when_front_blocked_count: int = 0
    fallback_when_front_blocked_action_histogram: Dict[int, int] = field(default_factory=dict)
    no_output_spike_count: int = 0
    output_threshold_mean_per_action: Dict[int, float] = field(default_factory=dict)
    output_current_mean_per_action: Dict[int, float] = field(default_factory=dict)
    action_mean_reward: Dict[int, float] = field(default_factory=dict)
    action_event_counts: Dict[int, Dict[str, int]] = field(default_factory=dict)
    delayed_credit_update_count: int = 0
    delayed_credit_action_histogram: Dict[int, int] = field(default_factory=dict)
    direct_credit_action_histogram: Dict[int, int] = field(default_factory=dict)
    previous_turn_credit_action_histogram: Dict[int, int] = field(default_factory=dict)
    learning_update_action_histogram: Dict[int, int] = field(default_factory=dict)
    potentiation_action_histogram: Dict[int, int] = field(default_factory=dict)
    depression_action_histogram: Dict[int, int] = field(default_factory=dict)
    turn_sensor_reward_count: int = 0
    turn_sensor_reward_action_histogram: Dict[int, int] = field(default_factory=dict)
    turn_sensor_penalty_count: int = 0
    turn_sensor_penalty_action_histogram: Dict[int, int] = field(default_factory=dict)
    learning_reason_histogram: Dict[str, int] = field(default_factory=dict)
    wall_avoid_forward_collision_count: int = 0
    wall_avoid_forward_after_clearance_count: int = 0
    wall_avoid_repeated_turn_spin_count: int = 0
    wall_avoid_forward_positive_count: int = 0
    wall_avoid_forward_positive_skipped_count: int = 0
    wall_avoid_turn_reward_count: int = 0
    wall_avoid_action0_selected_when_front_blocked_count: int = 0
    wall_avoid_forward_recovery_action_histogram: Dict[int, int] = field(default_factory=dict)
    wall_avoid_spin_penalty_action_histogram: Dict[int, int] = field(default_factory=dict)
    selected_step_action_histogram: Dict[int, Dict[int, int]] = field(default_factory=dict)
    input_hidden_current_abs_mean_by_action: Dict[int, float] = field(default_factory=dict)
    recurrent_hidden_current_abs_mean_by_action: Dict[int, float] = field(default_factory=dict)
    recurrent_to_input_current_ratio_by_action: Dict[int, float] = field(default_factory=dict)


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

    @staticmethod
    def _spike_histogram(spikes: Any) -> List[int]:
        if spikes is None:
            return []

        arr = np.asarray(spikes, dtype=int)
        if arr.size == 0:
            return []

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        else:
            arr = arr.reshape(-1, arr.shape[-1])

        return [int(v) for v in arr.sum(axis=0).tolist()]

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

    @classmethod
    def _safe_float_list(cls, value: Any) -> List[float]:
        if value is None:
            return []
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
        except Exception:
            return []
        return [float(v) for v in arr.tolist()]

    @classmethod
    def _safe_int_list(cls, value: Any, default: int = -1) -> List[int]:
        if value is None:
            return []
        out: List[int] = []
        try:
            values = list(value)
        except Exception:
            values = [value]
        for item in values:
            out.append(cls._safe_int(item, default))
        return out

    @staticmethod
    def _safe_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return bool(default)

        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False

        try:
            return bool(value)
        except Exception:
            return bool(default)

    @classmethod
    def _safe_int_histogram(cls, value: Any) -> Dict[int, int]:
        if not isinstance(value, dict):
            return {}

        out: Dict[int, int] = {}
        for key, count in value.items():
            action = cls._safe_int(key, -1)
            if action < 0:
                continue
            out[action] = out.get(action, 0) + cls._safe_int(count, 0)
        return out

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
        selected_by_spike = self._safe_bool(
            rollout_info.get("selected_by_spike", not used_fallback),
            not used_fallback,
        )
        fallback_action = self._safe_int(
            rollout_info.get("fallback_action", -1),
            -1,
        )
        fallback_scores = self._safe_float_list(
            rollout_info.get("fallback_scores", [])
        )
        output_scores = self._safe_float_list(
            rollout_info.get("output_scores", [])
        )

        selected_step = self._safe_int(
            rollout_info.get("selected_step", -1),
            -1,
        )
        first_spike_step = self._safe_int_list(
            rollout_info.get("first_spike_step", []),
            -1,
        )

        action = self._safe_int(
            rollout_info.get("action", -1),
            -1,
        )

        env_info_raw = rollout_info.get("env_info", {})
        env_info = dict(env_info_raw) if isinstance(env_info_raw, dict) else {}

        def info_value(name: str, default: Any) -> Any:
            return env_info.get(name, rollout_info.get(name, default))

        found_victim = self._safe_bool(info_value("found_victim", False), False)
        collision = self._safe_bool(info_value("collision", False), False)
        moved = self._safe_bool(info_value("moved", False), False)
        distance_to_nearest_victim = self._safe_float(
            info_value("distance_to_nearest_victim", 0.0),
            0.0,
        )
        rescued_count = self._safe_int(info_value("rescued_count", 0), 0)
        remaining_victims = self._safe_int(info_value("remaining_victims", 0), 0)
        snn_action = self._safe_int(info_value("snn_action", action), action)
        executed_action = self._safe_int(info_value("executed_action", action), action)
        delayed_credit_update_count = self._safe_int(
            info_value("delayed_credit_update_count", 0),
            0,
        )
        delayed_credit_action_histogram = self._safe_int_histogram(
            info_value("delayed_credit_action_histogram", {})
        )
        direct_credit_action_histogram = self._safe_int_histogram(
            info_value("direct_credit_action_histogram", {})
        )
        previous_turn_credit_action_histogram = self._safe_int_histogram(
            info_value("previous_turn_credit_action_histogram", {})
        )
        learning_reward = self._safe_float(info_value("learning_reward", 0.0), 0.0)
        learning_reason = str(info_value("learning_reason", "none"))
        turn_sensor_reward_applied = self._safe_bool(
            info_value("turn_sensor_reward_applied", False),
            False,
        )
        turn_sensor_penalty_applied = self._safe_bool(
            info_value("turn_sensor_penalty_applied", False),
            False,
        )
        pre_step_front_clearance = self._safe_float(
            info_value("pre_step_front_clearance", 1.0),
            1.0,
        )
        is_wall_avoid_phase = self._safe_bool(info_value("is_wall_avoid_phase", False), False)
        wall_avoid_forward_collision = self._safe_bool(
            info_value("wall_avoid_forward_collision", False),
            False,
        )
        wall_avoid_forward_after_clearance = self._safe_bool(
            info_value("wall_avoid_forward_after_clearance", False),
            False,
        )
        wall_avoid_forward_positive_applied = self._safe_bool(
            info_value("wall_avoid_forward_positive_applied", False),
            False,
        )
        wall_avoid_forward_positive_skipped = self._safe_bool(
            info_value("wall_avoid_forward_positive_skipped", False),
            False,
        )
        wall_avoid_turn_reward_applied = self._safe_bool(
            info_value("wall_avoid_turn_reward_applied", False),
            False,
        )
        wall_avoid_repeated_turn_spin = self._safe_bool(
            info_value("wall_avoid_repeated_turn_spin", False),
            False,
        )

        # spike 개수 계산
        hidden_spike_count = self._count_spikes(
            rollout_info.get("hidden_spikes")
        )
        input_hidden_current_abs_mean = self._safe_float(
            rollout_info.get("input_hidden_current_abs_mean", 0.0),
            0.0,
        )
        recurrent_hidden_current_abs_mean = self._safe_float(
            rollout_info.get("recurrent_hidden_current_abs_mean", 0.0),
            0.0,
        )
        recurrent_to_input_current_ratio = self._safe_float(
            rollout_info.get("recurrent_to_input_current_ratio", 0.0),
            0.0,
        )
        input_hidden_current_norm = self._safe_float(
            rollout_info.get("input_hidden_current_norm", 0.0),
            0.0,
        )
        recurrent_hidden_current_norm = self._safe_float(
            rollout_info.get("recurrent_hidden_current_norm", 0.0),
            0.0,
        )
        total_hidden_current_norm = self._safe_float(
            rollout_info.get("total_hidden_current_norm", 0.0),
            0.0,
        )
        prev_hidden_spike_count = self._safe_int(
            rollout_info.get("prev_hidden_spike_count", 0),
            0,
        )
        hidden_trace_mean = self._safe_float(
            rollout_info.get("hidden_trace_mean", 0.0),
            0.0,
        )
        hidden_trace_nonzero_count = self._safe_int(
            rollout_info.get("hidden_trace_nonzero_count", 0),
            0,
        )

        output_spike_count = self._count_spikes(
            rollout_info.get("output_spikes")
        )
        output_spike_histogram = self._spike_histogram(
            rollout_info.get("output_spikes")
        )
        output_winners = [
            self._safe_int(w, -1)
            for w in rollout_info.get("output_winners", [])
            if self._safe_int(w, -1) >= 0
        ]
        output_threshold_mean = self._safe_float_list(
            rollout_info.get("output_threshold_mean", [])
        )
        output_current_mean = self._safe_float_list(
            rollout_info.get("output_current_mean", [])
        )

        # silent 여부
        hidden_silent = hidden_spike_count == 0
        output_silent = output_spike_count == 0
        no_action_due_to_no_spike = self._safe_bool(
            rollout_info.get("no_action_due_to_no_spike", False),
            False,
        )

        env_info = rollout_info.get("env_info", {})

        collision = bool(
            rollout_info.get("collision", env_info.get("collision", False))
        )

        moved = bool(
            rollout_info.get("moved", env_info.get("moved", False))
        )

        danger_zone = bool(
            rollout_info.get("danger_zone", env_info.get("danger_zone", False))
        )

        found_victim = bool(
            rollout_info.get("found_victim", env_info.get("found_victim", False))
        )

        # learning 없는 경우
        if learning_event is None:
            pulses_plus = 0
            pulses_minus = 0
            n_refresh = 0

<<<<<<< HEAD
            reward = self._safe_float(rollout_info.get("reward", 0.0), 0.0)
=======
            reward = self._safe_float(
                rollout_info.get("reward", 0.0),
                0.0,
            )
>>>>>>> 791ac2528253af560762704ec2a507d4af97e674

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

            reward = self._safe_float(rollout_info.get("reward", 0.0), 0.0)

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
            selected_by_spike=selected_by_spike,
            fallback_action=fallback_action,
            fallback_scores=fallback_scores,
            output_scores=output_scores,
            selected_step=selected_step,
            first_spike_step=first_spike_step,
            action=action,

            output_spike_count=output_spike_count,
            hidden_spike_count=hidden_spike_count,
            input_hidden_current_abs_mean=input_hidden_current_abs_mean,
            recurrent_hidden_current_abs_mean=recurrent_hidden_current_abs_mean,
            recurrent_to_input_current_ratio=recurrent_to_input_current_ratio,
            input_hidden_current_norm=input_hidden_current_norm,
            recurrent_hidden_current_norm=recurrent_hidden_current_norm,
            total_hidden_current_norm=total_hidden_current_norm,
            prev_hidden_spike_count=prev_hidden_spike_count,
            hidden_trace_mean=hidden_trace_mean,
            hidden_trace_nonzero_count=hidden_trace_nonzero_count,
            output_spike_histogram=output_spike_histogram,
            output_winners=output_winners,
            output_threshold_mean=output_threshold_mean,
            output_current_mean=output_current_mean,

            output_silent=output_silent,
            hidden_silent=hidden_silent,
            no_action_due_to_no_spike=no_action_due_to_no_spike,

            pulses_plus=pulses_plus,
            pulses_minus=pulses_minus,
            n_refresh=n_refresh,

            updated_pair_count=updated_pair_count,
            potentiation_count=potentiation_count,
            depression_count=depression_count,

            winner=winner,
            target=target,

            reward=reward,
<<<<<<< HEAD
            found_victim=found_victim,
            collision=collision,
            moved=moved,
            distance_to_nearest_victim=distance_to_nearest_victim,
            rescued_count=rescued_count,
            remaining_victims=remaining_victims,
            snn_action=snn_action,
            executed_action=executed_action,
            delayed_credit_update_count=delayed_credit_update_count,
            delayed_credit_action_histogram=delayed_credit_action_histogram,
            direct_credit_action_histogram=direct_credit_action_histogram,
            previous_turn_credit_action_histogram=previous_turn_credit_action_histogram,
            learning_reward=learning_reward,
            learning_reason=learning_reason,
            turn_sensor_reward_applied=turn_sensor_reward_applied,
            turn_sensor_penalty_applied=turn_sensor_penalty_applied,
            pre_step_front_clearance=pre_step_front_clearance,
            is_wall_avoid_phase=is_wall_avoid_phase,
            wall_avoid_forward_collision=wall_avoid_forward_collision,
            wall_avoid_forward_after_clearance=wall_avoid_forward_after_clearance,
            wall_avoid_forward_positive_applied=wall_avoid_forward_positive_applied,
            wall_avoid_forward_positive_skipped=wall_avoid_forward_positive_skipped,
            wall_avoid_turn_reward_applied=wall_avoid_turn_reward_applied,
            wall_avoid_repeated_turn_spin=wall_avoid_repeated_turn_spin,
            env_info=env_info,
=======

            collision=collision,
            moved=moved,
            danger_zone=danger_zone,
            found_victim=found_victim,
>>>>>>> 791ac2528253af560762704ec2a507d4af97e674
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
                mean_abs_input_hidden_current=0.0,
                mean_abs_recurrent_hidden_current=0.0,
                mean_recurrent_to_input_current_ratio=0.0,
                median_recurrent_to_input_current_ratio=0.0,
                max_recurrent_to_input_current_ratio=0.0,
                mean_input_hidden_current_norm=0.0,
                mean_recurrent_hidden_current_norm=0.0,
                mean_total_hidden_current_norm=0.0,
                mean_prev_hidden_spike_count=0.0,
                mean_hidden_spike_count=0.0,
                hidden_trace_mean=0.0,
                hidden_trace_nonzero_count=0.0,

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

<<<<<<< HEAD
                found_victim_count=0,
                collision_count=0,
                moved_count=0,
                mean_distance_to_victim=0.0,
                min_distance_to_victim=0.0,
                final_distance_to_victim=0.0,
                final_rescued_count=0,
                final_remaining_victims=0,
=======
                collision_rate=0.0,
                moved_rate=0.0,
                danger_zone_rate=0.0,
                found_victim_count=0,

>>>>>>> 791ac2528253af560762704ec2a507d4af97e674
                action_histogram={},
                snn_action_histogram={},
                executed_action_histogram={},
                output_neuron_spike_histogram={},
                output_winner_histogram={},
                spike_selected_action_histogram={},
                fallback_action_histogram={},
                fallback_count_by_action={},
                fallback_when_front_blocked_count=0,
                fallback_when_front_blocked_action_histogram={},
                no_output_spike_count=0,
                output_threshold_mean_per_action={},
                output_current_mean_per_action={},
                action_mean_reward={},
                action_event_counts={},
                delayed_credit_update_count=0,
                delayed_credit_action_histogram={},
                direct_credit_action_histogram={},
                previous_turn_credit_action_histogram={},
                learning_update_action_histogram={},
                potentiation_action_histogram={},
                depression_action_histogram={},
                turn_sensor_reward_count=0,
                turn_sensor_reward_action_histogram={},
                turn_sensor_penalty_count=0,
                turn_sensor_penalty_action_histogram={},
                learning_reason_histogram={},
                wall_avoid_forward_collision_count=0,
                wall_avoid_forward_after_clearance_count=0,
                wall_avoid_repeated_turn_spin_count=0,
                wall_avoid_forward_positive_count=0,
                wall_avoid_forward_positive_skipped_count=0,
                wall_avoid_turn_reward_count=0,
                wall_avoid_action0_selected_when_front_blocked_count=0,
                wall_avoid_forward_recovery_action_histogram={},
                wall_avoid_spin_penalty_action_histogram={},
                selected_step_action_histogram={},
                input_hidden_current_abs_mean_by_action={},
                recurrent_hidden_current_abs_mean_by_action={},
                recurrent_to_input_current_ratio_by_action={},
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
        input_hidden_abs = np.asarray(
            [s.input_hidden_current_abs_mean for s in self.steps],
            dtype=float,
        )
        recurrent_hidden_abs = np.asarray(
            [s.recurrent_hidden_current_abs_mean for s in self.steps],
            dtype=float,
        )
        recurrent_to_input_ratios = np.asarray(
            [s.recurrent_to_input_current_ratio for s in self.steps],
            dtype=float,
        )
        input_hidden_norms = np.asarray(
            [s.input_hidden_current_norm for s in self.steps],
            dtype=float,
        )
        recurrent_hidden_norms = np.asarray(
            [s.recurrent_hidden_current_norm for s in self.steps],
            dtype=float,
        )
        total_hidden_norms = np.asarray(
            [s.total_hidden_current_norm for s in self.steps],
            dtype=float,
        )
        prev_hidden_spike_counts = np.asarray(
            [s.prev_hidden_spike_count for s in self.steps],
            dtype=float,
        )
        hidden_trace_means = np.asarray(
            [s.hidden_trace_mean for s in self.steps],
            dtype=float,
        )
        hidden_trace_nonzero_counts = np.asarray(
            [s.hidden_trace_nonzero_count for s in self.steps],
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

<<<<<<< HEAD
        found_victims = np.asarray(
            [s.found_victim for s in self.steps],
            dtype=float,
        )

=======
>>>>>>> 791ac2528253af560762704ec2a507d4af97e674
        collisions = np.asarray(
            [s.collision for s in self.steps],
            dtype=float,
        )

<<<<<<< HEAD
        moved = np.asarray(
=======
        moved_arr = np.asarray(
>>>>>>> 791ac2528253af560762704ec2a507d4af97e674
            [s.moved for s in self.steps],
            dtype=float,
        )

<<<<<<< HEAD
        distances = np.asarray(
            [s.distance_to_nearest_victim for s in self.steps],
            dtype=float,
        )
        delayed_credit_counts = np.asarray(
            [s.delayed_credit_update_count for s in self.steps],
=======
        danger_zones = np.asarray(
            [s.danger_zone for s in self.steps],
            dtype=float,
        )

        found_victims = np.asarray(
            [s.found_victim for s in self.steps],
>>>>>>> 791ac2528253af560762704ec2a507d4af97e674
            dtype=float,
        )

        # action 분포 계산
        action_histogram: Dict[int, int] = {}
        snn_action_histogram: Dict[int, int] = {}
        executed_action_histogram: Dict[int, int] = {}
        output_neuron_spike_histogram: Dict[int, int] = {}
        output_winner_histogram: Dict[int, int] = {}
        spike_selected_action_histogram: Dict[int, int] = {}
        fallback_action_histogram: Dict[int, int] = {}
        fallback_count_by_action: Dict[int, int] = {}
        fallback_when_front_blocked_count = 0
        fallback_when_front_blocked_action_histogram: Dict[int, int] = {}
        output_threshold_sum: Dict[int, float] = {}
        output_threshold_count: Dict[int, int] = {}
        output_current_sum: Dict[int, float] = {}
        output_current_count: Dict[int, int] = {}
        action_reward_sum: Dict[int, float] = {}
        action_event_counts: Dict[int, Dict[str, int]] = {}
        delayed_credit_action_histogram: Dict[int, int] = {}
        direct_credit_action_histogram: Dict[int, int] = {}
        previous_turn_credit_action_histogram: Dict[int, int] = {}
        learning_update_action_histogram: Dict[int, int] = {}
        potentiation_action_histogram: Dict[int, int] = {}
        depression_action_histogram: Dict[int, int] = {}
        turn_sensor_reward_action_histogram: Dict[int, int] = {}
        turn_sensor_penalty_action_histogram: Dict[int, int] = {}
        learning_reason_histogram: Dict[str, int] = {}
        turn_sensor_reward_count = 0
        turn_sensor_penalty_count = 0
        wall_avoid_forward_collision_count = 0
        wall_avoid_forward_after_clearance_count = 0
        wall_avoid_repeated_turn_spin_count = 0
        wall_avoid_forward_positive_count = 0
        wall_avoid_forward_positive_skipped_count = 0
        wall_avoid_turn_reward_count = 0
        wall_avoid_action0_selected_when_front_blocked_count = 0
        wall_avoid_forward_recovery_action_histogram: Dict[int, int] = {}
        wall_avoid_spin_penalty_action_histogram: Dict[int, int] = {}
        selected_step_action_histogram: Dict[int, Dict[int, int]] = {}
        input_current_sum_by_action: Dict[int, float] = {}
        recurrent_current_sum_by_action: Dict[int, float] = {}
        recurrent_ratio_sum_by_action: Dict[int, float] = {}
        current_count_by_action: Dict[int, int] = {}

        for s in self.steps:
            action_histogram[s.action] = (
                action_histogram.get(s.action, 0) + 1
            )
            snn_action_histogram[s.snn_action] = (
                snn_action_histogram.get(s.snn_action, 0) + 1
            )
            executed_action_histogram[s.executed_action] = (
                executed_action_histogram.get(s.executed_action, 0) + 1
            )
            for neuron_idx, spike_count in enumerate(s.output_spike_histogram):
                output_neuron_spike_histogram[neuron_idx] = (
                    output_neuron_spike_histogram.get(neuron_idx, 0)
                    + int(spike_count)
                )
            for winner in s.output_winners:
                output_winner_histogram[winner] = (
                    output_winner_histogram.get(winner, 0) + 1
                )
            if s.selected_by_spike and s.action >= 0:
                spike_selected_action_histogram[s.action] = (
                    spike_selected_action_histogram.get(s.action, 0) + 1
                )
            if s.used_fallback and s.fallback_action >= 0:
                fallback_action_histogram[s.fallback_action] = (
                    fallback_action_histogram.get(s.fallback_action, 0) + 1
                )
                fallback_count_by_action[s.fallback_action] = (
                    fallback_count_by_action.get(s.fallback_action, 0) + 1
                )
                if s.pre_step_front_clearance < 0.45:
                    fallback_when_front_blocked_count += 1
                    fallback_when_front_blocked_action_histogram[s.fallback_action] = (
                        fallback_when_front_blocked_action_histogram.get(
                            s.fallback_action,
                            0,
                        )
                        + 1
                    )
            for action_idx, value in enumerate(s.output_threshold_mean):
                output_threshold_sum[action_idx] = (
                    output_threshold_sum.get(action_idx, 0.0) + float(value)
                )
                output_threshold_count[action_idx] = (
                    output_threshold_count.get(action_idx, 0) + 1
                )
            for action_idx, value in enumerate(s.output_current_mean):
                output_current_sum[action_idx] = (
                    output_current_sum.get(action_idx, 0.0) + float(value)
                )
                output_current_count[action_idx] = (
                    output_current_count.get(action_idx, 0) + 1
                )
            action_reward_sum[s.action] = (
                action_reward_sum.get(s.action, 0.0) + float(s.reward)
            )
            input_current_sum_by_action[s.action] = (
                input_current_sum_by_action.get(s.action, 0.0)
                + float(s.input_hidden_current_abs_mean)
            )
            recurrent_current_sum_by_action[s.action] = (
                recurrent_current_sum_by_action.get(s.action, 0.0)
                + float(s.recurrent_hidden_current_abs_mean)
            )
            recurrent_ratio_sum_by_action[s.action] = (
                recurrent_ratio_sum_by_action.get(s.action, 0.0)
                + float(s.recurrent_to_input_current_ratio)
            )
            current_count_by_action[s.action] = current_count_by_action.get(s.action, 0) + 1
            if s.action not in action_event_counts:
                action_event_counts[s.action] = {
                    "found_victim": 0,
                    "collision": 0,
                    "moved": 0,
                }
            action_event_counts[s.action]["found_victim"] += int(s.found_victim)
            action_event_counts[s.action]["collision"] += int(s.collision)
            action_event_counts[s.action]["moved"] += int(s.moved)
            for action, count in s.delayed_credit_action_histogram.items():
                delayed_credit_action_histogram[action] = (
                    delayed_credit_action_histogram.get(action, 0)
                    + int(count)
                )
            for action, count in s.direct_credit_action_histogram.items():
                direct_credit_action_histogram[action] = (
                    direct_credit_action_histogram.get(action, 0)
                    + int(count)
                )
            for action, count in s.previous_turn_credit_action_histogram.items():
                previous_turn_credit_action_histogram[action] = (
                    previous_turn_credit_action_histogram.get(action, 0)
                    + int(count)
                )
            if s.updated_pair_count > 0 and s.target >= 0:
                learning_update_action_histogram[s.target] = (
                    learning_update_action_histogram.get(s.target, 0) + 1
                )
                if s.potentiation_count > 0:
                    potentiation_action_histogram[s.target] = (
                        potentiation_action_histogram.get(s.target, 0)
                        + int(s.potentiation_count)
                    )
                if s.depression_count > 0:
                    depression_action_histogram[s.target] = (
                        depression_action_histogram.get(s.target, 0)
                        + int(s.depression_count)
                    )
            if s.turn_sensor_reward_applied:
                turn_sensor_reward_count += 1
                turn_sensor_reward_action_histogram[s.executed_action] = (
                    turn_sensor_reward_action_histogram.get(s.executed_action, 0) + 1
                )
            if s.turn_sensor_penalty_applied:
                turn_sensor_penalty_count += 1
                turn_sensor_penalty_action_histogram[s.executed_action] = (
                    turn_sensor_penalty_action_histogram.get(s.executed_action, 0) + 1
                )
            if s.learning_reason and s.learning_reason != "none":
                learning_reason_histogram[s.learning_reason] = (
                    learning_reason_histogram.get(s.learning_reason, 0) + 1
                )
            selected_step_action_histogram.setdefault(s.selected_step, {})
            selected_step_action_histogram[s.selected_step][s.executed_action] = (
                selected_step_action_histogram[s.selected_step].get(s.executed_action, 0) + 1
            )
            wall_avoid_forward_collision_count += int(s.wall_avoid_forward_collision)
            wall_avoid_forward_after_clearance_count += int(
                s.wall_avoid_forward_after_clearance
            )
            wall_avoid_repeated_turn_spin_count += int(
                s.wall_avoid_repeated_turn_spin
            )
            if s.wall_avoid_forward_after_clearance:
                wall_avoid_forward_recovery_action_histogram[s.executed_action] = (
                    wall_avoid_forward_recovery_action_histogram.get(
                        s.executed_action,
                        0,
                    )
                    + 1
                )
            if s.wall_avoid_repeated_turn_spin:
                wall_avoid_spin_penalty_action_histogram[s.executed_action] = (
                    wall_avoid_spin_penalty_action_histogram.get(s.executed_action, 0)
                    + 1
                )
            wall_avoid_forward_positive_count += int(s.wall_avoid_forward_positive_applied)
            wall_avoid_forward_positive_skipped_count += int(
                s.wall_avoid_forward_positive_skipped
            )
            wall_avoid_turn_reward_count += int(s.wall_avoid_turn_reward_applied)
            if (
                s.is_wall_avoid_phase
                and s.snn_action == 0
                and s.pre_step_front_clearance < 0.45
            ):
                wall_avoid_action0_selected_when_front_blocked_count += 1

        action_mean_reward = {
            action: float(action_reward_sum[action] / count)
            for action, count in action_histogram.items()
            if count > 0
        }
        input_hidden_current_abs_mean_by_action = {
            action: float(input_current_sum_by_action[action] / count)
            for action, count in current_count_by_action.items()
            if count > 0
        }
        recurrent_hidden_current_abs_mean_by_action = {
            action: float(recurrent_current_sum_by_action[action] / count)
            for action, count in current_count_by_action.items()
            if count > 0
        }
        recurrent_to_input_current_ratio_by_action = {
            action: float(recurrent_ratio_sum_by_action[action] / count)
            for action, count in current_count_by_action.items()
            if count > 0
        }
        output_threshold_mean_per_action = {
            action: float(total / output_threshold_count[action])
            for action, total in output_threshold_sum.items()
            if output_threshold_count.get(action, 0) > 0
        }
        output_current_mean_per_action = {
            action: float(total / output_current_count[action])
            for action, total in output_current_sum.items()
            if output_current_count.get(action, 0) > 0
        }
        final_distance_to_victim = self._safe_float(
            self.steps[-1].env_info.get(
                "post_step_distance_to_victim",
                self.steps[-1].distance_to_nearest_victim,
            ),
            0.0,
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
            mean_abs_input_hidden_current=float(input_hidden_abs.mean()),
            mean_abs_recurrent_hidden_current=float(recurrent_hidden_abs.mean()),
            mean_recurrent_to_input_current_ratio=float(recurrent_to_input_ratios.mean()),
            median_recurrent_to_input_current_ratio=float(
                np.median(recurrent_to_input_ratios)
            ),
            max_recurrent_to_input_current_ratio=float(recurrent_to_input_ratios.max()),
            mean_input_hidden_current_norm=float(input_hidden_norms.mean()),
            mean_recurrent_hidden_current_norm=float(recurrent_hidden_norms.mean()),
            mean_total_hidden_current_norm=float(total_hidden_norms.mean()),
            mean_prev_hidden_spike_count=float(prev_hidden_spike_counts.mean()),
            mean_hidden_spike_count=float(hidden_spikes.mean()),
            hidden_trace_mean=float(hidden_trace_means.mean()),
            hidden_trace_nonzero_count=float(hidden_trace_nonzero_counts.mean()),

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

<<<<<<< HEAD
            found_victim_count=int(found_victims.sum()),
            collision_count=int(collisions.sum()),
            moved_count=int(moved.sum()),
            mean_distance_to_victim=float(distances.mean()),
            min_distance_to_victim=float(distances.min()),
            final_distance_to_victim=float(final_distance_to_victim),
            final_rescued_count=int(self.steps[-1].rescued_count),
            final_remaining_victims=int(self.steps[-1].remaining_victims),
=======
            collision_rate=float(collisions.mean()),
            moved_rate=float(moved_arr.mean()),
            danger_zone_rate=float(danger_zones.mean()),
            found_victim_count=int(found_victims.sum()),

>>>>>>> 791ac2528253af560762704ec2a507d4af97e674
            action_histogram=action_histogram,
            snn_action_histogram=snn_action_histogram,
            executed_action_histogram=executed_action_histogram,
            output_neuron_spike_histogram=output_neuron_spike_histogram,
            output_winner_histogram=output_winner_histogram,
            spike_selected_action_histogram=spike_selected_action_histogram,
            fallback_action_histogram=fallback_action_histogram,
            fallback_count_by_action=fallback_count_by_action,
            fallback_when_front_blocked_count=int(fallback_when_front_blocked_count),
            fallback_when_front_blocked_action_histogram=fallback_when_front_blocked_action_histogram,
            no_output_spike_count=int(output_silent.sum()),
            output_threshold_mean_per_action=output_threshold_mean_per_action,
            output_current_mean_per_action=output_current_mean_per_action,
            action_mean_reward=action_mean_reward,
            action_event_counts=action_event_counts,
            delayed_credit_update_count=int(delayed_credit_counts.sum()),
            delayed_credit_action_histogram=delayed_credit_action_histogram,
            direct_credit_action_histogram=direct_credit_action_histogram,
            previous_turn_credit_action_histogram=previous_turn_credit_action_histogram,
            learning_update_action_histogram=learning_update_action_histogram,
            potentiation_action_histogram=potentiation_action_histogram,
            depression_action_histogram=depression_action_histogram,
            turn_sensor_reward_count=int(turn_sensor_reward_count),
            turn_sensor_reward_action_histogram=turn_sensor_reward_action_histogram,
            turn_sensor_penalty_count=int(turn_sensor_penalty_count),
            turn_sensor_penalty_action_histogram=turn_sensor_penalty_action_histogram,
            learning_reason_histogram=learning_reason_histogram,
            wall_avoid_forward_collision_count=int(wall_avoid_forward_collision_count),
            wall_avoid_forward_after_clearance_count=int(
                wall_avoid_forward_after_clearance_count
            ),
            wall_avoid_repeated_turn_spin_count=int(
                wall_avoid_repeated_turn_spin_count
            ),
            wall_avoid_forward_positive_count=int(wall_avoid_forward_positive_count),
            wall_avoid_forward_positive_skipped_count=int(
                wall_avoid_forward_positive_skipped_count
            ),
            wall_avoid_turn_reward_count=int(wall_avoid_turn_reward_count),
            wall_avoid_action0_selected_when_front_blocked_count=int(
                wall_avoid_action0_selected_when_front_blocked_count
            ),
            wall_avoid_forward_recovery_action_histogram=wall_avoid_forward_recovery_action_histogram,
            wall_avoid_spin_penalty_action_histogram=wall_avoid_spin_penalty_action_histogram,
            selected_step_action_histogram=selected_step_action_histogram,
            input_hidden_current_abs_mean_by_action=input_hidden_current_abs_mean_by_action,
            recurrent_hidden_current_abs_mean_by_action=recurrent_hidden_current_abs_mean_by_action,
            recurrent_to_input_current_ratio_by_action=recurrent_to_input_current_ratio_by_action,
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
            "mean_abs_input_hidden_current": s.mean_abs_input_hidden_current,
            "mean_abs_recurrent_hidden_current": s.mean_abs_recurrent_hidden_current,
            "mean_recurrent_to_input_current_ratio": s.mean_recurrent_to_input_current_ratio,
            "median_recurrent_to_input_current_ratio": s.median_recurrent_to_input_current_ratio,
            "max_recurrent_to_input_current_ratio": s.max_recurrent_to_input_current_ratio,
            "mean_input_hidden_current_norm": s.mean_input_hidden_current_norm,
            "mean_recurrent_hidden_current_norm": s.mean_recurrent_hidden_current_norm,
            "mean_total_hidden_current_norm": s.mean_total_hidden_current_norm,
            "mean_prev_hidden_spike_count": s.mean_prev_hidden_spike_count,
            "mean_hidden_spike_count": s.mean_hidden_spike_count,
            "hidden_trace_mean": s.hidden_trace_mean,
            "hidden_trace_nonzero_count": s.hidden_trace_nonzero_count,

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

<<<<<<< HEAD
            "found_victim_count": s.found_victim_count,
            "collision_count": s.collision_count,
            "moved_count": s.moved_count,
            "mean_distance_to_victim": s.mean_distance_to_victim,
            "min_distance_to_victim": s.min_distance_to_victim,
            "final_distance_to_victim": s.final_distance_to_victim,
            "final_rescued_count": s.final_rescued_count,
            "final_remaining_victims": s.final_remaining_victims,
=======
            "collision_rate": s.collision_rate,
            "moved_rate": s.moved_rate,
            "danger_zone_rate": s.danger_zone_rate,
            "found_victim_count": s.found_victim_count,
>>>>>>> 791ac2528253af560762704ec2a507d4af97e674

            "action_histogram": s.action_histogram,
            "snn_action_histogram": s.snn_action_histogram,
            "executed_action_histogram": s.executed_action_histogram,
            "output_neuron_spike_histogram": s.output_neuron_spike_histogram,
            "output_winner_histogram": s.output_winner_histogram,
            "spike_selected_action_histogram": s.spike_selected_action_histogram,
            "fallback_action_histogram": s.fallback_action_histogram,
            "fallback_count_by_action": s.fallback_count_by_action,
            "fallback_when_front_blocked_count": s.fallback_when_front_blocked_count,
            "fallback_when_front_blocked_action_histogram": s.fallback_when_front_blocked_action_histogram,
            "no_output_spike_count": s.no_output_spike_count,
            "output_threshold_mean_per_action": s.output_threshold_mean_per_action,
            "output_current_mean_per_action": s.output_current_mean_per_action,
            "action_mean_reward": s.action_mean_reward,
            "action_event_counts": s.action_event_counts,
            "delayed_credit_update_count": s.delayed_credit_update_count,
            "delayed_credit_action_histogram": s.delayed_credit_action_histogram,
            "direct_credit_action_histogram": s.direct_credit_action_histogram,
            "previous_turn_credit_action_histogram": s.previous_turn_credit_action_histogram,
            "learning_update_action_histogram": s.learning_update_action_histogram,
            "potentiation_action_histogram": s.potentiation_action_histogram,
            "depression_action_histogram": s.depression_action_histogram,
            "turn_sensor_reward_count": s.turn_sensor_reward_count,
            "turn_sensor_reward_action_histogram": s.turn_sensor_reward_action_histogram,
            "turn_sensor_penalty_count": s.turn_sensor_penalty_count,
            "turn_sensor_penalty_action_histogram": s.turn_sensor_penalty_action_histogram,
            "learning_reason_histogram": s.learning_reason_histogram,
            "wall_avoid_forward_collision_count": s.wall_avoid_forward_collision_count,
            "wall_avoid_forward_after_clearance_count": s.wall_avoid_forward_after_clearance_count,
            "wall_avoid_repeated_turn_spin_count": s.wall_avoid_repeated_turn_spin_count,
            "wall_avoid_forward_positive_count": s.wall_avoid_forward_positive_count,
            "wall_avoid_forward_positive_skipped_count": s.wall_avoid_forward_positive_skipped_count,
            "wall_avoid_turn_reward_count": s.wall_avoid_turn_reward_count,
            "wall_avoid_action0_selected_when_front_blocked_count": s.wall_avoid_action0_selected_when_front_blocked_count,
            "wall_avoid_forward_recovery_action_histogram": s.wall_avoid_forward_recovery_action_histogram,
            "wall_avoid_spin_penalty_action_histogram": s.wall_avoid_spin_penalty_action_histogram,
            "selected_step_action_histogram": s.selected_step_action_histogram,
            "input_hidden_current_abs_mean_by_action": s.input_hidden_current_abs_mean_by_action,
            "recurrent_hidden_current_abs_mean_by_action": s.recurrent_hidden_current_abs_mean_by_action,
            "recurrent_to_input_current_ratio_by_action": s.recurrent_to_input_current_ratio_by_action,
        }

    def compact_summary_dict(self) -> Dict[str, Any]:
        """Small terminal/logging view for stage-level progress."""
        s = self.summary()
        return {
            "num_steps": s.num_steps,
            "fallback_rate": s.fallback_rate,
            "found_victim_count": s.found_victim_count,
            "collision_count": s.collision_count,
            "moved_count": s.moved_count,
            "mean_abs_input_hidden_current": s.mean_abs_input_hidden_current,
            "mean_abs_recurrent_hidden_current": s.mean_abs_recurrent_hidden_current,
            "mean_recurrent_to_input_current_ratio": s.mean_recurrent_to_input_current_ratio,
            "median_recurrent_to_input_current_ratio": s.median_recurrent_to_input_current_ratio,
            "max_recurrent_to_input_current_ratio": s.max_recurrent_to_input_current_ratio,
            "mean_prev_hidden_spike_count": s.mean_prev_hidden_spike_count,
            "mean_hidden_spike_count": s.mean_hidden_spike_count,
            "hidden_trace_mean": s.hidden_trace_mean,
            "hidden_trace_nonzero_count": s.hidden_trace_nonzero_count,
            "action_histogram": s.action_histogram,
            "snn_action_histogram": s.snn_action_histogram,
            "executed_action_histogram": s.executed_action_histogram,
            "output_winner_histogram": s.output_winner_histogram,
            "spike_selected_action_histogram": s.spike_selected_action_histogram,
            "fallback_action_histogram": s.fallback_action_histogram,
            "fallback_count_by_action": s.fallback_count_by_action,
            "fallback_when_front_blocked_count": s.fallback_when_front_blocked_count,
            "fallback_when_front_blocked_action_histogram": s.fallback_when_front_blocked_action_histogram,
            "no_output_spike_count": s.no_output_spike_count,
            "output_threshold_mean_per_action": s.output_threshold_mean_per_action,
            "output_current_mean_per_action": s.output_current_mean_per_action,
            "delayed_credit_update_count": s.delayed_credit_update_count,
            "delayed_credit_action_histogram": s.delayed_credit_action_histogram,
            "direct_credit_action_histogram": s.direct_credit_action_histogram,
            "previous_turn_credit_action_histogram": s.previous_turn_credit_action_histogram,
            "learning_update_action_histogram": s.learning_update_action_histogram,
            "potentiation_action_histogram": s.potentiation_action_histogram,
            "depression_action_histogram": s.depression_action_histogram,
            "turn_sensor_reward_count": s.turn_sensor_reward_count,
            "turn_sensor_reward_action_histogram": s.turn_sensor_reward_action_histogram,
            "turn_sensor_penalty_count": s.turn_sensor_penalty_count,
            "turn_sensor_penalty_action_histogram": s.turn_sensor_penalty_action_histogram,
            "learning_reason_histogram": s.learning_reason_histogram,
            "wall_avoid_forward_collision_count": s.wall_avoid_forward_collision_count,
            "wall_avoid_forward_after_clearance_count": s.wall_avoid_forward_after_clearance_count,
            "wall_avoid_repeated_turn_spin_count": s.wall_avoid_repeated_turn_spin_count,
            "wall_avoid_forward_positive_count": s.wall_avoid_forward_positive_count,
            "wall_avoid_forward_positive_skipped_count": s.wall_avoid_forward_positive_skipped_count,
            "wall_avoid_turn_reward_count": s.wall_avoid_turn_reward_count,
            "wall_avoid_action0_selected_when_front_blocked_count": s.wall_avoid_action0_selected_when_front_blocked_count,
            "wall_avoid_forward_recovery_action_histogram": s.wall_avoid_forward_recovery_action_histogram,
            "wall_avoid_spin_penalty_action_histogram": s.wall_avoid_spin_penalty_action_histogram,
            "selected_step_action_histogram": s.selected_step_action_histogram,
            "input_hidden_current_abs_mean_by_action": s.input_hidden_current_abs_mean_by_action,
            "recurrent_hidden_current_abs_mean_by_action": s.recurrent_hidden_current_abs_mean_by_action,
            "recurrent_to_input_current_ratio_by_action": s.recurrent_to_input_current_ratio_by_action,
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
