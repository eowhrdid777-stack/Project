from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal, Protocol, Tuple

import numpy as np
import config as cfg


Side = Literal["plus", "minus"]
PulsePolarity = Literal["pot", "dep"]
MemoryType = Literal["ltm", "stm"]


class PairAccessProtocol(Protocol):
    def read_pair(self, pair_id: Hashable) -> Tuple[float, float]:
        ...

    def apply_pulse(
        self,
        pair_id: Hashable,
        side: Side,
        polarity: PulsePolarity,
        n_pulses: int = 1,
    ) -> None:
        ...

    def get_pair_bounds(
        self,
        pair_id: Hashable,
    ) -> Tuple[float, float, float, float]:
        ...


@dataclass
class ProgrammingResult:
    success: bool
    n_pulses_plus: int
    n_pulses_minus: int
    g_plus_final: float
    g_minus_final: float
    g_plus_target: float
    g_minus_target: float
    weight_before: float
    weight_after: float
    message: str


@dataclass
class RecenterPlan:
    should_recenter: bool
    g_plus_now: float
    g_minus_now: float
    g_plus_target: float
    g_minus_target: float
    g_plus_min: float
    g_plus_max: float
    g_minus_min: float
    g_minus_max: float
    weight_before: float
    desired_shift: float
    actual_shift: float
    feasible_full_shift: bool
    message: str


class ConductanceModulationController:
    """
    현실적인 measured-read 기반 open-loop controller.
    linear 소자 가정하에:
      read 1회 -> pulse 수 계산 -> 한 번에 program -> audit read 1회
    """

    def __init__(
        self,
        access: PairAccessProtocol,
        memory_type: MemoryType = "ltm",
    ) -> None:
        self.access = access
        self.memory_type = memory_type.lower()

        if self.memory_type == "stm":
            self.g_min = float(getattr(cfg, "STM_G_MIN", cfg.G_MIN))
            self.g_max = float(getattr(cfg, "STM_G_MAX", cfg.G_MAX))
            self.p_max = int(getattr(cfg, "STM_P_MAX", cfg.P_MAX))

            self.recenter_trigger_fraction = float(
                getattr(cfg, "STM_RECENTER_TRIGGER_FRACTION", 0.90)
            )
            self.recenter_target_fraction = float(
                getattr(cfg, "STM_RECENTER_TARGET_FRACTION", 0.60)
            )
            self.program_tolerance = float(
                getattr(cfg, "STM_PROGRAM_TOLERANCE", 0.5e-6)
            )
            self.weight_correction_tol = float(
                getattr(cfg, "STM_WEIGHT_CORRECTION_TOL", 1e-6)
            )
        else:
            self.g_min = float(cfg.G_MIN)
            self.g_max = float(cfg.G_MAX)
            self.p_max = int(cfg.P_MAX)

            self.recenter_trigger_fraction = float(
                getattr(cfg, "RECENTER_TRIGGER_FRACTION", 0.90)
            )
            self.recenter_target_fraction = float(
                getattr(cfg, "RECENTER_TARGET_FRACTION", 0.60)
            )
            self.program_tolerance = float(
                getattr(cfg, "PROGRAM_TOLERANCE", 0.5e-6)
            )
            self.weight_correction_tol = float(
                getattr(cfg, "WEIGHT_CORRECTION_TOL", 1e-6)
            )

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    def _weight(self, g_plus: float, g_minus: float) -> float:
        return float(g_plus - g_minus)

    def _clip(self, g: float, g_lo: float, g_hi: float) -> float:
        return float(np.clip(g, g_lo, g_hi))

    def _get_pair_bounds(
        self,
        pair_id: Hashable,
    ) -> Tuple[float, float, float, float]:
        return self.access.get_pair_bounds(pair_id)

    def _avg_step(self, g_lo: float, g_hi: float) -> float:
        return max((g_hi - g_lo) / max((self.p_max - 1), 1), 1e-18)

    def _estimate_pulses(
        self,
        g_now: float,
        g_target: float,
        g_lo: float,
        g_hi: float,
    ) -> int:
        delta = abs(g_target - g_now)
        if delta <= self.program_tolerance:
            return 0
        return int(np.ceil(delta / self._avg_step(g_lo, g_hi)))

    def _needs_recentering(
        self,
        g_plus: float,
        g_minus: float,
        g_plus_max: float,
        g_minus_max: float,
    ) -> bool:
        plus_trigger = self.recenter_trigger_fraction * g_plus_max
        minus_trigger = self.recenter_trigger_fraction * g_minus_max
        return (g_plus >= plus_trigger) or (g_minus >= minus_trigger)

    # ------------------------------------------------------------
    # target programming
    # ------------------------------------------------------------
    def program_pair_to_targets(
        self,
        pair_id: Hashable,
        g_plus_target: float,
        g_minus_target: float,
    ) -> ProgrammingResult:
        g_plus_now, g_minus_now = self.access.read_pair(pair_id)
        g_plus_min, g_plus_max, g_minus_min, g_minus_max = self._get_pair_bounds(pair_id)

        g_plus_target = self._clip(g_plus_target, g_plus_min, g_plus_max)
        g_minus_target = self._clip(g_minus_target, g_minus_min, g_minus_max)

        weight_before = self._weight(g_plus_now, g_minus_now)

        n_plus = self._estimate_pulses(g_plus_now, g_plus_target, g_plus_min, g_plus_max)
        n_minus = self._estimate_pulses(g_minus_now, g_minus_target, g_minus_min, g_minus_max)

        if n_plus > 0:
            self.access.apply_pulse(
                pair_id=pair_id,
                side="plus",
                polarity="pot" if g_plus_target > g_plus_now else "dep",
                n_pulses=n_plus,
            )

        if n_minus > 0:
            self.access.apply_pulse(
                pair_id=pair_id,
                side="minus",
                polarity="pot" if g_minus_target > g_minus_now else "dep",
                n_pulses=n_minus,
            )

        # audit read 1회
        g_plus_final, g_minus_final = self.access.read_pair(pair_id)
        weight_after = self._weight(g_plus_final, g_minus_final)

        success = (
            abs(g_plus_final - g_plus_target) <= self._avg_step(g_plus_min, g_plus_max) + self.program_tolerance
            and abs(g_minus_final - g_minus_target) <= self._avg_step(g_minus_min, g_minus_max) + self.program_tolerance
        )

        return ProgrammingResult(
            success=success,
            n_pulses_plus=n_plus,
            n_pulses_minus=n_minus,
            g_plus_final=g_plus_final,
            g_minus_final=g_minus_final,
            g_plus_target=g_plus_target,
            g_minus_target=g_minus_target,
            weight_before=weight_before,
            weight_after=weight_after,
            message="One-shot linear programming complete.",
        )

    def program_pair_to_weight(
        self,
        pair_id: Hashable,
        target_weight: float,
        common_mode: float | None = None,
    ) -> ProgrammingResult:
        g_plus_min, g_plus_max, g_minus_min, g_minus_max = self._get_pair_bounds(pair_id)

        if common_mode is None:
            common_mode = 0.5 * (
                max(g_plus_min, g_minus_min) + min(g_plus_max, g_minus_max)
            )

        g_plus_target = common_mode + 0.5 * target_weight
        g_minus_target = common_mode - 0.5 * target_weight

        return self.program_pair_to_targets(
            pair_id=pair_id,
            g_plus_target=g_plus_target,
            g_minus_target=g_minus_target,
        )
    
    # ------------------------------------------------------------
    # symmetric differential update
    # ------------------------------------------------------------
    def apply_symmetric_weight_update(
        self,
        pair_id: Hashable,
        direction: int,
        n_pulses: int = 1,
        recenter: bool = True,
        recenter_check_period: int | None = None,
        step_idx: int | None = None,
    ) -> ProgrammingResult:
        """
        direction:
            +1 -> weight increase  : plus pot,  minus dep
            -1 -> weight decrease  : plus dep,  minus pot

        n_pulses:
            one-side pulse count before position factor in crossbar

        recenter:
            True면 필요 시 recenter 수행

        recenter_check_period:
            None이면 매 호출마다 recenter 가능 여부 검사
            정수이면 step_idx가 그 주기에 맞을 때만 recenter 검사

        step_idx:
            recenter_check_period와 함께 사용
        """
        if direction not in (+1, -1):
            raise ValueError(f"direction must be +1 or -1, got {direction}")

        n_pulses = int(n_pulses)
        if n_pulses <= 0:
            g_plus_now, g_minus_now = self.access.read_pair(pair_id)
            w_now = self._weight(g_plus_now, g_minus_now)
            return ProgrammingResult(
                success=True,
                n_pulses_plus=0,
                n_pulses_minus=0,
                g_plus_final=g_plus_now,
                g_minus_final=g_minus_now,
                g_plus_target=g_plus_now,
                g_minus_target=g_minus_now,
                weight_before=w_now,
                weight_after=w_now,
                message="No programming pulse requested.",
            )

        # update 전 measured read 1회
        g_plus_before, g_minus_before = self.access.read_pair(pair_id)
        weight_before = self._weight(g_plus_before, g_minus_before)

        if direction > 0:
            plus_pol = "pot"
            minus_pol = "dep"
        else:
            plus_pol = "dep"
            minus_pol = "pot"

        # symmetric differential update
        self.access.apply_pulse(pair_id, side="plus", polarity=plus_pol, n_pulses=n_pulses)
        self.access.apply_pulse(pair_id, side="minus", polarity=minus_pol, n_pulses=n_pulses)

        # update 후 measured audit 1회
        g_plus_after, g_minus_after = self.access.read_pair(pair_id)
        weight_after = self._weight(g_plus_after, g_minus_after)

        result = ProgrammingResult(
            success=True,
            n_pulses_plus=n_pulses,
            n_pulses_minus=n_pulses,
            g_plus_final=g_plus_after,
            g_minus_final=g_minus_after,
            g_plus_target=g_plus_after,
            g_minus_target=g_minus_after,
            weight_before=weight_before,
            weight_after=weight_after,
            message=f"Symmetric differential update applied (direction={direction}).",
        )

        if not recenter:
            return result

        # recenter 검사 주기 제어
        should_check_recenter = True
        if recenter_check_period is not None:
            if step_idx is None:
                should_check_recenter = False
            else:
                should_check_recenter = (step_idx % int(recenter_check_period) == 0)

        if not should_check_recenter:
            return result

        # 필요할 때만 recenter
        rec = self.execute_recenter(pair_id, weight_correct=True)

        # recenter 결과를 반영해서 최종 result 갱신
        result.n_pulses_plus += rec.n_pulses_plus
        result.n_pulses_minus += rec.n_pulses_minus
        result.g_plus_final = rec.g_plus_final
        result.g_minus_final = rec.g_minus_final
        result.g_plus_target = rec.g_plus_target
        result.g_minus_target = rec.g_minus_target
        result.weight_after = rec.weight_after
        result.success = result.success and rec.success
        result.message = (
            result.message + " | " + rec.message
        )
        return result
    
    # ------------------------------------------------------------
    # recenter
    # ------------------------------------------------------------
    def make_recenter_plan(self, pair_id: Hashable) -> RecenterPlan:
        g_plus, g_minus = self.access.read_pair(pair_id)
        g_plus_min, g_plus_max, g_minus_min, g_minus_max = self._get_pair_bounds(pair_id)

        weight_before = self._weight(g_plus, g_minus)

        if not self._needs_recentering(g_plus, g_minus, g_plus_max, g_minus_max):
            return RecenterPlan(
                should_recenter=False,
                g_plus_now=g_plus,
                g_minus_now=g_minus,
                g_plus_target=g_plus,
                g_minus_target=g_minus,
                g_plus_min=g_plus_min,
                g_plus_max=g_plus_max,
                g_minus_min=g_minus_min,
                g_minus_max=g_minus_max,
                weight_before=weight_before,
                desired_shift=0.0,
                actual_shift=0.0,
                feasible_full_shift=True,
                message="Recenter not needed.",
            )

        target_plus_high = self.recenter_target_fraction * g_plus_max
        target_minus_high = self.recenter_target_fraction * g_minus_max

        desired_shift_plus = max(0.0, g_plus - target_plus_high)
        desired_shift_minus = max(0.0, g_minus - target_minus_high)
        desired_shift = max(desired_shift_plus, desired_shift_minus)

        available_shift_plus = max(0.0, g_plus - g_plus_min)
        available_shift_minus = max(0.0, g_minus - g_minus_min)
        available_shift = min(available_shift_plus, available_shift_minus)

        feasible_full_shift = desired_shift <= available_shift + 1e-18
        actual_shift = min(desired_shift, available_shift)

        g_plus_target = self._clip(g_plus - actual_shift, g_plus_min, g_plus_max)
        g_minus_target = self._clip(g_minus - actual_shift, g_minus_min, g_minus_max)

        if actual_shift <= 0.0:
            msg = "Recenter requested but no common downward room is available."
        elif feasible_full_shift:
            msg = "Full common downward recenter is feasible."
        else:
            msg = "Only partial common downward recenter is feasible."

        return RecenterPlan(
            should_recenter=True,
            g_plus_now=g_plus,
            g_minus_now=g_minus,
            g_plus_target=g_plus_target,
            g_minus_target=g_minus_target,
            g_plus_min=g_plus_min,
            g_plus_max=g_plus_max,
            g_minus_min=g_minus_min,
            g_minus_max=g_minus_max,
            weight_before=weight_before,
            desired_shift=desired_shift,
            actual_shift=actual_shift,
            feasible_full_shift=feasible_full_shift,
            message=msg,
        )
    
    def needs_recentering_now(self, pair_id: Hashable) -> bool:
        """
        measured read 기반 recenter 필요 여부만 확인
        """
        g_plus, g_minus = self.access.read_pair(pair_id)
        _, g_plus_max, _, g_minus_max = self._get_pair_bounds(pair_id)
        return self._needs_recentering(g_plus, g_minus, g_plus_max, g_minus_max)
    
    def correct_weight_error(
        self,
        pair_id: Hashable,
        target_weight: float,
    ) -> ProgrammingResult:
        g_plus_now, g_minus_now = self.access.read_pair(pair_id)
        g_plus_min, g_plus_max, _, _ = self._get_pair_bounds(pair_id)

        current_weight = self._weight(g_plus_now, g_minus_now)
        weight_error = target_weight - current_weight

        if abs(weight_error) <= self.weight_correction_tol:
            return ProgrammingResult(
                success=True,
                n_pulses_plus=0,
                n_pulses_minus=0,
                g_plus_final=g_plus_now,
                g_minus_final=g_minus_now,
                g_plus_target=g_plus_now,
                g_minus_target=g_minus_now,
                weight_before=current_weight,
                weight_after=current_weight,
                message="No weight correction needed.",
            )

        g_plus_target = self._clip(g_plus_now + weight_error, g_plus_min, g_plus_max)

        return self.program_pair_to_targets(
            pair_id=pair_id,
            g_plus_target=g_plus_target,
            g_minus_target=g_minus_now,
        )

    def execute_recenter(
        self,
        pair_id: Hashable,
        weight_correct: bool = True,
    ) -> ProgrammingResult:
        plan = self.make_recenter_plan(pair_id)

        if not plan.should_recenter:
            return ProgrammingResult(
                success=True,
                n_pulses_plus=0,
                n_pulses_minus=0,
                g_plus_final=plan.g_plus_now,
                g_minus_final=plan.g_minus_now,
                g_plus_target=plan.g_plus_target,
                g_minus_target=plan.g_minus_target,
                weight_before=plan.weight_before,
                weight_after=plan.weight_before,
                message=plan.message,
            )

        result = self.program_pair_to_targets(
            pair_id=pair_id,
            g_plus_target=plan.g_plus_target,
            g_minus_target=plan.g_minus_target,
        )

        if weight_correct:
            corrected = self.correct_weight_error(
                pair_id=pair_id,
                target_weight=plan.weight_before,
            )
            result.n_pulses_plus += corrected.n_pulses_plus
            result.n_pulses_minus += corrected.n_pulses_minus
            result.g_plus_final = corrected.g_plus_final
            result.g_minus_final = corrected.g_minus_final
            result.weight_after = corrected.weight_after
            result.success = result.success and corrected.success

        result.message = plan.message
        return result

    # ------------------------------------------------------------
    # debug
    # ------------------------------------------------------------
    def get_pair_status(self, pair_id: Hashable) -> dict:
        g_plus, g_minus = self.access.read_pair(pair_id)
        g_plus_min, g_plus_max, g_minus_min, g_minus_max = self._get_pair_bounds(pair_id)

        return {
            "memory_type": self.memory_type,
            "g_plus_measured": float(g_plus),
            "g_minus_measured": float(g_minus),
            "weight_measured": self._weight(g_plus, g_minus),
            "g_plus_min": float(g_plus_min),
            "g_plus_max": float(g_plus_max),
            "g_minus_min": float(g_minus_min),
            "g_minus_max": float(g_minus_max),
            "needs_recentering": self._needs_recentering(
                g_plus, g_minus, g_plus_max, g_minus_max
            ),
        }
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    from crossbar import DifferentialCrossbar

    pair = (0, 0)
    n_steps = 120

    cb = DifferentialCrossbar(
        n_rows=4,
        n_cols=4,
        seed=cfg.SEED,
        memory_type="ltm",
    )

    ctrl = ConductanceModulationController(
        access=cb,
        memory_type="ltm",
    )

    # 시작 common-mode를 중간으로 맞춤
    g_plus_min, g_plus_max, g_minus_min, g_minus_max = cb.get_pair_bounds(pair)
    g_plus_mid = 0.5 * (g_plus_min + g_plus_max)
    g_minus_mid = 0.5 * (g_minus_min + g_minus_max)
    cb.set_pair_conductance(pair, g_plus=g_plus_mid, g_minus=g_minus_mid)

    step_hist = []
    gplus_hist = []
    gminus_hist = []
    weight_hist = []

    learn_plus_hist = []
    learn_minus_hist = []

    recenter_plus_hist = []
    recenter_minus_hist = []
    recenter_step_hist = []

    # 예시:
    #   0~59 step  : potentiation
    #   60~119 step: depression
    for step in range(n_steps):
        direction = +1 if step < (n_steps // 2) else -1

        result = ctrl.apply_symmetric_weight_update(
            pair_id=pair,
            direction=direction,
            n_pulses=1,
            recenter=True,
            recenter_check_period=5,   # 5 step마다만 recenter 검사
            step_idx=step,
        )

        step_hist.append(step)
        gplus_hist.append(result.g_plus_final)
        gminus_hist.append(result.g_minus_final)
        weight_hist.append(result.weight_after)

        # 학습 pulse 자체는 항상 양쪽 1개씩
        learn_plus_hist.append(1)
        learn_minus_hist.append(1)

        # recenter에 추가로 들어간 pulse 수만 따로 기록
        extra_plus = max(0, result.n_pulses_plus - 1)
        extra_minus = max(0, result.n_pulses_minus - 1)
        recenter_plus_hist.append(extra_plus)
        recenter_minus_hist.append(extra_minus)

        if extra_plus > 0 or extra_minus > 0:
            recenter_step_hist.append(step)

        print(
            f"[step {step:03d}] "
            f"dir={direction:+d}, "
            f"G+= {result.g_plus_final:.6e}, "
            f"G-= {result.g_minus_final:.6e}, "
            f"W= {result.weight_after:.6e}, "
            f"learn(+,-)= (1,1), "
            f"extra_recenter(+,-)= ({extra_plus},{extra_minus}), "
            f"msg= {result.message}"
        )

    plt.figure(figsize=(9, 5))
    plt.plot(step_hist, weight_hist, label="weight measured")
    for s in recenter_step_hist:
        plt.axvline(s, linestyle=":", alpha=0.25)
    plt.title("Weight evolution (symmetric differential update + rare recenter)")
    plt.xlabel("step")
    plt.ylabel("weight")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(step_hist, gplus_hist, label="G+ measured")
    plt.plot(step_hist, gminus_hist, label="G- measured")
    for s in recenter_step_hist:
        plt.axvline(s, linestyle=":", alpha=0.25)
    plt.title("Conductance evolution")
    plt.xlabel("step")
    plt.ylabel("conductance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(step_hist, recenter_plus_hist, label="extra recenter pulses on plus")
    plt.plot(step_hist, recenter_minus_hist, label="extra recenter pulses on minus")
    plt.title("Additional recenter pulses only")
    plt.xlabel("step")
    plt.ylabel("number of pulses")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\n=== Final status ===")
    print(ctrl.get_pair_status(pair))