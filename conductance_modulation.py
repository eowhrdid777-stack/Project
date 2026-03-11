# 단일 memristor 소자의 내부 conductance 상태 변화 제어
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal, Optional, Protocol, Tuple

import numpy as np
import config as cfg


Side = Literal["plus", "minus"] # device 종류 구별
PulsePolarity = Literal["pot", "dep"] # pulse 종류 구별


class PairAccessProtocol(Protocol):
    # device_model.py에서 conductance 상태를 읽기 위한 class로, 실제 구현은 crossbar.py에서 진행

    def read_pair(self, pair_id: Hashable) -> Tuple[float, float]: ...

    def apply_pulse(
        self,
        pair_id: Hashable,
        side: Side,
        polarity: PulsePolarity,
        n_pulses: int = 1
    ) -> None:
        ...

# 결과를 담는 dataclass 정의
@dataclass
class ProgrammingResult:
    success: bool
    n_trials: int
    g_plus_final: float
    g_minus_final: float
    g_plus_target: float
    g_minus_target: float
    message: str

# recenter를 실제로 수행하기 전에 계획을 담는 dataclass 정의
@dataclass
class RecenterPlan:
    should_recenter: bool
    g_plus_now: float
    g_minus_now: float
    g_plus_target: float
    g_minus_target: float
    weight_before: float
    shift_amount: float
    feasible_full_shift: bool
    message: str

# recenter target 계산과 pulse-read-repeat 수행
class ConductanceModulationController:
    def __init__(self, access: PairAccessProtocol) -> None:
        self.access = access

        self.g_min = float(cfg.G_MIN)
        self.g_max = float(cfg.G_MAX)

        self.recenter_trigger_fraction = float(cfg.RECENTER_TRIGGER_FRACTION)
        self.recenter_target_fraction = float(cfg.RECENTER_TARGET_FRACTION)

        self.program_tolerance = float(cfg.PROGRAM_TOLERANCE)
        self.program_max_trials = int(cfg.PROGRAM_MAX_TRIALS)

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def _clip(self, g: float) -> float:
        return float(np.clip(g, self.g_min, self.g_max))

    def _weight(self, g_plus: float, g_minus: float) -> float:
        return float(g_plus - g_minus)

    def _needs_recentering(self, g_plus: float, g_minus: float) -> bool:
        trigger = self.recenter_trigger_fraction * self.g_max
        return max(g_plus, g_minus) >= trigger

    def _choose_n_pulses(self, error: float, tolerance: float) -> int:
        ae = abs(error)

        if ae > 40e-6:
            return 50
        elif ae > 20e-6:
            return 40
        elif ae > 10e-6:
            return 30
        elif ae > 4e-6:
            return 15
        elif ae > 2e-6:
            return 5
        elif ae > tolerance:
            return 2
        else:
            return 0
    # ------------------------------------------------------------------
    # Recenter planning
    # ------------------------------------------------------------------
    def make_recenter_plan(self, pair_id: Hashable) -> RecenterPlan:
        # 실제로 pulse를 넣지는 않고 현재 상태를 읽어서 recenter가 필요한지 판단하고, 필요하다면 목표 지점 계산
        g_plus, g_minus = self.access.read_pair(pair_id)
        w_before = self._weight(g_plus, g_minus)

        if not self._needs_recentering(g_plus, g_minus):
            return RecenterPlan(
                should_recenter=False,
                g_plus_now=g_plus,
                g_minus_now=g_minus,
                g_plus_target=g_plus,
                g_minus_target=g_minus,
                weight_before=w_before,
                shift_amount=0.0,
                feasible_full_shift=True,
                message="Recenter not needed.",
            )

        target_high = self.recenter_target_fraction * self.g_max
        current_high = max(g_plus, g_minus)

        # Desired common downward shift to bring the higher conductance
        # near target_high.
        desired_shift = max(0.0, current_high - target_high)

        # Feasibility check:
        # since both sides are moved downward by the same amount,
        # the smaller one limits the maximum shift before hitting G_MIN.
        available_shift = min(g_plus, g_minus) - self.g_min
        available_shift = max(0.0, float(available_shift))

        feasible_full_shift = desired_shift <= available_shift + 1e-18
        actual_shift = min(desired_shift, available_shift)

        g_plus_target = self._clip(g_plus - actual_shift)
        g_minus_target = self._clip(g_minus - actual_shift)

        if actual_shift <= 0.0:
            msg = "Recenter requested but no downward room is available."
        elif feasible_full_shift:
            msg = "Full recenter shift is feasible."
        else:
            msg = "Only partial recenter shift is feasible due to lower-side limit."

        return RecenterPlan(
            should_recenter=True,
            g_plus_now=g_plus,
            g_minus_now=g_minus,
            g_plus_target=g_plus_target,
            g_minus_target=g_minus_target,
            weight_before=w_before,
            shift_amount=actual_shift,
            feasible_full_shift=feasible_full_shift,
            message=msg,
        )

    # ------------------------------------------------------------------
    # Generic target programming
    # ------------------------------------------------------------------
    def program_pair_to_targets(
    self,
    pair_id,
    g_plus_target,
    g_minus_target,
    max_trials=None,
    tolerance=None,
    ):
        g_plus_target = self._clip(float(g_plus_target))
        g_minus_target = self._clip(float(g_minus_target))

        max_trials = self.program_max_trials if max_trials is None else int(max_trials)
        tolerance = self.program_tolerance if tolerance is None else float(tolerance)

        for trial in range(1, max_trials + 1):
            g_plus, g_minus = self.access.read_pair(pair_id)

            err_plus = g_plus_target - g_plus
            err_minus = g_minus_target - g_minus

            done_plus = abs(err_plus) <= tolerance
            done_minus = abs(err_minus) <= tolerance

            if done_plus and done_minus:
                return ProgrammingResult(
                    success=True,
                    n_trials=trial - 1,
                    g_plus_final=g_plus,
                    g_minus_final=g_minus,
                    g_plus_target=g_plus_target,
                    g_minus_target=g_minus_target,
                    message="Targets reached within tolerance.",
                )

            # plus side
            if not done_plus:
                n_plus = self._choose_n_pulses(err_plus, tolerance)
                if err_plus > 0:
                    self.access.apply_pulse(pair_id, side="plus", polarity="pot", n_pulses=n_plus)
                elif err_plus < 0:
                    self.access.apply_pulse(pair_id, side="plus", polarity="dep", n_pulses=n_plus)

            # minus side
            if not done_minus:
                n_minus = self._choose_n_pulses(err_minus, tolerance)
                if err_minus > 0:
                    self.access.apply_pulse(pair_id, side="minus", polarity="pot", n_pulses=n_minus)
                elif err_minus < 0:
                    self.access.apply_pulse(pair_id, side="minus", polarity="dep", n_pulses=n_minus)

        g_plus_final, g_minus_final = self.access.read_pair(pair_id)
        return ProgrammingResult(
            success=False,
            n_trials=max_trials,
            g_plus_final=g_plus_final,
            g_minus_final=g_minus_final,
            g_plus_target=g_plus_target,
            g_minus_target=g_minus_target,
            message="Max trials reached before hitting targets.",
        )

    # ------------------------------------------------------------------
    # Recenter execution
    # ------------------------------------------------------------------
    def recenter_pair_if_needed(self, pair_id: Hashable) -> ProgrammingResult:
        """If needed, move both conductances downward while preserving weight as much as possible."""
        plan = self.make_recenter_plan(pair_id)

        if not plan.should_recenter:
            return ProgrammingResult(
                success=True,
                n_trials=0,
                g_plus_final=plan.g_plus_now,
                g_minus_final=plan.g_minus_now,
                g_plus_target=plan.g_plus_now,
                g_minus_target=plan.g_minus_now,
                message=plan.message,
            )

        return self.program_pair_to_targets(
            pair_id=pair_id,
            g_plus_target=plan.g_plus_target,
            g_minus_target=plan.g_minus_target,
        )

    # ------------------------------------------------------------------
    # Optional helper
    # ------------------------------------------------------------------
    def get_pair_status(self, pair_id: Hashable) -> dict:
        g_plus, g_minus = self.access.read_pair(pair_id)
        return {
            "g_plus": g_plus,
            "g_minus": g_minus,
            "weight": self._weight(g_plus, g_minus),
            "needs_recentering": self._needs_recentering(g_plus, g_minus),
        }


# ======================================================================
# TEST / DEBUG ONLY
# Delete or comment out this whole section later if not needed.
# ======================================================================
if __name__ == "__main__":
    from device_model import MemristorDevice

    class MockPairAccess:
        """Temporary mock access layer for testing before crossbar.py exists.

        Later, replace this whole class with a real crossbar access class
        implementing the same interface.
        """

        def __init__(self) -> None:
            self.plus = MemristorDevice(seed=cfg.SEED)
            self.minus = MemristorDevice(seed=cfg.SEED + 1 if cfg.SEED is not None else None)

            self.plus.reset("init")
            self.minus.reset("init")

        def read_pair(self, pair_id: Hashable) -> Tuple[float, float]:
            _ = pair_id
            return self.plus.g, self.minus.g

        def apply_pulse(
            self,
            pair_id: Hashable,
            side: Side,
            polarity: PulsePolarity,
            n_pulses: int = 1,
        ) -> None:
            _ = pair_id

            dev = self.plus if side == "plus" else self.minus
            dev.apply_pulse(polarity=polarity, n_pulses=n_pulses)

    access = MockPairAccess()
    ctrl = ConductanceModulationController(access=access)

    pair_id = (0, 0)

    # Make plus side approach saturation
    for _ in range(120):
        access.apply_pulse(pair_id, side="plus", polarity="pot", n_pulses=1)

    before = ctrl.get_pair_status(pair_id)
    print("Before recenter:", before)

    plan = ctrl.make_recenter_plan(pair_id)
    print("Plan:", plan)

    result = ctrl.recenter_pair_if_needed(pair_id)
    print("Result:", result)

    after = ctrl.get_pair_status(pair_id)
    print("After recenter:", after)