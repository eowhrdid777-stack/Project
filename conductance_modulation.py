from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal, Protocol, Tuple

import numpy as np
import config as cfg


Side = Literal["plus", "minus"]
PulsePolarity = Literal["pot", "dep"]


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
        # (g_plus_min, g_plus_max, g_minus_min, g_minus_max)


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
    Common-downward recenter for a differential pair.

    Important:
    - This preserves weight approximately by shifting plus/minus together.
    - With init=min and pot-only usage, recenter may become infeasible
      when one side is already near its lower bound.
    - That is a structural limitation, not a code bug.
    """

    def __init__(self, access: PairAccessProtocol) -> None:
        self.access = access

        self.g_min = float(cfg.G_MIN)
        self.g_max = float(cfg.G_MAX)

        self.recenter_trigger_fraction = float(
            getattr(cfg, "RECENTER_TRIGGER_FRACTION", 0.90)
        )
        self.recenter_target_fraction = float(
            getattr(cfg, "RECENTER_TARGET_FRACTION", 0.60)
        )
        self.program_tolerance = float(
            getattr(cfg, "PROGRAM_TOLERANCE", 0.5e-6)
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
        if hasattr(self.access, "get_pair_bounds"):
            return self.access.get_pair_bounds(pair_id)

        return (
            self.g_min,
            self.g_max,
            self.g_min,
            self.g_max,
        )

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

    def _estimate_pulses_from_curve(
        self,
        g_now: float,
        g_target: float,
        g_lo: float,
        g_hi: float,
    ) -> int:
        # simple average step estimate
        avg_step = max((g_hi - g_lo) / max((cfg.P_MAX - 1), 1), 1e-18)
        delta = max(0.0, g_now - g_target)

        if delta <= self.program_tolerance:
            return 0

        return int(np.ceil(delta / avg_step))

    # ------------------------------------------------------------
    # planning
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

    # ------------------------------------------------------------
    # execution
    # ------------------------------------------------------------
    def execute_recenter(self, pair_id: Hashable) -> ProgrammingResult:
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

        delta_plus = max(0.0, plan.g_plus_now - plan.g_plus_target)
        delta_minus = max(0.0, plan.g_minus_now - plan.g_minus_target)

        n_plus = self._estimate_pulses_from_curve(
            g_now=plan.g_plus_now,
            g_target=plan.g_plus_target,
            g_lo=plan.g_plus_min,
            g_hi=plan.g_plus_max,
        )
        n_minus = self._estimate_pulses_from_curve(
            g_now=plan.g_minus_now,
            g_target=plan.g_minus_target,
            g_lo=plan.g_minus_min,
            g_hi=plan.g_minus_max,
        )

        if n_plus > 0:
            self.access.apply_pulse(
                pair_id=pair_id,
                side="plus",
                polarity="dep",
                n_pulses=n_plus,
            )

        if n_minus > 0:
            self.access.apply_pulse(
                pair_id=pair_id,
                side="minus",
                polarity="dep",
                n_pulses=n_minus,
            )

        g_plus_final, g_minus_final = self.access.read_pair(pair_id)
        weight_after = self._weight(g_plus_final, g_minus_final)

        success = (
            abs(g_plus_final - plan.g_plus_target)
            <= max(self.program_tolerance, abs(delta_plus) + 1e-18)
            and abs(g_minus_final - plan.g_minus_target)
            <= max(self.program_tolerance, abs(delta_minus) + 1e-18)
        )

        return ProgrammingResult(
            success=success,
            n_pulses_plus=n_plus,
            n_pulses_minus=n_minus,
            g_plus_final=g_plus_final,
            g_minus_final=g_minus_final,
            g_plus_target=plan.g_plus_target,
            g_minus_target=plan.g_minus_target,
            weight_before=plan.weight_before,
            weight_after=weight_after,
            message=plan.message,
        )

    def get_pair_status(self, pair_id: Hashable) -> dict:
        g_plus, g_minus = self.access.read_pair(pair_id)
        g_plus_min, g_plus_max, g_minus_min, g_minus_max = self._get_pair_bounds(pair_id)

        return {
            "g_plus": float(g_plus),
            "g_minus": float(g_minus),
            "weight": self._weight(g_plus, g_minus),
            "g_plus_min": float(g_plus_min),
            "g_plus_max": float(g_plus_max),
            "g_minus_min": float(g_minus_min),
            "g_minus_max": float(g_minus_max),
            "needs_recentering": self._needs_recentering(
                g_plus, g_minus, g_plus_max, g_minus_max
            ),
        }