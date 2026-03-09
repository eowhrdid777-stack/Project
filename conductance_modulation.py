from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal, Optional, Protocol, Tuple

import numpy as np
import config as cfg


Side = Literal["plus", "minus"]
PulsePolarity = Literal["pot", "dep"]


class PairAccessProtocol(Protocol):
    """Abstract access interface.

    This module does NOT assume direct device-level read by itself.
    Instead, it asks an external access layer to:
        - read a selected differential pair
        - apply a pulse to one side of that pair

    Later, crossbar.py can implement this same interface.
    """

    def read_pair(self, pair_id: Hashable) -> Tuple[float, float]:
        """Return current (g_plus, g_minus) of the selected pair."""
        ...

    def apply_pulse(
        self,
        pair_id: Hashable,
        side: Side,
        polarity: PulsePolarity,
        n_pulses: int = 1,
    ) -> None:
        """Apply pulse(s) to one side of the selected pair."""
        ...


@dataclass
class ProgrammingResult:
    success: bool
    n_trials: int
    g_plus_final: float
    g_minus_final: float
    g_plus_target: float
    g_minus_target: float
    message: str


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


class ConductanceModulationController:
    """Programming / verify-after-write controller for a differential pair.

    Responsibilities:
        - compute recenter targets
        - check feasibility
        - perform pulse-read-repeat loop
        - keep effective weight approximately preserved while moving both
          conductances away from saturation

    Not responsible for:
        - actual crossbar read circuitry
        - device physics itself
        - array-level VMM
    """

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

    # ------------------------------------------------------------------
    # Recenter planning
    # ------------------------------------------------------------------
    def make_recenter_plan(self, pair_id: Hashable) -> RecenterPlan:
        """Plan a weight-preserving downward shift of both conductances.

        Idea:
            If one side approaches saturation, try moving BOTH sides downward
            by the same conductance amount. This preserves weight:

                W = G+ - G-

            If full desired shift is not feasible because the lower side would
            hit G_MIN first, reduce the shift to the maximum feasible amount.
        """
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
        pair_id: Hashable,
        g_plus_target: float,
        g_minus_target: float,
        max_trials: Optional[int] = None,
        tolerance: Optional[float] = None,
    ) -> ProgrammingResult:
        """Pulse-read-repeat programming to reach pair targets.

        This function supports both upward and downward adjustment.
        For each side:
            if current < target - tol: apply pot pulse
            if current > target + tol: apply dep pulse
        """
        g_plus_target = self._clip(float(g_plus_target))
        g_minus_target = self._clip(float(g_minus_target))

        max_trials = self.program_max_trials if max_trials is None else int(max_trials)
        tolerance = self.program_tolerance if tolerance is None else float(tolerance)

        for trial in range(1, max_trials + 1):
            g_plus, g_minus = self.access.read_pair(pair_id)

            done_plus = abs(g_plus - g_plus_target) <= tolerance
            done_minus = abs(g_minus - g_minus_target) <= tolerance

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
            if g_plus < g_plus_target - tolerance:
                self.access.apply_pulse(pair_id, side="plus", polarity="pot", n_pulses=1)
            elif g_plus > g_plus_target + tolerance:
                self.access.apply_pulse(pair_id, side="plus", polarity="dep", n_pulses=1)

            # minus side
            if g_minus < g_minus_target - tolerance:
                self.access.apply_pulse(pair_id, side="minus", polarity="pot", n_pulses=1)
            elif g_minus > g_minus_target + tolerance:
                self.access.apply_pulse(pair_id, side="minus", polarity="dep", n_pulses=1)

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