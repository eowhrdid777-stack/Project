from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal

import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from crossbar import DifferentialCrossbar

Direction = Literal[+1, -1]


@dataclass
class ProgrammingResult:
    success: bool
    n_pulses_plus: int
    n_pulses_minus: int
    g_plus_final: float
    g_minus_final: float
    weight_before: float
    weight_after: float
    common_mode_after: float
    did_refresh: bool
    chosen_action: str
    message: str


class ConductanceModulationController:
    """Measured-only controller for differential FeTFT synapses.

    Policy:
    1. Read the pair through the array (with nonidealities).
    2. For a requested weight direction, choose only one physical action from
       {plus-pot, plus-dep, minus-pot, minus-dep} that both changes the weight
       in the correct direction and keeps common-mode near the middle.
    3. Refresh/remap only when headroom becomes too small or periodically.

    This is intentionally conservative so the linear region of the device can be
    used for longer. It avoids the previous "common-downward recenter" idea.
    """

    def __init__(self, access: DifferentialCrossbar) -> None:
        self.access = access
        self.program_tolerance = float(cfg.PROGRAM_TOLERANCE)
        self.max_verify_steps = int(cfg.MAX_VERIFY_STEPS)
        self.pulses_per_verify_step = int(cfg.PULSES_PER_VERIFY_STEP)

        self.cm_target = float(cfg.COMMON_MODE_TARGET)
        self.cm_band_fraction = float(cfg.COMMON_MODE_BAND_FRACTION)
        self.headroom_trigger_fraction = float(cfg.HEADROOM_TRIGGER_FRACTION)
        self.refresh_check_period = int(cfg.REFRESH_CHECK_PERIOD)
        self.refresh_min_interval = int(cfg.REFRESH_MIN_INTERVAL)
        self.last_refresh_step = -10**9

    # ------------------------------------------------------------------
    # Helper calculations
    # ------------------------------------------------------------------
    @staticmethod
    def _weight(g_plus: float, g_minus: float) -> float:
        return float(g_plus - g_minus)

    @staticmethod
    def _common_mode(g_plus: float, g_minus: float) -> float:
        return 0.5 * float(g_plus + g_minus)

    def _pair_bounds(self, pair_id: Hashable) -> tuple[float, float, float, float]:
        return self.access.get_pair_bounds(pair_id)

    def _headrooms(self, pair_id: Hashable, g_plus: float, g_minus: float) -> dict:
        gp_min, gp_max, gm_min, gm_max = self._pair_bounds(pair_id)
        return {
            "gp_up": max(0.0, gp_max - g_plus),
            "gp_down": max(0.0, g_plus - gp_min),
            "gm_up": max(0.0, gm_max - g_minus),
            "gm_down": max(0.0, g_minus - gm_min),
            "gp_min": gp_min,
            "gp_max": gp_max,
            "gm_min": gm_min,
            "gm_max": gm_max,
        }

    def _cm_band(self, pair_id: Hashable) -> tuple[float, float]:
        gp_min, gp_max, gm_min, gm_max = self._pair_bounds(pair_id)
        global_min = 0.5 * (gp_min + gm_min)
        global_max = 0.5 * (gp_max + gm_max)
        full_span = global_max - global_min
        half_band = 0.5 * self.cm_band_fraction * full_span
        return self.cm_target - half_band, self.cm_target + half_band

    def _should_refresh(
        self,
        pair_id: Hashable,
        step_idx: int,
        g_plus: float,
        g_minus: float,
    ) -> bool:
        h = self._headrooms(pair_id, g_plus, g_minus)
        gp_span = max(1e-18, h["gp_max"] - h["gp_min"])
        gm_span = max(1e-18, h["gm_max"] - h["gm_min"])

        low_headroom = (
            h["gp_up"] <= self.headroom_trigger_fraction * gp_span
            or h["gp_down"] <= self.headroom_trigger_fraction * gp_span
            or h["gm_up"] <= self.headroom_trigger_fraction * gm_span
            or h["gm_down"] <= self.headroom_trigger_fraction * gm_span
        )
        periodic = (step_idx % max(1, self.refresh_check_period) == 0)
        spaced = (step_idx - self.last_refresh_step) >= self.refresh_min_interval
        return bool(spaced and (low_headroom or periodic))

    # ------------------------------------------------------------------
    # One-step online update policy
    # ------------------------------------------------------------------
    def _candidate_score(self, cm_after: float, target_bias: float) -> float:
        lo, hi = self.cm_target - target_bias, self.cm_target + target_bias
        if lo <= cm_after <= hi:
            return abs(cm_after - self.cm_target)
        return 10.0 * abs(cm_after - self.cm_target)

    def choose_one_sided_action(
        self,
        pair_id: Hashable,
        direction: int,
        g_plus: float,
        g_minus: float,
    ) -> tuple[str, str]:
        h = self._headrooms(pair_id, g_plus, g_minus)
        cm = self._common_mode(g_plus, g_minus)
        band_lo, band_hi = self._cm_band(pair_id)
        target_bias = max(self.cm_target - band_lo, band_hi - self.cm_target)

        candidates: list[tuple[float, str, str]] = []

        def add_if_feasible(side: str, polarity: str, gp_delta_sign: int, gm_delta_sign: int, headroom_key: str) -> None:
            if h[headroom_key] <= 0.0:
                return
            # Crude one-pulse CM prediction; enough for action selection.
            step_mag = 0.5 * min(h[headroom_key], max(1e-9, 0.04 * (cfg.G_MAX - cfg.G_MIN)))
            gp_after = g_plus + gp_delta_sign * step_mag
            gm_after = g_minus + gm_delta_sign * step_mag
            cm_after = self._common_mode(gp_after, gm_after)
            score = self._candidate_score(cm_after, target_bias)
            candidates.append((score, side, polarity))

        if direction > 0:
            add_if_feasible("plus", "pot", +1, 0, "gp_up")
            add_if_feasible("minus", "dep", 0, -1, "gm_down")
        else:
            add_if_feasible("plus", "dep", -1, 0, "gp_down")
            add_if_feasible("minus", "pot", 0, +1, "gm_up")

        if not candidates:
            # Fallback to whichever physically exists.
            if direction > 0:
                if h["gp_up"] > 0.0:
                    return "plus", "pot"
                if h["gm_down"] > 0.0:
                    return "minus", "dep"
            else:
                if h["gp_down"] > 0.0:
                    return "plus", "dep"
                if h["gm_up"] > 0.0:
                    return "minus", "pot"
            raise RuntimeError("No feasible one-sided programming action for this pair state.")

        candidates.sort(key=lambda x: x[0])
        _, side, polarity = candidates[0]
        return side, polarity

    # ------------------------------------------------------------------
    # Measured remap / verify programming
    # ------------------------------------------------------------------
    def _program_side_to_target(
        self,
        pair_id: Hashable,
        side: str,
        target: float,
    ) -> int:
        pulses = 0
        for _ in range(self.max_verify_steps):
            gp, gm = self.access.read_pair(pair_id)
            g_now = gp if side == "plus" else gm
            err = target - g_now
            if abs(err) <= self.program_tolerance:
                break
            polarity = "pot" if err > 0.0 else "dep"
            pulses += self.access.apply_pulse(pair_id, side=side, polarity=polarity, n_pulses=self.pulses_per_verify_step)
        return int(pulses)

    def refresh_remap(self, pair_id: Hashable, step_idx: int) -> ProgrammingResult:
        gp_before, gm_before = self.access.read_pair(pair_id)
        w_before = self._weight(gp_before, gm_before)

        gp_min, gp_max, gm_min, gm_max = self._pair_bounds(pair_id)
        cm_target = float(np.clip(self.cm_target, 0.5 * (gp_min + gm_min), 0.5 * (gp_max + gm_max)))
        gp_tgt = float(np.clip(cm_target + 0.5 * w_before, gp_min, gp_max))
        gm_tgt = float(np.clip(cm_target - 0.5 * w_before, gm_min, gm_max))

        # If clipping changed the represented weight too much, back off toward the
        # feasible center while preserving sign as much as possible.
        feasible_span_pos = gp_max - gm_min
        feasible_span_neg = gp_min - gm_max
        w_tgt = float(np.clip(w_before, feasible_span_neg, feasible_span_pos))
        gp_tgt = float(np.clip(cm_target + 0.5 * w_tgt, gp_min, gp_max))
        gm_tgt = float(np.clip(cm_target - 0.5 * w_tgt, gm_min, gm_max))

        n_plus = self._program_side_to_target(pair_id, "plus", gp_tgt)
        n_minus = self._program_side_to_target(pair_id, "minus", gm_tgt)

        gp_after, gm_after = self.access.read_pair(pair_id)
        self.last_refresh_step = int(step_idx)
        return ProgrammingResult(
            success=True,
            n_pulses_plus=n_plus,
            n_pulses_minus=n_minus,
            g_plus_final=gp_after,
            g_minus_final=gm_after,
            weight_before=w_before,
            weight_after=self._weight(gp_after, gm_after),
            common_mode_after=self._common_mode(gp_after, gm_after),
            did_refresh=True,
            chosen_action="refresh-remap",
            message="Measured refresh/remap executed.",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_weight(self, pair_id: Hashable, direction: int, step_idx: int) -> ProgrammingResult:
        if direction not in (+1, -1):
            raise ValueError(f"direction must be +1 or -1, got {direction}")

        gp_before, gm_before = self.access.read_pair(pair_id)
        w_before = self._weight(gp_before, gm_before)

        if self._should_refresh(pair_id, step_idx, gp_before, gm_before):
            return self.refresh_remap(pair_id, step_idx)

        side, polarity = self.choose_one_sided_action(pair_id, direction, gp_before, gm_before)
        n_eff = self.access.apply_pulse(pair_id, side=side, polarity=polarity, n_pulses=1)
        gp_after, gm_after = self.access.read_pair(pair_id)
        return ProgrammingResult(
            success=True,
            n_pulses_plus=n_eff if side == "plus" else 0,
            n_pulses_minus=n_eff if side == "minus" else 0,
            g_plus_final=gp_after,
            g_minus_final=gm_after,
            weight_before=w_before,
            weight_after=self._weight(gp_after, gm_after),
            common_mode_after=self._common_mode(gp_after, gm_after),
            did_refresh=False,
            chosen_action=f"{side}-{polarity}",
            message="One-sided balanced differential update.",
        )

    def get_pair_status(self, pair_id: Hashable) -> dict:
        gp, gm = self.access.read_pair(pair_id)
        h = self._headrooms(pair_id, gp, gm)
        return {
            "g_plus_measured": float(gp),
            "g_minus_measured": float(gm),
            "weight_measured": self._weight(gp, gm),
            "common_mode_measured": self._common_mode(gp, gm),
            "headrooms": h,
        }

