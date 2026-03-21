from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from conductance_modulation import ProgrammingResult
from network import MemristiveSNNNetwork


@dataclass
class RSTDPConfig:
    """
    Reward-modulated STDP parameters.

    delta_t = t_post - t_pre

    If delta_t > 0:
        delta_w ~ +A_plus * exp(-delta_t / tau_plus)

    If delta_t < 0:
        delta_w ~ -A_minus * exp(+delta_t / tau_minus)

    Final physical programming direction is determined by:
        sign(reward * eligibility)
    """
    tau_plus: float = 2.0
    tau_minus: float = 2.0
    a_plus: float = 1.0
    a_minus: float = 0.8
    eligibility_threshold: float = 1e-6

    # Conservative default:
    # if output decision came only from fallback, do not fabricate a fake post spike.
    use_surrogate_post_on_fallback: bool = False

    # Hidden-layer R-STDP is optional because it is harder to stabilize.
    enable_hidden_rstdp: bool = False


@dataclass
class RSTDPUpdateEvent:
    layer_name: str
    updated_pairs: List[Tuple[int, int]]
    delta_t_records: List[float]
    eligibility_values: List[float]
    directions: List[int]
    actions: List[str]
    n_pulses_plus: int
    n_pulses_minus: int
    n_refresh: int
    reward: float
    winner: int
    target: Optional[int]
    message: str


class RewardModulatedSTDPLearner:
    """
    Hardware-aware R-STDP learner.

    Important:
    - Spike timing -> eligibility
    - Reward -> global gate
    - Final update is NOT an ideal floating-point write
    - Actual programming goes through ConductanceModulationController.update_weight()
    """

    def __init__(self, config: Optional[RSTDPConfig] = None) -> None:
        self.cfg = RSTDPConfig() if config is None else config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def learn(
        self,
        net: MemristiveSNNNetwork,
        reward: float,
        target: Optional[int] = None,
    ) -> Dict[str, Optional[RSTDPUpdateEvent]]:
        if net.last_decision is None:
            raise RuntimeError("No decision available. Call net.decide(obs) before learner.learn(...).")

        output_event = self._learn_output(net=net, reward=reward, target=target)
        hidden_event = None
        if self.cfg.enable_hidden_rstdp:
            hidden_event = self._learn_hidden(net=net, reward=reward)

        return {
            "output": output_event,
            "hidden": hidden_event,
        }

    # ------------------------------------------------------------------
    # Output-layer R-STDP
    # ------------------------------------------------------------------
    def _learn_output(
        self,
        net: MemristiveSNNNetwork,
        reward: float,
        target: Optional[int],
    ) -> RSTDPUpdateEvent:
        decision = net.last_decision
        assert decision is not None

        step_records = decision.step_records
        n_pre = net.hidden_dim
        n_post = net.n_actions

        pre_spikes = np.array(
            [np.asarray(rec.hidden_result.spikes, dtype=int) for rec in step_records],
            dtype=int,
        )  # [T, hidden_dim]

        post_spikes = np.array(
            [np.asarray(rec.output_result.spikes, dtype=int) for rec in step_records],
            dtype=int,
        )  # [T, n_actions]

        winner = int(decision.action)
        post_col = winner if target is None else int(target)
        if not (0 <= post_col < n_post):
            raise ValueError(f"Target/post column out of range: {post_col}")

        post_times = np.flatnonzero(post_spikes[:, post_col] > 0).astype(int).tolist()

        if (not post_times) and bool(decision.used_fallback) and self.cfg.use_surrogate_post_on_fallback:
            post_times = [int(decision.selected_step)]

        if not post_times:
            return RSTDPUpdateEvent(
                layer_name="output",
                updated_pairs=[],
                delta_t_records=[],
                eligibility_values=[],
                directions=[],
                actions=[],
                n_pulses_plus=0,
                n_pulses_minus=0,
                n_refresh=0,
                reward=float(reward),
                winner=winner,
                target=target,
                message="No postsynaptic output spike available; skipped output R-STDP.",
            )

        reward_sign = self._reward_sign(reward)
        if reward_sign == 0:
            return RSTDPUpdateEvent(
                layer_name="output",
                updated_pairs=[],
                delta_t_records=[],
                eligibility_values=[],
                directions=[],
                actions=[],
                n_pulses_plus=0,
                n_pulses_minus=0,
                n_refresh=0,
                reward=float(reward),
                winner=winner,
                target=target,
                message="Reward is zero; no gated output R-STDP update applied.",
            )

        updated_pairs: List[Tuple[int, int]] = []
        delta_t_records: List[float] = []
        eligibility_values: List[float] = []
        directions: List[int] = []
        actions: List[str] = []
        n_pulses_plus = 0
        n_pulses_minus = 0
        n_refresh = 0

        for row in range(n_pre):
            pre_times = np.flatnonzero(pre_spikes[:, row] > 0).astype(int).tolist()
            if not pre_times:
                continue

            best_dt, elig = self._pair_eligibility(pre_times, post_times)
            if abs(elig) < self.cfg.eligibility_threshold:
                continue

            direction = self._eligibility_to_direction(elig=elig, reward=reward)
            if direction == 0:
                continue

            result: ProgrammingResult = net.output_layer.controller.update_weight(
                (int(row), int(post_col)),
                int(direction),
                int(net.global_step),
            )

            updated_pairs.append((int(row), int(post_col)))
            delta_t_records.append(float(best_dt))
            eligibility_values.append(float(elig))
            directions.append(int(direction))
            actions.append(str(result.chosen_action))
            n_pulses_plus += int(result.n_pulses_plus)
            n_pulses_minus += int(result.n_pulses_minus)
            if result.did_refresh:
                n_refresh += 1

        return RSTDPUpdateEvent(
            layer_name="output",
            updated_pairs=updated_pairs,
            delta_t_records=delta_t_records,
            eligibility_values=eligibility_values,
            directions=directions,
            actions=actions,
            n_pulses_plus=n_pulses_plus,
            n_pulses_minus=n_pulses_minus,
            n_refresh=n_refresh,
            reward=float(reward),
            winner=winner,
            target=target,
            message="Output R-STDP pulse update executed." if updated_pairs else "No output pair crossed eligibility threshold.",
        )

    # ------------------------------------------------------------------
    # Optional hidden-layer R-STDP
    # ------------------------------------------------------------------
    def _learn_hidden(
        self,
        net: MemristiveSNNNetwork,
        reward: float,
    ) -> RSTDPUpdateEvent:
        decision = net.last_decision
        assert decision is not None

        step_records = decision.step_records
        selected_step = int(decision.selected_step)
        if not (0 <= selected_step < len(step_records)):
            return RSTDPUpdateEvent(
                layer_name="hidden",
                updated_pairs=[],
                delta_t_records=[],
                eligibility_values=[],
                directions=[],
                actions=[],
                n_pulses_plus=0,
                n_pulses_minus=0,
                n_refresh=0,
                reward=float(reward),
                winner=-1,
                target=None,
                message="Selected step out of range; skipped hidden R-STDP.",
            )

        hidden_winner = int(step_records[selected_step].hidden_result.winner)
        if hidden_winner < 0:
            return RSTDPUpdateEvent(
                layer_name="hidden",
                updated_pairs=[],
                delta_t_records=[],
                eligibility_values=[],
                directions=[],
                actions=[],
                n_pulses_plus=0,
                n_pulses_minus=0,
                n_refresh=0,
                reward=float(reward),
                winner=-1,
                target=None,
                message="No hidden winner at selected step; skipped hidden R-STDP.",
            )

        pre_spikes = np.array(
            [np.asarray(rec.hidden_input_vector, dtype=int) for rec in step_records],
            dtype=int,
        )  # [T, hidden_input_dim]

        post_spikes = np.array(
            [np.asarray(rec.hidden_result.spikes, dtype=int) for rec in step_records],
            dtype=int,
        )  # [T, hidden_dim]

        post_times = np.flatnonzero(post_spikes[:, hidden_winner] > 0).astype(int).tolist()
        if not post_times:
            return RSTDPUpdateEvent(
                layer_name="hidden",
                updated_pairs=[],
                delta_t_records=[],
                eligibility_values=[],
                directions=[],
                actions=[],
                n_pulses_plus=0,
                n_pulses_minus=0,
                n_refresh=0,
                reward=float(reward),
                winner=hidden_winner,
                target=None,
                message="No hidden postsynaptic spike available; skipped hidden R-STDP.",
            )

        reward_sign = self._reward_sign(reward)
        if reward_sign == 0:
            return RSTDPUpdateEvent(
                layer_name="hidden",
                updated_pairs=[],
                delta_t_records=[],
                eligibility_values=[],
                directions=[],
                actions=[],
                n_pulses_plus=0,
                n_pulses_minus=0,
                n_refresh=0,
                reward=float(reward),
                winner=hidden_winner,
                target=None,
                message="Reward is zero; no gated hidden R-STDP update applied.",
            )

        updated_pairs: List[Tuple[int, int]] = []
        delta_t_records: List[float] = []
        eligibility_values: List[float] = []
        directions: List[int] = []
        actions: List[str] = []
        n_pulses_plus = 0
        n_pulses_minus = 0
        n_refresh = 0

        n_pre = pre_spikes.shape[1]
        for row in range(n_pre):
            pre_times = np.flatnonzero(pre_spikes[:, row] > 0).astype(int).tolist()
            if not pre_times:
                continue

            best_dt, elig = self._pair_eligibility(pre_times, post_times)
            if abs(elig) < self.cfg.eligibility_threshold:
                continue

            direction = self._eligibility_to_direction(elig=elig, reward=reward)
            if direction == 0:
                continue

            result: ProgrammingResult = net.hidden_layer.controller.update_weight(
                (int(row), int(hidden_winner)),
                int(direction),
                int(net.global_step),
            )

            updated_pairs.append((int(row), int(hidden_winner)))
            delta_t_records.append(float(best_dt))
            eligibility_values.append(float(elig))
            directions.append(int(direction))
            actions.append(str(result.chosen_action))
            n_pulses_plus += int(result.n_pulses_plus)
            n_pulses_minus += int(result.n_pulses_minus)
            if result.did_refresh:
                n_refresh += 1

        return RSTDPUpdateEvent(
            layer_name="hidden",
            updated_pairs=updated_pairs,
            delta_t_records=delta_t_records,
            eligibility_values=eligibility_values,
            directions=directions,
            actions=actions,
            n_pulses_plus=n_pulses_plus,
            n_pulses_minus=n_pulses_minus,
            n_refresh=n_refresh,
            reward=float(reward),
            winner=hidden_winner,
            target=None,
            message="Hidden R-STDP pulse update executed." if updated_pairs else "No hidden pair crossed eligibility threshold.",
        )

    # ------------------------------------------------------------------
    # STDP core
    # ------------------------------------------------------------------
    def _pair_eligibility(self, pre_times: Sequence[int], post_times: Sequence[int]) -> Tuple[float, float]:
        """
        Returns:
            best_dt: delta_t of the strongest pair contribution
            elig: summed STDP eligibility over all pre/post pairs
        """
        elig = 0.0
        best_dt = 0.0
        best_mag = -1.0

        for t_pre in pre_times:
            for t_post in post_times:
                dt = float(t_post - t_pre)
                contrib = self._stdp_kernel(dt)
                elig += contrib

                if abs(contrib) > best_mag:
                    best_mag = abs(contrib)
                    best_dt = dt

        return float(best_dt), float(elig)

    def _stdp_kernel(self, dt: float) -> float:
        if dt > 0.0:
            return float(self.cfg.a_plus * np.exp(-dt / max(self.cfg.tau_plus, 1e-12)))
        if dt < 0.0:
            return float(-self.cfg.a_minus * np.exp(-(-dt) / max(self.cfg.tau_minus, 1e-12)))
        return 0.0

    @staticmethod
    def _reward_sign(reward: float) -> int:
        if reward > 0.0:
            return 1
        if reward < 0.0:
            return -1
        return 0

    def _eligibility_to_direction(self, elig: float, reward: float) -> int:
        """
        Convert sign(reward * eligibility) to physical programming direction.

        +1: strengthen effective synaptic influence
        -1: weaken effective synaptic influence
         0: no update
        """
        value = float(elig) * float(reward)
        if value > 0.0:
            return +1
        if value < 0.0:
            return -1
        return 0


if __name__ == "__main__":
    print("learning.py ready: explicit delta_t -> eligibility -> pulse direction path enabled.")
