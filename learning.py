from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from conductance_modulation import ProgrammingResult
from network import MemristiveSNNNetwork


@dataclass
class RSTDPConfig:
    """Reward-modulated STDP configuration.

    Hardware interpretation
    -----------------------
    1. Spike timing creates an eligibility value.
    2. Reward acts as a global modulation gate.
    3. The requested signed update is converted to a pulse direction and a
       small integer pulse count.
    4. The actual conductance change is performed by the crossbar controller;
       this file never applies an ideal floating-point weight update.

    Three-crossbar network note
    ---------------------------
    The normal training target is W_out, i.e. hidden -> output. Hidden-layer
    updates are optional. If enabled, input -> hidden and recurrent hidden ->
    hidden updates are controlled separately because the recurrent path may be
    an STM/volatile path rather than an LTM policy-storage path.
    """

    tau_plus: float = 2.0
    tau_minus: float = 2.0
    a_plus: float = 1.0
    a_minus: float = 0.8
    eligibility_threshold: float = 1e-6

    # Fallback is a readout decision, not a postsynaptic spike by default.
    use_surrogate_post_on_fallback: bool = False
    use_surrogate_post_on_target: bool = False
    use_abs_eligibility_on_target_negative: bool = False

    # Main learning path is output R-STDP. Hidden R-STDP is optional.
    enable_hidden_rstdp: bool = False

    # In the three-crossbar network:
    # - input->hidden is normally LTM/differential and may optionally learn.
    # - hidden->hidden recurrent is normally STM and should remain OFF unless
    #   the STM crossbar explicitly supports programmable pair updates.
    hidden_update_input_path: bool = True
    hidden_update_recurrent_path: bool = False

    # delta_w = reward * eligibility -> pulse count mapping
    delta_w_scale: float = 1.0
    pulse_base: int = 1
    pulse_max: int = 4
    delta_w_per_pulse: float = 0.05
    delta_w_min_abs: float = 0.0
    output_depression_scale: float = 1.0
    anti_target_depression: bool = False
    anti_target_depression_scale: float = 0.25


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
    used_surrogate_post: bool = False


class RewardModulatedSTDPLearner:
    """Hardware-aware R-STDP learner.

    Conservative defaults:
    - no ideal weight writes;
    - no fake postsynaptic spike unless explicitly enabled;
    - no update when reward is zero;
    - no recurrent STM-path learning unless explicitly enabled.
    """

    def __init__(self, config: Optional[RSTDPConfig] = None) -> None:
        self.cfg = RSTDPConfig() if config is None else config
        self._validate_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def learn(
        self,
        net: MemristiveSNNNetwork,
        reward: float,
        target: Optional[int] = None,
        use_abs_eligibility_on_target_negative: Optional[bool] = None,
        use_surrogate_post_on_target_negative: bool = False,
    ) -> Dict[str, Optional[RSTDPUpdateEvent]]:
        if net.last_decision is None:
            raise RuntimeError("No decision available. Call net.decide(obs) before learner.learn(...).")

        output_event = self._learn_output(
            net=net,
            reward=float(reward),
            target=target,
            use_abs_eligibility_on_target_negative=(
                use_abs_eligibility_on_target_negative
            ),
            use_surrogate_post_on_target_negative=bool(
                use_surrogate_post_on_target_negative
            ),
        )
        hidden_event = None
        if bool(self.cfg.enable_hidden_rstdp):
            hidden_event = self._learn_hidden(net=net, reward=float(reward))

        return {"output": output_event, "hidden": hidden_event}

    # ------------------------------------------------------------------
    # Output-layer R-STDP: hidden -> output
    # ------------------------------------------------------------------
    def _learn_output(
        self,
        net: MemristiveSNNNetwork,
        reward: float,
        target: Optional[int],
        use_abs_eligibility_on_target_negative: Optional[bool],
        use_surrogate_post_on_target_negative: bool,
    ) -> RSTDPUpdateEvent:
        decision = net.last_decision
        assert decision is not None

        step_records = decision.step_records
        if not step_records:
            return self._empty_event("output", reward, -1, target, "No step records; skipped output R-STDP.")

        n_pre = int(net.hidden_dim)
        n_post = int(net.n_actions)

        pre_spikes = self._stack_step_vectors(
            [rec.hidden_result.spikes for rec in step_records],
            expected_dim=n_pre,
            name="output pre hidden spikes",
            dtype=int,
        )
        post_spikes = self._stack_step_vectors(
            [rec.output_result.spikes for rec in step_records],
            expected_dim=n_post,
            name="output post spikes",
            dtype=int,
        )

        winner = int(decision.action)
        post_col = winner if target is None else int(target)
        if not (0 <= post_col < n_post):
            raise ValueError(f"Target/post column out of range: {post_col}")

        post_times = np.flatnonzero(post_spikes[:, post_col] > 0).astype(int).tolist()
        used_surrogate_post = False
        if (not post_times) and bool(decision.used_fallback) and bool(self.cfg.use_surrogate_post_on_fallback):
            post_times = [int(decision.selected_step)]
            used_surrogate_post = True
        if (
            (not post_times)
            and target is not None
            and float(reward) > 0.0
            and bool(self.cfg.use_surrogate_post_on_target)
        ):
            post_times = [int(decision.selected_step)]
            used_surrogate_post = True
        if (
            (not post_times)
            and target is not None
            and float(reward) < 0.0
            and bool(use_surrogate_post_on_target_negative)
        ):
            post_times = [int(decision.selected_step)]
            used_surrogate_post = True

        if not post_times:
            return self._empty_event(
                "output",
                reward,
                winner,
                target,
                "No real postsynaptic output spike available; skipped output R-STDP.",
            )
        if reward == 0.0:
            return self._empty_event(
                "output",
                reward,
                winner,
                target,
                "Reward is zero; no gated output R-STDP update applied.",
            )

        effective_reward = float(reward)
        if effective_reward < 0.0:
            effective_reward *= float(self.cfg.output_depression_scale)

        event = self._apply_projection_rstdp(
            layer_name="output",
            controller=getattr(net.output_layer, "controller", None),
            pre_spikes=pre_spikes,
            post_col=post_col,
            post_times=post_times,
            reward=effective_reward,
            winner=winner,
            target=target,
            use_abs_eligibility_on_target_negative=(
                use_abs_eligibility_on_target_negative
            ),
            step_idx=int(net.global_step),
            no_update_message="No output pair crossed eligibility threshold.",
            success_message="Output R-STDP pulse update executed.",
        )
        event.used_surrogate_post = bool(used_surrogate_post)
        if used_surrogate_post:
            event.message += " surrogate target post used at selected_step."
        if (
            bool(self.cfg.anti_target_depression)
            and target is not None
            and float(reward) > 0.0
            and 0 <= int(target) < n_post
        ):
            anti_reward = -abs(float(reward)) * float(self.cfg.anti_target_depression_scale)
            anti_reward *= float(self.cfg.output_depression_scale)
            anti_times = [int(decision.selected_step)]
            for anti_col in range(n_post):
                if int(anti_col) == int(target):
                    continue
                anti_event = self._apply_projection_rstdp(
                    layer_name="output/anti_target",
                    controller=getattr(net.output_layer, "controller", None),
                    pre_spikes=pre_spikes,
                    post_col=int(anti_col),
                    post_times=anti_times,
                    reward=anti_reward,
                    winner=winner,
                    target=int(target),
                    use_abs_eligibility_on_target_negative=True,
                    step_idx=int(net.global_step),
                    no_update_message="No anti-target output pair crossed eligibility threshold.",
                    success_message="Anti-target output R-STDP depression executed.",
                )
                self._merge_event(event, anti_event, pair_prefix=f"anti{anti_col}")
            if event.updated_pairs:
                event.message += " anti-target depression path evaluated."
        return event

    # ------------------------------------------------------------------
    # Optional hidden-layer R-STDP
    # ------------------------------------------------------------------
    def _learn_hidden(self, net: MemristiveSNNNetwork, reward: float) -> RSTDPUpdateEvent:
        decision = net.last_decision
        assert decision is not None

        step_records = decision.step_records
        if not step_records:
            return self._empty_event("hidden", reward, -1, None, "No step records; skipped hidden R-STDP.")

        selected_step = int(decision.selected_step)
        if not (0 <= selected_step < len(step_records)):
            return self._empty_event("hidden", reward, -1, None, "Selected step out of range; skipped hidden R-STDP.")

        hidden_winner = int(step_records[selected_step].hidden_result.winner)
        if hidden_winner < 0:
            return self._empty_event("hidden", reward, -1, None, "No hidden winner at selected step; skipped hidden R-STDP.")

        hidden_dim = int(net.hidden_dim)
        post_spikes = self._stack_step_vectors(
            [rec.hidden_result.spikes for rec in step_records],
            expected_dim=hidden_dim,
            name="hidden post spikes",
            dtype=int,
        )
        post_times = np.flatnonzero(post_spikes[:, hidden_winner] > 0).astype(int).tolist()
        if not post_times:
            return self._empty_event(
                "hidden",
                reward,
                hidden_winner,
                None,
                "No hidden postsynaptic spike available; skipped hidden R-STDP.",
            )
        if reward == 0.0:
            return self._empty_event(
                "hidden",
                reward,
                hidden_winner,
                None,
                "Reward is zero; no gated hidden R-STDP update applied.",
            )

        combined = self._new_accumulator(layer_name="hidden", reward=reward, winner=hidden_winner, target=None)

        # Three-crossbar network: W_in and W_rec are separate physical arrays.
        has_three_crossbar_records = all(hasattr(rec, "recurrent_feedback_used") for rec in step_records)

        if has_three_crossbar_records and hasattr(net, "recurrent_layer"):
            if bool(self.cfg.hidden_update_input_path):
                input_dim = int(net.input_dim)
                input_pre = self._stack_step_vectors(
                    [rec.hidden_input_vector for rec in step_records],
                    expected_dim=input_dim,
                    name="input->hidden pre spikes",
                    dtype=int,
                )
                event_in = self._apply_projection_rstdp(
                    layer_name="hidden/input",
                    controller=getattr(net.hidden_layer, "controller", None),
                    pre_spikes=input_pre,
                    post_col=hidden_winner,
                    post_times=post_times,
                    reward=reward,
                    winner=hidden_winner,
                    target=None,
                    step_idx=int(net.global_step),
                    no_update_message="No input->hidden pair crossed eligibility threshold.",
                    success_message="Input->hidden R-STDP pulse update executed.",
                )
                self._merge_event(combined, event_in, pair_prefix="in")

            if bool(self.cfg.hidden_update_recurrent_path):
                rec_pre = self._stack_step_vectors(
                    [rec.recurrent_feedback_used for rec in step_records],
                    expected_dim=hidden_dim,
                    name="recurrent hidden pre spikes",
                    dtype=int,
                )
                event_rec = self._apply_projection_rstdp(
                    layer_name="hidden/recurrent",
                    controller=getattr(net.recurrent_layer, "controller", None),
                    pre_spikes=rec_pre,
                    post_col=hidden_winner,
                    post_times=post_times,
                    reward=reward,
                    winner=hidden_winner,
                    target=None,
                    step_idx=int(net.global_step),
                    no_update_message="No recurrent hidden pair crossed eligibility threshold.",
                    success_message="Recurrent hidden R-STDP pulse update executed.",
                )
                self._merge_event(combined, event_rec, pair_prefix="rec")

            if not combined.updated_pairs:
                combined.message = "Hidden R-STDP enabled, but no enabled hidden path produced an update."
            else:
                combined.message = "Hidden R-STDP pulse update executed on enabled hidden path(s)."
            return combined

        # Legacy two-crossbar network: hidden_input_vector already contains
        # [input, previous hidden] and is updated through net.hidden_layer.
        legacy_dim = int(getattr(net, "hidden_input_dim", 0))
        if legacy_dim <= 0:
            return self._empty_event("hidden", reward, hidden_winner, None, "Unsupported network hidden structure; skipped hidden R-STDP.")

        legacy_pre = self._stack_step_vectors(
            [rec.hidden_input_vector for rec in step_records],
            expected_dim=legacy_dim,
            name="legacy hidden pre spikes",
            dtype=int,
        )
        return self._apply_projection_rstdp(
            layer_name="hidden",
            controller=getattr(net.hidden_layer, "controller", None),
            pre_spikes=legacy_pre,
            post_col=hidden_winner,
            post_times=post_times,
            reward=reward,
            winner=hidden_winner,
            target=None,
            step_idx=int(net.global_step),
            no_update_message="No hidden pair crossed eligibility threshold.",
            success_message="Hidden R-STDP pulse update executed.",
        )

    # ------------------------------------------------------------------
    # Projection update core
    # ------------------------------------------------------------------
    def _apply_projection_rstdp(
        self,
        *,
        layer_name: str,
        controller: Any,
        pre_spikes: np.ndarray,
        post_col: int,
        post_times: Sequence[int],
        reward: float,
        winner: int,
        target: Optional[int],
        use_abs_eligibility_on_target_negative: Optional[bool] = None,
        step_idx: int,
        no_update_message: str,
        success_message: str,
    ) -> RSTDPUpdateEvent:
        if controller is None or not hasattr(controller, "update_weight"):
            return self._empty_event(layer_name, reward, winner, target, f"{layer_name} controller is not programmable; skipped R-STDP.")

        n_pre = int(pre_spikes.shape[1])
        updated_pairs: List[Tuple[int, int]] = []
        delta_t_records: List[float] = []
        eligibility_values: List[float] = []
        directions: List[int] = []
        actions: List[str] = []
        n_pulses_plus = 0
        n_pulses_minus = 0
        n_refresh = 0
        target_positive_abs_mode = target is not None and float(reward) > 0.0
        target_negative_abs_mode = (
            target is not None
            and float(reward) < 0.0
            and (
                bool(self.cfg.use_abs_eligibility_on_target_negative)
                if use_abs_eligibility_on_target_negative is None
                else bool(use_abs_eligibility_on_target_negative)
            )
        )

        for row in range(n_pre):
            pre_times = np.flatnonzero(pre_spikes[:, row] > 0).astype(int).tolist()
            if not pre_times:
                continue

            best_dt, elig = self._pair_eligibility(pre_times, post_times)
            if abs(elig) < float(self.cfg.eligibility_threshold):
                continue

            effective_elig = (
                abs(elig)
                if target_positive_abs_mode or target_negative_abs_mode
                else elig
            )
            delta_w = self._delta_w(elig=effective_elig, reward=reward)
            direction = self._delta_w_to_direction(delta_w)
            n_pulses = self._delta_w_to_pulse_count(delta_w)
            if direction == 0 or n_pulses == 0:
                continue

            result: ProgrammingResult = controller.update_weight(
                pair_id=(int(row), int(post_col)),
                direction=int(direction),
                step_idx=int(step_idx),
                n_pulses=int(n_pulses),
            )

            updated_pairs.append((int(row), int(post_col)))
            delta_t_records.append(float(best_dt))
            eligibility_values.append(float(effective_elig))
            directions.append(int(direction))
            actions.append(str(result.chosen_action))
            n_pulses_plus += int(result.n_pulses_plus)
            n_pulses_minus += int(result.n_pulses_minus)
            if bool(result.did_refresh):
                n_refresh += 1

        mode_message = (
            " target-positive abs eligibility mode."
            if target_positive_abs_mode
            else ""
        )
        if target_negative_abs_mode:
            mode_message += " target-negative abs eligibility mode."

        return RSTDPUpdateEvent(
            layer_name=str(layer_name),
            updated_pairs=updated_pairs,
            delta_t_records=delta_t_records,
            eligibility_values=eligibility_values,
            directions=directions,
            actions=actions,
            n_pulses_plus=n_pulses_plus,
            n_pulses_minus=n_pulses_minus,
            n_refresh=n_refresh,
            reward=float(reward),
            winner=int(winner),
            target=None if target is None else int(target),
            message=(success_message if updated_pairs else no_update_message) + mode_message,
        )

    # ------------------------------------------------------------------
    # STDP core
    # ------------------------------------------------------------------
    def _pair_eligibility(self, pre_times: Sequence[int], post_times: Sequence[int]) -> Tuple[float, float]:
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
            return float(self.cfg.a_plus * np.exp(-dt / max(float(self.cfg.tau_plus), 1e-12)))
        if dt < 0.0:
            return float(-self.cfg.a_minus * np.exp(-(-dt) / max(float(self.cfg.tau_minus), 1e-12)))
        # Simultaneous pre/post spikes are left neutral to avoid inserting an
        # arbitrary tie rule.
        return 0.0

    def _delta_w(self, elig: float, reward: float) -> float:
        return float(self.cfg.delta_w_scale * float(reward) * float(elig))

    @staticmethod
    def _delta_w_to_direction(delta_w: float) -> int:
        if delta_w > 0.0:
            return +1
        if delta_w < 0.0:
            return -1
        return 0

    def _delta_w_to_pulse_count(self, delta_w: float) -> int:
        mag = abs(float(delta_w))
        if mag <= float(self.cfg.delta_w_min_abs):
            return 0

        pulses = int(np.ceil(mag / max(float(self.cfg.delta_w_per_pulse), 1e-12)))
        pulses = max(int(self.cfg.pulse_base), pulses)
        pulses = min(int(self.cfg.pulse_max), pulses)
        return int(pulses)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        if self.cfg.tau_plus <= 0.0 or self.cfg.tau_minus <= 0.0:
            raise ValueError("RSTDP tau_plus and tau_minus must be positive.")
        if self.cfg.pulse_base < 0:
            raise ValueError("pulse_base must be >= 0")
        if self.cfg.pulse_max < 0:
            raise ValueError("pulse_max must be >= 0")
        if self.cfg.pulse_max < self.cfg.pulse_base:
            raise ValueError("pulse_max must be >= pulse_base")
        if self.cfg.delta_w_per_pulse <= 0.0:
            raise ValueError("delta_w_per_pulse must be positive")
        if self.cfg.eligibility_threshold < 0.0:
            raise ValueError("eligibility_threshold must be >= 0")
        if self.cfg.output_depression_scale < 0.0:
            raise ValueError("output_depression_scale must be >= 0")
        if self.cfg.anti_target_depression_scale < 0.0:
            raise ValueError("anti_target_depression_scale must be >= 0")

    @staticmethod
    def _stack_step_vectors(
        vectors: Sequence[Any],
        *,
        expected_dim: int,
        name: str,
        dtype: Any,
    ) -> np.ndarray:
        arrs = []
        for k, v in enumerate(vectors):
            a = np.asarray(v, dtype=dtype).reshape(-1)
            if a.size != int(expected_dim):
                raise ValueError(f"{name} at step {k} has dim {a.size}, expected {expected_dim}")
            if not np.all(np.isfinite(a.astype(float))):
                raise ValueError(f"{name} at step {k} contains NaN or inf")
            arrs.append(a)
        if not arrs:
            return np.zeros((0, int(expected_dim)), dtype=dtype)
        return np.asarray(arrs, dtype=dtype)

    @staticmethod
    def _empty_event(
        layer_name: str,
        reward: float,
        winner: int,
        target: Optional[int],
        message: str,
    ) -> RSTDPUpdateEvent:
        return RSTDPUpdateEvent(
            layer_name=str(layer_name),
            updated_pairs=[],
            delta_t_records=[],
            eligibility_values=[],
            directions=[],
            actions=[],
            n_pulses_plus=0,
            n_pulses_minus=0,
            n_refresh=0,
            reward=float(reward),
            winner=int(winner),
            target=None if target is None else int(target),
            message=str(message),
        )

    @staticmethod
    def _new_accumulator(layer_name: str, reward: float, winner: int, target: Optional[int]) -> RSTDPUpdateEvent:
        return RSTDPUpdateEvent(
            layer_name=str(layer_name),
            updated_pairs=[],
            delta_t_records=[],
            eligibility_values=[],
            directions=[],
            actions=[],
            n_pulses_plus=0,
            n_pulses_minus=0,
            n_refresh=0,
            reward=float(reward),
            winner=int(winner),
            target=None if target is None else int(target),
            message="",
        )

    @staticmethod
    def _merge_event(dst: RSTDPUpdateEvent, src: RSTDPUpdateEvent, pair_prefix: str) -> None:
        # updated_pairs remains numeric for compatibility with metrics/debugging.
        # The path identity is stored in actions as a prefix.
        dst.updated_pairs.extend(src.updated_pairs)
        dst.delta_t_records.extend(src.delta_t_records)
        dst.eligibility_values.extend(src.eligibility_values)
        dst.directions.extend(src.directions)
        dst.actions.extend([f"{pair_prefix}:{a}" for a in src.actions])
        dst.n_pulses_plus += int(src.n_pulses_plus)
        dst.n_pulses_minus += int(src.n_pulses_minus)
        dst.n_refresh += int(src.n_refresh)


