from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

import config as cfg
from crossbar import DifferentialCrossbar
from encoding import EncoderOutput, SensorSpikeEncoder
from neuron import LearningEvent, MemristiveLIFOutputLayer, NeuronStepResult

ObsType = Union[Dict[str, float], Sequence[float], np.ndarray]

@dataclass
class RecurrentStepRecord:
    """Per-timestep trace for the recurrent network."""

    t: int
    encoder_output: EncoderOutput
    hidden_input_vector: np.ndarray
    hidden_result: NeuronStepResult
    output_result: NeuronStepResult
    recurrent_feedback_used: np.ndarray


@dataclass
class NetworkDecision:
    """Summary of one observation-to-action pass through the recurrent network."""

    action: int
    selected_step: int
    used_fallback: bool
    hidden_pre_spikes_for_learning: np.ndarray
    output_pre_spikes_for_learning: np.ndarray
    integrated_input: np.ndarray
    integrated_hidden_spikes: np.ndarray
    hidden_scores_fallback: Optional[np.ndarray]
    output_scores_fallback: Optional[np.ndarray]
    step_records: List[RecurrentStepRecord]
    encoder_outputs: List[EncoderOutput]


class MemristiveSNNNetwork:
    """Hardware-aware recurrent SNN wrapper for the uploaded codebase.

    Architecture
    ------------
    observation -> encoder -> [input spikes over time]
                                |
                                v
                     concat([input_t, hidden_{t-1}])
                                |
                                v
                        hidden recurrent layer
                                |
                                v
                           output layer
                                |
                                v
                              action

    Notes
    -----
    - Recurrence is implemented as a one-step delayed hidden-state feedback.
      This is physically plausible and easy to schedule in discrete timesteps.
    - No ideal floating-point weight writes are introduced. All learning still
      goes through the pulse-based programming path already implemented in
      ``MemristiveLIFOutputLayer``.
    - The existing neuron layer class does not need to change because recurrence
      is handled by how the network constructs the presynaptic vector for the
      hidden layer.
    """

    def __init__(
        self,
        encoder: SensorSpikeEncoder,
        n_actions: int,
        hidden_dim: Optional[int] = None,
        seed: Optional[int] = None,
        hidden_input_crossbar: Optional[DifferentialCrossbar] = None,
        hidden_layer: Optional[MemristiveLIFOutputLayer] = None,
        output_crossbar: Optional[DifferentialCrossbar] = None,
        output_layer: Optional[MemristiveLIFOutputLayer] = None,
        reset_neuron_state_each_decision: Optional[bool] = None,
        reset_neuron_state_each_episode: Optional[bool] = None,
        force_action_on_no_spike: Optional[bool] = None,
        learn_hidden_layer: Optional[bool] = None,
    ) -> None:
        self.encoder = encoder
        self.n_actions = int(n_actions)
        self.seed = int(getattr(cfg, "SEED", 42) if seed is None else seed)
        self.rng = np.random.default_rng(self.seed)

        self.reset_neuron_state_each_decision = bool(
            getattr(cfg, "NETWORK_RESET_NEURON_STATE_EACH_DECISION", True)
            if reset_neuron_state_each_decision is None
            else reset_neuron_state_each_decision
        )
        self.reset_neuron_state_each_episode = bool(
            getattr(cfg, "NETWORK_RESET_NEURON_STATE_EACH_EPISODE", True)
            if reset_neuron_state_each_episode is None
            else reset_neuron_state_each_episode
        )
        self.force_action_on_no_spike = bool(
            getattr(cfg, "NETWORK_FORCE_ACTION_ON_NO_SPIKE", True)
            if force_action_on_no_spike is None
            else force_action_on_no_spike
        )
        self.learn_hidden_layer = bool(
            getattr(cfg, "NETWORK_LEARN_HIDDEN_LAYER", False)
            if learn_hidden_layer is None
            else learn_hidden_layer
        )

        default_hidden_dim = int(getattr(cfg, "NETWORK_HIDDEN_DIM", 8))
        self.hidden_dim = int(default_hidden_dim if hidden_dim is None else hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be >= 1 for the recurrent network.")

        self.input_dim = int(self.encoder.output_dim)
        self.hidden_input_dim = self.input_dim + self.hidden_dim

        if hidden_input_crossbar is None and hidden_layer is None:
            hidden_input_crossbar = DifferentialCrossbar(
                n_rows=self.hidden_input_dim,
                n_cols=self.hidden_dim,
                seed=self.seed + 101,
            )
        if hidden_layer is None:
            if hidden_input_crossbar is None:
                raise ValueError("hidden_input_crossbar must be provided when hidden_layer is None")
            hidden_layer = MemristiveLIFOutputLayer(crossbar=hidden_input_crossbar, seed=self.seed + 201)

        if output_crossbar is None and output_layer is None:
            output_crossbar = DifferentialCrossbar(
                n_rows=self.hidden_dim,
                n_cols=self.n_actions,
                seed=self.seed + 301,
            )
        if output_layer is None:
            if output_crossbar is None:
                raise ValueError("output_crossbar must be provided when output_layer is None")
            output_layer = MemristiveLIFOutputLayer(crossbar=output_crossbar, seed=self.seed + 401)

        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.hidden_input_crossbar = self.hidden_layer.crossbar
        self.output_crossbar = self.output_layer.crossbar

        if int(self.hidden_input_crossbar.n_rows) != self.hidden_input_dim:
            raise ValueError(
                f"Hidden crossbar rows ({self.hidden_input_crossbar.n_rows}) must equal input_dim + hidden_dim ({self.hidden_input_dim})."
            )
        if int(self.hidden_input_crossbar.n_logical_cols) != self.hidden_dim:
            raise ValueError(
                f"Hidden crossbar logical cols ({self.hidden_input_crossbar.n_logical_cols}) must equal hidden_dim ({self.hidden_dim})."
            )
        if int(self.output_crossbar.n_rows) != self.hidden_dim:
            raise ValueError(
                f"Output crossbar rows ({self.output_crossbar.n_rows}) must equal hidden_dim ({self.hidden_dim})."
            )
        if int(self.output_crossbar.n_logical_cols) != self.n_actions:
            raise ValueError(
                f"Output crossbar logical cols ({self.output_crossbar.n_logical_cols}) must equal n_actions ({self.n_actions})."
            )

        self.prev_hidden_spikes = np.zeros(self.hidden_dim, dtype=float)
        self.episode_index = 0
        self.global_step = 0
        self.last_decision: Optional[NetworkDecision] = None
        self.last_observation: Optional[ObsType] = None
        self.last_reward: Optional[float] = None
        self.last_learning_event_output: Optional[LearningEvent] = None
        self.last_learning_event_hidden: Optional[LearningEvent] = None
        self.action_history: List[int] = []
        self.reward_history: List[float] = []

    # ------------------------------------------------------------------
    # State control
    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        self.episode_index += 1
        self.last_decision = None
        self.last_observation = None
        self.last_reward = None
        self.last_learning_event_output = None
        self.last_learning_event_hidden = None
        self.prev_hidden_spikes.fill(0.0)
        if self.reset_neuron_state_each_episode:
            self.hidden_layer.reset_state()
            self.output_layer.reset_state()

    def reset_network_state(self) -> None:
        self.hidden_layer.reset_state()
        self.output_layer.reset_state()
        self.prev_hidden_spikes.fill(0.0)
        self.last_decision = None
        self.last_observation = None
        self.last_reward = None
        self.last_learning_event_output = None
        self.last_learning_event_hidden = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_window(self, observation: ObsType) -> List[EncoderOutput]:
        window = self.encoder.encode_window(observation)
        if len(window) == 0:
            raise RuntimeError("Encoder produced an empty window.")
        return window

    def _hidden_input_vector(self, input_spikes: Sequence[float]) -> np.ndarray:
        inp = np.asarray(input_spikes, dtype=float).reshape(-1)
        if inp.size != self.input_dim:
            raise ValueError(f"Expected input spike length {self.input_dim}, got {inp.size}")
        return np.concatenate([inp, self.prev_hidden_spikes.astype(float)], axis=0)

    def _fallback_action(self, integrated_hidden_spikes: np.ndarray) -> tuple[int, np.ndarray]:
        scores = self.output_layer._measured_vmm(integrated_hidden_spikes)
        return int(np.argmax(scores)), scores

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def decide(self, observation: ObsType) -> NetworkDecision:
        """Run one complete observation-to-action pass through the recurrent net.

        Primary policy
        --------------
        - First output spike winner over the encoding window.

        Fallback policy
        ---------------
        - If no output spike appears, use measured argmax on the integrated
          hidden spike vector.
        """
        if self.reset_neuron_state_each_decision:
            self.hidden_layer.reset_state()
            self.output_layer.reset_state()
            self.prev_hidden_spikes.fill(0.0)

        self.last_observation = observation
        window = self._prepare_window(observation)

        step_records: List[RecurrentStepRecord] = []
        selected_action = -1
        selected_step = -1
        used_fallback = False
        selected_hidden_pre: Optional[np.ndarray] = None
        selected_output_pre: Optional[np.ndarray] = None

        integrated_input = np.zeros(self.input_dim, dtype=float)
        integrated_hidden_spikes = np.zeros(self.hidden_dim, dtype=float)
        hidden_scores_fallback: Optional[np.ndarray] = None
        output_scores_fallback: Optional[np.ndarray] = None

        for local_t, enc_out in enumerate(window):
            hidden_pre = self._hidden_input_vector(enc_out.spikes)
            feedback_used = self.prev_hidden_spikes.copy()

            hidden_out = self.hidden_layer.step(hidden_pre, step_idx=self.global_step + local_t)
            hidden_spikes = np.asarray(hidden_out.spikes, dtype=float)

            output_out = self.output_layer.step(hidden_spikes, step_idx=self.global_step + local_t)

            step_records.append(
                RecurrentStepRecord(
                    t=int(local_t),
                    encoder_output=enc_out,
                    hidden_input_vector=hidden_pre.copy(),
                    hidden_result=hidden_out,
                    output_result=output_out,
                    recurrent_feedback_used=feedback_used,
                )
            )

            integrated_input += np.asarray(enc_out.spikes, dtype=float)
            integrated_hidden_spikes += hidden_spikes

            if selected_action < 0 and output_out.winner >= 0:
                selected_action = int(output_out.winner)
                selected_step = int(local_t)
                selected_hidden_pre = hidden_pre.copy()
                selected_output_pre = hidden_spikes.copy()

            # One-step delayed recurrence for next timestep.
            self.prev_hidden_spikes = hidden_spikes.copy()

        if selected_action < 0:
            if not self.force_action_on_no_spike:
                raise RuntimeError("No output spike winner found in current window and fallback is disabled.")
            # Optional hidden score inspection for debugging.
            if selected_hidden_pre is None:
                last_hidden_pre = self._hidden_input_vector(np.zeros(self.input_dim, dtype=float))
                hidden_scores_fallback = self.hidden_layer._measured_vmm(last_hidden_pre)
            selected_action, output_scores_fallback = self._fallback_action(integrated_hidden_spikes)
            selected_step = len(window) - 1
            selected_hidden_pre = step_records[-1].hidden_input_vector.copy()
            selected_output_pre = integrated_hidden_spikes.copy()
            used_fallback = True

        if selected_hidden_pre is None or selected_output_pre is None:
            raise RuntimeError("Internal error: no presynaptic vectors selected for learning.")

        self.global_step += len(window)

        decision = NetworkDecision(
            action=int(selected_action),
            selected_step=int(selected_step),
            used_fallback=bool(used_fallback),
            hidden_pre_spikes_for_learning=np.asarray(selected_hidden_pre, dtype=float),
            output_pre_spikes_for_learning=np.asarray(selected_output_pre, dtype=float),
            integrated_input=np.asarray(integrated_input, dtype=float),
            integrated_hidden_spikes=np.asarray(integrated_hidden_spikes, dtype=float),
            hidden_scores_fallback=None if hidden_scores_fallback is None else np.asarray(hidden_scores_fallback, dtype=float),
            output_scores_fallback=None if output_scores_fallback is None else np.asarray(output_scores_fallback, dtype=float),
            step_records=step_records,
            encoder_outputs=window,
        )
        self.last_decision = decision
        self.action_history.append(int(selected_action))
        return decision

    def act(self, observation: ObsType) -> int:
        return int(self.decide(observation).action)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def learn(
        self,
        reward: float,
        target: Optional[int] = None,
        update_all_active_to_target: bool = True,
        punish_wrong_winner: bool = True,
    ) -> Dict[str, Optional[LearningEvent]]:
        """Apply pulse-based learning.

        Default behavior trains the hidden->output projection only, which is the
        safest physically plausible choice without introducing backpropagation or
        BPTT. Hidden-layer learning is optional and uses the same scalar reward
        on the hidden winner for local reinforcement.
        """
        if self.last_decision is None:
            raise RuntimeError("No prior decision available. Call decide()/act() before learn().")

        output_event = self.output_layer.apply_reward_modulated_update(
            pre_spikes=self.last_decision.output_pre_spikes_for_learning,
            winner=int(self.last_decision.action),
            reward=float(reward),
            step_idx=int(self.global_step),
            target=target,
            update_all_active_to_target=bool(update_all_active_to_target),
            punish_wrong_winner=bool(punish_wrong_winner),
        )
        self.last_learning_event_output = output_event

        hidden_event: Optional[LearningEvent] = None
        if self.learn_hidden_layer:
            selected_record = self.last_decision.step_records[self.last_decision.selected_step]
            hidden_winner = int(selected_record.hidden_result.winner)
            if hidden_winner >= 0:
                hidden_event = self.hidden_layer.apply_reward_modulated_update(
                    pre_spikes=self.last_decision.hidden_pre_spikes_for_learning,
                    winner=hidden_winner,
                    reward=float(reward),
                    step_idx=int(self.global_step),
                    target=None,
                    update_all_active_to_target=False,
                    punish_wrong_winner=False,
                )
        self.last_learning_event_hidden = hidden_event

        self.last_reward = float(reward)
        self.reward_history.append(float(reward))
        return {"output": output_event, "hidden": hidden_event}

    def act_and_learn(
        self,
        observation: ObsType,
        reward: float,
        target: Optional[int] = None,
        update_all_active_to_target: bool = True,
        punish_wrong_winner: bool = True,
    ) -> tuple[int, Dict[str, Optional[LearningEvent]]]:
        decision = self.decide(observation)
        events = self.learn(
            reward=reward,
            target=target,
            update_all_active_to_target=update_all_active_to_target,
            punish_wrong_winner=punish_wrong_winner,
        )
        return int(decision.action), events

    # ------------------------------------------------------------------
    # Debug / inspection
    # ------------------------------------------------------------------
    def get_debug_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "episode_index": int(self.episode_index),
            "global_step": int(self.global_step),
            "input_dim": int(self.input_dim),
            "hidden_dim": int(self.hidden_dim),
            "n_actions": int(self.n_actions),
            "prev_hidden_spikes": self.prev_hidden_spikes.copy(),
            "action_history": list(self.action_history),
            "reward_history": list(self.reward_history),
            "last_reward": self.last_reward,
        }
        if self.last_decision is not None:
            state["last_action"] = int(self.last_decision.action)
            state["last_selected_step"] = int(self.last_decision.selected_step)
            state["last_used_fallback"] = bool(self.last_decision.used_fallback)
            state["last_integrated_input"] = self.last_decision.integrated_input.copy()
            state["last_integrated_hidden_spikes"] = self.last_decision.integrated_hidden_spikes.copy()
        if self.last_learning_event_output is not None:
            state["last_learning_event_output"] = self.last_learning_event_output
        if self.last_learning_event_hidden is not None:
            state["last_learning_event_hidden"] = self.last_learning_event_hidden
        return state