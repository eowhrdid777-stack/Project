from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Type, Union

import numpy as np

import config as cfg
from crossbar import DifferentialCrossbar
from encoding import EncoderOutput, SensorSpikeEncoder
from neuron import LearningEvent, MemristiveLIFOutputLayer, NeuronStepResult

try:
    from stm_crossbar import STMCrossbar
except Exception:  # STM backend is optional until stm_crossbar.py is made compatible.
    STMCrossbar = None  # type: ignore[assignment]


ObsType = Union[Dict[str, float], Sequence[float], np.ndarray]


def _cfg(name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


@dataclass
class RecurrentStepRecord:
    """One local timestep record used by learning.py and debugging.

    hidden_input_vector keeps only the input->hidden presynaptic vector.
    recurrent_feedback_used keeps the hidden(t-1)->hidden presynaptic vector.
    This is intentional because the hidden layer is now physically split into
    W_in and W_rec rather than one concatenated crossbar.
    """

    t: int
    encoder_output: EncoderOutput
    hidden_input_vector: np.ndarray
    hidden_result: NeuronStepResult
    output_result: NeuronStepResult
    recurrent_feedback_used: np.ndarray
    input_hidden_current: np.ndarray
    recurrent_hidden_current: np.ndarray
    prev_hidden_spikes_used: np.ndarray
    hidden_trace_used: np.ndarray
    recurrent_input_vector: np.ndarray


@dataclass
class NetworkDecision:
    """Decision result for one observation window."""

    action: int
    selected_step: int
    selection_mode: str
    selected_by_spike: bool
    used_fallback: bool
    no_output_spike: bool
    hidden_pre_spikes_for_learning: np.ndarray
    output_pre_spikes_for_learning: np.ndarray
    integrated_input: np.ndarray
    integrated_hidden_spikes: np.ndarray
    hidden_scores_fallback: Optional[np.ndarray]
    output_scores_fallback: Optional[np.ndarray]
    output_spike_counts: np.ndarray
    first_spike_steps: np.ndarray
    step_records: List[RecurrentStepRecord]
    encoder_outputs: List[EncoderOutput]


class MemristiveSNNNetwork:
    """Hardware-aware recurrent SNN wrapper with three physical crossbar blocks.

    Physical structure
    ------------------
    W_in  : input spikes        -> hidden neurons  (normally LTM/differential)
    W_rec : hidden(t-1) spikes  -> hidden neurons  (normally STM/differential)
    W_out : hidden spikes       -> output neurons  (normally LTM/differential)

    Hidden neuron current is computed as:

        I_hidden = W_in(input_spikes) + W_rec(prev_hidden_spikes)

    The two synaptic currents are summed before a single hidden LIF neuron
    state update.  This means STM can be used only in the recurrent feedback
    path while input->hidden and hidden->output remain LTM-compatible.
    """

    def __init__(
        self,
        encoder: SensorSpikeEncoder,
        n_actions: int,
        hidden_dim: Optional[int] = None,
        seed: Optional[int] = None,
        input_hidden_crossbar: Optional[Any] = None,
        recurrent_hidden_crossbar: Optional[Any] = None,
        output_crossbar: Optional[Any] = None,
        hidden_layer: Optional[MemristiveLIFOutputLayer] = None,
        recurrent_layer: Optional[MemristiveLIFOutputLayer] = None,
        output_layer: Optional[MemristiveLIFOutputLayer] = None,
        reset_neuron_state_each_decision: Optional[bool] = None,
        reset_neuron_state_each_episode: Optional[bool] = None,
        force_action_on_no_spike: Optional[bool] = None,
        learn_hidden_layer: Optional[bool] = None,
    ) -> None:
        self.encoder = encoder
        self.n_actions = int(n_actions)
        if self.n_actions <= 0:
            raise ValueError("n_actions must be >= 1")

        self.seed = int(_cfg("SEED", 42) if seed is None else seed)
        self.rng = np.random.default_rng(self.seed)

        self.reset_neuron_state_each_decision = bool(
            _cfg("NETWORK_RESET_NEURON_STATE_EACH_DECISION", True)
            if reset_neuron_state_each_decision is None
            else reset_neuron_state_each_decision
        )
        self.reset_neuron_state_each_episode = bool(
            _cfg("NETWORK_RESET_NEURON_STATE_EACH_EPISODE", True)
            if reset_neuron_state_each_episode is None
            else reset_neuron_state_each_episode
        )
        self.force_action_on_no_spike = bool(
            _cfg("NETWORK_FORCE_ACTION_ON_NO_SPIKE", True)
            if force_action_on_no_spike is None
            else force_action_on_no_spike
        )
        self.output_action_selection_mode = self._normalize_action_selection_mode(
            _cfg("OUTPUT_ACTION_SELECTION_MODE", "first_spike")
        )
        self.stm_recurrent_current_scale = float(_cfg("STM_RECURRENT_CURRENT_SCALE", 1.0))
        self.mask_recurrent_hidden_current = False
        self.enable_cross_decision_hidden_trace = bool(
            _cfg("ENABLE_CROSS_DECISION_HIDDEN_TRACE", True)
        )
        self.hidden_trace_decay = float(_cfg("HIDDEN_TRACE_DECAY", 0.85))
        self.hidden_trace_input_scale = float(_cfg("HIDDEN_TRACE_INPUT_SCALE", 1.0))
        self.hidden_trace_clip_max = float(_cfg("HIDDEN_TRACE_CLIP_MAX", 1.0))
        self.hidden_trace_reset_each_episode = bool(
            _cfg("HIDDEN_TRACE_RESET_EACH_EPISODE", True)
        )
        tie_order = np.random.default_rng(self.seed + 9091).permutation(self.n_actions)
        self.action_tiebreak_rank = np.empty(self.n_actions, dtype=int)
        for rank, action in enumerate(tie_order):
            self.action_tiebreak_rank[int(action)] = int(rank)
        self.learn_hidden_layer = bool(
            _cfg("NETWORK_LEARN_HIDDEN_LAYER", False)
            if learn_hidden_layer is None
            else learn_hidden_layer
        )

        self.hidden_dim = int(_cfg("NETWORK_HIDDEN_DIM", 8) if hidden_dim is None else hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be >= 1")

        self.input_dim = int(self.encoder.output_dim)
        if self.input_dim <= 0:
            raise ValueError("encoder.output_dim must be >= 1")

        # Kept for backward/debug compatibility.  The actual hidden computation
        # is no longer one concatenated crossbar; it is W_in + W_rec.
        self.hidden_input_dim = self.input_dim + self.hidden_dim

        # Backend selection.
        # Defaults implement the intended architecture:
        # input->hidden=LTM, recurrent hidden->hidden=STM, hidden->output=LTM.
        self.input_hidden_crossbar_type = str(
            _cfg("NETWORK_INPUT_HIDDEN_CROSSBAR_TYPE", _cfg("NETWORK_CROSSBAR_TYPE", "differential"))
        ).lower()
        self.recurrent_hidden_crossbar_type = str(
            _cfg("NETWORK_RECURRENT_HIDDEN_CROSSBAR_TYPE", "stm")
        ).lower()
        self.output_crossbar_type = str(
            _cfg("NETWORK_OUTPUT_CROSSBAR_TYPE", _cfg("NETWORK_CROSSBAR_TYPE", "differential"))
        ).lower()

        # ------------------------------------------------------------
        # Crossbar construction
        # ------------------------------------------------------------
        if input_hidden_crossbar is None:
            input_hidden_crossbar = self._make_crossbar(
                kind=self.input_hidden_crossbar_type,
                n_rows=self.input_dim,
                n_cols=self.hidden_dim,
                seed=self.seed + 101,
                name="input_hidden_crossbar",
            )
        if recurrent_hidden_crossbar is None:
            recurrent_hidden_crossbar = self._make_crossbar(
                kind=self.recurrent_hidden_crossbar_type,
                n_rows=self.hidden_dim,
                n_cols=self.hidden_dim,
                seed=self.seed + 151,
                name="recurrent_hidden_crossbar",
            )
        if output_crossbar is None:
            output_crossbar = self._make_crossbar(
                kind=self.output_crossbar_type,
                n_rows=self.hidden_dim,
                n_cols=self.n_actions,
                seed=self.seed + 301,
                name="output_crossbar",
            )

        self._validate_crossbar_interface(input_hidden_crossbar, "input_hidden_crossbar")
        self._validate_crossbar_interface(recurrent_hidden_crossbar, "recurrent_hidden_crossbar")
        self._validate_crossbar_interface(output_crossbar, "output_crossbar")

        # ------------------------------------------------------------
        # Layer construction
        # ------------------------------------------------------------
        # hidden_layer carries the actual hidden neuron state.  Its crossbar is
        # W_in.  W_rec is read through recurrent_layer, but recurrent_layer's
        # neuron state is not used.
        if hidden_layer is None:
            hidden_layer = MemristiveLIFOutputLayer(
                crossbar=input_hidden_crossbar,
                seed=self.seed + 201,
                base_threshold=float(
                    _cfg(
                        "NETWORK_HIDDEN_BASE_THRESHOLD",
                        _cfg("NEURON_BASE_THRESHOLD", 3.5e-6),
                    )
                ),
            )
        if recurrent_layer is None:
            recurrent_layer = MemristiveLIFOutputLayer(crossbar=recurrent_hidden_crossbar, seed=self.seed + 251)
        if output_layer is None:
            output_layer = MemristiveLIFOutputLayer(
                crossbar=output_crossbar,
                seed=self.seed + 401,
                base_threshold=float(
                    _cfg(
                        "NETWORK_OUTPUT_BASE_THRESHOLD",
                        _cfg("NEURON_BASE_THRESHOLD", 3.5e-6),
                    )
                ),
                inhibit_on_spike=bool(
                    _cfg("NETWORK_OUTPUT_ENABLE_WTA", _cfg("NEURON_ENABLE_WTA", True))
                ),
                lateral_inhibition_strength=float(
                    _cfg(
                        "NETWORK_OUTPUT_LATERAL_INHIBITION",
                        _cfg("NEURON_LATERAL_INHIBITION", 0.0),
                    )
                ),
            )

        self.hidden_layer = hidden_layer
        self.recurrent_layer = recurrent_layer
        self.output_layer = output_layer

        self.input_hidden_crossbar = self.hidden_layer.crossbar
        self.recurrent_hidden_crossbar = self.recurrent_layer.crossbar
        self.output_crossbar = self.output_layer.crossbar

        self._validate_dimensions()

        # Recurrent state.
        self.prev_hidden_spikes = np.zeros(self.hidden_dim, dtype=float)
        self.hidden_trace = np.zeros(self.hidden_dim, dtype=float)
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
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _crossbar_class(kind: str) -> Type[Any]:
        kind = str(kind).lower()
        if kind in {"differential", "ltm", "fett", "fetft", "memristive"}:
            return DifferentialCrossbar
        if kind == "stm":
            if STMCrossbar is None:
                raise ImportError("NETWORK_*_CROSSBAR_TYPE='stm' requires stm_crossbar.py with STMCrossbar.")
            return STMCrossbar  # type: ignore[return-value]
        raise ValueError(f"Unknown crossbar backend: {kind!r}")

    @classmethod
    def _make_crossbar(cls, kind: str, n_rows: int, n_cols: int, seed: int, name: str) -> Any:
        cb_cls = cls._crossbar_class(kind)
        try:
            cb = cb_cls(n_rows=int(n_rows), n_cols=int(n_cols), seed=int(seed))
        except TypeError:
            cb = cb_cls(int(n_rows), int(n_cols), seed=int(seed))
        cls._validate_crossbar_interface(cb, name)
        return cb

    @staticmethod
    def _validate_crossbar_interface(crossbar: Any, name: str) -> None:
        missing = []
        for attr in ("n_rows", "n_logical_cols"):
            if not hasattr(crossbar, attr):
                missing.append(attr)
        for method in ("read_pair", "apply_pulse", "get_pair_bounds"):
            if not callable(getattr(crossbar, method, None)):
                missing.append(method)
        if missing:
            raise TypeError(
                f"{name} must expose a differential-pair compatible interface. "
                f"Missing: {', '.join(missing)}. "
                "For STM, implement it as plus/minus STM cells per logical synapse."
            )

    def _validate_dimensions(self) -> None:
        checks = [
            ("input_hidden_crossbar.n_rows", int(self.input_hidden_crossbar.n_rows), self.input_dim),
            ("input_hidden_crossbar.n_logical_cols", int(self.input_hidden_crossbar.n_logical_cols), self.hidden_dim),
            ("recurrent_hidden_crossbar.n_rows", int(self.recurrent_hidden_crossbar.n_rows), self.hidden_dim),
            ("recurrent_hidden_crossbar.n_logical_cols", int(self.recurrent_hidden_crossbar.n_logical_cols), self.hidden_dim),
            ("output_crossbar.n_rows", int(self.output_crossbar.n_rows), self.hidden_dim),
            ("output_crossbar.n_logical_cols", int(self.output_crossbar.n_logical_cols), self.n_actions),
        ]
        for label, got, expected in checks:
            if got != int(expected):
                raise ValueError(f"{label} must be {expected}, got {got}.")

        if int(self.hidden_layer.n_inputs) != self.input_dim or int(self.hidden_layer.n_neurons) != self.hidden_dim:
            raise ValueError("hidden_layer must map input_dim -> hidden_dim.")
        if int(self.recurrent_layer.n_inputs) != self.hidden_dim or int(self.recurrent_layer.n_neurons) != self.hidden_dim:
            raise ValueError("recurrent_layer must map hidden_dim -> hidden_dim.")
        if int(self.output_layer.n_inputs) != self.hidden_dim or int(self.output_layer.n_neurons) != self.n_actions:
            raise ValueError("output_layer must map hidden_dim -> n_actions.")

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
        if self.hidden_trace_reset_each_episode:
            self.hidden_trace.fill(0.0)
        if self.reset_neuron_state_each_episode:
            reset_threshold_devices = bool(_cfg("NEURON_RESET_THRESHOLD_ADAPTATION_EACH_EPISODE", False))
            self.hidden_layer.reset_state(reset_threshold_devices=reset_threshold_devices)
            self.recurrent_layer.reset_state(reset_threshold_devices=reset_threshold_devices)
            self.output_layer.reset_state(reset_threshold_devices=reset_threshold_devices)

    def reset_network_state(self, reset_threshold_devices: Optional[bool] = None) -> None:
        if reset_threshold_devices is None:
            reset_threshold_devices = bool(_cfg("NEURON_RESET_THRESHOLD_ADAPTATION_EACH_EPISODE", False))
        self.hidden_layer.reset_state(reset_threshold_devices=bool(reset_threshold_devices))
        self.recurrent_layer.reset_state(reset_threshold_devices=bool(reset_threshold_devices))
        self.output_layer.reset_state(reset_threshold_devices=bool(reset_threshold_devices))
        self.prev_hidden_spikes.fill(0.0)
        self.hidden_trace.fill(0.0)
        self.last_decision = None
        self.last_observation = None
        self.last_reward = None
        self.last_learning_event_output = None
        self.last_learning_event_hidden = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_vector(x: Sequence[float], expected: int, name: str) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.size != int(expected):
            raise ValueError(f"{name} length must be {expected}, got {arr.size}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or inf.")
        return arr

    def _prepare_window(self, observation: ObsType) -> List[EncoderOutput]:
        window = self.encoder.encode_window(observation)
        if len(window) == 0:
            raise RuntimeError("Encoder produced an empty window.")
        if isinstance(observation, dict):
            silenced = observation.get("_silenced_features", ())
            if isinstance(silenced, str):
                silenced = (silenced,)
            if silenced:
                window = self._silence_encoded_features(window, silenced)
        for k, enc in enumerate(window):
            spikes = self._validate_vector(enc.spikes, self.input_dim, f"encoder_output[{k}].spikes")
            if spikes.shape != np.asarray(enc.spikes).reshape(-1).shape:
                pass
        return window

    def _silence_encoded_features(
        self,
        window: List[EncoderOutput],
        feature_names: Sequence[str],
    ) -> List[EncoderOutput]:
        raw_names = list(getattr(self.encoder, "feature_names", []))
        neurons_per_feature = int(getattr(self.encoder, "neurons_per_feature", 1))
        mask = np.zeros(self.input_dim, dtype=bool)
        for name in feature_names:
            if str(name) not in raw_names:
                continue
            idx = raw_names.index(str(name))
            start = idx * neurons_per_feature
            end = min(start + neurons_per_feature, self.input_dim)
            mask[start:end] = True
        if not np.any(mask):
            return window

        silenced_window: List[EncoderOutput] = []
        for enc in window:
            spikes = np.asarray(enc.spikes, dtype=float).copy()
            firing_rates = np.asarray(enc.firing_rates, dtype=float).copy()
            spike_times = np.asarray(enc.spike_times, dtype=float).copy()
            analog_values = np.asarray(enc.analog_values, dtype=float).copy()
            spikes[mask] = 0.0
            firing_rates[mask] = 0.0
            spike_times[mask] = np.inf
            analog_values[mask] = 0.0
            silenced_window.append(
                EncoderOutput(
                    spikes=spikes,
                    firing_rates=firing_rates,
                    spike_times=spike_times,
                    analog_values=analog_values,
                    feature_names=list(enc.feature_names),
                    mode=str(enc.mode),
                )
            )
        return silenced_window

    def _recurrent_input_vector(self) -> np.ndarray:
        prev = self._validate_vector(
            self.prev_hidden_spikes,
            self.hidden_dim,
            "prev_hidden_spikes",
        )
        if not self.enable_cross_decision_hidden_trace:
            return prev.copy()
        trace = self._validate_vector(
            self.hidden_trace,
            self.hidden_dim,
            "hidden_trace",
        )
        recurrent_input = prev + float(self.hidden_trace_input_scale) * trace
        return self._validate_vector(
            recurrent_input,
            self.hidden_dim,
            "recurrent_input_vector",
        )

    def _update_hidden_trace_after_decision(self, integrated_hidden_spikes: np.ndarray, window_len: int) -> None:
        if not self.enable_cross_decision_hidden_trace:
            return
        spikes = self._validate_vector(
            integrated_hidden_spikes,
            self.hidden_dim,
            "integrated_hidden_spikes",
        )
        window = max(1, int(window_len))
        normalized_spikes = spikes / float(window)
        next_trace = float(self.hidden_trace_decay) * self.hidden_trace + normalized_spikes
        clip_max = float(self.hidden_trace_clip_max)
        if clip_max > 0.0:
            next_trace = np.clip(next_trace, 0.0, clip_max)
        self.hidden_trace = self._validate_vector(next_trace, self.hidden_dim, "hidden_trace")

    def _hidden_currents(self, input_spikes: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inp = self._validate_vector(input_spikes, self.input_dim, "input_spikes")
        rec = self._recurrent_input_vector()
        input_current = np.asarray(self.hidden_layer._measured_vmm(inp), dtype=float).reshape(-1)
        recurrent_current = np.asarray(self.recurrent_layer._measured_vmm(rec), dtype=float).reshape(-1)
        input_current = self._validate_vector(input_current, self.hidden_dim, "input_hidden_current")
        recurrent_current = self._validate_vector(recurrent_current, self.hidden_dim, "recurrent_hidden_current")
        recurrent_current = recurrent_current * float(self.stm_recurrent_current_scale)
        if bool(self.mask_recurrent_hidden_current):
            recurrent_current = np.zeros_like(recurrent_current)
        total_current = input_current + recurrent_current
        total_current = self._validate_vector(total_current, self.hidden_dim, "total_hidden_current")
        return input_current, recurrent_current, total_current

    def _step_layer_from_current(
        self,
        layer: MemristiveLIFOutputLayer,
        synaptic_current: Sequence[float],
        step_idx: int,
    ) -> NeuronStepResult:
        """Advance one LIF layer using a precomputed synaptic current vector.

        This mirrors MemristiveLIFOutputLayer.step(), but bypasses its internal
        single-crossbar VMM so W_in and W_rec currents can be summed first.
        """
        syn = self._validate_vector(synaptic_current, layer.n_neurons, "synaptic_current")
        thresholds_before = np.asarray(layer.get_thresholds(), dtype=float).reshape(-1)
        thresholds_before = self._validate_vector(thresholds_before, layer.n_neurons, "thresholds")

        layer.spike_trace *= layer.trace_decay
        spikes = np.zeros(layer.n_neurons, dtype=np.int8)

        for j in range(layer.n_neurons):
            if layer.refractory[j] > 0:
                layer.refractory[j] -= 1
                layer.vmem[j] = layer.reset_voltage
                continue

            layer.vmem[j] = layer.membrane_decay * layer.vmem[j] + layer.input_gain * syn[j]
            if layer.vmem[j] >= thresholds_before[j]:
                spikes[j] = 1

        spiking_idx = np.flatnonzero(spikes > 0)
        winner = -1

        if spiking_idx.size > 0:
            winner = int(spiking_idx[np.argmax(layer.vmem[spiking_idx])])

            if layer.inhibit_on_spike:
                spikes[:] = 0
                spikes[winner] = 1
                effective_spikes = np.array([winner], dtype=int)
            else:
                effective_spikes = spiking_idx.astype(int)

            for j in effective_spikes:
                layer.spike_trace[j] += 1.0
                layer.refractory[j] = layer.refractory_steps
                layer.last_spike_step[j] = int(step_idx)
                layer.vmem[j] = layer.reset_voltage

                if layer.enable_threshold_adaptation and layer.threshold_pot_pulses_on_spike > 0:
                    layer.threshold_devices[j].apply_pot_pulse(layer.threshold_pot_pulses_on_spike)

            if layer.inhibit_on_spike and layer.lateral_inhibition_strength > 0.0:
                mask = np.ones(layer.n_neurons, dtype=bool)
                mask[winner] = False
                layer.vmem[mask] -= layer.lateral_inhibition_strength * np.maximum(syn[mask], 0.0)

        layer._recover_threshold_devices(int(step_idx))

        return NeuronStepResult(
            synaptic_currents=syn.copy(),
            membrane_potentials=layer.vmem.copy(),
            thresholds=np.asarray(layer.get_thresholds(), dtype=float).copy(),
            spikes=spikes.copy(),
            spike_trace=layer.spike_trace.copy(),
            refractory_counters=layer.refractory.copy(),
            winner=int(winner),
        )

    def _fallback_action(self, integrated_hidden_spikes: np.ndarray) -> tuple[int, np.ndarray]:
        scores = np.asarray(self.output_layer._measured_vmm(integrated_hidden_spikes), dtype=float).reshape(-1)
        scores = self._validate_vector(scores, self.n_actions, "output_fallback_scores")
        return int(np.argmax(scores)), scores

    @staticmethod
    def _normalize_action_selection_mode(mode: Any) -> str:
        text = str(mode).strip().lower()
        aliases = {
            "first": "first_spike",
            "latency": "first_spike",
            "count": "spike_count",
            "rate": "spike_count",
            "hybrid": "hybrid_spike_count_latency",
            "hybrid_spike_count": "hybrid_spike_count_latency",
        }
        text = aliases.get(text, text)
        valid = {"first_spike", "spike_count", "hybrid_spike_count_latency"}
        if text not in valid:
            raise ValueError(
                "OUTPUT_ACTION_SELECTION_MODE must be one of "
                f"{sorted(valid)}, got {mode!r}"
            )
        return text

    def set_output_action_selection_mode(self, mode: Any) -> None:
        self.output_action_selection_mode = self._normalize_action_selection_mode(mode)

    def set_recurrent_current_diagnostic_mask(self, masked: bool) -> None:
        """Mask W_rec read current for eval diagnostics without changing conductance."""
        self.mask_recurrent_hidden_current = bool(masked)

    def set_recurrent_current_scale(self, scale: float) -> None:
        """Set recurrent read-current gain for diagnostics or circuit gain sweeps."""
        self.stm_recurrent_current_scale = float(scale)

    def set_hidden_trace_params(
        self,
        *,
        decay: Optional[float] = None,
        input_scale: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        if decay is not None:
            self.hidden_trace_decay = float(decay)
        if input_scale is not None:
            self.hidden_trace_input_scale = float(input_scale)
        if enabled is not None:
            self.enable_cross_decision_hidden_trace = bool(enabled)

    def _break_action_tie(self, candidates: np.ndarray) -> int:
        candidates = np.asarray(candidates, dtype=int).reshape(-1)
        if candidates.size == 1:
            return int(candidates[0])
        ranks = self.action_tiebreak_rank[candidates]
        return int(candidates[int(np.argmin(ranks))])

    def _select_from_counts_and_latency(
        self,
        spike_counts: np.ndarray,
        first_spike_steps: np.ndarray,
    ) -> tuple[int, int]:
        counts = self._validate_vector(spike_counts, self.n_actions, "output_spike_counts")
        first_steps = np.asarray(first_spike_steps, dtype=float).reshape(-1)
        if first_steps.size != self.n_actions:
            raise ValueError(
                f"first_spike_steps must have size {self.n_actions}, got {first_steps.size}."
            )
        if np.any(np.isnan(first_steps)):
            raise ValueError("first_spike_steps contains NaN.")
        valid = np.flatnonzero(counts > 0)
        if valid.size == 0:
            return -1, -1

        mode = self._normalize_action_selection_mode(self.output_action_selection_mode)
        finite_steps = np.where(np.isfinite(first_steps), first_steps, np.inf)

        if mode == "spike_count":
            max_count = np.max(counts[valid])
            candidates = valid[np.isclose(counts[valid], max_count)]
            min_step = np.min(finite_steps[candidates])
            candidates = candidates[np.isclose(finite_steps[candidates], min_step)]
            action = self._break_action_tie(candidates)
            return int(action), int(finite_steps[action])

        count_weight = float(_cfg("OUTPUT_COUNT_WEIGHT", 1.0))
        latency_weight = float(_cfg("OUTPUT_LATENCY_WEIGHT", 0.05))
        scores = np.full(self.n_actions, -np.inf, dtype=float)
        scores[valid] = counts[valid] * count_weight - finite_steps[valid] * latency_weight
        max_score = np.max(scores[valid])
        candidates = valid[np.isclose(scores[valid], max_score)]
        min_step = np.min(finite_steps[candidates])
        candidates = candidates[np.isclose(finite_steps[candidates], min_step)]
        action = self._break_action_tie(candidates)
        return int(action), int(finite_steps[action])

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def decide(self, observation: ObsType) -> NetworkDecision:
        if self.reset_neuron_state_each_decision:
            self.hidden_layer.reset_state()
            self.recurrent_layer.reset_state()
            self.output_layer.reset_state()
            self.prev_hidden_spikes.fill(0.0)

        self.last_observation = observation
        window = self._prepare_window(observation)

        step_records: List[RecurrentStepRecord] = []
        first_spike_action = -1
        first_spike_step = -1
        used_fallback = False

        integrated_input = np.zeros(self.input_dim, dtype=float)
        integrated_hidden_spikes = np.zeros(self.hidden_dim, dtype=float)
        hidden_scores_fallback: Optional[np.ndarray] = None
        output_scores_fallback: Optional[np.ndarray] = None
        output_spike_counts = np.zeros(self.n_actions, dtype=float)
        first_spike_steps = np.full(self.n_actions, np.inf, dtype=float)
        selection_mode = self._normalize_action_selection_mode(
            self.output_action_selection_mode
        )

        for local_t, enc_out in enumerate(window):
            input_spikes = self._validate_vector(enc_out.spikes, self.input_dim, "enc_out.spikes")
            prev_hidden_used = self.prev_hidden_spikes.copy()
            hidden_trace_used = self.hidden_trace.copy()
            feedback_used = self._recurrent_input_vector()

            i_in, i_rec, i_hidden = self._hidden_currents(input_spikes)
            hidden_out = self._step_layer_from_current(
                self.hidden_layer,
                i_hidden,
                step_idx=self.global_step + local_t,
            )
            hidden_spikes = self._validate_vector(hidden_out.spikes, self.hidden_dim, "hidden_spikes")

            output_out = self.output_layer.step(hidden_spikes, step_idx=self.global_step + local_t)
            output_spikes = self._validate_vector(output_out.spikes, self.n_actions, "output_spikes")

            step_records.append(
                RecurrentStepRecord(
                    t=int(local_t),
                    encoder_output=enc_out,
                    hidden_input_vector=input_spikes.copy(),
                    hidden_result=hidden_out,
                    output_result=output_out,
                    recurrent_feedback_used=feedback_used,
                    input_hidden_current=i_in.copy(),
                    recurrent_hidden_current=i_rec.copy(),
                    prev_hidden_spikes_used=prev_hidden_used,
                    hidden_trace_used=hidden_trace_used,
                    recurrent_input_vector=feedback_used.copy(),
                )
            )

            integrated_input += input_spikes
            integrated_hidden_spikes += hidden_spikes
            output_spike_counts += output_spikes
            newly_spiked = np.flatnonzero((output_spikes > 0) & ~np.isfinite(first_spike_steps))
            for action_idx in newly_spiked:
                first_spike_steps[int(action_idx)] = float(local_t)

            if first_spike_action < 0 and output_out.winner >= 0:
                first_spike_action = int(output_out.winner)
                first_spike_step = int(local_t)

            self.prev_hidden_spikes = hidden_spikes.copy()

        self._update_hidden_trace_after_decision(integrated_hidden_spikes, len(window))

        if selection_mode == "first_spike":
            selected_action = int(first_spike_action)
            selected_step = int(first_spike_step)
        else:
            selected_action, selected_step = self._select_from_counts_and_latency(
                output_spike_counts,
                first_spike_steps,
            )

        no_output_spike = bool(selected_action < 0)
        if selected_action < 0:
            # Debug score only: what the hidden neuron would receive from the
            # accumulated input path without recurrent feedback.  The output
            # score is still measured through the output crossbar; when
            # fallback is disabled it is logged but not used to choose action.
            hidden_scores_fallback = np.asarray(self.hidden_layer._measured_vmm(integrated_input), dtype=float)
            fallback_action, output_scores_fallback = self._fallback_action(integrated_hidden_spikes)
            selected_step = len(window) - 1
            selected_hidden_pre = integrated_input.copy()
            selected_output_pre = integrated_hidden_spikes.copy()
            if self.force_action_on_no_spike:
                selected_action = int(fallback_action)
                used_fallback = True
            else:
                selected_action = -1
                used_fallback = False

        selected_hidden_pre: Optional[np.ndarray] = None
        selected_output_pre: Optional[np.ndarray] = None
        if no_output_spike:
            selected_hidden_pre = integrated_input.copy()
            selected_output_pre = integrated_hidden_spikes.copy()
        elif 0 <= selected_step < len(step_records):
            selected_hidden_pre = step_records[selected_step].hidden_input_vector.copy()
            selected_output_pre = step_records[selected_step].hidden_result.spikes.copy()

        if selected_hidden_pre is None or selected_output_pre is None:
            raise RuntimeError("Internal error: missing presynaptic vectors for learning.")

        self.global_step += len(window)

        decision = NetworkDecision(
            action=int(selected_action),
            selected_step=int(selected_step),
            selection_mode=str(selection_mode),
            selected_by_spike=bool(
                int(selected_action) >= 0 and not used_fallback and not no_output_spike
            ),
            used_fallback=bool(used_fallback),
            no_output_spike=bool(no_output_spike),
            hidden_pre_spikes_for_learning=np.asarray(selected_hidden_pre, dtype=float),
            output_pre_spikes_for_learning=np.asarray(selected_output_pre, dtype=float),
            integrated_input=np.asarray(integrated_input, dtype=float),
            integrated_hidden_spikes=np.asarray(integrated_hidden_spikes, dtype=float),
            hidden_scores_fallback=None if hidden_scores_fallback is None else np.asarray(hidden_scores_fallback, dtype=float),
            output_scores_fallback=None if output_scores_fallback is None else np.asarray(output_scores_fallback, dtype=float),
            output_spike_counts=np.asarray(output_spike_counts, dtype=float),
            first_spike_steps=np.asarray(first_spike_steps, dtype=float),
            step_records=step_records,
            encoder_outputs=window,
        )
        self.last_decision = decision
        self.action_history.append(int(selected_action))
        return decision

    def act(self, observation: ObsType) -> int:
        return int(self.decide(observation).action)
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
            "hidden_input_dim_legacy": int(self.hidden_input_dim),
            "input_hidden_crossbar_type": self.input_hidden_crossbar_type,
            "recurrent_hidden_crossbar_type": self.recurrent_hidden_crossbar_type,
            "output_crossbar_type": self.output_crossbar_type,
            "reset_neuron_state_each_decision": bool(self.reset_neuron_state_each_decision),
            "stm_recurrent_current_scale": float(self.stm_recurrent_current_scale),
            "mask_recurrent_hidden_current": bool(self.mask_recurrent_hidden_current),
            "enable_cross_decision_hidden_trace": bool(self.enable_cross_decision_hidden_trace),
            "hidden_trace_decay": float(self.hidden_trace_decay),
            "hidden_trace_input_scale": float(self.hidden_trace_input_scale),
            "hidden_trace": self.hidden_trace.copy(),
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
