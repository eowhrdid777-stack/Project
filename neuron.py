from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

import config as cfg
from conductance_modulation import ConductanceModulationController, ProgrammingResult
from crossbar import DifferentialCrossbar
from device_model import MemristorDevice


@dataclass
class NeuronStepResult:
    """Container for one simulation step of the output-neuron layer."""

    synaptic_currents: np.ndarray
    membrane_potentials: np.ndarray
    thresholds: np.ndarray
    spikes: np.ndarray
    spike_trace: np.ndarray
    refractory_counters: np.ndarray
    winner: int


@dataclass
class LearningEvent:
    """Summary of physically plausible synaptic programming actions."""

    updated_pairs: List[tuple[int, int]]
    directions: List[int]
    actions: List[str]
    n_pulses_plus: int
    n_pulses_minus: int
    n_refresh: int
    reward: float
    winner: int
    target: Optional[int]
    message: str


class MemristiveLIFOutputLayer:
    """Measured-read output neuron layer compatible with the uploaded codebase.

    Design choices
    --------------
    1. Synaptic accumulation uses *measured* differential pair reads only.
       No direct ideal-weight access is used during inference.
    2. Online programming uses ConductanceModulationController so all weight
       updates are executed as one-sided pulse operations or refresh/remap,
       matching the physical FeTFT programming style already implemented in the
       project.
    3. Optional threshold adaptation is stored in a small per-neuron memristive
       state device. Threshold changes happen through pulses only, not by
       directly setting an arbitrary value.

    The layer is therefore intentionally conservative and hardware-aware rather
    than mathematically idealized.
    """

    def __init__(
        self,
        crossbar: DifferentialCrossbar,
        seed: Optional[int] = None,
        membrane_decay: Optional[float] = None,
        input_gain: Optional[float] = None,
        base_threshold: Optional[float] = None,
        reset_voltage: Optional[float] = None,
        refractory_steps: Optional[int] = None,
        trace_decay: Optional[float] = None,
        inhibit_on_spike: Optional[bool] = None,
        lateral_inhibition_strength: Optional[float] = None,
        enable_threshold_adaptation: Optional[bool] = None,
        threshold_scale: Optional[float] = None,
        threshold_pot_pulses_on_spike: Optional[int] = None,
        threshold_dep_pulses_recovery: Optional[int] = None,
        threshold_recovery_period: Optional[int] = None,
    ) -> None:
        self.crossbar = crossbar
        self.controller = ConductanceModulationController(crossbar)
        self.rng = np.random.default_rng(getattr(cfg, "SEED", 42) if seed is None else seed)

        self.n_inputs = int(crossbar.n_rows)
        self.n_neurons = int(crossbar.n_logical_cols)

        self.membrane_decay = float(getattr(cfg, "NEURON_MEMBRANE_DECAY", 0.90 if membrane_decay is None else membrane_decay)) if membrane_decay is None else float(membrane_decay)
        self.input_gain = float(getattr(cfg, "NEURON_INPUT_GAIN", 1.0 if input_gain is None else input_gain)) if input_gain is None else float(input_gain)
        self.base_threshold = float(getattr(cfg, "NEURON_BASE_THRESHOLD", 8.0e-6 if base_threshold is None else base_threshold)) if base_threshold is None else float(base_threshold)
        self.reset_voltage = float(getattr(cfg, "NEURON_RESET_VOLTAGE", 0.0 if reset_voltage is None else reset_voltage)) if reset_voltage is None else float(reset_voltage)
        self.refractory_steps = int(getattr(cfg, "NEURON_REFRACTORY_STEPS", 1 if refractory_steps is None else refractory_steps)) if refractory_steps is None else int(refractory_steps)
        self.trace_decay = float(getattr(cfg, "NEURON_TRACE_DECAY", 0.85 if trace_decay is None else trace_decay)) if trace_decay is None else float(trace_decay)
        self.inhibit_on_spike = bool(getattr(cfg, "NEURON_ENABLE_WTA", True if inhibit_on_spike is None else inhibit_on_spike)) if inhibit_on_spike is None else bool(inhibit_on_spike)
        self.lateral_inhibition_strength = float(getattr(cfg, "NEURON_LATERAL_INHIBITION", 0.5 if lateral_inhibition_strength is None else lateral_inhibition_strength)) if lateral_inhibition_strength is None else float(lateral_inhibition_strength)

        self.enable_threshold_adaptation = bool(getattr(cfg, "NEURON_ENABLE_THRESHOLD_ADAPTATION", True if enable_threshold_adaptation is None else enable_threshold_adaptation)) if enable_threshold_adaptation is None else bool(enable_threshold_adaptation)
        self.threshold_scale = float(getattr(cfg, "NEURON_THRESHOLD_SCALE", 2.0e-6 if threshold_scale is None else threshold_scale)) if threshold_scale is None else float(threshold_scale)
        self.threshold_pot_pulses_on_spike = int(getattr(cfg, "NEURON_THRESHOLD_POT_PULSES_ON_SPIKE", 1 if threshold_pot_pulses_on_spike is None else threshold_pot_pulses_on_spike)) if threshold_pot_pulses_on_spike is None else int(threshold_pot_pulses_on_spike)
        self.threshold_dep_pulses_recovery = int(getattr(cfg, "NEURON_THRESHOLD_DEP_PULSES_RECOVERY", 1 if threshold_dep_pulses_recovery is None else threshold_dep_pulses_recovery)) if threshold_dep_pulses_recovery is None else int(threshold_dep_pulses_recovery)
        self.threshold_recovery_period = int(getattr(cfg, "NEURON_THRESHOLD_RECOVERY_PERIOD", 4 if threshold_recovery_period is None else threshold_recovery_period)) if threshold_recovery_period is None else int(threshold_recovery_period)

        self.vmem = np.full(self.n_neurons, self.reset_voltage, dtype=float)
        self.spike_trace = np.zeros(self.n_neurons, dtype=float)
        self.refractory = np.zeros(self.n_neurons, dtype=int)
        self.last_spike_step = np.full(self.n_neurons, -10**9, dtype=int)

        self.threshold_devices: List[MemristorDevice] = []
        if self.enable_threshold_adaptation:
            base_seed = getattr(cfg, "SEED", 42) if seed is None else int(seed)
            for j in range(self.n_neurons):
                dev = MemristorDevice(seed=base_seed + 50000 + 97 * j)
                dev.reset("mid")
                self.threshold_devices.append(dev)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        self.vmem.fill(self.reset_voltage)
        self.spike_trace.fill(0.0)
        self.refractory.fill(0)
        self.last_spike_step.fill(-10**9)
        for dev in self.threshold_devices:
            dev.reset("mid")

    def _measured_vmm(self, x: Sequence[float]) -> np.ndarray:
        """Measured-only synaptic accumulation.

        Each pair is read through the nonideal read path. This is slower than an
        ideal matrix multiply, but it respects the current hardware-oriented
        modeling assumptions in the uploaded files.
        """
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        if x_arr.size != self.n_inputs:
            raise ValueError(f"Expected input length {self.n_inputs}, got {x_arr.size}")

        out = np.zeros(self.n_neurons, dtype=float)
        for j in range(self.n_neurons):
            acc = 0.0
            for i in range(self.n_inputs):
                gp, gm = self.crossbar.read_pair((i, j))
                acc += float(x_arr[i]) * (float(gp) - float(gm))
            out[j] = acc
        return out

    def _threshold_offsets(self) -> np.ndarray:
        if not self.enable_threshold_adaptation:
            return np.zeros(self.n_neurons, dtype=float)

        offsets = np.zeros(self.n_neurons, dtype=float)
        for j, dev in enumerate(self.threshold_devices):
            span = max(dev.g_max_eff - dev.g_min_eff, 1e-18)
            norm = (float(dev.g) - dev.g_min_eff) / span
            offsets[j] = self.threshold_scale * norm
        return offsets

    def get_thresholds(self) -> np.ndarray:
        return self.base_threshold + self._threshold_offsets()

    def _recover_threshold_devices(self, step_idx: int) -> None:
        if not self.enable_threshold_adaptation:
            return
        if self.threshold_dep_pulses_recovery <= 0:
            return
        if step_idx % max(1, self.threshold_recovery_period) != 0:
            return

        for j, dev in enumerate(self.threshold_devices):
            if self.refractory[j] > 0:
                continue
            if dev.state.level_idx > 0:
                dev.apply_dep_pulse(self.threshold_dep_pulses_recovery)

    # ------------------------------------------------------------------
    # Inference dynamics
    # ------------------------------------------------------------------
    def step(self, pre_spikes: Sequence[float], step_idx: int) -> NeuronStepResult:
        syn = self._measured_vmm(pre_spikes)
        thresholds = self.get_thresholds()

        # Decay trace first.
        self.spike_trace *= self.trace_decay

        spikes = np.zeros(self.n_neurons, dtype=np.int8)

        # Integrate each neuron using only current + membrane state.
        for j in range(self.n_neurons):
            if self.refractory[j] > 0:
                self.refractory[j] -= 1
                self.vmem[j] = self.reset_voltage
                continue

            self.vmem[j] = self.membrane_decay * self.vmem[j] + self.input_gain * syn[j]

            if self.vmem[j] >= thresholds[j]:
                spikes[j] = 1

        # Winner-take-all arbitration, physically plausible at a block level via
        # a shared inhibitory/reset line. Only the strongest spiking neuron is
        # allowed to emit when WTA is enabled.
        winner = -1
        spiking_idx = np.flatnonzero(spikes > 0)
        if spiking_idx.size > 0:
            if self.inhibit_on_spike:
                winner = int(spiking_idx[np.argmax(self.vmem[spiking_idx])])
                spikes[:] = 0
                spikes[winner] = 1
            else:
                winner = int(spiking_idx[np.argmax(self.vmem[spiking_idx])])

        if winner >= 0:
            self.spike_trace[winner] += 1.0
            self.refractory[winner] = self.refractory_steps
            self.last_spike_step[winner] = int(step_idx)

            # Reset spiking neuron.
            self.vmem[winner] = self.reset_voltage

            # Shared inhibition for competing neurons.
            if self.inhibit_on_spike and self.lateral_inhibition_strength > 0.0:
                mask = np.ones(self.n_neurons, dtype=bool)
                mask[winner] = False
                self.vmem[mask] -= self.lateral_inhibition_strength * np.maximum(syn[mask], 0.0)
                self.vmem = np.maximum(self.vmem, self.reset_voltage)

            if self.enable_threshold_adaptation:
                self.threshold_devices[winner].apply_pot_pulse(self.threshold_pot_pulses_on_spike)

        self._recover_threshold_devices(step_idx)

        return NeuronStepResult(
            synaptic_currents=syn.copy(),
            membrane_potentials=self.vmem.copy(),
            thresholds=self.get_thresholds(),
            spikes=spikes,
            spike_trace=self.spike_trace.copy(),
            refractory_counters=self.refractory.copy(),
            winner=winner,
        )

    # ------------------------------------------------------------------
    # Learning / programming
    # ------------------------------------------------------------------
    def apply_reward_modulated_update(
        self,
        pre_spikes: Sequence[float],
        winner: int,
        reward: float,
        step_idx: int,
        target: Optional[int] = None,
        update_all_active_to_target: bool = True,
        punish_wrong_winner: bool = True,
    ) -> LearningEvent:
        """Program synapses using the measured conductance controller.

        Parameters
        ----------
        pre_spikes:
            Presynaptic activity vector. Only active rows are eligible.
        winner:
            Post neuron selected by the layer. ``-1`` means no output spike.
        reward:
            Positive reward strengthens active synapses onto the reinforced
            action. Negative reward weakens them.
        target:
            Optional target action. If provided, positive reward can reinforce
            the target column even when it did not spike.
        update_all_active_to_target:
            When True and ``target`` is valid, all active rows are updated on the
            target column. This is useful for supervised imitation or labeled
            episodes.
        punish_wrong_winner:
            When target is provided and the winner is wrong, active rows on the
            wrong winner can be depressed.
        """
        pre = np.asarray(pre_spikes, dtype=float).reshape(-1)
        if pre.size != self.n_inputs:
            raise ValueError(f"Expected input length {self.n_inputs}, got {pre.size}")

        active_rows = [int(i) for i, x in enumerate(pre) if x > 0.0]
        if not active_rows:
            return LearningEvent([], [], [], 0, 0, 0, float(reward), int(winner), target, "No active presynaptic rows.")

        updates: List[tuple[int, int, int]] = []  # (row, col, direction)

        if target is not None and not (0 <= int(target) < self.n_neurons):
            raise ValueError(f"target must be in [0, {self.n_neurons - 1}] or None")

        sign = +1 if reward >= 0.0 else -1

        if target is not None and update_all_active_to_target:
            for i in active_rows:
                updates.append((i, int(target), sign))

        elif winner >= 0:
            for i in active_rows:
                updates.append((i, int(winner), sign))

        if target is not None and punish_wrong_winner and winner >= 0 and int(winner) != int(target):
            for i in active_rows:
                updates.append((i, int(winner), -1))

        if not updates:
            return LearningEvent([], [], [], 0, 0, 0, float(reward), int(winner), target, "No eligible synaptic updates.")

        updated_pairs: List[tuple[int, int]] = []
        directions: List[int] = []
        actions: List[str] = []
        n_pulses_plus = 0
        n_pulses_minus = 0
        n_refresh = 0

        for row, col, direction in updates:
            result: ProgrammingResult = self.controller.update_weight((row, col), int(direction), int(step_idx))
            updated_pairs.append((int(row), int(col)))
            directions.append(int(direction))
            actions.append(str(result.chosen_action))
            n_pulses_plus += int(result.n_pulses_plus)
            n_pulses_minus += int(result.n_pulses_minus)
            if result.did_refresh:
                n_refresh += 1

        return LearningEvent(
            updated_pairs=updated_pairs,
            directions=directions,
            actions=actions,
            n_pulses_plus=n_pulses_plus,
            n_pulses_minus=n_pulses_minus,
            n_refresh=n_refresh,
            reward=float(reward),
            winner=int(winner),
            target=None if target is None else int(target),
            message="Measured pulse-based synaptic update executed.",
        )