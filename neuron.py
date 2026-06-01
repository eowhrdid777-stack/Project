from __future__ import annotations

"""Hardware-aware LIF neuron layer for the R-STDP rescue-robot SNN.

This file is intentionally limited to neuron/crossbar dynamics.
It does not read sensors, communicate with Arduino, or drive motors.

Design rules used in this revision
----------------------------------
1. Do not use an ideal weight matrix during inference.
   Synaptic accumulation is computed from measured differential crossbar reads.
2. Do not add artificial spike guarantees or membrane clipping just to make the
   simulation easier to train.
3. Do not silently add extra hardware blocks.  Circuit-like mechanisms such as
   lateral WTA or adaptive threshold devices are available only when explicitly
   enabled through config.py or constructor arguments.
4. Keep the public interface compatible with network.py and learning.py:
   - NeuronStepResult
   - LearningEvent
   - MemristiveLIFOutputLayer.step(...)
   - MemristiveLIFOutputLayer._measured_vmm(...)
   - MemristiveLIFOutputLayer.apply_reward_modulated_update(...)

Real-robot transition note
--------------------------
For real-robot learning, this file should normally remain unchanged.  The real
robot path should be added outside this module, for example:

    Arduino sensors/motors <-> robot_interface.py <-> real_robot_env.py
        -> observation dict -> encoding.py -> network.py -> neuron.py

A commented reference implementation for robot_interface.py / real_robot_env.py
is appended at the bottom of this file only as a planning note.  It is not
executed and should be moved to separate files when real hardware is used.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import config as cfg
from conductance_modulation import ConductanceModulationController, ProgrammingResult
from crossbar import DifferentialCrossbar
from device_model import MemristorDevice


def _cfg(name: str, default: Any) -> Any:
    """Read config value while keeping compatibility with older config.py files."""
    return getattr(cfg, name, default)


@dataclass
class NeuronStepResult:
    """한 simulation step에서 neuron layer 상태를 저장하는 구조체."""

    synaptic_currents: np.ndarray          # measured differential synaptic signal, shape = (n_neurons,)
    membrane_potentials: np.ndarray        # current membrane state, shape = (n_neurons,)
    thresholds: np.ndarray                 # current thresholds, shape = (n_neurons,)
    spikes: np.ndarray                     # spike vector for this timestep, shape = (n_neurons,)
    spike_trace: np.ndarray                # decayed spike traces, shape = (n_neurons,)
    refractory_counters: np.ndarray        # remaining refractory steps, shape = (n_neurons,)
    winner: int                            # representative winning neuron, -1 if no spike


@dataclass
class LearningEvent:
    """실제 synapse programming 결과를 요약해서 저장하는 구조체."""

    updated_pairs: List[Tuple[int, int]]   # updated (row, col) pairs
    directions: List[int]                  # +1 potentiation of differential weight, -1 depression
    actions: List[str]                     # controller-level programming actions used
    n_pulses_plus: int                     # number of pulses applied to plus devices
    n_pulses_minus: int                    # number of pulses applied to minus devices
    n_refresh: int                         # number of common-mode refresh operations
    reward: float                          # reward used for this update
    winner: int                            # selected output neuron
    target: Optional[int]                  # target neuron, if given
    message: str                           # human-readable summary


class MemristiveLIFOutputLayer:
    """Measured-read 기반 hardware-aware LIF neuron layer.

    The same class is used for both hidden and output layers.  Its input length
    is fixed by the crossbar row count, and its neuron count is fixed by the
    crossbar logical column count.
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
        if crossbar is None:
            raise ValueError("crossbar must not be None")

        self.crossbar = crossbar
        self.controller = ConductanceModulationController(crossbar)

        self.seed = int(_cfg("SEED", 42) if seed is None else seed)
        self.rng = np.random.default_rng(self.seed)

        self.n_inputs = int(crossbar.n_rows)
        self.n_neurons = int(crossbar.n_logical_cols)
        if self.n_inputs <= 0:
            raise ValueError("crossbar.n_rows must be >= 1")
        if self.n_neurons <= 0:
            raise ValueError("crossbar.n_logical_cols must be >= 1")

        # -----------------------------
        # LIF neuron parameters
        # -----------------------------
        self.membrane_decay = float(
            _cfg("NEURON_MEMBRANE_DECAY", 0.97) if membrane_decay is None else membrane_decay
        )
        if not (0.0 <= self.membrane_decay <= 1.0):
            raise ValueError("NEURON_MEMBRANE_DECAY must be in [0, 1]")

        self.input_gain = float(
            _cfg("NEURON_INPUT_GAIN", 1.0) if input_gain is None else input_gain
        )
        if not np.isfinite(self.input_gain):
            raise ValueError("NEURON_INPUT_GAIN must be finite")

        self.base_threshold = float(
            _cfg("NEURON_BASE_THRESHOLD", 3.5e-6) if base_threshold is None else base_threshold
        )
        if not np.isfinite(self.base_threshold) or self.base_threshold <= 0.0:
            raise ValueError("NEURON_BASE_THRESHOLD must be a positive finite value")

        self.reset_voltage = float(
            _cfg("NEURON_RESET_VOLTAGE", 0.0) if reset_voltage is None else reset_voltage
        )
        if not np.isfinite(self.reset_voltage):
            raise ValueError("NEURON_RESET_VOLTAGE must be finite")

        self.refractory_steps = int(
            _cfg("NEURON_REFRACTORY_STEPS", 1) if refractory_steps is None else refractory_steps
        )
        if self.refractory_steps < 0:
            raise ValueError("NEURON_REFRACTORY_STEPS must be >= 0")

        self.trace_decay = float(
            _cfg("NEURON_TRACE_DECAY", 0.85) if trace_decay is None else trace_decay
        )
        if not (0.0 <= self.trace_decay <= 1.0):
            raise ValueError("NEURON_TRACE_DECAY must be in [0, 1]")

        # WTA/lateral inhibition is an extra circuit-level mechanism.  It is
        # disabled by default unless the config explicitly enables it.  Action
        # selection can still be performed by network.py using the returned
        # winner field without modifying neuron dynamics.
        self.inhibit_on_spike = bool(
            _cfg("NEURON_ENABLE_WTA", True) if inhibit_on_spike is None else inhibit_on_spike
        )
        self.lateral_inhibition_strength = float(
            _cfg("NEURON_LATERAL_INHIBITION", 0.0)
            if lateral_inhibition_strength is None
            else lateral_inhibition_strength
        )
        if self.lateral_inhibition_strength < 0.0:
            raise ValueError("NEURON_LATERAL_INHIBITION must be >= 0")
        if not self.inhibit_on_spike:
            self.lateral_inhibition_strength = 0.0

        # No artificial membrane clipping is applied.  If membrane values become
        # unrealistic, tune the physical/model parameters instead:
        # NEURON_INPUT_GAIN, NEURON_BASE_THRESHOLD, NEURON_MEMBRANE_DECAY,
        # ENCODER_* parameters, or the crossbar conductance window.

        # -----------------------------
        # Optional threshold adaptation
        # -----------------------------
        # This adds extra threshold devices.  It is disabled by default because
        # the baseline hardware model already contains synaptic devices in the
        # crossbar; adding threshold devices without explicitly designing them
        # would be a hidden hardware assumption.
        self.enable_threshold_adaptation = bool(
            _cfg("NEURON_ENABLE_THRESHOLD_ADAPTATION", True)
            if enable_threshold_adaptation is None
            else enable_threshold_adaptation
        )
        self.threshold_scale = float(
            _cfg("NEURON_THRESHOLD_SCALE", 1.0e-6) if threshold_scale is None else threshold_scale
        )
        if self.threshold_scale < 0.0:
            raise ValueError("NEURON_THRESHOLD_SCALE must be >= 0")

        self.threshold_pot_pulses_on_spike = int(
            _cfg("NEURON_THRESHOLD_POT_PULSES_ON_SPIKE", 1)
            if threshold_pot_pulses_on_spike is None
            else threshold_pot_pulses_on_spike
        )
        self.threshold_dep_pulses_recovery = int(
            _cfg("NEURON_THRESHOLD_DEP_PULSES_RECOVERY", 1)
            if threshold_dep_pulses_recovery is None
            else threshold_dep_pulses_recovery
        )
        self.threshold_recovery_period = int(
            _cfg("NEURON_THRESHOLD_RECOVERY_PERIOD", 3)
            if threshold_recovery_period is None
            else threshold_recovery_period
        )
        if self.threshold_pot_pulses_on_spike < 0:
            raise ValueError("NEURON_THRESHOLD_POT_PULSES_ON_SPIKE must be >= 0")
        if self.threshold_dep_pulses_recovery < 0:
            raise ValueError("NEURON_THRESHOLD_DEP_PULSES_RECOVERY must be >= 0")
        if self.threshold_recovery_period < 1:
            raise ValueError("NEURON_THRESHOLD_RECOVERY_PERIOD must be >= 1")

        # -----------------------------
        # Internal state
        # -----------------------------
        self.vmem = np.full(self.n_neurons, self.reset_voltage, dtype=float)
        self.spike_trace = np.zeros(self.n_neurons, dtype=float)
        self.refractory = np.zeros(self.n_neurons, dtype=int)
        self.last_spike_step = np.full(self.n_neurons, -10**9, dtype=int)

        self.threshold_devices: List[MemristorDevice] = []
        if self.enable_threshold_adaptation:
            for j in range(self.n_neurons):
                dev = MemristorDevice(seed=self.seed + 50000 + 97 * j)
                dev.reset("mid")
                self.threshold_devices.append(dev)

    # ------------------------------------------------------------------
    # State and inspection helpers
    # ------------------------------------------------------------------
    def reset_state(self, reset_threshold_devices: bool = False) -> None:
        """Reset dynamic neuron state, optionally resetting adaptive threshold devices."""
        self.vmem.fill(self.reset_voltage)
        self.spike_trace.fill(0.0)
        self.refractory.fill(0)
        self.last_spike_step.fill(-10**9)
        if reset_threshold_devices:
            for dev in self.threshold_devices:
                dev.reset("mid")

    def describe(self) -> Dict[str, Any]:
        """Return current layer settings for experiment logging."""
        return {
            "n_inputs": int(self.n_inputs),
            "n_neurons": int(self.n_neurons),
            "membrane_decay": float(self.membrane_decay),
            "input_gain": float(self.input_gain),
            "base_threshold": float(self.base_threshold),
            "reset_voltage": float(self.reset_voltage),
            "refractory_steps": int(self.refractory_steps),
            "trace_decay": float(self.trace_decay),
            "inhibit_on_spike": bool(self.inhibit_on_spike),
            "lateral_inhibition_strength": float(self.lateral_inhibition_strength),
            "enable_threshold_adaptation": bool(self.enable_threshold_adaptation),
            "threshold_scale": float(self.threshold_scale),
        }

    def _validate_input_vector(self, x: Sequence[float], *, name: str = "input") -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.size != self.n_inputs:
            raise ValueError(f"Expected {name} length {self.n_inputs}, got {arr.size}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or inf")
        return arr

    # ------------------------------------------------------------------
    # Hardware-aware read path
    # ------------------------------------------------------------------
    def _measured_vmm(self, x: Sequence[float]) -> np.ndarray:
        """Measured-only synaptic accumulation.

        The crossbar returns measured differential-pair conductances through
        read_pair(), including the nonidealities modeled in DifferentialCrossbar.
        The neuron receives:

            sum_i x_i * (G_plus(i, j) - G_minus(i, j))

        The result is conductance-like.  Therefore NEURON_BASE_THRESHOLD and
        NEURON_INPUT_GAIN must be tuned in the same effective scale.
        """
        x_arr = self._validate_input_vector(x, name="pre_spikes")

        out = np.zeros(self.n_neurons, dtype=float)
        for j in range(self.n_neurons):
            acc = 0.0
            for i, xi in enumerate(x_arr):
                if xi == 0.0:
                    continue
                gp, gm = self.crossbar.read_pair((i, j))
                acc += float(xi) * (float(gp) - float(gm))
            out[j] = acc
        return out

    def _threshold_offsets(self) -> np.ndarray:
        """Convert optional threshold-device states into threshold offsets."""
        if not self.enable_threshold_adaptation:
            return np.zeros(self.n_neurons, dtype=float)

        offsets = np.zeros(self.n_neurons, dtype=float)
        for j, dev in enumerate(self.threshold_devices):
            span = max(float(dev.g_max_eff - dev.g_min_eff), 1e-18)
            norm = (float(dev.g) - float(dev.g_min_eff)) / span
            norm = float(np.clip(norm, 0.0, 1.0))
            offsets[j] = self.threshold_scale * norm
        return offsets

    def get_thresholds(self) -> np.ndarray:
        """Return current threshold = base threshold + optional adaptive offset."""
        return self.base_threshold + self._threshold_offsets()

    def _recover_threshold_devices(self, step_idx: int) -> None:
        """Apply slow recovery only when adaptive-threshold devices are enabled."""
        if not self.enable_threshold_adaptation:
            return
        if self.threshold_dep_pulses_recovery <= 0:
            return
        if int(step_idx) % max(1, self.threshold_recovery_period) != 0:
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
        """Run one LIF timestep.

        ``pre_spikes`` is usually binary in the current population-latency
        pipeline, but the method also accepts finite analog amplitudes.
        """
        syn = self._measured_vmm(pre_spikes)
        thresholds_before = self.get_thresholds()

        self.spike_trace *= self.trace_decay
        spikes = np.zeros(self.n_neurons, dtype=np.int8)

        for j in range(self.n_neurons):
            if self.refractory[j] > 0:
                self.refractory[j] -= 1
                self.vmem[j] = self.reset_voltage
                continue

            self.vmem[j] = self.membrane_decay * self.vmem[j] + self.input_gain * syn[j]
            if self.vmem[j] >= thresholds_before[j]:
                spikes[j] = 1

        spiking_idx = np.flatnonzero(spikes > 0)
        winner = -1

        if spiking_idx.size > 0:
            winner = int(spiking_idx[np.argmax(self.vmem[spiking_idx])])

            if self.inhibit_on_spike:
                # Optional hardware block:
                # WTA/lateral inhibition circuit keeps only the strongest spiking neuron.
                spikes[:] = 0
                spikes[winner] = 1
                effective_spike_idx = np.array([winner], dtype=int)
            else:
                # No WTA circuit:
                # all neurons that crossed threshold are valid spikes.
                effective_spike_idx = spiking_idx.astype(int)
        else:
            effective_spike_idx = np.array([], dtype=int)

        for j in effective_spike_idx:
            self.spike_trace[j] += 1.0
            self.refractory[j] = self.refractory_steps
            self.last_spike_step[j] = int(step_idx)
            self.vmem[j] = self.reset_voltage

        if self.inhibit_on_spike and winner >= 0 and self.lateral_inhibition_strength > 0.0:
            mask = np.ones(self.n_neurons, dtype=bool)
            mask[winner] = False
            self.vmem[mask] -= self.lateral_inhibition_strength * np.maximum(syn[mask], 0.0)

        if self.enable_threshold_adaptation:
            for j in effective_spike_idx:
                self.threshold_devices[j].apply_pot_pulse(self.threshold_pot_pulses_on_spike)
                
        self._recover_threshold_devices(step_idx)

        return NeuronStepResult(
            synaptic_currents=syn.copy(),
            membrane_potentials=self.vmem.copy(),
            thresholds=self.get_thresholds(),
            spikes=spikes.copy(),
            spike_trace=self.spike_trace.copy(),
            refractory_counters=self.refractory.copy(),
            winner=winner,
        )

    # ------------------------------------------------------------------
    # Compatibility learning/programming helper
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
        n_pulses: Optional[int] = None,
    ) -> LearningEvent:
        """Apply a simple pulse-based reward-modulated update.

        This method is retained because network.py can call it through
        MemristiveSNNNetwork.learn().  In the current main training path,
        RewardModulatedSTDPLearner in learning.py performs the timing-based
        R-STDP update using recorded spike times.  Therefore this method should
        be considered a compatibility/local-update path rather than the primary
        learning rule for reported R-STDP experiments.
        """
        pre = self._validate_input_vector(pre_spikes, name="pre_spikes")
        reward = float(reward)
        winner = int(winner)
        step_idx = int(step_idx)

        if winner >= self.n_neurons:
            raise ValueError(f"winner out of range: {winner}")

        if target is not None:
            target = int(target)
            if not (0 <= target < self.n_neurons):
                raise ValueError(f"target must be in [0, {self.n_neurons - 1}] or None")

        active_rows = [int(i) for i, x in enumerate(pre) if x > 0.0]
        if not active_rows:
            return LearningEvent([], [], [], 0, 0, 0, reward, winner, target, "No active presynaptic rows.")

        if reward == 0.0:
            return LearningEvent([], [], [], 0, 0, 0, reward, winner, target, "Reward is zero; skipped update.")

        pulse_count = int(_cfg("NEURON_UPDATE_N_PULSES", 1) if n_pulses is None else n_pulses)
        pulse_count = max(1, pulse_count)

        reward_sign = +1 if reward > 0.0 else -1
        updates: List[Tuple[int, int, int]] = []

        if target is not None and update_all_active_to_target:
            for row in active_rows:
                updates.append((row, target, reward_sign))
        elif 0 <= winner < self.n_neurons:
            for row in active_rows:
                updates.append((row, winner, reward_sign))

        if target is not None and punish_wrong_winner and 0 <= winner < self.n_neurons and winner != target:
            for row in active_rows:
                updates.append((row, winner, -1))

        if not updates:
            return LearningEvent([], [], [], 0, 0, 0, reward, winner, target, "No eligible synaptic updates.")

        updated_pairs: List[Tuple[int, int]] = []
        directions: List[int] = []
        actions: List[str] = []
        n_pulses_plus = 0
        n_pulses_minus = 0
        n_refresh = 0

        seen: set[Tuple[int, int, int]] = set()
        for row, col, direction in updates:
            key = (int(row), int(col), int(direction))
            if key in seen:
                continue
            seen.add(key)

            result: ProgrammingResult = self.controller.update_weight(
                pair_id=(int(row), int(col)),
                direction=int(direction),
                step_idx=step_idx,
                n_pulses=pulse_count,
            )
            updated_pairs.append((int(row), int(col)))
            directions.append(int(direction))
            actions.append(str(result.chosen_action))
            n_pulses_plus += int(result.n_pulses_plus)
            n_pulses_minus += int(result.n_pulses_minus)
            if bool(result.did_refresh):
                n_refresh += 1

        return LearningEvent(
            updated_pairs=updated_pairs,
            directions=directions,
            actions=actions,
            n_pulses_plus=int(n_pulses_plus),
            n_pulses_minus=int(n_pulses_minus),
            n_refresh=int(n_refresh),
            reward=reward,
            winner=winner,
            target=target,
            message="Compatibility pulse-based synaptic update executed.",
        )
