from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal, Optional, Tuple

import numpy as np

import config as cfg
from stm_device_model import STMDeviceModel

Side = Literal["plus", "minus"]
PulsePolarity = Literal["pot", "dep"]


def _cfg(name: str, default):
    return getattr(cfg, name, default)


@dataclass
class STMPulseResult:
    """Result of programming or reading one physical STM cell."""

    row: int
    col: int
    applied_amplitude_v: float
    pulse_width_s: float
    gap_after_s: float
    measured_g: float
    measured_i: float
    z: float
    x: float
    r: float


@dataclass
class STMPairPulseResult:
    """Debug record for one logical differential-pair STM synapse."""

    row: int
    logical_col: int
    side: str
    polarity: str
    requested_pulses: int
    effective_pulses: int
    g_plus: float
    g_minus: float
    weight: float


class STMCrossbar:
    """Differential-pair STM crossbar compatible with the LTM crossbar interface.

    Network/neuron/learning code expects each logical synapse to expose a signed
    weight through a plus/minus pair:

        w_ij = G_plus(i, j) - G_minus(i, j)

    This class keeps that same interface while using STMDeviceModel as each
    physical cell.  Therefore, n_cols means the number of logical columns, and
    the physical array has 2*n_cols columns.

    Required compatibility API
    --------------------------
    - n_rows
    - n_logical_cols
    - n_phys_cols
    - read_pair(pair_id) -> (g_plus, g_minus)
    - apply_pulse(pair_id, side, polarity, n_pulses) -> int
    - get_pair_bounds(pair_id) -> (gp_min, gp_max, gm_min, gm_max)

    Notes
    -----
    STM devices remain volatile.  The crossbar does not automatically advance
    physical time on every network step.  A caller that wants real-time STM
    decay should call relax_all(dt_s) with the elapsed physical time.
    """

    def __init__(self, n_rows: int, n_cols: int, seed: Optional[int] = None) -> None:
        self.n_rows = int(n_rows)
        self.n_logical_cols = int(n_cols)
        self.n_cols = self.n_logical_cols  # alias for older scripts
        self.n_phys_cols = 2 * self.n_logical_cols

        if self.n_rows <= 0:
            raise ValueError("n_rows must be >= 1")
        if self.n_logical_cols <= 0:
            raise ValueError("n_cols/n_logical_cols must be >= 1")

        self.rng = np.random.default_rng(seed)

        # Read path.
        self.read_voltage = float(_cfg("STM_READ_VOLTAGE", 0.1))
        self.read_avg_samples = int(_cfg("STM_READ_AVG_SAMPLES", 1))
        self.read_ir_drop_alpha = float(_cfg("STM_READ_IR_DROP_ALPHA", 0.04))
        self.enable_sneak_path = bool(_cfg("STM_ENABLE_SNEAK_PATH", True))
        self.sneak_ratio = float(_cfg("STM_SNEAK_RATIO", 0.0015))

        # Programming path.
        self.prog_ir_drop_alpha = float(_cfg("STM_PROG_IR_DROP_ALPHA", 0.04))
        self.pulse_width_s = float(_cfg("STM_PULSE_WIDTH_S", 1.0e-3))
        self.pot_pulse_v = float(_cfg("STM_POT_PULSE_V", 0.45))
        self.dep_pulse_v = float(_cfg("STM_DEP_PULSE_V", -0.45))
        self.pulse_gap_s = float(_cfg("STM_PROGRAM_GAP_AFTER_PULSE_S", 0.0))
        self.relax_unselected_during_program = bool(_cfg("STM_RELAX_UNSELECTED_DURING_PROGRAM", True))

        # Optional read disturb for STM.  Disabled by default; if enabled, it is
        # modeled as a tiny passive elapsed time during a read, not as an
        # artificial conductance clamp.
        self.enable_read_relax = bool(_cfg("STM_ENABLE_READ_RELAX", False))
        self.read_relax_s = float(_cfg("STM_READ_RELAX_S", 0.0))

        self.devices = np.empty((self.n_rows, self.n_phys_cols), dtype=object)
        base_seed = None if seed is None else int(seed)
        for i in range(self.n_rows):
            for pcol in range(self.n_phys_cols):
                dev_seed = None if base_seed is None else base_seed + 1009 * i + 37 * pcol
                self.devices[i, pcol] = STMDeviceModel(seed=dev_seed)

    # ------------------------------------------------------------------
    # Pair / physical-column helpers
    # ------------------------------------------------------------------
    def _parse_pair_id(self, pair_id: Hashable) -> Tuple[int, int]:
        if not isinstance(pair_id, tuple) or len(pair_id) != 2:
            raise ValueError("pair_id must be a tuple (row, logical_col)")
        row, logical_col = int(pair_id[0]), int(pair_id[1])
        if not (0 <= row < self.n_rows and 0 <= logical_col < self.n_logical_cols):
            raise IndexError(f"pair_id out of range: {(row, logical_col)}")
        return row, logical_col

    def _validate_cell(self, cell_id: tuple[int, int]) -> tuple[int, int]:
        if not isinstance(cell_id, tuple) or len(cell_id) != 2:
            raise ValueError("cell_id must be (row, physical_col)")
        row, phys_col = int(cell_id[0]), int(cell_id[1])
        if not (0 <= row < self.n_rows and 0 <= phys_col < self.n_phys_cols):
            raise IndexError(f"cell_id out of range: {(row, phys_col)}")
        return row, phys_col

    @staticmethod
    def _plus_col(logical_col: int) -> int:
        return 2 * int(logical_col)

    @staticmethod
    def _minus_col(logical_col: int) -> int:
        return 2 * int(logical_col) + 1

    def _read_position_factor(self, row: int, phys_col: int) -> float:
        rr = row / max(self.n_rows - 1, 1)
        cc = phys_col / max(self.n_phys_cols - 1, 1)
        factor = 1.0 - self.read_ir_drop_alpha * 0.5 * (rr + cc)
        return max(0.70, float(factor))

    def _program_position_factor(self, row: int, phys_col: int) -> float:
        rr = row / max(self.n_rows - 1, 1)
        cc = phys_col / max(self.n_phys_cols - 1, 1)
        factor = 1.0 - self.prog_ir_drop_alpha * 0.5 * (rr + cc)
        return max(0.70, float(factor))

    # ------------------------------------------------------------------
    # State controls
    # ------------------------------------------------------------------
    def reset_all(self, mode: str = "rest") -> None:
        for row in range(self.n_rows):
            for pcol in range(self.n_phys_cols):
                self.devices[row, pcol].reset(mode)

    def relax_all(self, dt_s: float) -> None:
        dt_s = float(dt_s)
        if dt_s <= 0.0:
            return
        for row in range(self.n_rows):
            for pcol in range(self.n_phys_cols):
                self.devices[row, pcol].relax(dt_s, record_history=False)

    def relax_pair(self, pair_id: Hashable, dt_s: float) -> None:
        row, logical_col = self._parse_pair_id(pair_id)
        if dt_s <= 0.0:
            return
        self.devices[row, self._plus_col(logical_col)].relax(float(dt_s), record_history=False)
        self.devices[row, self._minus_col(logical_col)].relax(float(dt_s), record_history=False)

    # ------------------------------------------------------------------
    # Ideal/internal state views
    # ------------------------------------------------------------------
    def read_pair_ideal(self, pair_id: Hashable) -> Tuple[float, float]:
        row, logical_col = self._parse_pair_id(pair_id)
        jp = self._plus_col(logical_col)
        jm = self._minus_col(logical_col)
        return float(self.devices[row, jp].g), float(self.devices[row, jm].g)

    def get_pair_bounds(self, pair_id: Hashable) -> Tuple[float, float, float, float]:
        row, logical_col = self._parse_pair_id(pair_id)
        dp = self.devices[row, self._plus_col(logical_col)]
        dm = self.devices[row, self._minus_col(logical_col)]
        return float(dp.g_rest_eff), float(dp.g_peak_eff), float(dm.g_rest_eff), float(dm.g_peak_eff)

    def get_pair_status(self, pair_id: Hashable) -> dict:
        gp, gm = self.read_pair(pair_id)
        gp_min, gp_max, gm_min, gm_max = self.get_pair_bounds(pair_id)
        return {
            "g_plus_measured": float(gp),
            "g_minus_measured": float(gm),
            "weight_measured": float(gp - gm),
            "common_mode_measured": float(0.5 * (gp + gm)),
            "bounds": {
                "gp_min": gp_min,
                "gp_max": gp_max,
                "gm_min": gm_min,
                "gm_max": gm_max,
            },
        }

    # ------------------------------------------------------------------
    # Measured read path
    # ------------------------------------------------------------------
    def _read_single_cell_conductance(self, row: int, phys_col: int) -> float:
        dev = self.devices[row, phys_col]
        g = float(dev.read_conductance(self.read_voltage))
        g *= self._read_position_factor(row, phys_col)

        if self.enable_sneak_path:
            # Same simple phenomenological form as the LTM crossbar: unselected
            # conductance headroom contributes a small leakage term.  This is a
            # crossbar-level parasitic, not a numerical stabilizer.
            leak_headroom = max(0.0, float(dev.g_peak_eff) - float(dev.g))
            g += self.sneak_ratio * leak_headroom * self._read_position_factor(row, phys_col)

        g = max(0.0, float(g))

        if self.enable_read_relax and self.read_relax_s > 0.0:
            dev.relax(self.read_relax_s, record_history=False)

        return g

    def read_cell(self, cell_id: tuple[int, int]) -> tuple[float, float]:
        row, phys_col = self._validate_cell(cell_id)
        n = max(1, int(self.read_avg_samples))
        gs = [self._read_single_cell_conductance(row, phys_col) for _ in range(n)]
        g_mean = float(np.mean(gs))
        i_mean = float(g_mean * self.read_voltage)
        return g_mean, i_mean

    def _read_pair_once(self, pair_id: Hashable) -> Tuple[float, float]:
        row, logical_col = self._parse_pair_id(pair_id)
        jp = self._plus_col(logical_col)
        jm = self._minus_col(logical_col)
        gp, _ = self.read_cell((row, jp))
        gm, _ = self.read_cell((row, jm))
        return float(gp), float(gm)

    def read_pair(self, pair_id: Hashable) -> Tuple[float, float]:
        n = max(1, int(self.read_avg_samples))
        vals = [self._read_pair_once(pair_id) for _ in range(n)]
        gp = float(np.mean([v[0] for v in vals]))
        gm = float(np.mean([v[1] for v in vals]))
        return gp, gm

    def read_weight_measured(self, pair_id: Hashable) -> float:
        gp, gm = self.read_pair(pair_id)
        return float(gp - gm)

    # ------------------------------------------------------------------
    # Programming
    # ------------------------------------------------------------------
    def _polarity_to_amplitude(self, polarity: PulsePolarity) -> float:
        if polarity == "pot":
            return float(self.pot_pulse_v)
        if polarity == "dep":
            return float(self.dep_pulse_v)
        raise ValueError(f"Unknown polarity: {polarity}")

    def apply_pulse_to_cell(
        self,
        cell_id: tuple[int, int],
        amplitude_v: float,
        pulse_width_s: Optional[float] = None,
        gap_after_s: Optional[float] = None,
        *,
        relax_unselected: Optional[bool] = None,
    ) -> STMPulseResult:
        row, phys_col = self._validate_cell(cell_id)
        dev = self.devices[row, phys_col]

        width_s = self.pulse_width_s if pulse_width_s is None else float(pulse_width_s)
        gap_s = self.pulse_gap_s if gap_after_s is None else float(gap_after_s)
        relax_all_unselected = (
            self.relax_unselected_during_program if relax_unselected is None else bool(relax_unselected)
        )

        amp_eff = float(amplitude_v) * self._program_position_factor(row, phys_col)
        dev.apply_pulse(amplitude_v=amp_eff, width_s=width_s, record_history=False)

        if gap_s > 0.0:
            if relax_all_unselected:
                self.relax_all(gap_s)
            else:
                dev.relax(gap_s, record_history=False)

        measured_g, measured_i = self.read_cell((row, phys_col))
        snap = dev.snapshot()
        return STMPulseResult(
            row=row,
            col=phys_col,
            applied_amplitude_v=amp_eff,
            pulse_width_s=width_s,
            gap_after_s=max(0.0, gap_s),
            measured_g=measured_g,
            measured_i=measured_i,
            z=snap.z,
            x=snap.x,
            r=snap.r,
        )

    def apply_pulse(
        self,
        pair_id: Hashable,
        side: Side,
        polarity: PulsePolarity,
        n_pulses: int = 1,
    ) -> int:
        n_pulses = int(n_pulses)
        if n_pulses <= 0:
            return 0
        if side not in ("plus", "minus"):
            raise ValueError(f"side must be 'plus' or 'minus', got {side!r}")

        row, logical_col = self._parse_pair_id(pair_id)
        phys_col = self._plus_col(logical_col) if side == "plus" else self._minus_col(logical_col)
        amp = self._polarity_to_amplitude(polarity)

        # Position-dependent voltage loss is modeled through pulse amplitude,
        # not by fabricating extra pulses.  Therefore the effective pulse count
        # remains equal to the requested count.
        for _ in range(n_pulses):
            self.apply_pulse_to_cell(
                (row, phys_col),
                amplitude_v=amp,
                pulse_width_s=self.pulse_width_s,
                gap_after_s=self.pulse_gap_s,
                relax_unselected=self.relax_unselected_during_program,
            )
        return int(n_pulses)

    def apply_pair_pulse_debug(
        self,
        pair_id: Hashable,
        side: Side,
        polarity: PulsePolarity,
        n_pulses: int = 1,
    ) -> STMPairPulseResult:
        row, logical_col = self._parse_pair_id(pair_id)
        n_eff = self.apply_pulse(pair_id, side=side, polarity=polarity, n_pulses=n_pulses)
        gp, gm = self.read_pair(pair_id)
        return STMPairPulseResult(
            row=row,
            logical_col=logical_col,
            side=str(side),
            polarity=str(polarity),
            requested_pulses=int(n_pulses),
            effective_pulses=int(n_eff),
            g_plus=float(gp),
            g_minus=float(gm),
            weight=float(gp - gm),
        )

    # ------------------------------------------------------------------
    # VMM / summaries
    # ------------------------------------------------------------------
    def vmm_ideal(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        if x_arr.size != self.n_rows:
            raise ValueError(f"Expected input of length {self.n_rows}, got {x_arr.size}")
        if not np.all(np.isfinite(x_arr)):
            raise ValueError("Input vector contains NaN or inf")

        out = np.zeros(self.n_logical_cols, dtype=float)
        for j in range(self.n_logical_cols):
            acc = 0.0
            for i in range(self.n_rows):
                if x_arr[i] == 0.0:
                    continue
                gp, gm = self.read_pair_ideal((i, j))
                acc += x_arr[i] * (gp - gm)
            out[j] = acc
        return out

    def vmm_measured(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        if x_arr.size != self.n_rows:
            raise ValueError(f"Expected input of length {self.n_rows}, got {x_arr.size}")
        if not np.all(np.isfinite(x_arr)):
            raise ValueError("Input vector contains NaN or inf")

        out = np.zeros(self.n_logical_cols, dtype=float)
        for j in range(self.n_logical_cols):
            acc = 0.0
            for i in range(self.n_rows):
                if x_arr[i] == 0.0:
                    continue
                gp, gm = self.read_pair((i, j))
                acc += x_arr[i] * (gp - gm)
            out[j] = acc
        return out

    def summary(self) -> dict:
        g_plus = []
        g_minus = []
        weights = []
        z_vals = []
        x_vals = []
        r_vals = []

        for i in range(self.n_rows):
            for j in range(self.n_logical_cols):
                jp = self._plus_col(j)
                jm = self._minus_col(j)
                gp, gm = self.read_pair_ideal((i, j))
                g_plus.append(gp)
                g_minus.append(gm)
                weights.append(gp - gm)
                for pcol in (jp, jm):
                    s = self.devices[i, pcol].snapshot()
                    z_vals.append(s.z)
                    x_vals.append(s.x)
                    r_vals.append(s.r)

        return {
            "type": "STM differential-pair crossbar",
            "n_rows": self.n_rows,
            "n_logical_cols": self.n_logical_cols,
            "n_phys_cols": self.n_phys_cols,
            "g_plus_mean": float(np.mean(g_plus)),
            "g_minus_mean": float(np.mean(g_minus)),
            "weight_mean": float(np.mean(weights)),
            "weight_std": float(np.std(weights)),
            "weight_min": float(np.min(weights)),
            "weight_max": float(np.max(weights)),
            "z_mean": float(np.mean(z_vals)),
            "x_mean": float(np.mean(x_vals)),
            "r_mean": float(np.mean(r_vals)),
            "read_voltage": float(self.read_voltage),
            "pot_pulse_v": float(self.pot_pulse_v),
            "dep_pulse_v": float(self.dep_pulse_v),
        }

    # ------------------------------------------------------------------
    # Legacy pulse-train helper for one physical cell
    # ------------------------------------------------------------------
    def run_pulse_train(
        self,
        cell_id: tuple[int, int],
        n_pulses: int,
        amplitude_v: float,
        pulse_width_s: Optional[float] = None,
        period_s: Optional[float] = None,
        tail_relax_s: float = 0.0,
        *,
        relax_unselected: bool = True,
    ) -> dict[str, np.ndarray]:
        row, phys_col = self._validate_cell(cell_id)
        dev = self.devices[row, phys_col]
        width_s = self.pulse_width_s if pulse_width_s is None else float(pulse_width_s)
        gap_s = 0.0 if period_s is None else max(0.0, float(period_s) - width_s)

        hist = dev._new_history()
        g0, i0 = self.read_cell((row, phys_col))
        snap0 = dev.snapshot()
        hist["time_s"].append(snap0.t_s)
        hist["conductance_s"].append(g0)
        hist["current_a"].append(i0)
        hist["z"].append(snap0.z)
        hist["x"].append(snap0.x)
        hist["r"].append(snap0.r)
        hist["event"].append("init")

        for _ in range(max(0, int(n_pulses))):
            amp_eff = float(amplitude_v) * self._program_position_factor(row, phys_col)
            ph = dev.apply_pulse(amplitude_v=amp_eff, width_s=width_s, record_history=True)
            self._extend_history(hist, ph, skip_first=True)
            if gap_s > 0.0:
                if relax_unselected:
                    self.relax_all(gap_s)
                    self._append_measured_point(hist, row, phys_col, "gap_end")
                else:
                    gh = dev.relax(gap_s, record_history=True)
                    self._extend_history(hist, gh, skip_first=True)

        if tail_relax_s > 0.0:
            if relax_unselected:
                self.relax_all(float(tail_relax_s))
                self._append_measured_point(hist, row, phys_col, "tail_relax_end")
            else:
                th = dev.relax(float(tail_relax_s), record_history=True)
                self._extend_history(hist, th, skip_first=True)

        return self._history_to_arrays(hist)

    def _append_measured_point(self, hist: dict[str, list], row: int, phys_col: int, event: str) -> None:
        g, i = self.read_cell((row, phys_col))
        snap = self.devices[row, phys_col].snapshot()
        hist["time_s"].append(float(snap.t_s))
        hist["conductance_s"].append(float(g))
        hist["current_a"].append(float(i))
        hist["z"].append(float(snap.z))
        hist["x"].append(float(snap.x))
        hist["r"].append(float(snap.r))
        hist["event"].append(str(event))

    @staticmethod
    def _extend_history(dst: dict[str, list], src: Optional[dict[str, np.ndarray]], *, skip_first: bool = False) -> None:
        if src is None:
            return
        start = 1 if skip_first else 0
        for key in dst:
            dst[key].extend(list(src[key][start:]))

    @staticmethod
    def _history_to_arrays(hist: dict[str, list]) -> dict[str, np.ndarray]:
        return {
            "time_s": np.asarray(hist["time_s"], dtype=float),
            "conductance_s": np.asarray(hist["conductance_s"], dtype=float),
            "current_a": np.asarray(hist["current_a"], dtype=float),
            "z": np.asarray(hist["z"], dtype=float),
            "x": np.asarray(hist["x"], dtype=float),
            "r": np.asarray(hist["r"], dtype=float),
            "event": np.asarray(hist["event"], dtype=object),
        }
