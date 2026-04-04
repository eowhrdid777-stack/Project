from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import config as cfg
from stm_device_model import STMDeviceModel


def _cfg(name: str, default):
    return getattr(cfg, name, default)


@dataclass
class STMPulseResult:
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


class STMCrossbar:
    """Simple crossbar wrapper for STMDeviceModel with lightweight read nonidealities."""

    def __init__(self, n_rows: int, n_cols: int, seed: Optional[int] = None) -> None:
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.rng = np.random.default_rng(seed)

        self.read_voltage = float(_cfg("STM_READ_VOLTAGE", 0.1))
        self.read_avg_samples = int(_cfg("STM_READ_AVG_SAMPLES", 1))
        self.read_ir_drop_alpha = float(_cfg("STM_READ_IR_DROP_ALPHA", 0.04))
        self.prog_ir_drop_alpha = float(_cfg("STM_PROG_IR_DROP_ALPHA", 0.04))
        self.enable_sneak_path = bool(_cfg("STM_ENABLE_SNEAK_PATH", True))
        self.sneak_ratio = float(_cfg("STM_SNEAK_RATIO", 0.0015))

        self.devices = np.empty((self.n_rows, self.n_cols), dtype=object)
        base_seed = None if seed is None else int(seed)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                dev_seed = None if base_seed is None else base_seed + 1009 * i + 37 * j
                self.devices[i, j] = STMDeviceModel(seed=dev_seed)

    def _validate_cell(self, cell_id: tuple[int, int]) -> tuple[int, int]:
        if not isinstance(cell_id, tuple) or len(cell_id) != 2:
            raise ValueError("cell_id must be (row, col)")
        row, col = int(cell_id[0]), int(cell_id[1])
        if not (0 <= row < self.n_rows and 0 <= col < self.n_cols):
            raise IndexError(f"cell_id out of range: {(row, col)}")
        return row, col

    def _read_position_factor(self, row: int, col: int) -> float:
        rr = row / max(self.n_rows - 1, 1)
        cc = col / max(self.n_cols - 1, 1)
        return max(0.70, 1.0 - self.read_ir_drop_alpha * 0.5 * (rr + cc))

    def _program_position_factor(self, row: int, col: int) -> float:
        rr = row / max(self.n_rows - 1, 1)
        cc = col / max(self.n_cols - 1, 1)
        return max(0.70, 1.0 - self.prog_ir_drop_alpha * 0.5 * (rr + cc))

    def reset_all(self, mode: str = "rest") -> None:
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                self.devices[row, col].reset(mode)

    def relax_all(self, dt_s: float) -> None:
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                self.devices[row, col].relax(dt_s, record_history=False)

    def read_cell(self, cell_id: tuple[int, int]) -> tuple[float, float]:
        row, col = self._validate_cell(cell_id)
        dev = self.devices[row, col]

        gs = []
        for _ in range(max(1, self.read_avg_samples)):
            g = dev.read_conductance(self.read_voltage)
            g *= self._read_position_factor(row, col)
            if self.enable_sneak_path:
                # simple leakage proportional to unused headroom in the selected cell
                leak_headroom = max(0.0, dev.g_peak_eff - dev.g)
                g += self.sneak_ratio * leak_headroom
            gs.append(max(0.0, float(g)))

        g_mean = float(np.mean(gs))
        i_mean = g_mean * self.read_voltage
        return g_mean, i_mean

    def apply_pulse_to_cell(
        self,
        cell_id: tuple[int, int],
        amplitude_v: float,
        pulse_width_s: Optional[float] = None,
        gap_after_s: float = 0.0,
        *,
        relax_unselected: bool = True,
    ) -> STMPulseResult:
        row, col = self._validate_cell(cell_id)
        dev = self.devices[row, col]

        amp_eff = float(amplitude_v) * self._program_position_factor(row, col)
        dev.apply_pulse(amplitude_v=amp_eff, width_s=pulse_width_s, record_history=False)

        if gap_after_s > 0.0:
            if relax_unselected:
                self.relax_all(gap_after_s)
            else:
                dev.relax(gap_after_s, record_history=False)

        measured_g, measured_i = self.read_cell((row, col))
        snap = dev.snapshot()
        return STMPulseResult(
            row=row,
            col=col,
            applied_amplitude_v=amp_eff,
            pulse_width_s=dev.default_pulse_width_s if pulse_width_s is None else float(pulse_width_s),
            gap_after_s=float(gap_after_s),
            measured_g=measured_g,
            measured_i=measured_i,
            z=snap.z,
            x=snap.x,
            r=snap.r,
        )

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
        row, col = self._validate_cell(cell_id)
        dev = self.devices[row, col]
        width_s = dev.default_pulse_width_s if pulse_width_s is None else float(pulse_width_s)
        if period_s is None:
            gap_s = 0.0
        else:
            gap_s = max(0.0, float(period_s) - width_s)

        hist = dev._new_history()
        g0, i0 = self.read_cell((row, col))
        snap0 = dev.snapshot()
        hist["time_s"].append(snap0.t_s)
        hist["conductance_s"].append(g0)
        hist["current_a"].append(i0)
        hist["z"].append(snap0.z)
        hist["x"].append(snap0.x)
        hist["r"].append(snap0.r)
        hist["event"].append("init")

        for _ in range(int(n_pulses)):
            amp_eff = float(amplitude_v) * self._program_position_factor(row, col)
            ph = dev.apply_pulse(amplitude_v=amp_eff, width_s=width_s, record_history=True)
            self._extend_measured_history(hist, ph, row=row, col=col)

            if gap_s > 0.0:
                if relax_unselected:
                    self.relax_all(gap_s)
                    gh = dev._new_history()
                    g_sel, i_sel = self.read_cell((row, col))
                    snap = dev.snapshot()
                    gh["time_s"].append(snap.t_s)
                    gh["conductance_s"].append(g_sel)
                    gh["current_a"].append(i_sel)
                    gh["z"].append(snap.z)
                    gh["x"].append(snap.x)
                    gh["r"].append(snap.r)
                    gh["event"].append("gap_end")
                    self._extend_history_lists(hist, gh, skip_first=False)
                else:
                    gh = dev.relax(gap_s, record_history=True)
                    self._extend_measured_history(hist, gh, row=row, col=col, skip_first=True)

        if tail_relax_s > 0.0:
            if relax_unselected:
                self.relax_all(float(tail_relax_s))
                g_tail, i_tail = self.read_cell((row, col))
                snap = dev.snapshot()
                hist["time_s"].append(snap.t_s)
                hist["conductance_s"].append(g_tail)
                hist["current_a"].append(i_tail)
                hist["z"].append(snap.z)
                hist["x"].append(snap.x)
                hist["r"].append(snap.r)
                hist["event"].append("tail_relax_end")
            else:
                th = dev.relax(float(tail_relax_s), record_history=True)
                self._extend_measured_history(hist, th, row=row, col=col, skip_first=True)

        return {
            "time_s": np.asarray(hist["time_s"], dtype=float),
            "conductance_s": np.asarray(hist["conductance_s"], dtype=float),
            "current_a": np.asarray(hist["current_a"], dtype=float),
            "z": np.asarray(hist["z"], dtype=float),
            "x": np.asarray(hist["x"], dtype=float),
            "r": np.asarray(hist["r"], dtype=float),
            "event": np.asarray(hist["event"], dtype=object),
        }

    def _extend_measured_history(
        self,
        dst: dict[str, list],
        src: Optional[dict[str, np.ndarray]],
        *,
        row: int,
        col: int,
        skip_first: bool = True,
    ) -> None:
        if src is None:
            return
        start = 1 if skip_first else 0
        for idx in range(start, len(src["time_s"])):
            g_meas, i_meas = self.read_cell((row, col))
            dst["time_s"].append(float(src["time_s"][idx]))
            dst["conductance_s"].append(g_meas)
            dst["current_a"].append(i_meas)
            dst["z"].append(float(src["z"][idx]))
            dst["x"].append(float(src["x"][idx]))
            dst["r"].append(float(src["r"][idx]))
            dst["event"].append(str(src["event"][idx]))

    @staticmethod
    def _extend_history_lists(dst: dict[str, list], src: dict[str, list], *, skip_first: bool = False) -> None:
        start = 1 if skip_first else 0
        for key in dst:
            dst[key].extend(src[key][start:])
