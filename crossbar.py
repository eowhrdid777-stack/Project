from __future__ import annotations

from typing import Hashable, Literal, Optional, Tuple

import numpy as np

import config as cfg
from device_model import MemristorDevice

Side = Literal["plus", "minus"]
PulsePolarity = Literal["pot", "dep"]


class DifferentialCrossbar:
    """Differential crossbar array with measured readout.

    Each logical synapse is represented by a pair of physical cells:
        W[i, j] = G_plus[i, j] - G_minus[i, j]
    """

    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        seed: Optional[int] = None,
    ) -> None:
        self.n_rows = int(n_rows)
        self.n_logical_cols = int(n_cols)
        self.n_phys_cols = 2 * self.n_logical_cols
        self.rng = np.random.default_rng(seed)

        self.read_voltage = float(cfg.READ_VOLTAGE)
        self.program_voltage = float(cfg.PROGRAM_VOLTAGE)
        self.read_avg_samples = int(cfg.READ_AVG_SAMPLES)

        self.read_ir_drop_alpha = float(cfg.READ_IR_DROP_ALPHA)
        self.prog_ir_drop_alpha = float(cfg.PROG_IR_DROP_ALPHA)

        self.enable_read_noise = bool(cfg.ENABLE_READ_NOISE)
        self.read_noise_rel_sigma = float(cfg.READ_NOISE_REL_SIGMA)

        self.enable_read_disturb = bool(cfg.ENABLE_READ_DISTURB)
        self.read_disturb_step = float(cfg.READ_DISTURB_STEP)

        self.enable_sneak_path = bool(cfg.ENABLE_SNEAK_PATH)
        self.sneak_ratio = float(cfg.SNEAK_RATIO)

        self.g_min = float(cfg.G_MIN)
        self.g_max = float(cfg.G_MAX)

        self.devices = np.empty((self.n_rows, self.n_phys_cols), dtype=object)
        base_seed = None if seed is None else int(seed)
        for i in range(self.n_rows):
            for pcol in range(self.n_phys_cols):
                dev_seed = None if base_seed is None else base_seed + 1009 * i + 37 * pcol
                self.devices[i, pcol] = MemristorDevice(seed=dev_seed)

    # ------------------------------------------------------------------
    # Pair/column helpers
    # ------------------------------------------------------------------
    def _parse_pair_id(self, pair_id: Hashable) -> Tuple[int, int]:
        if not isinstance(pair_id, tuple) or len(pair_id) != 2:
            raise ValueError("pair_id must be a tuple (row, logical_col)")
        i, j = int(pair_id[0]), int(pair_id[1])
        if not (0 <= i < self.n_rows and 0 <= j < self.n_logical_cols):
            raise IndexError(f"pair_id out of range: {(i, j)}")
        return i, j

    @staticmethod
    def _plus_col(j: int) -> int:
        return 2 * j

    @staticmethod
    def _minus_col(j: int) -> int:
        return 2 * j + 1

    def _read_position_factor(self, i: int, phys_col: int) -> float:
        r = i / max(self.n_rows - 1, 1)
        c = phys_col / max(self.n_phys_cols - 1, 1)
        factor = 1.0 - self.read_ir_drop_alpha * (0.5 * (r + c))
        return max(0.70, float(factor))

    def _program_position_factor(self, i: int, phys_col: int) -> float:
        r = i / max(self.n_rows - 1, 1)
        c = phys_col / max(self.n_phys_cols - 1, 1)
        factor = 1.0 - self.prog_ir_drop_alpha * (0.5 * (r + c))
        return max(0.70, float(factor))

    # ------------------------------------------------------------------
    # Ideal/internal state views
    # ------------------------------------------------------------------
    def read_pair_ideal(self, pair_id: Hashable) -> Tuple[float, float]:
        i, j = self._parse_pair_id(pair_id)
        jp = self._plus_col(j)
        jm = self._minus_col(j)
        return float(self.devices[i, jp].g), float(self.devices[i, jm].g)

    def get_pair_bounds(self, pair_id: Hashable) -> Tuple[float, float, float, float]:
        i, j = self._parse_pair_id(pair_id)
        jp = self._plus_col(j)
        jm = self._minus_col(j)
        dp = self.devices[i, jp]
        dm = self.devices[i, jm]
        return dp.g_min_eff, dp.g_max_eff, dm.g_min_eff, dm.g_max_eff

    def set_pair_conductance(self, pair_id: Hashable, g_plus: float, g_minus: float) -> None:
        i, j = self._parse_pair_id(pair_id)
        self.devices[i, self._plus_col(j)].set_g(float(g_plus))
        self.devices[i, self._minus_col(j)].set_g(float(g_minus))

    # ------------------------------------------------------------------
    # Measured read path
    # ------------------------------------------------------------------
    def _read_single_cell_current(self, i: int, phys_col: int) -> float:
        dev = self.devices[i, phys_col]
        g_true = float(dev.g)
        v_eff = self.read_voltage * self._read_position_factor(i, phys_col)
        i_cell = g_true * v_eff

        if self.enable_sneak_path:
            leak = max(0.0, self.g_max - g_true)
            i_cell += self.sneak_ratio * leak * v_eff

        if self.enable_read_noise and self.read_noise_rel_sigma > 0.0:
            i_cell *= float(1.0 + self.rng.normal(0.0, self.read_noise_rel_sigma))

        return float(max(i_cell, 0.0))

    def _apply_read_disturb(self, i: int, phys_col: int) -> None:
        if not self.enable_read_disturb or self.read_disturb_step <= 0.0:
            return
        dev = self.devices[i, phys_col]
        new_g = dev.g + self.read_disturb_step * (dev.g_min_eff - dev.g)
        dev.set_g(new_g)

    def _read_pair_once(self, pair_id: Hashable) -> Tuple[float, float]:
        i, j = self._parse_pair_id(pair_id)
        jp = self._plus_col(j)
        jm = self._minus_col(j)

        i_plus = self._read_single_cell_current(i, jp)
        i_minus = self._read_single_cell_current(i, jm)

        # Convert with the nominal read voltage. Position losses stay embedded in
        # the estimate to mimic real measured underestimation.
        g_plus_est = i_plus / max(self.read_voltage, 1e-18)
        g_minus_est = i_minus / max(self.read_voltage, 1e-18)

        self._apply_read_disturb(i, jp)
        self._apply_read_disturb(i, jm)

        return float(g_plus_est), float(g_minus_est)

    def read_pair(self, pair_id: Hashable) -> Tuple[float, float]:
        n = max(1, self.read_avg_samples)
        vals = [self._read_pair_once(pair_id) for _ in range(n)]
        gp = float(np.mean([v[0] for v in vals]))
        gm = float(np.mean([v[1] for v in vals]))
        return gp, gm

    # ------------------------------------------------------------------
    # Programming
    # ------------------------------------------------------------------
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

        i, j = self._parse_pair_id(pair_id)
        phys_col = self._plus_col(j) if side == "plus" else self._minus_col(j)
        dev = self.devices[i, phys_col]

        factor = self._program_position_factor(i, phys_col)
        n_eff = max(0, int(round(n_pulses * factor)))
        if n_eff <= 0:
            n_eff = 1

        dev.apply_pulse(polarity=polarity, n_pulses=n_eff)
        return int(n_eff)

    # ------------------------------------------------------------------
    # Ideal VMM (for algorithm debug) and measured pair current
    # ------------------------------------------------------------------
    def read_weight_measured(self, pair_id: Hashable) -> float:
        gp, gm = self.read_pair(pair_id)
        return float(gp - gm)

    def vmm_ideal(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.n_rows:
            raise ValueError(f"Expected input of length {self.n_rows}, got {x.size}")
        out = np.zeros(self.n_logical_cols, dtype=float)
        for j in range(self.n_logical_cols):
            acc = 0.0
            for i in range(self.n_rows):
                gp, gm = self.read_pair_ideal((i, j))
                acc += x[i] * (gp - gm)
            out[j] = acc
        return out

    def summary(self) -> dict:
        g_plus = []
        g_minus = []
        w = []
        for i in range(self.n_rows):
            for j in range(self.n_logical_cols):
                gp, gm = self.read_pair_ideal((i, j))
                g_plus.append(gp)
                g_minus.append(gm)
                w.append(gp - gm)
        return {
            "n_rows": self.n_rows,
            "n_logical_cols": self.n_logical_cols,
            "n_phys_cols": self.n_phys_cols,
            "g_plus_mean": float(np.mean(g_plus)),
            "g_minus_mean": float(np.mean(g_minus)),
            "weight_mean": float(np.mean(w)),
            "weight_std": float(np.std(w)),
            "weight_min": float(np.min(w)),
            "weight_max": float(np.max(w)),
        }
