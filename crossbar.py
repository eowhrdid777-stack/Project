from __future__ import annotations

from typing import Hashable, Literal, Tuple, Optional

import numpy as np
import config as cfg
from device_model import MemristorDevice

Side = Literal["plus", "minus"]
PulsePolarity = Literal["pot", "dep"]


class DifferentialCrossbar:
    # W[i, j] = G_plus[i, j] - G_minus[i, j]
    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        seed: Optional[int] = cfg.SEED,
        memory_type: str = "ltm",
    ) -> None:
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.memory_type = memory_type.lower()

        if self.n_rows <= 0 or self.n_cols <= 0:
            raise ValueError("n_rows and n_cols must be positive.")

        self.n_phys_cols = 2 * self.n_cols
        self.devices = np.empty((self.n_rows, self.n_phys_cols), dtype=object)

        self.g_min = float(cfg.G_MIN)
        self.g_max = float(cfg.G_MAX)

        self.read_voltage = float(getattr(cfg, "READ_VOLTAGE", 0.1))
        self.program_voltage = float(getattr(cfg, "PROGRAM_VOLTAGE", 1.0))

        self.read_ir_drop_alpha = float(getattr(cfg, "READ_IR_DROP_ALPHA", 0.05))
        self.prog_ir_drop_alpha = float(getattr(cfg, "PROG_IR_DROP_ALPHA", 0.05))

        self.enable_read_noise = bool(getattr(cfg, "ENABLE_READ_NOISE", True))
        self.read_noise_rel_sigma = float(getattr(cfg, "READ_NOISE_REL_SIGMA", 0.02))

        self.enable_read_disturb = bool(getattr(cfg, "ENABLE_READ_DISTURB", True))
        self.read_disturb_step = float(getattr(cfg, "READ_DISTURB_STEP", 0.0))

        self.enable_sneak_path = bool(getattr(cfg, "ENABLE_SNEAK_PATH", True))
        self.sneak_ratio = float(getattr(cfg, "SNEAK_RATIO", 0.001))

        # 현실적으로 여러 번 읽어 평균내는 경우도 있음
        # 단, disturb는 sample마다 실제로 반영
        self.read_avg_samples = int(getattr(cfg, "READ_AVG_SAMPLES", 1))

        self._build_array()

    # ------------------------------------------------------------
    # mapping
    # ------------------------------------------------------------
    def _plus_col(self, j: int) -> int:
        return 2 * j

    def _minus_col(self, j: int) -> int:
        return 2 * j + 1

    def _parse_pair_id(self, pair_id: Hashable) -> Tuple[int, int]:
        if not (isinstance(pair_id, tuple) and len(pair_id) == 2):
            raise ValueError(f"pair_id must be (row, logical_col), got {pair_id!r}")

        i, j = int(pair_id[0]), int(pair_id[1])

        if not (0 <= i < self.n_rows):
            raise IndexError(f"row index out of range: {i}")
        if not (0 <= j < self.n_cols):
            raise IndexError(f"logical col index out of range: {j}")

        return i, j

    # ------------------------------------------------------------
    # construction
    # ------------------------------------------------------------
    def _build_array(self) -> None:
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                jp = self._plus_col(j)
                jm = self._minus_col(j)

                s_plus = None if self.seed is None else int(self.rng.integers(1_000_000_000))
                s_minus = None if self.seed is None else int(self.rng.integers(1_000_000_000))

                self.devices[i, jp] = MemristorDevice(seed=s_plus, memory_type=self.memory_type)
                self.devices[i, jm] = MemristorDevice(seed=s_minus, memory_type=self.memory_type)

                self.devices[i, jp].reset("init")
                self.devices[i, jm].reset("init")

    # ------------------------------------------------------------
    # bounds
    # ------------------------------------------------------------
    def get_pair_bounds(self, pair_id: Hashable) -> Tuple[float, float, float, float]:
        i, j = self._parse_pair_id(pair_id)
        jp = self._plus_col(j)
        jm = self._minus_col(j)

        plus_dev = self.devices[i, jp]
        minus_dev = self.devices[i, jm]

        return (
            float(plus_dev.g_min_eff),
            float(plus_dev.g_max_eff),
            float(minus_dev.g_min_eff),
            float(minus_dev.g_max_eff),
        )

    # ------------------------------------------------------------
    # position factors
    # ------------------------------------------------------------
    def _read_position_factor(self, i: int, phys_col: int) -> float:
        r = i / max(self.n_rows - 1, 1)
        c = phys_col / max(self.n_phys_cols - 1, 1)
        factor = 1.0 - self.read_ir_drop_alpha * (0.5 * (r + c))
        return max(0.5, float(factor))

    def _program_position_factor(self, i: int, phys_col: int) -> float:
        r = i / max(self.n_rows - 1, 1)
        c = phys_col / max(self.n_phys_cols - 1, 1)
        factor = 1.0 - self.prog_ir_drop_alpha * (0.5 * (r + c))
        return max(0.5, float(factor))

    # ------------------------------------------------------------
    # ideal internal read (debug only)
    # ------------------------------------------------------------
    def read_pair_ideal(self, pair_id: Hashable) -> Tuple[float, float]:
        i, j = self._parse_pair_id(pair_id)
        jp = self._plus_col(j)
        jm = self._minus_col(j)
        return float(self.devices[i, jp].g), float(self.devices[i, jm].g)

    # ------------------------------------------------------------
    # measured current
    # ------------------------------------------------------------
    def _read_single_cell_current(self, i: int, phys_col: int) -> float:
        dev = self.devices[i, phys_col]
        g_true = float(dev.g)

        v_eff = self.read_voltage * self._read_position_factor(i, phys_col)
        i_cell = g_true * v_eff

        if self.enable_sneak_path:
            g_leak = max(0.0, self.g_max - g_true)
            i_cell += self.sneak_ratio * g_leak * v_eff

        if self.enable_read_noise:
            i_cell *= (1.0 + self.rng.normal(0.0, self.read_noise_rel_sigma))

        return float(max(i_cell, 0.0))

    def _apply_read_disturb(self, i: int, phys_col: int) -> None:
        if not self.enable_read_disturb:
            return
        if self.read_disturb_step <= 0.0:
            return

        dev = self.devices[i, phys_col]
        new_g = dev.g + self.read_disturb_step * (self.g_min - dev.g)
        dev.set_g(new_g)

    def _read_pair_once(self, pair_id: Hashable) -> Tuple[float, float]:
        i, j = self._parse_pair_id(pair_id)
        jp = self._plus_col(j)
        jm = self._minus_col(j)

        i_plus = self._read_single_cell_current(i, jp)
        i_minus = self._read_single_cell_current(i, jm)

        g_plus_est = i_plus / max(self.read_voltage, 1e-18)
        g_minus_est = i_minus / max(self.read_voltage, 1e-18)

        # 현실적으로 read disturb는 매 read마다 발생
        self._apply_read_disturb(i, jp)
        self._apply_read_disturb(i, jm)

        return float(g_plus_est), float(g_minus_est)

    # ------------------------------------------------------------
    # public measured read
    # ------------------------------------------------------------
    def read_pair(self, pair_id: Hashable) -> Tuple[float, float]:
        n = max(1, self.read_avg_samples)
        vals = [self._read_pair_once(pair_id) for _ in range(n)]

        g_plus = float(np.mean([v[0] for v in vals]))
        g_minus = float(np.mean([v[1] for v in vals]))
        return g_plus, g_minus

    # ------------------------------------------------------------
    # programming
    # ------------------------------------------------------------
    def apply_pulse(
        self,
        pair_id: Hashable,
        side: Side,
        polarity: PulsePolarity,
        n_pulses: int = 1,
    ) -> None:
        if n_pulses <= 0:
            return

        i, j = self._parse_pair_id(pair_id)
        phys_col = self._plus_col(j) if side == "plus" else self._minus_col(j)
        dev = self.devices[i, phys_col]

        prog_factor = self._program_position_factor(i, phys_col)
        n_eff = max(1, int(round(n_pulses * prog_factor)))
        dev.apply_pulse(polarity=polarity, n_pulses=n_eff)

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    def get_conductance_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        g_plus = np.zeros((self.n_rows, self.n_cols), dtype=np.float64)
        g_minus = np.zeros((self.n_rows, self.n_cols), dtype=np.float64)

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                jp = self._plus_col(j)
                jm = self._minus_col(j)
                g_plus[i, j] = self.devices[i, jp].g
                g_minus[i, j] = self.devices[i, jm].g

        return g_plus, g_minus

    def weight_matrix(self) -> np.ndarray:
        g_plus, g_minus = self.get_conductance_matrices()
        return g_plus - g_minus

    def get_pair_weight_ideal(self, pair_id: Hashable) -> float:
        g_plus, g_minus = self.read_pair_ideal(pair_id)
        return float(g_plus - g_minus)

    def get_pair_weight_measured(self, pair_id: Hashable) -> float:
        g_plus, g_minus = self.read_pair(pair_id)
        return float(g_plus - g_minus)

    def relax_all(self, dt: float = 1.0) -> None:
        for i in range(self.n_rows):
            for k in range(self.n_phys_cols):
                self.devices[i, k].relax(dt=dt)

    def reset(self, mode: str = "init") -> None:
        for i in range(self.n_rows):
            for k in range(self.n_phys_cols):
                self.devices[i, k].reset(mode=mode)

    def set_pair_conductance(
        self,
        pair_id: Hashable,
        g_plus: float,
        g_minus: float,
    ) -> None:
        i, j = self._parse_pair_id(pair_id)
        jp = self._plus_col(j)
        jm = self._minus_col(j)

        self.devices[i, jp].set_g(g_plus)
        self.devices[i, jm].set_g(g_minus)