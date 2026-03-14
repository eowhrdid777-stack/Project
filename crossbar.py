# crossbar_array

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
    ) -> None:
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)  # logical differential columns
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if self.n_rows <= 0 or self.n_cols <= 0:
            raise ValueError(
                f"n_rows and n_cols must be positive, got n_rows={n_rows}, n_cols={n_cols}"
            )

        self.n_phys_cols = 2 * self.n_cols
        self.devices = np.empty((self.n_rows, self.n_phys_cols), dtype=object)

        self.g_min = float(cfg.G_MIN)
        self.g_max = float(cfg.G_MAX)

        # Read / program voltages
        self.read_voltage = float(getattr(cfg, "READ_VOLTAGE", 0.1))
        self.program_voltage = float(getattr(cfg, "PROGRAM_VOLTAGE", 1.0))

        # IR-drop proxy strengths
        self.read_ir_drop_alpha = float(getattr(cfg, "READ_IR_DROP_ALPHA", 0.15))
        self.prog_ir_drop_alpha = float(getattr(cfg, "PROG_IR_DROP_ALPHA", 0.15))

        self._build_array()

    # ------------------------------------------------------------------
    # Physical mapping helpers
    # ------------------------------------------------------------------
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

    def _get_device(self, pair_id: Hashable, side: Side) -> MemristorDevice:
        i, j = self._parse_pair_id(pair_id)
        phys_col = self._plus_col(j) if side == "plus" else self._minus_col(j)
        return self.devices[i, phys_col]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_array(self) -> None:
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                jp = self._plus_col(j)
                jm = self._minus_col(j)

                s_plus = None if self.seed is None else int(self.rng.integers(1_000_000_000))
                s_minus = None if self.seed is None else int(self.rng.integers(1_000_000_000))

                self.devices[i, jp] = MemristorDevice(seed=s_plus)
                self.devices[i, jm] = MemristorDevice(seed=s_minus)

                self.devices[i, jp].reset("init")
                self.devices[i, jm].reset("init")

    # ------------------------------------------------------------------
    # Position-dependent proxy factors
    # ------------------------------------------------------------------
    def _read_position_factor(self, i: int, phys_col: int) -> float:
        # factor = 1 − alpha*(row_position + col_position)/2 > 먼 cell일수록 sensing이 약해져 conductance underestimate
        r = i / max(self.n_rows - 1, 1)
        c = phys_col / max(self.n_phys_cols - 1, 1)

        factor = 1.0 - self.read_ir_drop_alpha * (0.5 * (r + c))
        return max(0.5, float(factor))

    def _program_position_factor(self, i: int, phys_col: int) -> float:
        # factor = 1 − alpha*(row_position + col_position)/2 > 먼 cell일수록 programming이 약해져 conductance change underestimate
        r = i / max(self.n_rows - 1, 1)
        c = phys_col / max(self.n_phys_cols - 1, 1)

        factor = 1.0 - self.prog_ir_drop_alpha * (0.5 * (r + c))
        return max(0.5, float(factor))

    # ------------------------------------------------------------------
    # Ideal internal state read
    # ------------------------------------------------------------------
    def read_pair_ideal(self, pair_id: Hashable) -> Tuple[float, float]:
        # Return ideal internal conductances (not sensed values).
        i, j = self._parse_pair_id(pair_id)
        jp = self._plus_col(j)
        jm = self._minus_col(j)
        return float(self.devices[i, jp].g), float(self.devices[i, jm].g)

    # ------------------------------------------------------------------
    # Current-based cell read helper
    # ------------------------------------------------------------------
    def _read_single_cell_current(
        self,
        i: int,
        phys_col: int,
        noisy: bool = True,
    ) -> float:
        # I = G_true * V_eff with noisy read and optional sneak path proxy.
        dev = self.devices[i, phys_col]
        g_ideal = float(dev.g)

        # Effective read voltage after simple IR-drop proxy
        v_eff = self.read_voltage * self._read_position_factor(i, phys_col)

        # Ideal selected-cell current
        i_cell = g_ideal * v_eff

        # Sneak current proxy: add a fraction of current from "background"
        if bool(getattr(cfg, "ENABLE_SNEAK_PATH", False)):
            sneak_ratio = float(getattr(cfg, "SNEAK_RATIO", 0.02))
            # simple proxy: leakage proportional to unused conductance headroom
            g_leak = max(0.0, self.g_max - g_ideal)
            i_cell += sneak_ratio * g_leak * v_eff

        # Read noise on current measurement
        if noisy and bool(getattr(cfg, "ENABLE_READ_NOISE", False)):
            sigma_rel = float(getattr(cfg, "READ_NOISE_REL_SIGMA", 0.02))
            i_cell *= 1.0 + self.rng.normal(0.0, sigma_rel) # i_cell *= (1 + normal noise)

        return float(max(i_cell, 0.0))

    def _apply_read_disturb(self, i: int, phys_col: int) -> None:
        # Simple read disturb proxy: repeated reads drift conductance toward G_MIN.
        if not bool(getattr(cfg, "ENABLE_READ_DISTURB", False)):
            return

        disturb_step = float(getattr(cfg, "READ_DISTURB_STEP", 0.0))
        if disturb_step <= 0.0:
            return

        dev = self.devices[i, phys_col]
        dev.state.g += disturb_step * (self.g_min - dev.g) # g ← g + step*(g_min − g)
        dev.state.g = float(np.clip(dev.state.g, self.g_min, self.g_max))

    # ------------------------------------------------------------------
    # Measured pair read for controller
    # ------------------------------------------------------------------
    def read_pair(
        self,
        pair_id: Hashable,
        noisy: bool = True,
        disturb: bool = True,
    ) -> Tuple[float, float]:
        # ideal G -> read current I = G * V_eff -> estimate G_est = I / V_read
        i, j = self._parse_pair_id(pair_id)
        jp = self._plus_col(j)
        jm = self._minus_col(j)

        i_plus = self._read_single_cell_current(i, jp, noisy=noisy)
        i_minus = self._read_single_cell_current(i, jm, noisy=noisy)

        # Convert measured current back to estimated conductance using nominal V_read
        g_plus_est = i_plus / max(self.read_voltage, 1e-18)
        g_minus_est = i_minus / max(self.read_voltage, 1e-18)

        # Optional disturb modifies ideal internal state after the read
        if disturb:
            self._apply_read_disturb(i, jp)
            self._apply_read_disturb(i, jm)

        return float(g_plus_est), float(g_minus_est)

    # ------------------------------------------------------------------
    # Programming pulse routing
    # ------------------------------------------------------------------
    def apply_pulse(
        self,
        pair_id: Hashable,
        side: Side,
        polarity: PulsePolarity,
        n_pulses: int = 1,
    ) -> None:
        # Apply pulse(s) to one selected device.
        if n_pulses <= 0:
            return

        i, j = self._parse_pair_id(pair_id)
        phys_col = self._plus_col(j) if side == "plus" else self._minus_col(j)
        dev = self.devices[i, phys_col]

        prog_factor = self._program_position_factor(i, phys_col)
        n_eff = max(1, int(round(n_pulses * prog_factor))) # n_eff = n_pulses * position_factor

        dev.apply_pulse(polarity=polarity, n_pulses=n_eff)

    # ------------------------------------------------------------------
    # Array inspection helpers
    # ------------------------------------------------------------------
    def get_conductance_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        # Return ideal internal conductance matrices (G_plus, G_minus).
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

    def get_pair_weight_measured(
        self,
        pair_id: Hashable,
        noisy: bool = True,
        disturb: bool = True,
    ) -> float:
        g_plus, g_minus = self.read_pair(pair_id, noisy=noisy, disturb=disturb)
        return float(g_plus - g_minus)

    def snapshot_pair(self, pair_id: Hashable) -> dict:
        i, j = self._parse_pair_id(pair_id)
        jp = self._plus_col(j)
        jm = self._minus_col(j)

        plus_snap = self.devices[i, jp].snapshot()
        minus_snap = self.devices[i, jm].snapshot()

        return {
            "pair_id": (i, j),
            "phys_plus_col": jp,
            "phys_minus_col": jm,
            "g_plus_ideal": plus_snap.g,
            "g_minus_ideal": minus_snap.g,
            "weight_ideal": plus_snap.g - minus_snap.g,
            "plus_pulse_count_pot": plus_snap.pulse_count_pot,
            "plus_pulse_count_dep": plus_snap.pulse_count_dep,
            "minus_pulse_count_pot": minus_snap.pulse_count_pot,
            "minus_pulse_count_dep": minus_snap.pulse_count_dep,
        }

    # ------------------------------------------------------------------
    # VMM
    # ------------------------------------------------------------------
    def vmm(
        self,
        x: np.ndarray,
        measured: bool = False,
        noisy: bool = True,
        disturb: bool = False,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)

        if x.shape != (self.n_rows,):
            raise ValueError(f"x must have shape ({self.n_rows},), got {x.shape}")

        y = np.zeros(self.n_cols, dtype=np.float64)

        # for j:
        #    for i:
        #        acc += (G+ − G−)*x[i]
        for j in range(self.n_cols):
            acc = 0.0
            for i in range(self.n_rows):
                if measured:
                    g_p, g_m = self.read_pair((i, j), noisy=noisy, disturb=disturb)
                else:
                    g_p, g_m = self.read_pair_ideal((i, j))

                acc += (g_p - g_m) * x[i]

            y[j] = acc

        return y

    # ------------------------------------------------------------------
    # Maintenance helpers
    # ------------------------------------------------------------------
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

    def summary(self) -> dict:
        g_plus, g_minus = self.get_conductance_matrices()
        w = g_plus - g_minus

        return {
            "n_rows": self.n_rows,
            "n_logical_cols": self.n_cols,
            "n_physical_cols": self.n_phys_cols,
            "g_plus_mean": float(np.mean(g_plus)),
            "g_minus_mean": float(np.mean(g_minus)),
            "weight_mean": float(np.mean(w)),
            "weight_std": float(np.std(w)),
            "weight_min": float(np.min(w)),
            "weight_max": float(np.max(w)),
        }


if __name__ == "__main__":
    cb = DifferentialCrossbar(n_rows=3, n_cols=2, seed=cfg.SEED)

    print("Initial summary:")
    print(cb.summary())

    pair_id = (0, 1)
    print("\nSnapshot before programming:")
    print(cb.snapshot_pair(pair_id))

    for _ in range(20):
        cb.apply_pulse(pair_id, side="plus", polarity="pot", n_pulses=1)

    print("\nTrue pair read:")
    print(cb.read_pair_ideal(pair_id))

    print("\nMeasured pair read:")
    print(cb.read_pair(pair_id, noisy=True, disturb=False))

    x = np.array([1.0, 0.0, 1.0], dtype=np.float64)

    print("\nIdeal VMM:")
    print(cb.vmm(x, measured=False))

    print("\nMeasured VMM:")
    print(cb.vmm(x, measured=True, noisy=True, disturb=False))