# crossbar_array

from __future__ import annotations

from typing import Hashable, Literal, Tuple, Optional

import numpy as np
import config as cfg
from device_model import MemristorDevice

Side = Literal["plus", "minus"]
PulsePolarity = Literal["pot", "dep"]


class DifferentialCrossbar:
    """
    Differential memristor crossbar with physical '+ - + -' column layout.

    Logical synapse:
        W[i, j] = G_plus[i, j] - G_minus[i, j]

    Physical layout:
        plus  column of logical j -> physical col = 2*j
        minus column of logical j -> physical col = 2*j + 1

    Abstraction level:
        - device state is stored as internal conductance in MemristorDevice
        - measured read is current-based:
              I = G * V_eff
              G_est = I / V_read
        - simple array-level non-idealities are approximated:
              IR drop proxy
              sneak current proxy
              read noise
              read disturb
        - exact nodal analysis / full resistive network solving is NOT included
    """

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
        """
        Proxy for read IR drop.
        Farther cells see a weaker effective read voltage.
        """
        r = i / max(self.n_rows - 1, 1)
        c = phys_col / max(self.n_phys_cols - 1, 1)

        factor = 1.0 - self.read_ir_drop_alpha * (0.5 * (r + c))
        return max(0.5, float(factor))

    def _program_position_factor(self, i: int, phys_col: int) -> float:
        """
        Proxy for programming IR drop / write inefficiency.
        """
        r = i / max(self.n_rows - 1, 1)
        c = phys_col / max(self.n_phys_cols - 1, 1)

        factor = 1.0 - self.prog_ir_drop_alpha * (0.5 * (r + c))
        return max(0.5, float(factor))

    # ------------------------------------------------------------------
    # Ideal internal state read
    # ------------------------------------------------------------------
    def read_pair_true(self, pair_id: Hashable) -> Tuple[float, float]:
        """
        Return true internal conductances (not sensed values).
        """
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
        """
        Read one cell current using:
            I = G_true * V_eff
        with optional read noise and sneak current proxy.
        """
        dev = self.devices[i, phys_col]
        g_true = float(dev.g)

        # Effective read voltage after simple IR-drop proxy
        v_eff = self.read_voltage * self._read_position_factor(i, phys_col)

        # Ideal selected-cell current
        i_cell = g_true * v_eff

        # Sneak current proxy: add a fraction of current from "background"
        if bool(getattr(cfg, "ENABLE_SNEAK_PATH", False)):
            sneak_ratio = float(getattr(cfg, "SNEAK_RATIO", 0.02))
            # simple proxy: leakage proportional to unused conductance headroom
            g_leak = max(0.0, self.g_max - g_true)
            i_cell += sneak_ratio * g_leak * v_eff

        # Read noise on current measurement
        if noisy and bool(getattr(cfg, "ENABLE_READ_NOISE", False)):
            sigma_rel = float(getattr(cfg, "READ_NOISE_REL_SIGMA", 0.02))
            i_cell *= 1.0 + self.rng.normal(0.0, sigma_rel)

        return float(max(i_cell, 0.0))

    def _apply_read_disturb(self, i: int, phys_col: int) -> None:
        """
        Simple read disturb proxy: repeated reads drift conductance toward G_MIN.
        This is only a compact behavioral approximation.
        """
        if not bool(getattr(cfg, "ENABLE_READ_DISTURB", False)):
            return

        disturb_step = float(getattr(cfg, "READ_DISTURB_STEP", 0.0))
        if disturb_step <= 0.0:
            return

        dev = self.devices[i, phys_col]
        dev.state.g += disturb_step * (self.g_min - dev.g)
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
        """
        Measured pair read.

        Flow:
            true G -> read current I = G * V_eff -> estimate G_est = I / V_read

        Returned values are sensed conductance estimates, not true internal states.
        """
        i, j = self._parse_pair_id(pair_id)
        jp = self._plus_col(j)
        jm = self._minus_col(j)

        i_plus = self._read_single_cell_current(i, jp, noisy=noisy)
        i_minus = self._read_single_cell_current(i, jm, noisy=noisy)

        # Convert measured current back to estimated conductance using nominal V_read
        g_plus_est = i_plus / max(self.read_voltage, 1e-18)
        g_minus_est = i_minus / max(self.read_voltage, 1e-18)

        # Optional disturb modifies true internal state after the read
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
        """
        Apply pulse(s) to one selected device.

        Array-level write non-ideality is approximated by converting requested
        pulse count to an effective pulse count via position-dependent factor.
        """
        if n_pulses <= 0:
            return

        i, j = self._parse_pair_id(pair_id)
        phys_col = self._plus_col(j) if side == "plus" else self._minus_col(j)
        dev = self.devices[i, phys_col]

        prog_factor = self._program_position_factor(i, phys_col)
        n_eff = max(1, int(round(n_pulses * prog_factor)))

        dev.apply_pulse(polarity=polarity, n_pulses=n_eff)

    # ------------------------------------------------------------------
    # Array inspection helpers
    # ------------------------------------------------------------------
    def get_conductance_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return true internal conductance matrices (G_plus, G_minus).
        Shape: (n_rows, n_cols)
        """
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

    def get_pair_weight_true(self, pair_id: Hashable) -> float:
        g_plus, g_minus = self.read_pair_true(pair_id)
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
            "g_plus_true": plus_snap.g,
            "g_minus_true": minus_snap.g,
            "weight_true": plus_snap.g - minus_snap.g,
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
        """
        Vector-matrix multiplication.

        measured=False:
            uses true internal conductances

        measured=True:
            uses current-based sensed conductance estimate per cell
            (therefore includes read non-idealities)
        """
        x = np.asarray(x, dtype=np.float64)

        if x.shape != (self.n_rows,):
            raise ValueError(f"x must have shape ({self.n_rows},), got {x.shape}")

        y = np.zeros(self.n_cols, dtype=np.float64)

        for j in range(self.n_cols):
            acc = 0.0
            for i in range(self.n_rows):
                if measured:
                    g_p, g_m = self.read_pair((i, j), noisy=noisy, disturb=disturb)
                else:
                    g_p, g_m = self.read_pair_true((i, j))

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
    print(cb.read_pair_true(pair_id))

    print("\nMeasured pair read:")
    print(cb.read_pair(pair_id, noisy=True, disturb=False))

    x = np.array([1.0, 0.0, 1.0], dtype=np.float64)

    print("\nIdeal VMM:")
    print(cb.vmm(x, measured=False))

    print("\nMeasured VMM:")
    print(cb.vmm(x, measured=True, noisy=True, disturb=False))