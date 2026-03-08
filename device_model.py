"""Pulse-number-based memristor device model.

Key idea
--------
Each conductance cell is represented by an integer pulse count n.
The actual conductance is obtained from a fitted monotonic pulse-response curve:

    G(n) = G0 + P_fast * (1 - exp(-p_fast * n))
             + P_slow * (1 - exp(-p_slow * n))

For a differential pair synapse,

    W = G_plus(n_plus) - G_minus(n_minus)

This keeps the model close to a pulse-programmed real device while still being easy
to integrate with STDP / network simulation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import config as cfg


class MemristorArray:
    """Pulse-count-based memristor array.

    Shape convention:
        rows = pre-synaptic inputs
        cols = post-synaptic outputs
    """

    def __init__(
        self,
        n_rows: int = cfg.DEVICE_ROWS,
        n_cols: int = cfg.DEVICE_COLS,
        use_differential: bool = cfg.USE_DIFFERENTIAL,
        seed: Optional[int] = cfg.SEED,
    ) -> None:
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.use_differential = bool(use_differential)
        self.rng = np.random.default_rng(seed)

        # Pulse count states
        self.n_plus = np.full((self.n_rows, self.n_cols), cfg.N_PULSE_INIT, dtype=int)
        self.n_minus = (
            np.full((self.n_rows, self.n_cols), cfg.N_PULSE_INIT, dtype=int)
            if self.use_differential
            else None
        )

        # Fixed D2D multiplicative factors (applied at read time)
        if cfg.ENABLE_D2D and cfg.CV_D2D > 0.0:
            self.d2d_plus = self._lognormal_factor(cfg.CV_D2D, size=(self.n_rows, self.n_cols))
            self.d2d_minus = (
                self._lognormal_factor(cfg.CV_D2D, size=(self.n_rows, self.n_cols))
                if self.use_differential
                else None
            )
        else:
            self.d2d_plus = np.ones((self.n_rows, self.n_cols), dtype=float)
            self.d2d_minus = (
                np.ones((self.n_rows, self.n_cols), dtype=float) if self.use_differential else None
            )

        # Solve P_slow so that G(N_PULSE_MAX) approximately reaches G_MAX.
        self.g0 = float(cfg.G_INIT)
        self.p_fast_amp = float(cfg.P_FAST_AMP)
        self.p_fast_rate = float(cfg.P_FAST_RATE)
        self.p_slow_rate = float(cfg.P_SLOW_RATE)
        self.p_slow_amp = self._solve_p_slow_amp()

    # -----------------------------------------------------------------
    # Core pulse-response model
    # -----------------------------------------------------------------
    def conductance_from_pulses(self, n: np.ndarray | int) -> np.ndarray:
        """Map pulse count -> conductance using a monotonic dual-exponential curve."""
        n = np.asarray(n, dtype=float)
        g = (
            self.g0
            + self.p_fast_amp * (1.0 - np.exp(-self.p_fast_rate * n))
            + self.p_slow_amp * (1.0 - np.exp(-self.p_slow_rate * n))
        )
        return np.clip(g, cfg.G_MIN, cfg.G_MAX)

    def _solve_p_slow_amp(self) -> float:
        """Choose P_slow so that G(N_PULSE_MAX) ~= G_MAX."""
        n_end = float(cfg.N_PULSE_MAX)
        denom = 1.0 - np.exp(-self.p_slow_rate * n_end)
        denom = max(denom, 1e-15)
        used_by_fast = self.p_fast_amp * (1.0 - np.exp(-self.p_fast_rate * n_end))
        p_slow = (cfg.G_MAX - self.g0 - used_by_fast) / denom
        return max(0.0, float(p_slow))

    # -----------------------------------------------------------------
    # Read path
    # -----------------------------------------------------------------
    def read_pair_conductance(self, noisy: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (G_plus, G_minus)."""
        g_plus = self.conductance_from_pulses(self.n_plus)
        g_plus = self._apply_read_variation(g_plus, self.d2d_plus, noisy=noisy)

        if not self.use_differential:
            return g_plus, None

        assert self.n_minus is not None
        assert self.d2d_minus is not None
        g_minus = self.conductance_from_pulses(self.n_minus)
        g_minus = self._apply_read_variation(g_minus, self.d2d_minus, noisy=noisy)
        return g_plus, g_minus

    def get_effective_weights(self, noisy: bool = True) -> np.ndarray:
        """Return effective synaptic weights.

        Differential mode:
            W = G_plus - G_minus
        Single mode:
            W = G_plus
        """
        g_plus, g_minus = self.read_pair_conductance(noisy=noisy)
        if g_minus is None:
            return g_plus
        return g_plus - g_minus

    # -----------------------------------------------------------------
    # Update path
    # -----------------------------------------------------------------
    def apply_weight_delta(self, delta_w: np.ndarray) -> None:
        """Map abstract learning delta to pulse-count updates.

        Differential mode:
            delta_w > 0 -> increment n_plus
            delta_w < 0 -> increment n_minus

        Single mode:
            delta_w > 0 -> increment n_plus
            delta_w < 0 -> decrement n_plus   (rough simplification)

        Note:
            The differential case is the main intended use.
        """
        delta_w = np.asarray(delta_w, dtype=float)
        if delta_w.shape != self.n_plus.shape:
            raise ValueError(
                f"delta_w shape {delta_w.shape} does not match array shape {self.n_plus.shape}"
            )

        n_step = self._delta_to_pulse_step(delta_w)

        pos_mask = delta_w > 0
        neg_mask = delta_w < 0

        if np.any(pos_mask):
            self.n_plus[pos_mask] += n_step[pos_mask]

        if self.use_differential:
            if np.any(neg_mask):
                assert self.n_minus is not None
                self.n_minus[neg_mask] += n_step[neg_mask]
        else:
            if np.any(neg_mask):
                self.n_plus[neg_mask] -= n_step[neg_mask]

        self.clip_pulse_counts()
        self.pair_soft_reset_if_needed()
        
    def pair_soft_reset_if_needed(self) -> None:
        if (not self.use_differential) or (not cfg.ENABLE_PAIR_RESET):
            return

        assert self.n_minus is not None

        g_plus, g_minus = self.read_pair_conductance(noisy=True)

        threshold = cfg.PAIR_RESET_THRESHOLD * cfg.G_MAX
        reset_step = cfg.PAIR_RESET_STEP

        mask = np.maximum(g_plus, g_minus) > threshold

        if np.any(mask):
            self.n_plus[mask] -= reset_step
            self.n_minus[mask] -= reset_step
            self.clip_pulse_counts()

    def apply_potentiation(self, mask: np.ndarray, n_step: int = 1, target: str = "plus") -> None:
        """Manual pulse update for debugging / controlled experiments."""
        self._validate_mask(mask)
        n_step = int(n_step)

        if target == "plus":
            self.n_plus[mask] += n_step
        elif target == "minus":
            if not self.use_differential or self.n_minus is None:
                raise ValueError("minus target requires differential mode")
            self.n_minus[mask] += n_step
        else:
            raise ValueError(f"Unknown target: {target}")

        self.clip_pulse_counts()
        self.pair_soft_reset_if_needed()

    def reset(self, mode: str = "init") -> None:
        """Reset pulse-count states.

        mode:
            zero   -> all counts = 0
            mid    -> half of N_PULSE_MAX
            random -> uniform integer in [N_PULSE_MIN, N_PULSE_MAX]
        """
        if mode == "init":
            self.n_plus.fill(cfg.N_PULSE_INIT)
            if self.use_differential and self.n_minus is not None:
                self.n_minus.fill(cfg.N_PULSE_INIT)

        elif mode == "mid":
            mid = (cfg.N_PULSE_MIN + cfg.N_PULSE_MAX) // 2
            self.n_plus.fill(mid)
            if self.use_differential and self.n_minus is not None:
                self.n_minus.fill(mid)

        elif mode == "random":
            self.n_plus = self.rng.integers(
                low=cfg.N_PULSE_MIN,
                high=cfg.N_PULSE_MAX + 1,
                size=(self.n_rows, self.n_cols),
                endpoint=True,
            )
            if self.use_differential and self.n_minus is not None:
                self.n_minus = self.rng.integers(
                    low=cfg.N_PULSE_MIN,
                    high=cfg.N_PULSE_MAX + 1,
                    size=(self.n_rows, self.n_cols),
                    endpoint=True,
                )
        else:
            raise ValueError(f"Unknown reset mode: {mode}")

        self.clip_pulse_counts()
        self.pair_soft_reset_if_needed()

    def clip_pulse_counts(self) -> None:
        self.n_plus = np.clip(self.n_plus, cfg.N_PULSE_MIN, cfg.N_PULSE_MAX)
        if self.use_differential and self.n_minus is not None:
            self.n_minus = np.clip(self.n_minus, cfg.N_PULSE_MIN, cfg.N_PULSE_MAX)

    # -----------------------------------------------------------------
    # Helper functions
    # -----------------------------------------------------------------
    def _delta_to_pulse_step(self, delta_w: np.ndarray) -> np.ndarray:
        """Convert abstract learning magnitude to integer pulse steps."""
        mag = np.abs(delta_w)
        n_step = np.ceil(mag * cfg.PULSE_SCALE).astype(int)
        n_step = np.clip(n_step, 0, cfg.MAX_PULSE_STEP)
        return n_step

    def _apply_read_variation(self, g: np.ndarray, d2d_factor: np.ndarray, noisy: bool) -> np.ndarray:
        out = g * d2d_factor

        if noisy and cfg.ENABLE_C2C and cfg.CV_C2C > 0.0:
            out = out * self._lognormal_factor(cfg.CV_C2C, size=out.shape)

        if noisy and cfg.READ_NOISE_STD > 0.0:
            out = out + self.rng.normal(0.0, cfg.READ_NOISE_STD, size=out.shape)

        return np.clip(out, cfg.G_MIN, cfg.G_MAX)

    def _lognormal_factor(self, cv: float, size=None) -> np.ndarray:
        cv = float(cv)
        sigma = np.sqrt(np.log(cv * cv + 1.0))
        mu = -0.5 * sigma * sigma
        return np.exp(mu + sigma * self.rng.standard_normal(size=size))

    def _validate_mask(self, mask: np.ndarray) -> None:
        if mask.shape != self.n_plus.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match array shape {self.n_plus.shape}"
            )


# ---------------------------------------------------------------------
# Plot / debug helpers
# ---------------------------------------------------------------------
def plot_device_example(save_path: Optional[str] = None) -> None:
    """Visualize pulse-count conductance curve and an example differential weight."""
    dev = MemristorArray(n_rows=1, n_cols=1, use_differential=True, seed=cfg.SEED)

    n_axis = np.arange(cfg.N_PULSE_MIN, cfg.N_PULSE_MAX + 1)
    g_axis = dev.conductance_from_pulses(n_axis)

    # Example: n_plus fixed sweep and n_minus fixed point
    n_minus_fixed = 40
    w_axis = g_axis - dev.conductance_from_pulses(np.full_like(n_axis, n_minus_fixed))

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), constrained_layout=True)

    axes[0].plot(n_axis, g_axis, linewidth=2)
    axes[0].set_title("Monotonic pulse-count conductance curve")
    axes[0].set_xlabel("Pulse count n")
    axes[0].set_ylabel("Conductance (S)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(n_axis, w_axis, linewidth=2)
    axes[1].axhline(0.0, linewidth=1)
    axes[1].set_title(f"Example differential weight: W = G+(n+) - G-(n-={n_minus_fixed})")
    axes[1].set_xlabel("n_plus")
    axes[1].set_ylabel("Effective weight (S)")
    axes[1].grid(True, alpha=0.3)

    if save_path is not None:
        fig.savefig(save_path, dpi=180)
    else:
        plt.show()

if __name__ == "__main__":
    dev = MemristorArray(n_rows=1, n_cols=1, use_differential=True, seed=cfg.SEED)

    # 초기화
    dev.reset(mode="init")

    total_cycles = 600   # N_PULSE_MAX=200이어도 soft reset 반복을 보기 위해 더 길게 봄
    n_step = 3           # 너무 작으면 reset이 늦고, 너무 크면 그래프가 거칠어짐

    cycle_axis = []
    g_plus_hist = []
    g_minus_hist = []
    weight_hist = []
    n_plus_hist = []
    n_minus_hist = []
    reset_cycles = []

    for t in range(total_cycles):
        # update 전 pulse count 저장
        n_plus_before = dev.n_plus.copy()
        n_minus_before = dev.n_minus.copy()

        # G+만 계속 potentiation
        mask = np.array([[True]])
        dev.apply_potentiation(mask=mask, n_step=n_step, target="plus")

        # update 후 읽기 (검증용이므로 noise 없이)
        g_plus, g_minus = dev.read_pair_conductance(noisy=True)
        weight = (g_plus - g_minus)[0, 0]

        cycle_axis.append(t)
        g_plus_hist.append(g_plus[0, 0])
        g_minus_hist.append(g_minus[0, 0])
        weight_hist.append(weight)
        n_plus_hist.append(dev.n_plus[0, 0])
        n_minus_hist.append(dev.n_minus[0, 0])

        # soft reset 발생 여부 판정
        # 원래라면 plus만 n_step만큼 증가해야 하는데,
        # 실제 결과가 그보다 작으면 reset으로 둘 다 감소한 것
        expected_plus_after = min(n_plus_before[0, 0] + n_step, cfg.N_PULSE_MAX)
        expected_minus_after = n_minus_before[0, 0]  # 원래 minus는 그대로여야 함

        if (dev.n_plus[0, 0] < expected_plus_after) or (dev.n_minus[0, 0] < expected_minus_after):
            reset_cycles.append(t)

    # 배열로 변환
    cycle_axis = np.array(cycle_axis)
    g_plus_hist = np.array(g_plus_hist)
    g_minus_hist = np.array(g_minus_hist)
    weight_hist = np.array(weight_hist)
    n_plus_hist = np.array(n_plus_hist)
    n_minus_hist = np.array(n_minus_hist)

    # threshold 선
    threshold = cfg.PAIR_RESET_THRESHOLD * cfg.G_MAX

    # ---------------- 그래프 ----------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

    # 1) G+
    axes[0].plot(cycle_axis, g_plus_hist, label="G+", linewidth=2)
    axes[0].axhline(threshold, linestyle="--", linewidth=1, label="Reset threshold")
    for rc in reset_cycles:
        axes[0].axvline(rc, linestyle=":", alpha=0.4)
    axes[0].set_title("G+ over cycles")
    axes[0].set_xlabel("Cycle")
    axes[0].set_ylabel("Conductance (S)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2) G-
    axes[1].plot(cycle_axis, g_minus_hist, label="G-", linewidth=2)
    for rc in reset_cycles:
        axes[1].axvline(rc, linestyle=":", alpha=0.4)
    axes[1].set_title("G- over cycles")
    axes[1].set_xlabel("Cycle")
    axes[1].set_ylabel("Conductance (S)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 3) Effective weight G = G+ - G-
    axes[2].plot(cycle_axis, weight_hist, label="G = G+ - G-", linewidth=2)
    for rc in reset_cycles:
        axes[2].axvline(rc, linestyle=":", alpha=0.4)
    axes[2].set_title("Effective weight over cycles")
    axes[2].set_xlabel("Cycle")
    axes[2].set_ylabel("Weight (S)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.show()

    print(
        f"before: n_plus={n_plus_before}, n_minus={n_minus_before} | "
        f"after: n_plus={dev.n_plus[0,0]}, n_minus={dev.n_minus[0,0]}"
    )