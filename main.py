# from __future__ import annotations

# import math
# import traceback

# import matplotlib.pyplot as plt
# import numpy as np

# import config as cfg
# from device_model import MemristorDevice
# from crossbar import DifferentialCrossbar
# from conductance_modulation import ConductanceModulationController


# def assert_true(cond: bool, msg: str) -> None:
#     if not cond:
#         raise AssertionError(msg)


# def print_header(title: str) -> None:
#     print("\n" + "=" * 72)
#     print(title)
#     print("=" * 72)


# def test_config_sanity() -> None:
#     print_header("TEST 1: config sanity")

#     print(f"G_MIN = {cfg.G_MIN:.6e}")
#     print(f"G_MAX = {cfg.G_MAX:.6e}")
#     print(f"P_MAX = {cfg.P_MAX}")
#     print(f"G_INIT_MODE = {cfg.G_INIT_MODE}")
#     print(f"COMMON_MODE_TARGET = {cfg.COMMON_MODE_TARGET:.6e}")
#     print(f"TEST_PAIR = {cfg.TEST_PAIR}")
#     print(f"TEST_ARRAY_ROWS = {cfg.TEST_ARRAY_ROWS}")
#     print(f"TEST_ARRAY_COLS = {cfg.TEST_ARRAY_COLS}")
#     print(f"TEST_N_STEPS = {cfg.TEST_N_STEPS}")

#     assert_true(cfg.G_MIN > 0.0, "G_MIN must be positive")
#     assert_true(cfg.G_MAX > cfg.G_MIN, "G_MAX must be larger than G_MIN")
#     assert_true(cfg.P_MAX >= 2, "P_MAX must be at least 2")
#     assert_true(cfg.G_INIT_MODE in ("min", "mid", "max"), "G_INIT_MODE must be min/mid/max")
#     assert_true(0 <= cfg.TEST_PAIR[0] < cfg.TEST_ARRAY_ROWS, "TEST_PAIR row out of range")
#     assert_true(0 <= cfg.TEST_PAIR[1] < cfg.TEST_ARRAY_COLS, "TEST_PAIR col out of range")

#     expected_cm = 0.5 * (cfg.G_MIN + cfg.G_MAX)
#     assert_true(
#         math.isclose(cfg.COMMON_MODE_TARGET, expected_cm, rel_tol=0.0, abs_tol=1e-18),
#         "COMMON_MODE_TARGET should be the midpoint of G_MIN and G_MAX"
#     )

#     print("PASS: config sanity")


# def test_device_model() -> None:
#     print_header("TEST 2: device_model basic behavior")

#     dev = MemristorDevice(seed=cfg.SEED)

#     print(f"effective window: [{dev.g_min_eff:.6e}, {dev.g_max_eff:.6e}]")
#     print(f"initial g = {dev.g:.6e}, level_idx = {dev.state.level_idx}")

#     assert_true(dev.g_min_eff > 0.0, "effective g_min must be positive")
#     assert_true(dev.g_max_eff > dev.g_min_eff, "effective g_max must exceed effective g_min")
#     assert_true(dev.n_levels == cfg.P_MAX, "device n_levels must match cfg.P_MAX")
#     assert_true(0 <= dev.state.level_idx < dev.n_levels, "initial level index out of range")
#     assert_true(dev.g_min_eff <= dev.g <= dev.g_max_eff, "initial conductance out of range")

#     # reset test
#     dev.reset("min")
#     g_min = dev.g
#     idx_min = dev.state.level_idx

#     dev.reset("mid")
#     g_mid = dev.g
#     idx_mid = dev.state.level_idx

#     dev.reset("max")
#     g_max = dev.g
#     idx_max = dev.state.level_idx

#     print(f"reset min: g={g_min:.6e}, idx={idx_min}")
#     print(f"reset mid: g={g_mid:.6e}, idx={idx_mid}")
#     print(f"reset max: g={g_max:.6e}, idx={idx_max}")

#     assert_true(idx_min == 0, "min reset should go to index 0")
#     assert_true(idx_max == dev.n_levels - 1, "max reset should go to last index")
#     assert_true(idx_min < idx_mid < idx_max, "mid index should lie between min and max")
#     assert_true(g_min <= g_mid <= g_max, "conductance ordering after reset is wrong")

#     # potentiation monotonicity
#     dev.reset("mid")
#     pot_trace = [dev.g]
#     for _ in range(10):
#         dev.apply_pot_pulse(1)
#         pot_trace.append(dev.g)

#     print("pot trace:", [f"{x:.3e}" for x in pot_trace])
#     assert_true(
#         all(pot_trace[k + 1] >= pot_trace[k] - 1e-18 for k in range(len(pot_trace) - 1)),
#         "potentiation trace must be nondecreasing"
#     )

#     # depression monotonicity
#     dev.reset("mid")
#     dep_trace = [dev.g]
#     for _ in range(10):
#         dev.apply_dep_pulse(1)
#         dep_trace.append(dev.g)

#     print("dep trace:", [f"{x:.3e}" for x in dep_trace])
#     assert_true(
#         all(dep_trace[k + 1] <= dep_trace[k] + 1e-18 for k in range(len(dep_trace) - 1)),
#         "depression trace must be nonincreasing"
#     )

#     # direct set test
#     target = 0.5 * (dev.g_min_eff + dev.g_max_eff)
#     dev.set_g(target)
#     print(f"set_g target={target:.6e}, actual={dev.g:.6e}, idx={dev.state.level_idx}")
#     assert_true(dev.g_min_eff <= dev.g <= dev.g_max_eff, "set_g put conductance outside window")

#     print("PASS: device_model")


# def test_crossbar_basic() -> None:
#     print_header("TEST 3: crossbar basic pair behavior")

#     cb = DifferentialCrossbar(
#         n_rows=cfg.TEST_ARRAY_ROWS,
#         n_cols=cfg.TEST_ARRAY_COLS,
#         seed=cfg.SEED,
#     )

#     print("summary:", cb.summary())

#     pair = tuple(cfg.TEST_PAIR)

#     gp_i0, gm_i0 = cb.read_pair_ideal(pair)
#     gp_m0, gm_m0 = cb.read_pair(pair)

#     print(f"initial ideal read   : G+={gp_i0:.6e}, G-={gm_i0:.6e}, W={gp_i0-gm_i0:.6e}")
#     print(f"initial measured read: G+={gp_m0:.6e}, G-={gm_m0:.6e}, W={gp_m0-gm_m0:.6e}")

#     bounds = cb.get_pair_bounds(pair)
#     print("pair bounds:", bounds)

#     assert_true(len(bounds) == 4, "get_pair_bounds must return four values")
#     assert_true(bounds[0] < bounds[1], "plus bounds invalid")
#     assert_true(bounds[2] < bounds[3], "minus bounds invalid")

#     # Set pair to known mid state
#     gp_mid = 0.5 * (bounds[0] + bounds[1])
#     gm_mid = 0.5 * (bounds[2] + bounds[3])
#     cb.set_pair_conductance(pair, gp_mid, gm_mid)

#     gp_i1, gm_i1 = cb.read_pair_ideal(pair)
#     print(f"after set_pair_conductance ideal: G+={gp_i1:.6e}, G-={gm_i1:.6e}")

#     assert_true(bounds[0] <= gp_i1 <= bounds[1], "set_pair_conductance made invalid G+")
#     assert_true(bounds[2] <= gm_i1 <= bounds[3], "set_pair_conductance made invalid G-")

#     # One pulse on plus-pot should increase ideal G+
#     gp_before, gm_before = cb.read_pair_ideal(pair)
#     n_eff = cb.apply_pulse(pair, side="plus", polarity="pot", n_pulses=1)
#     gp_after, gm_after = cb.read_pair_ideal(pair)

#     print(f"apply plus-pot: n_eff={n_eff}, G+ {gp_before:.6e} -> {gp_after:.6e}")
#     assert_true(n_eff >= 1, "effective pulse count should be >= 1")
#     assert_true(gp_after >= gp_before - 1e-18, "plus-pot should not decrease ideal G+")
#     assert_true(abs(gm_after - gm_before) < 1e-18, "plus-pot should not change ideal G-")

#     # One pulse on minus-dep should decrease ideal G-
#     gp_before2, gm_before2 = cb.read_pair_ideal(pair)
#     n_eff2 = cb.apply_pulse(pair, side="minus", polarity="dep", n_pulses=1)
#     gp_after2, gm_after2 = cb.read_pair_ideal(pair)

#     print(f"apply minus-dep: n_eff={n_eff2}, G- {gm_before2:.6e} -> {gm_after2:.6e}")
#     assert_true(n_eff2 >= 1, "effective pulse count should be >= 1")
#     assert_true(gm_after2 <= gm_before2 + 1e-18, "minus-dep should not increase ideal G-")
#     assert_true(abs(gp_after2 - gp_before2) < 1e-18, "minus-dep should not change ideal G+")

#     # Measured weight path
#     w_meas = cb.read_weight_measured(pair)
#     print(f"measured weight after programming = {w_meas:.6e}")

#     # Ideal VMM
#     x = np.ones(cb.n_rows, dtype=float)
#     y = cb.vmm_ideal(x)
#     print("ideal VMM output:", y)
#     assert_true(y.shape == (cb.n_logical_cols,), "vmm_ideal output shape mismatch")

#     print("PASS: crossbar")


# def test_controller_basic() -> None:
#     print_header("TEST 4: conductance modulation controller basic behavior")

#     cb = DifferentialCrossbar(
#         n_rows=cfg.TEST_ARRAY_ROWS,
#         n_cols=cfg.TEST_ARRAY_COLS,
#         seed=cfg.SEED,
#     )
#     ctrl = ConductanceModulationController(cb)
#     pair = tuple(cfg.TEST_PAIR)

#     bounds = cb.get_pair_bounds(pair)
#     gp_mid = 0.5 * (bounds[0] + bounds[1])
#     gm_mid = 0.5 * (bounds[2] + bounds[3])
#     cb.set_pair_conductance(pair, gp_mid, gm_mid)

#     status0 = ctrl.get_pair_status(pair)
#     print("initial status:", status0)

#     # Step 1: increase weight
#     res_up = ctrl.update_weight(pair_id=pair, direction=+1, step_idx=1)
#     print("update +1 result:", res_up)

#     assert_true(res_up.success, "update_weight(+1) should succeed")
#     assert_true(res_up.weight_after >= res_up.weight_before - 5e-6,
#                 "weight should generally increase for direction +1")
#     assert_true(res_up.chosen_action in (
#         "plus-pot", "minus-dep", "refresh-remap"
#     ), "unexpected chosen_action for +1")

#     # Step 2: decrease weight
#     res_dn = ctrl.update_weight(pair_id=pair, direction=-1, step_idx=2)
#     print("update -1 result:", res_dn)

#     assert_true(res_dn.success, "update_weight(-1) should succeed")
#     assert_true(res_dn.chosen_action in (
#         "plus-dep", "minus-pot", "refresh-remap"
#     ), "unexpected chosen_action for -1")

#     # Force refresh condition by pushing pair near an edge
#     gp_min, gp_max, gm_min, gm_max = cb.get_pair_bounds(pair)
#     cb.set_pair_conductance(pair, gp_max, gm_min)

#     res_refresh = ctrl.update_weight(pair_id=pair, direction=+1, step_idx=max(cfg.REFRESH_MIN_INTERVAL + 5, 20))
#     print("forced refresh result:", res_refresh)

#     assert_true(res_refresh.success, "refresh path should succeed")
#     if res_refresh.did_refresh:
#         print("refresh/remap was actually triggered as expected")
#     else:
#         print("refresh did not trigger; this can happen if measured read/noise made headroom still look acceptable")

#     print("PASS: controller basic")


# def run_demo_plots() -> None:
#     print_header("TEST 5: integration demo with plots")

#     cb = DifferentialCrossbar(
#         n_rows=cfg.TEST_ARRAY_ROWS,
#         n_cols=cfg.TEST_ARRAY_COLS,
#         seed=cfg.SEED,
#     )
#     ctrl = ConductanceModulationController(cb)
#     pair = tuple(cfg.TEST_PAIR)

#     # Start from center
#     gp_min, gp_max, gm_min, gm_max = cb.get_pair_bounds(pair)
#     gp_mid = 0.5 * (gp_min + gp_max)
#     gm_mid = 0.5 * (gm_min + gm_max)
#     cb.set_pair_conductance(pair, gp_mid, gm_mid)

#     # -------------------------------
#     # Modified schedule:
#     # short positive phase, long negative phase
#     # -------------------------------
#     n_steps = 320
#     change_step = 40   # 0~39: weight increase, 40~319: weight decrease

#     step_hist = []
#     gp_hist = []
#     gm_hist = []
#     w_hist = []
#     cm_hist = []
#     act_hist = []
#     refresh_hist = []
#     ideal_gp_hist = []
#     ideal_gm_hist = []
#     ideal_w_hist = []

#     for step in range(n_steps):
#         direction = +1 if step < change_step else -1
#         res = ctrl.update_weight(pair_id=pair, direction=direction, step_idx=step)

#         gp_i, gm_i = cb.read_pair_ideal(pair)
#         w_i = gp_i - gm_i

#         step_hist.append(step)
#         gp_hist.append(res.g_plus_final)
#         gm_hist.append(res.g_minus_final)
#         w_hist.append(res.weight_after)
#         cm_hist.append(res.common_mode_after)
#         act_hist.append(res.chosen_action)
#         refresh_hist.append(1 if res.did_refresh else 0)
#         ideal_gp_hist.append(gp_i)
#         ideal_gm_hist.append(gm_i)
#         ideal_w_hist.append(w_i)

#     print("last 20 actions:", act_hist[-20:])
#     print(f"num refreshes = {sum(refresh_hist)}")
#     print(f"measured weight min = {min(w_hist):.6e}")
#     print(f"measured weight max = {max(w_hist):.6e}")
#     print(f"ideal weight min    = {min(ideal_w_hist):.6e}")
#     print(f"ideal weight max    = {max(ideal_w_hist):.6e}")

#     if not cfg.TEST_ENABLE_PLOTS:
#         return

#     plt.figure(figsize=(10, 5))
#     plt.plot(step_hist, gp_hist, label="G+ measured")
#     plt.plot(step_hist, gm_hist, label="G- measured")
#     plt.plot(step_hist, ideal_gp_hist, "--", alpha=0.6, label="G+ ideal")
#     plt.plot(step_hist, ideal_gm_hist, "--", alpha=0.6, label="G- ideal")
#     for s, flag in zip(step_hist, refresh_hist):
#         if flag:
#             plt.axvline(s, linestyle=":", alpha=0.25)
#     plt.title("Pair conductance evolution")
#     plt.xlabel("step")
#     plt.ylabel("conductance (S)")
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     plt.figure(figsize=(10, 5))
#     plt.plot(step_hist, w_hist, label="Effective conductance measured (G+ - G-)")
#     plt.plot(step_hist, ideal_w_hist, "--", alpha=0.7, label="Effective conductance ideal")
#     plt.axhline(min(w_hist), linestyle=":", label=f"measured min = {min(w_hist):.3e}")
#     for s, flag in zip(step_hist, refresh_hist):
#         if flag:
#             plt.axvline(s, linestyle=":", alpha=0.25)
#     plt.title("Effective differential conductance")
#     plt.xlabel("step")
#     plt.ylabel("conductance (S)")
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     plt.figure(figsize=(10, 5))
#     plt.plot(step_hist, cm_hist, label="Common-mode measured")
#     plt.axhline(cfg.COMMON_MODE_TARGET, linestyle="--", label="CM target")
#     for s, flag in zip(step_hist, refresh_hist):
#         if flag:
#             plt.axvline(s, linestyle=":", alpha=0.25)
#     plt.title("Common-mode evolution")
#     plt.xlabel("step")
#     plt.ylabel("conductance (S)")
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     plt.figure(figsize=(10, 4))
#     action_to_num = {
#         "plus-pot": 0,
#         "minus-dep": 1,
#         "plus-dep": 2,
#         "minus-pot": 3,
#         "refresh-remap": 4,
#     }
#     y = [action_to_num.get(a, -1) for a in act_hist]
#     plt.plot(step_hist, y, ".", markersize=4)
#     plt.yticks(
#         [0, 1, 2, 3, 4],
#         ["plus-pot", "minus-dep", "plus-dep", "minus-pot", "refresh-remap"]
#     )
#     plt.title("Chosen controller action per step")
#     plt.xlabel("step")
#     plt.grid(True, alpha=0.3)

#     plt.show()

#     print("PASS: integration demo")


# def main() -> None:
#     np.set_printoptions(precision=4, suppress=True)

#     test_config_sanity()
#     test_device_model()
#     test_crossbar_basic()
#     test_controller_basic()
#     run_demo_plots()

#     print_header("ALL TESTS FINISHED")
#     print("If no AssertionError appeared, the four modules are at least mutually consistent.")
#     print("The plots should help you see whether:")
#     print("1) device pot/dep is monotonic,")
#     print("2) crossbar measured read is reasonable,")
#     print("3) controller uses one-sided balanced actions,")
#     print("4) refresh/remap happens when headroom becomes tight.")


# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         print("\nTEST FAILED")
#         print(type(e).__name__, e)
#         traceback.print_exc()
#         raise


from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from device_model import MemristorDevice


def collect_single_cycle_trace(dev: MemristorDevice, n_pulses: int = 64):
    """
    One cycle on ONE device:
      - reset to min, apply 64 pot pulses
      - reset to max, apply 64 dep pulses
    Returns:
      pot_trace: length n_pulses+1
      dep_trace: length n_pulses+1
    """
    dev.reset("min")
    pot_trace = [dev.g]
    for _ in range(n_pulses):
        dev.apply_pot_pulse(1)
        pot_trace.append(dev.g)

    dev.reset("max")
    dep_trace = [dev.g]
    for _ in range(n_pulses):
        dev.apply_dep_pulse(1)
        dep_trace.append(dev.g)

    return np.array(pot_trace), np.array(dep_trace)


def test_c2c_same_device(n_cycles: int = 100, n_pulses: int = 64):
    """
    C2C only:
    Use ONE device instance repeatedly.
    D2D is fixed because it is the same physical device.
    C2C appears because pulse response is sampled every cycle/pulse.
    """
    dev = MemristorDevice(seed=cfg.SEED)

    pot_cycles = []
    dep_cycles = []

    for _ in range(n_cycles):
        pot_trace, dep_trace = collect_single_cycle_trace(dev, n_pulses=n_pulses)
        pot_cycles.append(pot_trace)
        dep_cycles.append(dep_trace)

    pot_cycles = np.array(pot_cycles)   # shape: (n_cycles, n_pulses+1)
    dep_cycles = np.array(dep_cycles)

    pot_mean = pot_cycles.mean(axis=0)
    pot_std = pot_cycles.std(axis=0, ddof=1)

    dep_mean = dep_cycles.mean(axis=0)
    dep_std = dep_cycles.std(axis=0, ddof=1)

    x = np.arange(n_pulses + 1)

    print("\n===== C2C (same device, repeated cycles) =====")
    print(f"n_cycles = {n_cycles}, n_pulses = {n_pulses}")
    print(f"pot end mean = {pot_mean[-1]:.6e}, std = {pot_std[-1]:.6e}")
    print(f"dep end mean = {dep_mean[-1]:.6e}, std = {dep_std[-1]:.6e}")

    plt.figure(figsize=(9, 5))
    for k in range(min(n_cycles, 30)):
        plt.plot(x, pot_cycles[k], alpha=0.25)
    plt.plot(x, pot_mean, linewidth=2, label="mean")
    plt.fill_between(x, pot_mean - pot_std, pot_mean + pot_std, alpha=0.25, label="±1σ")
    plt.title("C2C only: 64 potentiation pulses on the same device")
    plt.xlabel("pulse count")
    plt.ylabel("conductance (S)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.figure(figsize=(9, 5))
    for k in range(min(n_cycles, 30)):
        plt.plot(x, dep_cycles[k], alpha=0.25)
    plt.plot(x, dep_mean, linewidth=2, label="mean")
    plt.fill_between(x, dep_mean - dep_std, dep_mean + dep_std, alpha=0.25, label="±1σ")
    plt.title("C2C only: 64 depression pulses on the same device")
    plt.xlabel("pulse count")
    plt.ylabel("conductance (S)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    return {
        "pot_cycles": pot_cycles,
        "dep_cycles": dep_cycles,
        "pot_mean": pot_mean,
        "pot_std": pot_std,
        "dep_mean": dep_mean,
        "dep_std": dep_std,
    }


def test_d2d_multiple_devices(n_devices: int = 40, n_pulses: int = 64):
    """
    D2D only-ish:
    For each device, run one 64-pot and one 64-dep trace.
    This mainly shows device-to-device window variation.

    Strictly speaking, current code still has C2C enabled during the run if cfg says so.
    If you want pure D2D, set:
        ENABLE_C2C_VARIATION = False
    before running.
    """
    pot_devices = []
    dep_devices = []
    gmin_list = []
    gmax_list = []

    for k in range(n_devices):
        dev = MemristorDevice(seed=cfg.SEED + k)
        gmin_list.append(dev.g_min_eff)
        gmax_list.append(dev.g_max_eff)

        pot_trace, dep_trace = collect_single_cycle_trace(dev, n_pulses=n_pulses)
        pot_devices.append(pot_trace)
        dep_devices.append(dep_trace)

    pot_devices = np.array(pot_devices)   # shape: (n_devices, n_pulses+1)
    dep_devices = np.array(dep_devices)
    gmin_arr = np.array(gmin_list)
    gmax_arr = np.array(gmax_list)

    pot_mean = pot_devices.mean(axis=0)
    pot_std = pot_devices.std(axis=0, ddof=1)

    dep_mean = dep_devices.mean(axis=0)
    dep_std = dep_devices.std(axis=0, ddof=1)

    x = np.arange(n_pulses + 1)

    print("\n===== D2D (multiple devices) =====")
    print(f"n_devices = {n_devices}, n_pulses = {n_pulses}")
    print(f"g_min_eff mean = {gmin_arr.mean():.6e}, std = {gmin_arr.std(ddof=1):.6e}")
    print(f"g_max_eff mean = {gmax_arr.mean():.6e}, std = {gmax_arr.std(ddof=1):.6e}")
    print(f"pot end mean   = {pot_mean[-1]:.6e}, std = {pot_std[-1]:.6e}")
    print(f"dep end mean   = {dep_mean[-1]:.6e}, std = {dep_std[-1]:.6e}")

    plt.figure(figsize=(9, 5))
    for k in range(min(n_devices, 30)):
        plt.plot(x, pot_devices[k], alpha=0.25)
    plt.plot(x, pot_mean, linewidth=2, label="mean")
    plt.fill_between(x, pot_mean - pot_std, pot_mean + pot_std, alpha=0.25, label="±1σ")
    plt.title("D2D: 64 potentiation pulses across multiple devices")
    plt.xlabel("pulse count")
    plt.ylabel("conductance (S)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.figure(figsize=(9, 5))
    for k in range(min(n_devices, 30)):
        plt.plot(x, dep_devices[k], alpha=0.25)
    plt.plot(x, dep_mean, linewidth=2, label="mean")
    plt.fill_between(x, dep_mean - dep_std, dep_mean + dep_std, alpha=0.25, label="±1σ")
    plt.title("D2D: 64 depression pulses across multiple devices")
    plt.xlabel("pulse count")
    plt.ylabel("conductance (S)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.figure(figsize=(9, 5))
    plt.plot(gmin_arr, label="g_min_eff")
    plt.plot(gmax_arr, label="g_max_eff")
    plt.title("D2D effective window across devices")
    plt.xlabel("device index")
    plt.ylabel("conductance (S)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    return {
        "pot_devices": pot_devices,
        "dep_devices": dep_devices,
        "gmin_arr": gmin_arr,
        "gmax_arr": gmax_arr,
        "pot_mean": pot_mean,
        "pot_std": pot_std,
        "dep_mean": dep_mean,
        "dep_std": dep_std,
    }


def test_both_d2d_and_c2c(n_devices: int = 20, n_cycles: int = 20, n_pulses: int = 64):
    """
    Both D2D and C2C:
    many devices, many cycles per device.
    """
    pot_all = []
    dep_all = []

    for d in range(n_devices):
        dev = MemristorDevice(seed=cfg.SEED + d)
        pot_cycles = []
        dep_cycles = []

        for _ in range(n_cycles):
            pot_trace, dep_trace = collect_single_cycle_trace(dev, n_pulses=n_pulses)
            pot_cycles.append(pot_trace)
            dep_cycles.append(dep_trace)

        pot_all.append(pot_cycles)
        dep_all.append(dep_cycles)

    pot_all = np.array(pot_all)   # (n_devices, n_cycles, n_pulses+1)
    dep_all = np.array(dep_all)

    pot_mean = pot_all.mean(axis=(0, 1))
    pot_std = pot_all.std(axis=(0, 1), ddof=1)

    dep_mean = dep_all.mean(axis=(0, 1))
    dep_std = dep_all.std(axis=(0, 1), ddof=1)

    x = np.arange(n_pulses + 1)

    print("\n===== BOTH D2D + C2C =====")
    print(f"n_devices = {n_devices}, n_cycles = {n_cycles}, n_pulses = {n_pulses}")
    print(f"pot end mean = {pot_mean[-1]:.6e}, std = {pot_std[-1]:.6e}")
    print(f"dep end mean = {dep_mean[-1]:.6e}, std = {dep_std[-1]:.6e}")

    plt.figure(figsize=(9, 5))
    plt.plot(x, pot_mean, linewidth=2, label="mean")
    plt.fill_between(x, pot_mean - pot_std, pot_mean + pot_std, alpha=0.25, label="±1σ")
    plt.title("D2D + C2C: potentiation")
    plt.xlabel("pulse count")
    plt.ylabel("conductance (S)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.figure(figsize=(9, 5))
    plt.plot(x, dep_mean, linewidth=2, label="mean")
    plt.fill_between(x, dep_mean - dep_std, dep_mean + dep_std, alpha=0.25, label="±1σ")
    plt.title("D2D + C2C: depression")
    plt.xlabel("pulse count")
    plt.ylabel("conductance (S)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    return {
        "pot_all": pot_all,
        "dep_all": dep_all,
        "pot_mean": pot_mean,
        "pot_std": pot_std,
        "dep_mean": dep_mean,
        "dep_std": dep_std,
    }


if __name__ == "__main__":
    # 1) same device repeated cycles -> C2C
    c2c = test_c2c_same_device(n_cycles=100, n_pulses=64)

    # 2) multiple devices -> D2D
    d2d = test_d2d_multiple_devices(n_devices=40, n_pulses=64)

    # 3) both together
    both = test_both_d2d_and_c2c(n_devices=20, n_cycles=20, n_pulses=64)

    plt.show()