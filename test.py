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

 # -------------------------------------------------------------------------
# from __future__ import annotations

# import matplotlib.pyplot as plt
# import numpy as np

# import config as cfg
# from device_model import MemristorDevice


# def collect_single_cycle_trace(dev: MemristorDevice, n_pulses: int = 64):
#     """
#     One cycle on ONE device:
#       - reset to min, apply 64 pot pulses
#       - reset to max, apply 64 dep pulses
#     Returns:
#       pot_trace: length n_pulses+1
#       dep_trace: length n_pulses+1
#     """
#     dev.reset("min")
#     pot_trace = [dev.g]
#     for _ in range(n_pulses):
#         dev.apply_pot_pulse(1)
#         pot_trace.append(dev.g)

#     dev.reset("max")
#     dep_trace = [dev.g]
#     for _ in range(n_pulses):
#         dev.apply_dep_pulse(1)
#         dep_trace.append(dev.g)

#     return np.array(pot_trace), np.array(dep_trace)


# def test_c2c_same_device(n_cycles: int = 100, n_pulses: int = 64):
#     """
#     C2C only:
#     Use ONE device instance repeatedly.
#     D2D is fixed because it is the same physical device.
#     C2C appears because pulse response is sampled every cycle/pulse.
#     """
#     dev = MemristorDevice(seed=cfg.SEED)

#     pot_cycles = []
#     dep_cycles = []

#     for _ in range(n_cycles):
#         pot_trace, dep_trace = collect_single_cycle_trace(dev, n_pulses=n_pulses)
#         pot_cycles.append(pot_trace)
#         dep_cycles.append(dep_trace)

#     pot_cycles = np.array(pot_cycles)   # shape: (n_cycles, n_pulses+1)
#     dep_cycles = np.array(dep_cycles)

#     pot_mean = pot_cycles.mean(axis=0)
#     pot_std = pot_cycles.std(axis=0, ddof=1)

#     dep_mean = dep_cycles.mean(axis=0)
#     dep_std = dep_cycles.std(axis=0, ddof=1)

#     x = np.arange(n_pulses + 1)

#     print("\n===== C2C (same device, repeated cycles) =====")
#     print(f"n_cycles = {n_cycles}, n_pulses = {n_pulses}")
#     print(f"pot end mean = {pot_mean[-1]:.6e}, std = {pot_std[-1]:.6e}")
#     print(f"dep end mean = {dep_mean[-1]:.6e}, std = {dep_std[-1]:.6e}")

#     plt.figure(figsize=(9, 5))
#     for k in range(min(n_cycles, 30)):
#         plt.plot(x, pot_cycles[k], alpha=0.25)
#     plt.plot(x, pot_mean, linewidth=2, label="mean")
#     plt.fill_between(x, pot_mean - pot_std, pot_mean + pot_std, alpha=0.25, label="±1σ")
#     plt.title("C2C only: 64 potentiation pulses on the same device")
#     plt.xlabel("pulse count")
#     plt.ylabel("conductance (S)")
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     plt.figure(figsize=(9, 5))
#     for k in range(min(n_cycles, 30)):
#         plt.plot(x, dep_cycles[k], alpha=0.25)
#     plt.plot(x, dep_mean, linewidth=2, label="mean")
#     plt.fill_between(x, dep_mean - dep_std, dep_mean + dep_std, alpha=0.25, label="±1σ")
#     plt.title("C2C only: 64 depression pulses on the same device")
#     plt.xlabel("pulse count")
#     plt.ylabel("conductance (S)")
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     return {
#         "pot_cycles": pot_cycles,
#         "dep_cycles": dep_cycles,
#         "pot_mean": pot_mean,
#         "pot_std": pot_std,
#         "dep_mean": dep_mean,
#         "dep_std": dep_std,
#     }


# def test_d2d_multiple_devices(n_devices: int = 40, n_pulses: int = 64):
#     """
#     D2D only-ish:
#     For each device, run one 64-pot and one 64-dep trace.
#     This mainly shows device-to-device window variation.

#     Strictly speaking, current code still has C2C enabled during the run if cfg says so.
#     If you want pure D2D, set:
#         ENABLE_C2C_VARIATION = False
#     before running.
#     """
#     pot_devices = []
#     dep_devices = []
#     gmin_list = []
#     gmax_list = []

#     for k in range(n_devices):
#         dev = MemristorDevice(seed=cfg.SEED + k)
#         gmin_list.append(dev.g_min_eff)
#         gmax_list.append(dev.g_max_eff)

#         pot_trace, dep_trace = collect_single_cycle_trace(dev, n_pulses=n_pulses)
#         pot_devices.append(pot_trace)
#         dep_devices.append(dep_trace)

#     pot_devices = np.array(pot_devices)   # shape: (n_devices, n_pulses+1)
#     dep_devices = np.array(dep_devices)
#     gmin_arr = np.array(gmin_list)
#     gmax_arr = np.array(gmax_list)

#     pot_mean = pot_devices.mean(axis=0)
#     pot_std = pot_devices.std(axis=0, ddof=1)

#     dep_mean = dep_devices.mean(axis=0)
#     dep_std = dep_devices.std(axis=0, ddof=1)

#     x = np.arange(n_pulses + 1)

#     print("\n===== D2D (multiple devices) =====")
#     print(f"n_devices = {n_devices}, n_pulses = {n_pulses}")
#     print(f"g_min_eff mean = {gmin_arr.mean():.6e}, std = {gmin_arr.std(ddof=1):.6e}")
#     print(f"g_max_eff mean = {gmax_arr.mean():.6e}, std = {gmax_arr.std(ddof=1):.6e}")
#     print(f"pot end mean   = {pot_mean[-1]:.6e}, std = {pot_std[-1]:.6e}")
#     print(f"dep end mean   = {dep_mean[-1]:.6e}, std = {dep_std[-1]:.6e}")

#     plt.figure(figsize=(9, 5))
#     for k in range(min(n_devices, 30)):
#         plt.plot(x, pot_devices[k], alpha=0.25)
#     plt.plot(x, pot_mean, linewidth=2, label="mean")
#     plt.fill_between(x, pot_mean - pot_std, pot_mean + pot_std, alpha=0.25, label="±1σ")
#     plt.title("D2D: 64 potentiation pulses across multiple devices")
#     plt.xlabel("pulse count")
#     plt.ylabel("conductance (S)")
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     plt.figure(figsize=(9, 5))
#     for k in range(min(n_devices, 30)):
#         plt.plot(x, dep_devices[k], alpha=0.25)
#     plt.plot(x, dep_mean, linewidth=2, label="mean")
#     plt.fill_between(x, dep_mean - dep_std, dep_mean + dep_std, alpha=0.25, label="±1σ")
#     plt.title("D2D: 64 depression pulses across multiple devices")
#     plt.xlabel("pulse count")
#     plt.ylabel("conductance (S)")
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     plt.figure(figsize=(9, 5))
#     plt.plot(gmin_arr, label="g_min_eff")
#     plt.plot(gmax_arr, label="g_max_eff")
#     plt.title("D2D effective window across devices")
#     plt.xlabel("device index")
#     plt.ylabel("conductance (S)")
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     return {
#         "pot_devices": pot_devices,
#         "dep_devices": dep_devices,
#         "gmin_arr": gmin_arr,
#         "gmax_arr": gmax_arr,
#         "pot_mean": pot_mean,
#         "pot_std": pot_std,
#         "dep_mean": dep_mean,
#         "dep_std": dep_std,
#     }


# def test_both_d2d_and_c2c(n_devices: int = 20, n_cycles: int = 20, n_pulses: int = 64):
#     """
#     Both D2D and C2C:
#     many devices, many cycles per device.
#     """
#     pot_all = []
#     dep_all = []

#     for d in range(n_devices):
#         dev = MemristorDevice(seed=cfg.SEED + d)
#         pot_cycles = []
#         dep_cycles = []

#         for _ in range(n_cycles):
#             pot_trace, dep_trace = collect_single_cycle_trace(dev, n_pulses=n_pulses)
#             pot_cycles.append(pot_trace)
#             dep_cycles.append(dep_trace)

#         pot_all.append(pot_cycles)
#         dep_all.append(dep_cycles)

#     pot_all = np.array(pot_all)   # (n_devices, n_cycles, n_pulses+1)
#     dep_all = np.array(dep_all)

#     pot_mean = pot_all.mean(axis=(0, 1))
#     pot_std = pot_all.std(axis=(0, 1), ddof=1)

#     dep_mean = dep_all.mean(axis=(0, 1))
#     dep_std = dep_all.std(axis=(0, 1), ddof=1)

#     x = np.arange(n_pulses + 1)

#     print("\n===== BOTH D2D + C2C =====")
#     print(f"n_devices = {n_devices}, n_cycles = {n_cycles}, n_pulses = {n_pulses}")
#     print(f"pot end mean = {pot_mean[-1]:.6e}, std = {pot_std[-1]:.6e}")
#     print(f"dep end mean = {dep_mean[-1]:.6e}, std = {dep_std[-1]:.6e}")

#     plt.figure(figsize=(9, 5))
#     plt.plot(x, pot_mean, linewidth=2, label="mean")
#     plt.fill_between(x, pot_mean - pot_std, pot_mean + pot_std, alpha=0.25, label="±1σ")
#     plt.title("D2D + C2C: potentiation")
#     plt.xlabel("pulse count")
#     plt.ylabel("conductance (S)")
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     plt.figure(figsize=(9, 5))
#     plt.plot(x, dep_mean, linewidth=2, label="mean")
#     plt.fill_between(x, dep_mean - dep_std, dep_mean + dep_std, alpha=0.25, label="±1σ")
#     plt.title("D2D + C2C: depression")
#     plt.xlabel("pulse count")
#     plt.ylabel("conductance (S)")
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     return {
#         "pot_all": pot_all,
#         "dep_all": dep_all,
#         "pot_mean": pot_mean,
#         "pot_std": pot_std,
#         "dep_mean": dep_mean,
#         "dep_std": dep_std,
#     }


# if __name__ == "__main__":
#     # 1) same device repeated cycles -> C2C
#     c2c = test_c2c_same_device(n_cycles=100, n_pulses=64)

#     # 2) multiple devices -> D2D
#     d2d = test_d2d_multiple_devices(n_devices=40, n_pulses=64)

#     # 3) both together
#     both = test_both_d2d_and_c2c(n_devices=20, n_cycles=20, n_pulses=64)

#     plt.show()

# ----------------------------------------------------------

# from __future__ import annotations

# import numpy as np

# import config as cfg
# from device_model import MemristorDevice
# from crossbar import DifferentialCrossbar
# from conductance_modulation import ConductanceModulationController
# from encoding import SensorSpikeEncoder
# from neuron import MemristiveLIFOutputLayer


# def section(title: str) -> None:
#     print("\n" + "=" * 80)
#     print(title)
#     print("=" * 80)



# def test_device_model() -> None:
#     section("1) device_model.py smoke test")
#     dev = MemristorDevice(seed=cfg.SEED)

#     s0 = dev.snapshot()
#     print(f"initial: g={s0.g:.6e}, level={s0.level_idx}")

#     dev.apply_pulse("pot", n_pulses=5)
#     s1 = dev.snapshot()
#     print(f"after 5 pot pulses: g={s1.g:.6e}, level={s1.level_idx}")
#     assert s1.level_idx >= s0.level_idx, "pot should not decrease level"

#     dev.apply_pulse("dep", n_pulses=3)
#     s2 = dev.snapshot()
#     print(f"after 3 dep pulses: g={s2.g:.6e}, level={s2.level_idx}")
#     assert s2.level_idx <= s1.level_idx, "dep should not increase level"

#     dev.set_g(0.5 * (dev.g_min_eff + dev.g_max_eff))
#     s3 = dev.snapshot()
#     print(f"after set_g(mid): g={s3.g:.6e}, level={s3.level_idx}")
#     assert dev.g_min_eff <= s3.g <= dev.g_max_eff



# def test_crossbar() -> None:
#     section("2) crossbar.py smoke test")
#     xbar = DifferentialCrossbar(n_rows=4, n_cols=3, seed=cfg.SEED)
#     print("summary:", xbar.summary())

#     pair_id = (1, 1)
#     gp0, gm0 = xbar.read_pair_ideal(pair_id)
#     print(f"pair {pair_id} ideal before: gp={gp0:.6e}, gm={gm0:.6e}, w={gp0-gm0:.6e}")

#     xbar.apply_pulse(pair_id, side="plus", polarity="pot", n_pulses=4)
#     xbar.apply_pulse(pair_id, side="minus", polarity="dep", n_pulses=2)
#     gp1, gm1 = xbar.read_pair_ideal(pair_id)
#     print(f"pair {pair_id} ideal after:  gp={gp1:.6e}, gm={gm1:.6e}, w={gp1-gm1:.6e}")
#     assert (gp1 - gm1) >= (gp0 - gm0), "plus-pot / minus-dep should tend to increase weight"

#     gp_meas, gm_meas = xbar.read_pair(pair_id)
#     print(f"pair {pair_id} measured read: gp={gp_meas:.6e}, gm={gm_meas:.6e}, w={gp_meas-gm_meas:.6e}")

#     x = np.array([1.0, 0.5, 0.0, 0.25])
#     y = xbar.vmm_ideal(x)
#     print("vmm_ideal input:", x)
#     print("vmm_ideal output:", y)
#     assert y.shape == (3,)



# def test_conductance_controller() -> None:
#     section("3) conductance_modulation.py smoke test")
#     xbar = DifferentialCrossbar(n_rows=4, n_cols=4, seed=cfg.SEED)
#     ctrl = ConductanceModulationController(xbar)
#     pair_id = (0, 0)

#     status0 = ctrl.get_pair_status(pair_id)
#     print("initial status:", status0)

#     for step in range(6):
#         res = ctrl.update_weight(pair_id, direction=+1, step_idx=step)
#         print(
#             f"step={step:02d} action={res.chosen_action:>14s} "
#             f"w_before={res.weight_before:.6e} w_after={res.weight_after:.6e} "
#             f"cm_after={res.common_mode_after:.6e} refresh={res.did_refresh}"
#         )

#     status1 = ctrl.get_pair_status(pair_id)
#     print("final status:", status1)
#     assert status1["weight_measured"] >= status0["weight_measured"] - 1e-12



# def test_encoding() -> None:
#     section("4) encoding.py smoke test")
#     obs = {"distance": 52.0, "temperature": 31.0, "gas": 0.18}
#     value_ranges = {
#         "distance": (0.0, 200.0),
#         "temperature": (0.0, 80.0),
#         "gas": (0.0, 1.0),
#     }

#     for mode in ("rate", "population_rate", "population_latency"):
#         print(f"\nmode={mode}")
#         enc = SensorSpikeEncoder(
#             feature_names=["distance", "temperature", "gas"],
#             value_ranges=value_ranges,
#             mode=mode,
#             neurons_per_feature=5,
#             latency_steps=8,
#             seed=cfg.SEED,
#             dt=1e-3,
#             max_rate_hz=200.0,
#         )

#         if mode == "population_latency":
#             window = enc.encode_window(obs)
#             print("feature names:", window[0].feature_names)
#             print("spike times :", window[0].spike_times.tolist())
#             total_spikes = np.sum([w.spikes for w in window], axis=0)
#             print("spikes per step:")
#             for t, out in enumerate(window):
#                 print(f"  t={t}: {out.spikes.tolist()}")
#             print("total emitted per channel:", total_spikes.tolist())
#             assert len(window) == 8
#             assert total_spikes.shape[0] == 15
#             assert np.all(total_spikes <= 1), "latency mode should emit at most one spike per channel"
#         else:
#             out = enc.encode(obs, sim_step=0)
#             print("feature names:", out.feature_names)
#             print("firing_rates :", np.round(out.firing_rates, 4).tolist())
#             print("spikes       :", out.spikes.tolist())
#             expected_dim = 3 if mode == "rate" else 15
#             assert out.spikes.shape == (expected_dim,)



# def test_end_to_end_five_file_integration() -> None:
#     section("5) end-to-end integration test (encoding -> crossbar -> controller)")
#     obs = {"distance": 52.0, "temperature": 31.0, "gas": 0.18}

#     enc = SensorSpikeEncoder(
#         feature_names=["distance", "temperature", "gas"],
#         value_ranges={
#             "distance": (0.0, 200.0),
#             "temperature": (0.0, 80.0),
#             "gas": (0.0, 1.0),
#         },
#         mode="population_latency",
#         neurons_per_feature=4,
#         latency_steps=6,
#         seed=cfg.SEED,
#         dt=1e-3,
#         max_rate_hz=200.0,
#     )

#     input_dim = enc.output_dim
#     n_actions = 4
#     xbar = DifferentialCrossbar(n_rows=input_dim, n_cols=n_actions, seed=cfg.SEED)
#     ctrl = ConductanceModulationController(xbar)

#     print(f"encoder output_dim={input_dim}, crossbar rows={xbar.n_rows}, actions={n_actions}")

#     for i in range(input_dim):
#         ctrl.update_weight((i, 0), direction=+1, step_idx=i)
#         ctrl.update_weight((i, 0), direction=+1, step_idx=i + 100)

#     window = enc.encode_window(obs)
#     spike_matrix = np.stack([w.spikes for w in window], axis=0)
#     integrated_input = spike_matrix.sum(axis=0).astype(float)
#     print("integrated input spikes:", integrated_input.tolist())

#     action_scores = xbar.vmm_ideal(integrated_input)
#     action_idx = int(np.argmax(action_scores))
#     print("action scores:", action_scores.tolist())
#     print("selected action:", action_idx)

#     assert action_scores.shape == (n_actions,)
#     assert 0 <= action_idx < n_actions



# def test_neuron_layer_integration() -> None:
#     section("6) neuron.py smoke test (encoding -> neuron -> learning)")
#     obs = {"distance": 52.0, "temperature": 31.0, "gas": 0.18}

#     enc = SensorSpikeEncoder(
#         feature_names=["distance", "temperature", "gas"],
#         value_ranges={
#             "distance": (0.0, 200.0),
#             "temperature": (0.0, 80.0),
#             "gas": (0.0, 1.0),
#         },
#         mode="population_latency",
#         neurons_per_feature=4,
#         latency_steps=6,
#         seed=cfg.SEED,
#         dt=1e-3,
#         max_rate_hz=200.0,
#     )

#     xbar = DifferentialCrossbar(n_rows=enc.output_dim, n_cols=4, seed=cfg.SEED)
#     layer = MemristiveLIFOutputLayer(xbar, seed=cfg.SEED)

#     window = enc.encode_window(obs)
#     selected_action = -1
#     selected_pre_spikes = None

#     for t, enc_out in enumerate(window):
#         out = layer.step(enc_out.spikes, step_idx=t)
#         print(f"[t={t}] I_syn={out.synaptic_currents.tolist()}")
#         print(f"       Vmem={out.membrane_potentials.tolist()}")
#         print(f"       thr ={out.thresholds.tolist()}")
#         print(f"       spk ={out.spikes.tolist()}, winner={out.winner}")

#         if selected_action < 0 and out.winner >= 0:
#             selected_action = int(out.winner)
#             selected_pre_spikes = np.asarray(enc_out.spikes, dtype=float)

#     if selected_action < 0:
#         total_spikes = np.sum([w.spikes for w in window], axis=0).astype(float)
#         selected_action = int(np.argmax(layer._measured_vmm(total_spikes)))
#         selected_pre_spikes = total_spikes
#         print(f"no spike winner in window -> fallback argmax action={selected_action}")

#     assert selected_pre_spikes is not None
#     assert 0 <= selected_action < layer.n_neurons

#     status_before = layer.controller.get_pair_status((2, selected_action))
#     learn = layer.apply_reward_modulated_update(
#         pre_spikes=selected_pre_spikes,
#         winner=selected_action,
#         reward=+1.0,
#         step_idx=len(window),
#         target=selected_action,
#     )
#     status_after = layer.controller.get_pair_status((2, selected_action))
#     print("learning:", learn)
#     print(
#         f"tracked pair (2,{selected_action}) weight before={status_before['weight_measured']:.6e}, "
#         f"after={status_after['weight_measured']:.6e}"
#     )

#     assert len(learn.updated_pairs) > 0
#     assert learn.n_pulses_plus + learn.n_pulses_minus + learn.n_refresh >= 0
#     assert learn.message.startswith("Measured pulse-based synaptic update")



# def main() -> None:
#     np.set_printoptions(precision=6, suppress=False)
#     test_device_model()
#     test_crossbar()
#     test_conductance_controller()
#     test_encoding()
#     test_end_to_end_five_file_integration()
#     test_neuron_layer_integration()
#     section("ALL 6-FILE TESTS PASSED")


# if __name__ == "__main__":
#     main()

#---------------------------------------------------------------------------
# from __future__ import annotations

# """Standalone smoke/integration test for the latest 7 uploaded files.

# Run this file directly:
#     python test_latest_7files.py

# Important:
# - Do NOT paste this code into network.py.
# - Save it as a separate file (for example, test_latest_7files.py or main.py)
#   in the same folder as config.py, crossbar.py, device_model.py, encoding.py,
#   neuron.py, conductance_modulation.py, and network.py.
# """

# import traceback
# import numpy as np

# import config as cfg
# from device_model import MemristorDevice
# from crossbar import DifferentialCrossbar
# from conductance_modulation import ConductanceModulationController
# from encoding import SensorSpikeEncoder
# from neuron import MemristiveLIFOutputLayer
# from network import MemristiveSNNNetwork


# SEED = int(getattr(cfg, "SEED", 42))


# def banner(title: str) -> None:
#     print("\n" + "=" * 88)
#     print(title)
#     print("=" * 88)



# def test_device_model() -> None:
#     banner("1) device_model.py smoke test")
#     dev = MemristorDevice(seed=SEED)
#     print(f"initial: g={dev.g:.6e}, level={dev.state.level_idx}")

#     for _ in range(5):
#         dev.apply_pot_pulse(1)
#     print(f"after 5 pot pulses: g={dev.g:.6e}, level={dev.state.level_idx}")

#     for _ in range(3):
#         dev.apply_dep_pulse(1)
#     print(f"after 3 dep pulses: g={dev.g:.6e}, level={dev.state.level_idx}")

#     mid = 0.5 * (dev.g_min_eff + dev.g_max_eff)
#     dev.set_g(mid)
#     print(f"after set_g(mid): g={dev.g:.6e}, level={dev.state.level_idx}")



# def test_crossbar() -> None:
#     banner("2) crossbar.py smoke test")
#     xbar = DifferentialCrossbar(n_rows=4, n_cols=3, seed=SEED)
#     summary = xbar.summary()
#     print("summary:", summary)

#     gp0, gm0 = xbar.read_pair_ideal((1, 1))
#     w0 = gp0 - gm0
#     print(f"pair (1,1) ideal before: gp={gp0:.6e}, gm={gm0:.6e}, w={w0:.6e}")

#     xbar.set_pair_conductance((1, 1), gp0 + 1.0e-05, gm0 - 5.0e-06)
#     gp1, gm1 = xbar.read_pair_ideal((1, 1))
#     w1 = gp1 - gm1
#     print(f"pair (1,1) ideal after : gp={gp1:.6e}, gm={gm1:.6e}, w={w1:.6e}")

#     gp_m, gm_m = xbar.read_pair((1, 1))
#     print(f"pair (1,1) measured   : gp={gp_m:.6e}, gm={gm_m:.6e}, w={gp_m-gm_m:.6e}")

#     x = np.array([1.0, 0.5, 0.0, 0.25], dtype=float)
#     y = xbar.vmm_ideal(x)
#     print("vmm_ideal input :", x.tolist())
#     print("vmm_ideal output:", [float(v) for v in y])



# def test_conductance_controller() -> None:
#     banner("3) conductance_modulation.py smoke test")
#     xbar = DifferentialCrossbar(n_rows=4, n_cols=3, seed=SEED)
#     ctrl = ConductanceModulationController(xbar)

#     pair = (2, 1)
#     print("initial status:", ctrl.get_pair_status(pair))

#     for step, direction in enumerate([+1, +1, -1, +1, -1, +1]):
#         before = ctrl.get_pair_status(pair)["weight_measured"]
#         result = ctrl.update_weight(pair, direction, step_idx=step)
#         after = ctrl.get_pair_status(pair)["weight_measured"]
#         print(
#             f"step={step:02d} action={result.chosen_action:>14s} "
#             f"w_before={before:.6e} w_after={after:.6e} "
#             f"cm_after={result.common_mode_after:.6e} refresh={result.did_refresh}"
#         )

#     print("final status:", ctrl.get_pair_status(pair))



# def build_encoder() -> SensorSpikeEncoder:
#     return SensorSpikeEncoder(
#         feature_names=["distance", "temperature", "gas"],
#         value_ranges={
#             "distance": (0.0, 200.0),
#             "temperature": (0.0, 80.0),
#             "gas": (0.0, 1.0),
#         },
#         mode="population_latency",
#         neurons_per_feature=int(getattr(cfg, "ENCODER_NEURONS_PER_FEATURE", 4)),
#         latency_steps=int(getattr(cfg, "ENCODER_LATENCY_STEPS", 6)),
#         seed=SEED,
#     )



# def test_encoding() -> SensorSpikeEncoder:
#     banner("4) encoding.py smoke test")
#     obs = {"distance": 52.0, "temperature": 31.0, "gas": 0.18}

#     for mode in ["rate", "population_rate", "population_latency"]:
#         encoder = SensorSpikeEncoder(
#             feature_names=["distance", "temperature", "gas"],
#             value_ranges={
#                 "distance": (0.0, 200.0),
#                 "temperature": (0.0, 80.0),
#                 "gas": (0.0, 1.0),
#             },
#             mode=mode,
#             neurons_per_feature=5,
#             latency_steps=int(getattr(cfg, "ENCODER_LATENCY_STEPS", 8)),
#             seed=SEED,
#         )
#         print(f"\nmode={mode}")
#         if mode == "population_latency":
#             window = encoder.encode_window(obs)
#             print("feature names:", window[0].feature_names)
#             print("spike times :", window[0].spike_times.tolist())
#             total = np.zeros_like(window[0].spikes, dtype=int)
#             print("spikes per step:")
#             for t, out in enumerate(window):
#                 total += out.spikes.astype(int)
#                 print(f"  t={t}: {out.spikes.tolist()}")
#             print("total emitted per channel:", total.tolist())
#         else:
#             out = encoder.encode(obs, sim_step=0)
#             print("feature names:", out.feature_names)
#             print("firing rates :", np.round(out.firing_rates, 4).tolist())
#             print("spikes       :", out.spikes.tolist())

#     return build_encoder()



# def test_neuron(encoder: SensorSpikeEncoder) -> None:
#     banner("5) neuron.py smoke test")
#     obs = {"distance": 52.0, "temperature": 31.0, "gas": 0.18}
#     window = encoder.encode_window(obs)

#     xbar = DifferentialCrossbar(n_rows=encoder.output_dim, n_cols=4, seed=SEED)
#     layer = MemristiveLIFOutputLayer(crossbar=xbar, seed=SEED)

#     first_winner = -1
#     first_pre = None
#     for t, enc in enumerate(window):
#         res = layer.step(enc.spikes, step_idx=t)
#         print(f"[t={t}] I_syn={res.synaptic_currents.tolist()}")
#         print(f"       Vmem={res.membrane_potentials.tolist()}")
#         print(f"       thr ={res.thresholds.tolist()}")
#         print(f"       spk ={res.spikes.tolist()}, winner={res.winner}")
#         if first_winner < 0 and res.winner >= 0:
#             first_winner = int(res.winner)
#             first_pre = np.asarray(enc.spikes, dtype=float)

#     if first_pre is None:
#         first_winner = 0
#         first_pre = np.asarray(window[-1].spikes, dtype=float)

#     event = layer.apply_reward_modulated_update(
#         pre_spikes=first_pre,
#         winner=first_winner,
#         reward=1.0,
#         step_idx=len(window),
#         target=first_winner,
#     )
#     print("learning:", event)



# def test_network(encoder: SensorSpikeEncoder) -> None:
#     banner("6) network.py recurrent integration test")
#     obs = {"distance": 52.0, "temperature": 31.0, "gas": 0.18}

#     net = MemristiveSNNNetwork(
#         encoder=encoder,
#         n_actions=4,
#         hidden_dim=6,
#         seed=SEED,
#     )

#     decision = net.decide(obs)
#     print("action:", decision.action)
#     print("selected_step:", decision.selected_step)
#     print("used_fallback:", decision.used_fallback)
#     for rec in decision.step_records:
#         print(
#             f"[t={rec.t}] hidden_spk={rec.hidden_result.spikes.tolist()} "
#             f"out_spk={rec.output_result.spikes.tolist()} "
#             f"out_winner={rec.output_result.winner}"
#         )

#     events = net.learn(reward=1.0, target=decision.action)
#     print("learning output:", events["output"])
#     print("learning hidden:", events["hidden"])

#     dbg = net.get_debug_state()
#     print("debug last_action:", dbg.get("last_action"))
#     print("debug last_used_fallback:", dbg.get("last_used_fallback"))



# def main() -> None:
#     test_device_model()
#     test_crossbar()
#     test_conductance_controller()
#     encoder = test_encoding()
#     test_neuron(encoder)
#     test_network(encoder)

#     banner("ALL LATEST TESTS PASSED")


# if __name__ == "__main__":
#     try:
#         main()
#     except Exception:
#         banner("TEST FAILED")
#         traceback.print_exc()
#         raise


# from __future__ import annotations

# import numpy as np

# import config as cfg
# from crossbar import DifferentialCrossbar


# def print_pair_status(xbar: DifferentialCrossbar, pair_id: tuple[int, int], tag: str = "") -> None:
#     gp_ideal, gm_ideal = xbar.read_pair_ideal(pair_id)
#     gp_meas, gm_meas = xbar.read_pair(pair_id)
#     w_ideal = gp_ideal - gm_ideal
#     w_meas = gp_meas - gm_meas

#     print(f"\n[{tag}] pair={pair_id}")
#     print(f"  ideal  : g+={gp_ideal:.6e}, g-={gm_ideal:.6e}, w={w_ideal:.6e}")
#     print(f"  meas   : g+={gp_meas:.6e}, g-={gm_meas:.6e}, w={w_meas:.6e}")


# def basic_pair_test() -> None:
#     print("=" * 60)
#     print("1) Basic pair test")
#     print("=" * 60)

#     xbar = DifferentialCrossbar(
#         n_rows=cfg.TEST_ARRAY_ROWS,
#         n_cols=cfg.TEST_ARRAY_COLS,
#         seed=cfg.SEED,
#     )

#     pair_id = cfg.TEST_PAIR

#     print_pair_status(xbar, pair_id, tag="initial")

#     # plus 쪽 potentiation
#     n_eff = xbar.apply_pulse(pair_id, side="plus", polarity="pot", n_pulses=5)
#     print(f"\nApplied PLUS-POT pulses, effective pulses = {n_eff}")
#     print_pair_status(xbar, pair_id, tag="after plus-pot")

#     # minus 쪽 potentiation -> differential weight 감소 효과 가능
#     n_eff = xbar.apply_pulse(pair_id, side="minus", polarity="pot", n_pulses=3)
#     print(f"\nApplied MINUS-POT pulses, effective pulses = {n_eff}")
#     print_pair_status(xbar, pair_id, tag="after minus-pot")

#     # minus 쪽 depression
#     n_eff = xbar.apply_pulse(pair_id, side="minus", polarity="dep", n_pulses=4)
#     print(f"\nApplied MINUS-DEP pulses, effective pulses = {n_eff}")
#     print_pair_status(xbar, pair_id, tag="after minus-dep")

#     measured_weight = xbar.read_weight_measured(pair_id)
#     print(f"\nMeasured differential weight = {measured_weight:.6e}")


# def bounds_and_set_test() -> None:
#     print("\n" + "=" * 60)
#     print("2) Bounds / direct set test")
#     print("=" * 60)

#     xbar = DifferentialCrossbar(
#         n_rows=cfg.TEST_ARRAY_ROWS,
#         n_cols=cfg.TEST_ARRAY_COLS,
#         seed=cfg.SEED,
#     )

#     pair_id = cfg.TEST_PAIR
#     gp_min, gp_max, gm_min, gm_max = xbar.get_pair_bounds(pair_id)

#     print(f"\nBounds for pair {pair_id}")
#     print(f"  g+ min/max = {gp_min:.6e} / {gp_max:.6e}")
#     print(f"  g- min/max = {gm_min:.6e} / {gm_max:.6e}")

#     # 중간쯤 값으로 강제 설정
#     gp_target = 0.75 * gp_max + 0.25 * gp_min
#     gm_target = 0.25 * gm_max + 0.75 * gm_min

#     xbar.set_pair_conductance(pair_id, gp_target, gm_target)
#     print_pair_status(xbar, pair_id, tag="after direct set")


# def vmm_test() -> None:
#     print("\n" + "=" * 60)
#     print("3) Ideal VMM test")
#     print("=" * 60)

#     xbar = DifferentialCrossbar(
#         n_rows=cfg.TEST_ARRAY_ROWS,
#         n_cols=cfg.TEST_ARRAY_COLS,
#         seed=cfg.SEED,
#     )

#     # 몇 개 pair에 임의 weight를 만들어줌
#     for i in range(cfg.TEST_ARRAY_ROWS):
#         for j in range(cfg.TEST_ARRAY_COLS):
#             gp_min, gp_max, gm_min, gm_max = xbar.get_pair_bounds((i, j))

#             # j가 짝수면 양의 weight, 홀수면 음의 weight 느낌으로 설정
#             if j % 2 == 0:
#                 gp = 0.8 * gp_max + 0.2 * gp_min
#                 gm = 0.3 * gm_max + 0.7 * gm_min
#             else:
#                 gp = 0.3 * gp_max + 0.7 * gp_min
#                 gm = 0.8 * gm_max + 0.2 * gm_min

#             xbar.set_pair_conductance((i, j), gp, gm)

#     x = np.array([0.2, 0.5, -0.3, 1.0], dtype=float)
#     y = xbar.vmm_ideal(x)

#     print(f"\nInput x = {x}")
#     print(f"Ideal VMM output y = {y}")


# def pulse_sweep_test() -> None:
#     print("\n" + "=" * 60)
#     print("4) Pulse sweep test")
#     print("=" * 60)

#     xbar = DifferentialCrossbar(
#         n_rows=cfg.TEST_ARRAY_ROWS,
#         n_cols=cfg.TEST_ARRAY_COLS,
#         seed=cfg.SEED,
#     )

#     pair_id = cfg.TEST_PAIR

#     print_pair_status(xbar, pair_id, tag="start sweep")

#     # plus-pot sweep
#     print("\n[PLUS-POT sweep]")
#     for step in range(8):
#         xbar.apply_pulse(pair_id, side="plus", polarity="pot", n_pulses=1)
#         gp, gm = xbar.read_pair(pair_id)
#         print(f"  step {step+1:02d}: g+={gp:.6e}, g-={gm:.6e}, w={gp-gm:.6e}")

#     # plus-dep sweep
#     print("\n[PLUS-DEP sweep]")
#     for step in range(8):
#         xbar.apply_pulse(pair_id, side="plus", polarity="dep", n_pulses=1)
#         gp, gm = xbar.read_pair(pair_id)
#         print(f"  step {step+1:02d}: g+={gp:.6e}, g-={gm:.6e}, w={gp-gm:.6e}")


# def summary_test() -> None:
#     print("\n" + "=" * 60)
#     print("5) Array summary test")
#     print("=" * 60)

#     xbar = DifferentialCrossbar(
#         n_rows=cfg.TEST_ARRAY_ROWS,
#         n_cols=cfg.TEST_ARRAY_COLS,
#         seed=cfg.SEED,
#     )

#     # 몇 번 pulse 줘서 상태 바꿔보기
#     xbar.apply_pulse((0, 0), side="plus", polarity="pot", n_pulses=4)
#     xbar.apply_pulse((1, 1), side="minus", polarity="pot", n_pulses=6)
#     xbar.apply_pulse((2, 2), side="plus", polarity="dep", n_pulses=2)

#     info = xbar.summary()

#     print("\nSummary:")
#     for k, v in info.items():
#         if isinstance(v, float):
#             print(f"  {k}: {v:.6e}")
#         else:
#             print(f"  {k}: {v}")


# if __name__ == "__main__":
#     basic_pair_test()
#     bounds_and_set_test()
#     vmm_test()
#     pulse_sweep_test()
#     summary_test()

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from stm_device_model import STMDeviceModel
from stm_crossbar import STMCrossbar


def _uS(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) * 1e6


def _ms(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) * 1e3


def plot_device_accumulation_and_gap_decay(seed: int = 0) -> None:
    dev = STMDeviceModel(seed=seed)
    dev.reset("rest")

    hist = dev.simulate_pulse_train(
        n_pulses=18,
        amplitude_v=0.62,
        pulse_width_s=1.0e-3,
        interval_s=1.2e-3,
        tail_relax_s=12e-3,
    )

    t = _ms(hist["time_s"])
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(t, _uS(hist["conductance_s"]), linewidth=1.5)
    axes[0].set_ylabel("Conductance (uS)")
    axes[0].set_title("STM device: accumulation with exponential gap/tail decay")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, hist["z"], linewidth=1.3, label="z")
    axes[1].plot(t, hist["x"], linewidth=1.3, label="x")
    axes[1].plot(t, hist["r"], linewidth=1.3, label="r")
    axes[1].set_ylabel("State")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, 1e6 * hist["current_a"], linewidth=1.5)
    axes[2].set_ylabel("Read current (uA)")
    axes[2].set_xlabel("Time (ms)")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()


def plot_device_near_saturation_then_relax(seed: int = 1) -> None:
    dev = STMDeviceModel(seed=seed)
    dev.reset("rest")

    hist = dev.simulate_pulse_train(
        n_pulses=90,
        amplitude_v=0.78,
        pulse_width_s=1.2e-3,
        interval_s=0.25e-3,
        tail_relax_s=80e-3,
    )

    t = _ms(hist["time_s"])
    # split at start of tail relaxation by detecting last pulse/gap activity before long relax
    split_idx = int(np.argmax(np.diff(hist["time_s"]) > 5 * dev.dt_internal)) + 1
    if split_idx <= 1:
        split_idx = len(t) - 1

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(t[:split_idx], _uS(hist["conductance_s"][:split_idx]), linewidth=1.6, label="pulse train")
    axes[0].plot(t[split_idx-1:], _uS(hist["conductance_s"][split_idx-1:]), linewidth=1.8, label="relax only")
    axes[0].axvline(t[split_idx-1], linestyle="--", alpha=0.8)
    axes[0].set_ylabel("Conductance (uS)")
    axes[0].set_title("STM device: smooth saturation and exponential relaxation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t[:split_idx], hist["x"][:split_idx], linewidth=1.4, label="x during pulse train")
    axes[1].plot(t[:split_idx], hist["r"][:split_idx], linewidth=1.4, label="r during pulse train")
    axes[1].plot(t[split_idx-1:], hist["x"][split_idx-1:], linewidth=1.6, label="x during relax only")
    axes[1].plot(t[split_idx-1:], hist["r"][split_idx-1:], linewidth=1.6, label="r during relax only")
    axes[1].axvline(t[split_idx-1], linestyle="--", alpha=0.8)
    axes[1].set_ylabel("State")
    axes[1].set_xlabel("Time (ms)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()


def plot_d2d_variation(num_devices: int = 10) -> None:
    plt.figure(figsize=(8, 5))
    for seed in range(num_devices):
        dev = STMDeviceModel(seed=seed)
        dev.reset("rest")
        hist = dev.simulate_pulse_train(
            n_pulses=16,
            amplitude_v=0.62,
            pulse_width_s=1.0e-3,
            interval_s=0.9e-3,
            tail_relax_s=0.0,
        )
        plt.plot(_ms(hist["time_s"]), _uS(hist["conductance_s"]), linewidth=1.0, alpha=0.9)
    plt.title("D2D variation across STM devices")
    plt.xlabel("Time (ms)")
    plt.ylabel("Conductance (uS)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_c2c_variation(seed: int = 7, n_runs: int = 8) -> None:
    plt.figure(figsize=(8, 5))
    for _ in range(n_runs):
        dev = STMDeviceModel(seed=seed)
        dev.reset("rest")
        hist = dev.simulate_pulse_train(
            n_pulses=16,
            amplitude_v=0.62,
            pulse_width_s=1.0e-3,
            interval_s=0.9e-3,
            tail_relax_s=0.0,
        )
        plt.plot(_ms(hist["time_s"]), _uS(hist["conductance_s"]), linewidth=1.0, alpha=0.8)
    plt.title("C2C variation for repeated runs of one STM device")
    plt.xlabel("Time (ms)")
    plt.ylabel("Conductance (uS)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_crossbar_selected_cell(seed: int = 2) -> None:
    cb = STMCrossbar(4, 4, seed=seed)
    cb.reset_all("rest")
    hist = cb.run_pulse_train(
        cell_id=(0, 0),
        n_pulses=22,
        amplitude_v=0.66,
        pulse_width_s=1.0e-3,
        period_s=1.9e-3,
        tail_relax_s=15e-3,
        relax_unselected=False,
    )

    t = _ms(hist["time_s"])
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(t, _uS(hist["conductance_s"]), linewidth=1.5)
    axes[0].set_ylabel("Measured G (uS)")
    axes[0].set_title("STMCrossbar selected-cell response")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, hist["z"], linewidth=1.3, label="z")
    axes[1].plot(t, hist["x"], linewidth=1.3, label="x")
    axes[1].plot(t, hist["r"], linewidth=1.3, label="r")
    axes[1].set_ylabel("State")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, 1e6 * hist["current_a"], linewidth=1.5)
    axes[2].set_ylabel("Measured I (uA)")
    axes[2].set_xlabel("Time (ms)")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()


def print_quick_summary() -> None:
    dev = STMDeviceModel(seed=123)
    dev.reset("rest")
    hist = dev.simulate_pulse_train(
        n_pulses=20,
        amplitude_v=0.62,
        pulse_width_s=1.0e-3,
        interval_s=1.0e-3,
        tail_relax_s=20e-3,
    )
    g = hist["conductance_s"]
    print("[Quick summary]")
    print(f"Initial G : {g[0]:.6e} S")
    print(f"Peak G    : {np.max(g):.6e} S")
    print(f"Final G   : {g[-1]:.6e} S")
    print(f"Peak/Init : {np.max(g)/max(g[0], 1e-30):.3f}x")


if __name__ == "__main__":
    print_quick_summary()
    plot_device_accumulation_and_gap_decay()
    plot_device_near_saturation_then_relax()
    plot_d2d_variation()
    plot_c2c_variation()
    plot_crossbar_selected_cell()
    plt.show()
