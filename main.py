

# import matplotlib.pyplot as plt

# import config as cfg
# from device_model import MemristorDevice
# from conductance_modulation import ConductanceModulationController


# class MockPairAccess:
#     def __init__(self):
#         self.plus = MemristorDevice(seed=cfg.SEED)
#         self.minus = MemristorDevice(seed=cfg.SEED + 1 if cfg.SEED is not None else None)

#         self.plus.reset("init")
#         self.minus.reset("init")

#     def read_pair(self, pair_id):
#         _ = pair_id
#         return self.plus.g, self.minus.g

#     def apply_pulse(self, pair_id, side, polarity, n_pulses=1):
#         _ = pair_id
#         dev = self.plus if side == "plus" else self.minus
#         dev.apply_pulse(polarity=polarity, n_pulses=n_pulses)


# # -------------------------------------------------
# # Basic logic tests
# # -------------------------------------------------

# def test_recenter_not_needed():
#     access = MockPairAccess()
#     ctrl = ConductanceModulationController(access)

#     pair_id = (0, 0)
#     status_before = ctrl.get_pair_status(pair_id)
#     result = ctrl.recenter_pair_if_needed(pair_id)
#     status_after = ctrl.get_pair_status(pair_id)

#     print("\n=== recenter not needed ===")
#     print("before:", status_before)
#     print("result:", result)
#     print("after :", status_after)


# def test_recenter_needed():
#     access = MockPairAccess()
#     ctrl = ConductanceModulationController(access)

#     pair_id = (0, 0)

#     for _ in range(150):
#         access.apply_pulse(pair_id, "plus", "pot", 1)

#     before = ctrl.get_pair_status(pair_id)
#     plan = ctrl.make_recenter_plan(pair_id)
#     result = ctrl.recenter_pair_if_needed(pair_id)
#     after = ctrl.get_pair_status(pair_id)

#     print("\n=== recenter needed ===")
#     print("before:", before)
#     print("plan  :", plan)
#     print("result:", result)
#     print("after :", after)
#     print("weight change:", after["weight"] - before["weight"])


# def test_partial_feasible_shift():
#     access = MockPairAccess()
#     ctrl = ConductanceModulationController(access)

#     pair_id = (0, 0)

#     access.plus.reset("max")
#     access.minus.reset("min")

#     before = ctrl.get_pair_status(pair_id)
#     plan = ctrl.make_recenter_plan(pair_id)
#     result = ctrl.recenter_pair_if_needed(pair_id)
#     after = ctrl.get_pair_status(pair_id)

#     print("\n=== partial feasible shift ===")
#     print("before:", before)
#     print("plan  :", plan)
#     print("result:", result)
#     print("after :", after)


# # -------------------------------------------------
# # Graph visualization test
# # -------------------------------------------------

# def test_recenter_graph():

#     access = MockPairAccess()
#     ctrl = ConductanceModulationController(access)

#     pair_id = (0, 0)

#     g_plus_hist = []
#     g_minus_hist = []
#     weight_hist = []

#     reset_step = None

#     # 먼저 plus를 saturation 쪽으로 몰기
#     for step in range(250):

#         access.apply_pulse(pair_id, "plus", "pot", 1)

#         g_plus, g_minus = access.read_pair(pair_id)

#         g_plus_hist.append(g_plus)
#         g_minus_hist.append(g_minus)
#         weight_hist.append(g_plus - g_minus)

#     # reset 실행
#     reset_step = len(g_plus_hist)

#     ctrl.recenter_pair_if_needed(pair_id)

#     # reset 이후 상태 기록
#     for _ in range(50):

#         g_plus, g_minus = access.read_pair(pair_id)

#         g_plus_hist.append(g_plus)
#         g_minus_hist.append(g_minus)
#         weight_hist.append(g_plus - g_minus)

#     # ---------------- 그래프 ----------------

#     steps = range(len(g_plus_hist))

#     plt.figure(figsize=(10, 8))

#     plt.subplot(3, 1, 1)
#     plt.plot(steps, g_plus_hist, label="G+")
#     plt.axvline(reset_step, color="red", linestyle="--", label="reset")
#     plt.axhline(cfg.G_MAX, linestyle="--", alpha=0.3)
#     plt.ylabel("G+")
#     plt.legend()

#     plt.subplot(3, 1, 2)
#     plt.plot(steps, g_minus_hist, label="G-")
#     plt.axvline(reset_step, color="red", linestyle="--")
#     plt.axhline(cfg.G_MIN, linestyle="--", alpha=0.3)
#     plt.ylabel("G-")
#     plt.legend()

#     plt.subplot(3, 1, 3)
#     plt.plot(steps, weight_hist, label="Weight (G+ - G-)")
#     plt.axvline(reset_step, color="red", linestyle="--")
#     plt.ylabel("Weight")
#     plt.xlabel("Step")
#     plt.legend()

#     plt.tight_layout()
#     plt.show()


# # -------------------------------------------------
# # main
# # -------------------------------------------------

# if __name__ == "__main__":

#     test_recenter_not_needed()
#     test_recenter_needed()
#     test_partial_feasible_shift()

#     # 그래프 확인
#     test_recenter_graph()


#--------------------------------------------
# from __future__ import annotations

# import numpy as np
# import matplotlib.pyplot as plt

# import config as cfg
# from crossbar import DifferentialCrossbar
# from conductance_modulation import ConductanceModulationController


# def print_pair_status(cb: DifferentialCrossbar, ctrl: ConductanceModulationController, pair_id):
#     gpt, gmt = cb.read_pair_true(pair_id)
#     gpm, gmm = cb.read_pair(pair_id, noisy=True, disturb=False)

#     print(f"pair_id = {pair_id}")
#     print(f"  true     : G+ = {gpt:.6e}, G- = {gmt:.6e}, W = {gpt-gmt:.6e}")
#     print(f"  measured : G+ = {gpm:.6e}, G- = {gmm:.6e}, W = {gpm-gmm:.6e}")
#     print(f"  ctrl status = {ctrl.get_pair_status(pair_id)}")


# def main():
#     # -------------------------------------------------
#     # 1. crossbar / controller 생성
#     # -------------------------------------------------
#     n_rows = 4
#     n_cols = 3

#     cb = DifferentialCrossbar(
#         n_rows=n_rows,
#         n_cols=n_cols,
#         seed=cfg.SEED,
#     )
#     ctrl = ConductanceModulationController(access=cb)

#     print("=" * 60)
#     print("Initial crossbar summary")
#     print(cb.summary())

#     pair_id = (1, 1)

#     print("\n" + "=" * 60)
#     print("Initial selected pair status")
#     print_pair_status(cb, ctrl, pair_id)

#     # -------------------------------------------------
#     # 2. 특정 pair를 일부러 saturation 쪽으로 보내기
#     # -------------------------------------------------
#     n_prog_steps = 80

#     hist_g_plus_true = []
#     hist_g_minus_true = []
#     hist_w_true = []

#     hist_g_plus_meas = []
#     hist_g_minus_meas = []
#     hist_w_meas = []

#     for t in range(n_prog_steps):
#         # 예시: plus만 potentiation해서 saturation 근처로 밀기
#         cb.apply_pulse(pair_id, side="plus", polarity="pot", n_pulses=1)

#         gpt, gmt = cb.read_pair_true(pair_id)
#         gpm, gmm = cb.read_pair(pair_id, noisy=True, disturb=False)

#         hist_g_plus_true.append(gpt)
#         hist_g_minus_true.append(gmt)
#         hist_w_true.append(gpt - gmt)

#         hist_g_plus_meas.append(gpm)
#         hist_g_minus_meas.append(gmm)
#         hist_w_meas.append(gpm - gmm)

#     print("\n" + "=" * 60)
#     print("After repeated plus-side potentiation")
#     print_pair_status(cb, ctrl, pair_id)

#     # -------------------------------------------------
#     # 3. recenter plan 확인
#     # -------------------------------------------------
#     print("\n" + "=" * 60)
#     print("Recenter plan")
#     plan = ctrl.make_recenter_plan(pair_id)
#     print(plan)

#     # -------------------------------------------------
#     # 4. recenter 실행
#     # -------------------------------------------------
#     print("\n" + "=" * 60)
#     print("Recenter execution")
#     result = ctrl.recenter_pair_if_needed(pair_id)
#     print(result)

#     print("\n" + "=" * 60)
#     print("After recenter")
#     print_pair_status(cb, ctrl, pair_id)

#     # -------------------------------------------------
#     # 5. target programming 예시
#     # -------------------------------------------------
#     print("\n" + "=" * 60)
#     print("Direct target programming example")

#     g_plus_target = 55e-6
#     g_minus_target = 20e-6

#     result2 = ctrl.program_pair_to_targets(
#         pair_id=pair_id,
#         g_plus_target=g_plus_target,
#         g_minus_target=g_minus_target,
#     )
#     print(result2)

#     print("\nAfter direct target programming")
#     print_pair_status(cb, ctrl, pair_id)

#     # -------------------------------------------------
#     # 6. VMM 테스트
#     # -------------------------------------------------
#     print("\n" + "=" * 60)
#     print("VMM test")

#     x = np.array([1.0, 0.0, 1.0, 0.5], dtype=np.float64)

#     y_ideal = cb.vmm(x, measured=False)
#     y_measured = cb.vmm(x, measured=True, noisy=True, disturb=False)

#     print("Input x       :", x)
#     print("Ideal VMM     :", y_ideal)
#     print("Measured VMM  :", y_measured)

#     # -------------------------------------------------
#     # 7. 그래프
#     # -------------------------------------------------
#     pulse_axis = np.arange(1, n_prog_steps + 1)

#     plt.figure(figsize=(8, 4))
#     plt.plot(pulse_axis, hist_g_plus_true, label="G+ true")
#     plt.plot(pulse_axis, hist_g_minus_true, label="G- true")
#     plt.xlabel("Programming step")
#     plt.ylabel("Conductance (S)")
#     plt.title("True conductance evolution before recenter")
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     plt.figure(figsize=(8, 4))
#     plt.plot(pulse_axis, hist_w_true, label="Weight true")
#     plt.plot(pulse_axis, hist_w_meas, label="Weight measured")
#     plt.xlabel("Programming step")
#     plt.ylabel("Differential weight (S)")
#     plt.title("Differential weight evolution before recenter")
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     main()

# ---------------------------------------

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import config as cfg
from crossbar import DifferentialCrossbar
from conductance_modulation import ConductanceModulationController


def print_pair_status(cb: DifferentialCrossbar, ctrl: ConductanceModulationController, pair_id):
    gpt, gmt = cb.read_pair_true(pair_id)
    gpm, gmm = cb.read_pair(pair_id, noisy=True, disturb=False)

    print(f"pair_id = {pair_id}")
    print(f"  true     : G+ = {gpt:.6e}, G- = {gmt:.6e}, W = {gpt-gmt:.6e}")
    print(f"  measured : G+ = {gpm:.6e}, G- = {gmm:.6e}, W = {gpm-gmm:.6e}")
    print(f"  ctrl status = {ctrl.get_pair_status(pair_id)}")


def record_history(cb, pair_id, step_idx, hist):
    gpt, gmt = cb.read_pair_true(pair_id)
    gpm, gmm = cb.read_pair(pair_id, noisy=True, disturb=False)

    hist["step"].append(step_idx)
    hist["g_plus_true"].append(gpt)
    hist["g_minus_true"].append(gmt)
    hist["w_true"].append(gpt - gmt)

    hist["g_plus_meas"].append(gpm)
    hist["g_minus_meas"].append(gmm)
    hist["w_meas"].append(gpm - gmm)


def main():
    n_rows = 4
    n_cols = 3

    cb = DifferentialCrossbar(n_rows=n_rows, n_cols=n_cols, seed=cfg.SEED)
    ctrl = ConductanceModulationController(access=cb)

    pair_id = (1, 1)

    print("=" * 60)
    print("Initial selected pair status")
    print_pair_status(cb, ctrl, pair_id)

    hist = {
        "step": [],
        "g_plus_true": [],
        "g_minus_true": [],
        "w_true": [],
        "g_plus_meas": [],
        "g_minus_meas": [],
        "w_meas": [],
    }

    recenter_trigger_step = None
    recenter_after_step = None

    step_idx = 0

    # -------------------------------------------------
    # 1) recenter 걸릴 때까지 potentiation
    # -------------------------------------------------
    max_steps_before_recenter = 1000

    for _ in range(max_steps_before_recenter):
        step_idx += 1
        cb.apply_pulse(pair_id, side="plus", polarity="pot", n_pulses=1)
        record_history(cb, pair_id, step_idx, hist)

        status = ctrl.get_pair_status(pair_id)
        if status["needs_recentering"]:
            recenter_trigger_step = step_idx
            print("\n" + "=" * 60)
            print(f"Recenter trigger detected at step {recenter_trigger_step}")
            print_pair_status(cb, ctrl, pair_id)
            break

    if recenter_trigger_step is None:
        print("\nRecenter was not triggered. Increase max_steps_before_recenter.")
        return

    # -------------------------------------------------
    # 2) recenter 실행
    # -------------------------------------------------
    print("\n" + "=" * 60)
    print("Recenter plan")
    plan = ctrl.make_recenter_plan(pair_id)
    print(plan)

    print("\n" + "=" * 60)
    print("Recenter execution")
    result = ctrl.recenter_pair_if_needed(pair_id)
    print(result)

    recenter_after_step = step_idx + 1
    record_history(cb, pair_id, recenter_after_step, hist)

    print("\n" + "=" * 60)
    print("After recenter")
    print_pair_status(cb, ctrl, pair_id)

    # -------------------------------------------------
    # 3) recenter 이후에도 계속 potentiation
    # -------------------------------------------------
    post_recenter_steps = 120

    for _ in range(post_recenter_steps):
        step_idx += 1
        cb.apply_pulse(pair_id, side="plus", polarity="pot", n_pulses=1)
        record_history(cb, pair_id, step_idx + 1, hist)

    print("\n" + "=" * 60)
    print("After post-recenter potentiation")
    print_pair_status(cb, ctrl, pair_id)

    # -------------------------------------------------
    # 4) VMM test
    # row 1을 바꿨으므로 x[1]은 0이 아니어야 영향이 보임
    # -------------------------------------------------
    x = np.array([1.0, 1.0, 1.0, 0.5], dtype=np.float64)

    print("\n" + "=" * 60)
    print("VMM test")
    print("Input x      :", x)
    print("Ideal VMM    :", cb.vmm(x, measured=False))
    print("Measured VMM :", cb.vmm(x, measured=True, noisy=True, disturb=False))

    # -------------------------------------------------
    # 5) 그래프 1: conductance evolution
    # -------------------------------------------------
    plt.figure(figsize=(10, 4.8))
    plt.plot(hist["step"], hist["g_plus_true"], label="G+ true")
    plt.plot(hist["step"], hist["g_minus_true"], label="G- true")

    plt.axvline(recenter_trigger_step, linestyle="--", label="recenter trigger")
    plt.axvline(recenter_after_step, linestyle=":", label="after recenter")

    plt.xlabel("Programming step")
    plt.ylabel("Conductance (S)")
    plt.title("Conductance evolution before and after recenter")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------
    # 6) 그래프 2: differential weight evolution
    # -------------------------------------------------
    plt.figure(figsize=(10, 4.8))
    plt.plot(hist["step"], hist["w_true"], label="Weight true")
    plt.plot(hist["step"], hist["w_meas"], label="Weight measured")

    plt.axvline(recenter_trigger_step, linestyle="--", label="recenter trigger")
    plt.axvline(recenter_after_step, linestyle=":", label="after recenter")

    plt.xlabel("Programming step")
    plt.ylabel("Differential weight (S)")
    plt.title("Weight evolution before and after recenter")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()