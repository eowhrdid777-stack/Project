import matplotlib.pyplot as plt

import config as cfg
from device_model import MemristorDevice
from conductance_modulation import ConductanceModulationController


class MockPairAccess:
    def __init__(self):
        self.plus = MemristorDevice(seed=cfg.SEED)
        self.minus = MemristorDevice(seed=cfg.SEED + 1 if cfg.SEED is not None else None)

        self.plus.reset("init")
        self.minus.reset("init")

    def read_pair(self, pair_id):
        _ = pair_id
        return self.plus.g, self.minus.g

    def apply_pulse(self, pair_id, side, polarity, n_pulses=1):
        _ = pair_id
        dev = self.plus if side == "plus" else self.minus
        dev.apply_pulse(polarity=polarity, n_pulses=n_pulses)


# -------------------------------------------------
# Basic logic tests
# -------------------------------------------------

def test_recenter_not_needed():
    access = MockPairAccess()
    ctrl = ConductanceModulationController(access)

    pair_id = (0, 0)
    status_before = ctrl.get_pair_status(pair_id)
    result = ctrl.recenter_pair_if_needed(pair_id)
    status_after = ctrl.get_pair_status(pair_id)

    print("\n=== recenter not needed ===")
    print("before:", status_before)
    print("result:", result)
    print("after :", status_after)


def test_recenter_needed():
    access = MockPairAccess()
    ctrl = ConductanceModulationController(access)

    pair_id = (0, 0)

    for _ in range(150):
        access.apply_pulse(pair_id, "plus", "pot", 1)

    before = ctrl.get_pair_status(pair_id)
    plan = ctrl.make_recenter_plan(pair_id)
    result = ctrl.recenter_pair_if_needed(pair_id)
    after = ctrl.get_pair_status(pair_id)

    print("\n=== recenter needed ===")
    print("before:", before)
    print("plan  :", plan)
    print("result:", result)
    print("after :", after)
    print("weight change:", after["weight"] - before["weight"])


def test_partial_feasible_shift():
    access = MockPairAccess()
    ctrl = ConductanceModulationController(access)

    pair_id = (0, 0)

    access.plus.reset("max")
    access.minus.reset("min")

    before = ctrl.get_pair_status(pair_id)
    plan = ctrl.make_recenter_plan(pair_id)
    result = ctrl.recenter_pair_if_needed(pair_id)
    after = ctrl.get_pair_status(pair_id)

    print("\n=== partial feasible shift ===")
    print("before:", before)
    print("plan  :", plan)
    print("result:", result)
    print("after :", after)


# -------------------------------------------------
# Graph visualization test
# -------------------------------------------------

def test_recenter_graph():

    access = MockPairAccess()
    ctrl = ConductanceModulationController(access)

    pair_id = (0, 0)

    g_plus_hist = []
    g_minus_hist = []
    weight_hist = []

    reset_step = None

    # 먼저 plus를 saturation 쪽으로 몰기
    for step in range(250):

        access.apply_pulse(pair_id, "plus", "pot", 1)

        g_plus, g_minus = access.read_pair(pair_id)

        g_plus_hist.append(g_plus)
        g_minus_hist.append(g_minus)
        weight_hist.append(g_plus - g_minus)

    # reset 실행
    reset_step = len(g_plus_hist)

    ctrl.recenter_pair_if_needed(pair_id)

    # reset 이후 상태 기록
    for _ in range(50):

        g_plus, g_minus = access.read_pair(pair_id)

        g_plus_hist.append(g_plus)
        g_minus_hist.append(g_minus)
        weight_hist.append(g_plus - g_minus)

    # ---------------- 그래프 ----------------

    steps = range(len(g_plus_hist))

    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(steps, g_plus_hist, label="G+")
    plt.axvline(reset_step, color="red", linestyle="--", label="reset")
    plt.axhline(cfg.G_MAX, linestyle="--", alpha=0.3)
    plt.ylabel("G+")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(steps, g_minus_hist, label="G-")
    plt.axvline(reset_step, color="red", linestyle="--")
    plt.axhline(cfg.G_MIN, linestyle="--", alpha=0.3)
    plt.ylabel("G-")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(steps, weight_hist, label="Weight (G+ - G-)")
    plt.axvline(reset_step, color="red", linestyle="--")
    plt.ylabel("Weight")
    plt.xlabel("Step")
    plt.legend()

    plt.tight_layout()
    plt.show()


# -------------------------------------------------
# main
# -------------------------------------------------

if __name__ == "__main__":

    test_recenter_not_needed()
    test_recenter_needed()
    test_partial_feasible_shift()

    # 그래프 확인
    test_recenter_graph()


