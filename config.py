from __future__ import annotations

SEED = 42

# Device window
G_MIN = 1.25e-8
G_MAX = 1.80e-7
P_MAX = 64
G_INIT_MODE = "mid"

# Smooth curve shaping constants used to emulate the measured analog trend.
A_POT = -0.8028
A_DEP = -0.6979

# Variation
ENABLE_D2D_VARIATION = True
CV_D2D = 0.0393
ENABLE_C2C_VARIATION = True
CV_C2C = 0.0236

# Retention
ENABLE_RETENTION = False
RETENTION_GAMMA = 0.0
G_RCP = 0.5 * (G_MIN + G_MAX)

# Bias / pulse scheme
READ_GATE_V = -1.0
READ_DRAIN_V = 1.0
POT_START_V = 2.7
POT_STOP_V = 4.3
DEP_START_V = -2.0
DEP_STOP_V = -3.6
PULSE_V_STEP = 0.025
PULSE_WIDTH_S = 10e-3
READ_AVG_SAMPLES = 1

READ_IR_DROP_ALPHA = 0.04
PROG_IR_DROP_ALPHA = 0.04

ENABLE_READ_NOISE = True
READ_NOISE_REL_SIGMA = 0.003
ENABLE_SNEAK_PATH = True
SNEAK_RATIO = 0.0015
ENABLE_READ_DISTURB = True
READ_DISTURB_STEP = 0.001

# Controller / modulation policy
COMMON_MODE_TARGET = 0.5 * (G_MIN + G_MAX)
COMMON_MODE_BAND_FRACTION = 0.18
HEADROOM_TRIGGER_FRACTION = 0.10
REFRESH_CHECK_PERIOD = 8
REFRESH_MIN_INTERVAL = 12
PROGRAM_TOLERANCE = 1.0e-8
MAX_VERIFY_STEPS = 96
PULSES_PER_VERIFY_STEP = 1

ENCODER_ENSURE_ONE_SPIKE_PER_FEATURE = False
ENCODER_USE_RANK_ORDER_TIE_BREAK = False

ENCODER_NEURONS_PER_FEATURE = 7
ENCODER_LATENCY_STEPS = 8
ENCODER_ACTIVATION_THRESHOLD = 0.03
ENCODER_SIGMA_SCALE = 0.75

# Neuron parameters
NEURON_ENABLE_WTA = True
NEURON_LATERAL_INHIBITION = 0.2
NEURON_ENABLE_THRESHOLD_ADAPTATION = True
NEURON_MEMBRANE_DECAY = 0.97
NEURON_INPUT_GAIN = 20.0
NEURON_BASE_THRESHOLD = 1.0e-7
NEURON_REFRACTORY_STEPS = 1
NEURON_THRESHOLD_SCALE = 1.0e-8
NEURON_THRESHOLD_POT_PULSES_ON_SPIKE = 1
NEURON_THRESHOLD_DEP_PULSES_RECOVERY = 1
NEURON_THRESHOLD_RECOVERY_PERIOD = 3

# Test/demo
TEST_PAIR = (0, 0)
TEST_ARRAY_ROWS = 4
TEST_ARRAY_COLS = 4
TEST_N_STEPS = 160
TEST_DIRECTION_CHANGE_STEP = 80
TEST_ENABLE_PLOTS = True

# ------------------------------------------------------------------
# STM diffusive device model
# ------------------------------------------------------------------
NETWORK_INPUT_HIDDEN_CROSSBAR_TYPE = "differential"
NETWORK_RECURRENT_HIDDEN_CROSSBAR_TYPE = "differential"
NETWORK_OUTPUT_CROSSBAR_TYPE = "differential"

# Recommended STM constants for config.py
# These values make STM recurrent-feedback devices close enough in conductance
# scale to the LTM array for network compatibility, while keeping STM volatile.

# ------------------------------------------------------------------
# STM diffusive / short-term-memory device model
# ------------------------------------------------------------------
STM_G_REST = 1.25e-8
STM_G_PEAK = 1.20e-7
STM_G_NONLINEARITY = 1.35

STM_DT_INTERNAL = 2.0e-5
STM_PULSE_WIDTH_S = 1.0e-3
STM_READ_VOLTAGE = 0.1
STM_READ_AVG_SAMPLES = 1

# Bipolar programming thresholds/scales
STM_POT_THRESHOLD_V = 0.20
STM_DEP_THRESHOLD_V = 0.20
STM_POT_SCALE_V = 0.18
STM_DEP_SCALE_V = 0.18

# Backward-compatible aliases; keep them if old stm_crossbar.py still refers to them.
STM_PULSE_THRESHOLD_V = STM_POT_THRESHOLD_V
STM_PULSE_SCALE_V = STM_POT_SCALE_V

# Potentiation dynamics
STM_Z_POT_GAIN = 150.0
STM_X_POT_GAIN = 55.0
STM_Z_TO_X_THRESHOLD = 0.25
STM_Z_TO_X_SLOPE = 10.0
STM_PULSE_LEAK_FACTOR = 0.02

# Depression/reset dynamics
STM_Z_DEP_GAIN = 210.0
STM_X_DEP_GAIN = 35.0
STM_R_RECOVERY_GAIN_DURING_DEP = 2.0

# Relaxation/recovery time constants
STM_TAU_Z_S = 5.0e-3
STM_TAU_X_S = 0.20
STM_TAU_R_S = 0.50
STM_R_DEPLETION_GAIN = 0.20

# Optional high-excitation instability; normally off.
STM_ENABLE_OVERLOAD_DECAY = False
STM_OVERLOAD_X_THRESHOLD = 0.985
STM_OVERLOAD_R_THRESHOLD = 0.10
STM_OVERLOAD_DECAY_GAIN = 0.20

# Conductance mapping
STM_FAST_WEIGHT = 0.35
STM_SLOW_WEIGHT = 0.65

# Variability/readout
STM_ENABLE_D2D_VARIATION = True
STM_CV_D2D = 0.03
STM_ENABLE_C2C_VARIATION = True
STM_CV_C2C = 0.025
STM_ENABLE_READ_NOISE = True
STM_READ_NOISE_REL_SIGMA = 0.003

# Crossbar nonidealities
STM_READ_IR_DROP_ALPHA = 0.04
STM_PROG_IR_DROP_ALPHA = 0.04
STM_ENABLE_SNEAK_PATH = True
STM_SNEAK_RATIO = 0.0015

# Recurrent-feedback timing hook to be used by stm_crossbar.py/network.py if needed.
# After each network decision/robot step, relaxing STM recurrent cells by the elapsed
# physical time makes the feedback memory time-scale meaningful.
STM_RELAX_PER_DECISION_S = 5.0e-3


# overload decay는 기본적으로 꺼둠
STM_ENABLE_OVERLOAD_DECAY = False
STM_OVERLOAD_X_THRESHOLD = 0.985
STM_OVERLOAD_R_THRESHOLD = 0.10
STM_OVERLOAD_DECAY_GAIN = 0.20

STM_FAST_WEIGHT = 0.18
STM_SLOW_WEIGHT = 0.82

STM_ENABLE_D2D_VARIATION = True
STM_CV_D2D = 0.015

STM_ENABLE_C2C_VARIATION = True
STM_CV_C2C = 0.025

STM_ENABLE_READ_NOISE = True
STM_READ_NOISE_REL_SIGMA = 0.003

STM_READ_IR_DROP_ALPHA = 0.04
STM_PROG_IR_DROP_ALPHA = 0.04
STM_ENABLE_SNEAK_PATH = True
STM_SNEAK_RATIO = 0.0015

#  -------------------------------------------------------------------
# R-STDP learning parameters
# ------------------------------------------------------------------

RSTDP_ELIGIBILITY_THRESHOLD = 1e-9

RSTDP_DELTA_W_SCALE = 5.0
RSTDP_DELTA_W_PER_PULSE = 0.005

RSTDP_PULSE_BASE = 1
RSTDP_PULSE_MAX = 3

RSTDP_TAU_PLUS = 4.0
RSTDP_TAU_MINUS = 4.0
RSTDP_A_PLUS = 1.0
RSTDP_A_MINUS = 0.8

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------
ENV_USE_FIXED_MAPS = True
ENV_MAP_SELECTION_MODE = "cycle"  # "cycle", "random", "fixed_index"

ENV_FIXED_MAPS = [
    {
        "name": "train_map_0",
        "grid": [
            "########",
            "#>.....#",
            "#..#...#",
            "#..#V..#",
            "#......#",
            "#.V.#..#",
            "#......#",
            "########",
        ],
    },
    {
        "name": "train_map_1",
        "grid": [
            "########",
            "#>..#..#",
            "#...#V.#",
            "#......#",
            "#.##...#",
            "#...V..#",
            "#......#",
            "########",
        ],
    },
]