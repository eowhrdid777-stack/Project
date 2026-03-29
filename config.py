from __future__ import annotations

"""Unified configuration for FeTFT differential-pair experiments.

Paper-based anchors from the uploaded supplement:
- 64 conductance states
- Gmax/Gmin = 14.4
- D2D variation = 3.93%
- C2C variation = 2.36%
- Read condition uses VG = -1 V, VD = 1 V
- Pot/depression pulse amplitudes sweep from 2.7->4.3 V and -2.0->-3.6 V
"""

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

# Array nonidealities
READ_VOLTAGE = 0.1
READ_AVG_SAMPLES = 1
PROGRAM_VOLTAGE = 1.0
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
PROGRAM_TOLERANCE = 1.0e-6
MAX_VERIFY_STEPS = 96
PULSES_PER_VERIFY_STEP = 1

# Neuron parameters
NEURON_MEMBRANE_DECAY = 0.97
NEURON_INPUT_GAIN = 1.0
NEURON_BASE_THRESHOLD = 3.5e-6
NEURON_REFRACTORY_STEPS = 1
NEURON_THRESHOLD_SCALE = 1.0e-6
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
