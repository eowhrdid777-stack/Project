#----------------------device_model---------------------
import numpy as np  
# -------------------------------
# FeTFT paper-based conductance model
# -------------------------------
G_MIN = 12e-6
G_MAX = 172.8e-6          # ratio 14.4 exactly
G_INIT = 0.5 * (G_MIN + G_MAX)        # 기본 erased state 기준

P_MAX = 64              # paper: 64 conductance states

A_POT = -0.8028
A_DEP = -0.6979

CONDUCTANCE_WINDOW = G_MAX / G_MIN    # G_MAX / G_MIN

ENABLE_D2D_VARIATION = True
CV_D2D = 0.0393    # paper D2D variation 3.93%

ENABLE_C2C_VARIATION = True
CV_C2C = 0.0236   # paper C2C variation 2.36%

ENABLE_RETENTION = False
RETENTION_GAMMA = 0.0

G_RCP = G_INIT 

# ---------------- STM device ----------------
STM_G_MIN = 12.5e-6
STM_G_MAX = 180e-6
STM_G_INIT = STM_G_MIN
STM_P_MAX = 64

STM_A_POT = -0.8028
STM_A_DEP = -0.6979

STM_ENABLE_D2D_VARIATION = False
STM_CV_D2D = 0.0

STM_ENABLE_C2C_VARIATION = False
STM_CV_C2C = 0.0

STM_ENABLE_RETENTION = True
STM_RETENTION_GAMMA = 0.2
STM_G_RCP = STM_G_MIN

# paper pulse scheme
POT_START_V = 2.7
POT_STOP_V = 4.3
DEP_START_V = -2.0
DEP_STOP_V = -3.6
PULSE_V_STEP = 0.025
PULSE_WIDTH_S = 10e-3
READ_GATE_V = -1.0
READ_DRAIN_V = 1.0

# ------------------------conductance_modulation--------------------------------
SEED = 42

# --------------------------LTM--------------------------------
RECENTER_TRIGGER_FRACTION = 0.90
RECENTER_TARGET_FRACTION = 0.50
PROGRAM_TOLERANCE = 0.5e-6

MAX_VERIFY_STEPS = 30
PULSES_PER_VERIFY_STEP = 1
WEIGHT_CORRECTION_TOL = 0.5e-6

# -------------------------STM --------------------------------
STM_RECENTER_TRIGGER_FRACTION = 0.90
STM_RECENTER_TARGET_FRACTION = 0.60
STM_PROGRAM_TOLERANCE = 0.5e-6

STM_MAX_VERIFY_STEPS = 30
STM_PULSES_PER_VERIFY_STEP = 1
STM_WEIGHT_CORRECTION_TOL = 1e-6

# -------------------------------crossbar--------------------------------------
READ_VOLTAGE = 0.1
PROGRAM_VOLTAGE = 1.0

READ_IR_DROP_ALPHA = 0.05
PROG_IR_DROP_ALPHA = 0.05

ENABLE_READ_NOISE = True
READ_NOISE_REL_SIGMA = 0.005

ENABLE_READ_DISTURB = True
READ_DISTURB_STEP = 0.005

ENABLE_SNEAK_PATH = True
SNEAK_RATIO = 0.002