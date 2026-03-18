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

RECENTER_TRIGGER_FRACTION = 0.90
RECENTER_TARGET_FRACTION = 0.60
PROGRAM_TOLERANCE = 0.5e-6

# linear-like pulse estimation용 평균 step
LINEAR_MEAN_STEP = (G_MAX - G_MIN) / (P_MAX - 1)

# -------------------------------crossbar--------------------------------------
READ_VOLTAGE = 0.1
PROGRAM_VOLTAGE = 1.0

READ_IR_DROP_ALPHA = 0.05
PROG_IR_DROP_ALPHA = 0.05

ENABLE_READ_NOISE = True
READ_NOISE_REL_SIGMA = 0.02

ENABLE_READ_DISTURB = True
READ_DISTURB_STEP = 0.002

ENABLE_SNEAK_PATH = True
SNEAK_RATIO = 0.001