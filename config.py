#----------------------device_model---------------------

# -------------------------------
# Device conductance range
# -------------------------------
G_MIN = 1e-6
G_MAX = 100e-6
G_INIT = 30e-6

# -------------------------------
# Pulse response
# -------------------------------
# 1 pulse당 기본 변화량
G_POT_STEP = 1.2e-6
G_DEP_STEP = 1.2e-6

# 1.0이면 거의 linear
# 1.2 ~ 1.5면 약한 soft-bound
# 2.0 이상이면 강한 비선형
G_POT_BETA = 1.2
G_DEP_BETA = 1.2

# pot/dep asymmetry용 추가 스케일
POT_SCALE = 1.0
DEP_SCALE = 1.0

# ----------D2D variation on initial state----------
ENABLE_D2D_INIT_VARIATION = True
CV_D2D_INIT = 0.05
INIT_VARIATION_MODE = "lognormal"   # or "normal"

# -------------------------------
# D2D / C2C variation on step size
# -------------------------------
ENABLE_D2D_STEP_VARIATION = True
CV_D2D_STEP = 0.05

ENABLE_C2C_STEP_NOISE = True
CV_C2C_STEP = 0.02
# -------------------------------
# Retention
# -------------------------------
ENABLE_RETENTION = True
RETENTION_GAMMA = 0.0
G_RCP = G_INIT

# -------------------------------
# Random seed
# -------------------------------
SEED = 42

# ------------------------conductance_modulation--------------------------------

# -------------------------------------------------
# Conductance modulation / programming controller
# -------------------------------------------------

# 이 값을 넘으면 recenter 시작
RECENTER_TRIGGER_FRACTION = 0.85

# recenter 후 높은 쪽 conductance를 대략 어디까지 내릴지
# 예: 0.65 * G_MAX 정도
RECENTER_TARGET_FRACTION = 0.4

# target 도달 판정 허용오차
PROGRAM_TOLERANCE = 1e-6

# verify-after-write 최대 반복 횟수
PROGRAM_MAX_TRIALS = 30

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