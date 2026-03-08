"""Global configuration for the memristor-SNN project.

Only project-wide settings are kept here.
Local temporary variables should stay inside each module.
"""

# =========================
# Random / reproducibility
# =========================
SEED = 42

# =========================
# Environment (placeholder)
# =========================
MAP_H = 12
MAP_W = 12
N_SURVIVORS = 3
MAX_STEPS = 300

# =========================
# Sensor / encoding
# =========================
N_DISTANCE_SENSORS = 3
SENSOR_LEVELS = 5
USE_DELTA_SENSOR = True

# Example input/output sizes
N_INPUT = 30
N_HIDDEN = 32
N_OUTPUT = 5

# =========================
# Differential memristor synapse
# =========================
USE_DIFFERENTIAL = True

# Crossbar shape for a single layer (example default)
DEVICE_ROWS = 4
DEVICE_COLS = 3

# Conductance range [S]
G_MIN = 1e-3
G_MAX = 10e-3
G_INIT = 5e-3

# Pulse-count model
N_PULSE_MIN = 0
N_PULSE_MAX = 200
N_PULSE_INIT = 70

# Monotonic potentiation curve:
# G(n) = G_0 + P_fast*(1-exp(-p_fast*n)) + P_slow*(1-exp(-p_slow*n))
# We solve P_slow internally so that G(N_PULSE_MAX) ~= G_MAX.
P_FAST_AMP = 1e-3
P_FAST_RATE = 0.05
P_SLOW_RATE = 0.005

# Mapping from abstract learning delta -> integer pulse count increment
# n_step = ceil(abs(delta_w) * PULSE_SCALE)
PULSE_SCALE = 4.0

# Optional clipping for a single update event
MAX_PULSE_STEP = 5

# =========================
# Read noise / variation
# Start with all-zero for ideal behavior.
# Turn these on later only after the full system works.
# =========================
ENABLE_D2D = False
ENABLE_C2C = False
READ_NOISE_STD = 0.2e-4   # absolute std [S]
CV_D2D = 0.05           # 0.05 means 5%
CV_C2C = 0.05           

# Pair normalization / soft reset
ENABLE_PAIR_RESET = True
PAIR_RESET_THRESHOLD = 0.9     # Gmax > 85%
PAIR_RESET_STEP = 30            # full range의 10%만큼 낮춤

# =========================
# Neuron (placeholder)
# =========================


# =========================
# Learning (placeholder)
# =========================

# =========================
# Simulation (placeholder)
# =========================
