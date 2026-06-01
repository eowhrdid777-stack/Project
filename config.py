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

ENABLE_READ_NOISE = False
READ_NOISE_REL_SIGMA = 0.003
ENABLE_SNEAK_PATH = False
SNEAK_RATIO = 0.0015
ENABLE_READ_DISTURB = False
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

# Real-robot-feasible sensor feature expansion.
# These are derived from distance/color/heading sensors and are fed only as SNN
# inputs. They must never be used as direct action override rules.
ENABLE_REAL_ROBOT_FEASIBLE_SENSOR_EXPANSION = True
CLEARANCE_OPEN_THRESHOLD = 0.6
CLEARANCE_BLOCKED_THRESHOLD = 0.25
BASIC_FEATURES = [
    "front_clearance",
    "left_clearance",
    "right_clearance",
    "victim_signal",
    "sound_signal",
]
OBSTACLE_BINARY_FEATURES = [
    "front_open",
    "front_blocked",
    "left_open",
    "left_blocked",
    "right_open",
    "right_blocked",
]
SIDE_COMPARISON_FEATURES = [
    "right_more_open_than_left",
    "left_more_open_than_right",
    "front_more_open_than_sides",
    "sides_more_open_than_front",
]
ENABLE_COLOR_RISK_FEATURES = False
COLOR_RISK_FEATURES = [
    "risk_signal",
    "safe_floor_signal",
]
ENABLE_HEADING_FEATURES = False
HEADING_FEATURES = [
    "heading_sin",
    "heading_cos",
]
ACTIVE_ENCODER_FEATURES = (
    BASIC_FEATURES + OBSTACLE_BINARY_FEATURES + SIDE_COMPARISON_FEATURES
)

# Neuron parameters
NEURON_ENABLE_WTA = False
NEURON_LATERAL_INHIBITION = 0.05
NEURON_ENABLE_THRESHOLD_ADAPTATION = True
NEURON_RESET_THRESHOLD_ADAPTATION_EACH_EPISODE = False
NEURON_MEMBRANE_DECAY = 0.97
NEURON_INPUT_GAIN = 20.0
NEURON_BASE_THRESHOLD = 1.0e-7
NETWORK_HIDDEN_DIM = 16
NETWORK_HIDDEN_BASE_THRESHOLD = NEURON_BASE_THRESHOLD
NETWORK_OUTPUT_BASE_THRESHOLD = 1.0e-7
NETWORK_OUTPUT_ENABLE_WTA = False
NETWORK_OUTPUT_LATERAL_INHIBITION = 0.05
OUTPUT_ACTION_SELECTION_MODE = "first_spike"
OUTPUT_ACTION_SELECTION_DIAGNOSTIC_MODES = (
    "first_spike",
    "spike_count",
    "hybrid_spike_count_latency",
)
OUTPUT_COUNT_WEIGHT = 1.0
OUTPUT_LATENCY_WEIGHT = 0.05
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
NETWORK_RECURRENT_HIDDEN_CROSSBAR_TYPE = "stm"
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
STM_ENABLE_READ_NOISE = False
STM_READ_NOISE_REL_SIGMA = 0.003

# Crossbar nonidealities
STM_READ_IR_DROP_ALPHA = 0.04
STM_PROG_IR_DROP_ALPHA = 0.04
STM_ENABLE_SNEAK_PATH = False
STM_SNEAK_RATIO = 0.0015

# Recurrent-feedback timing hook to be used by stm_crossbar.py/network.py if needed.
# After each network decision/robot step, relaxing STM recurrent cells by the elapsed
# physical time makes the feedback memory time-scale meaningful.
STM_RELAX_PER_DECISION_S = 5.0e-3

#  -------------------------------------------------------------------
# R-STDP learning parameters
# ------------------------------------------------------------------

RSTDP_ELIGIBILITY_THRESHOLD = 1e-9

RSTDP_DELTA_W_SCALE = 5.0
RSTDP_DELTA_W_PER_PULSE = 0.005

RSTDP_PULSE_BASE = 1
RSTDP_PULSE_MAX = 3
RSTDP_USE_SURROGATE_POST_ON_TARGET = True
RSTDP_USE_ABS_ELIGIBILITY_ON_TARGET_NEGATIVE = False
RSTDP_ENABLE_HIDDEN = False
RSTDP_ENABLE_HIDDEN_PHASE_KEYWORDS = ()
OUTPUT_DEPRESSION_SCALE = 1.0
ANTI_TARGET_DEPRESSION = False
ANTI_TARGET_DEPRESSION_SCALE = 0.25

RSTDP_TAU_PLUS = 4.0
RSTDP_TAU_MINUS = 4.0
RSTDP_A_PLUS = 1.0
RSTDP_A_MINUS = 0.8

# Train-only exploration
TRAIN_EXPLORATION_EPSILON = 0.10
WALL_AVOID_TRAIN_EXPLORATION_EPSILON = 0.20
RUN_MODE = "curriculum"  # "single" or "curriculum"
ENABLE_TURN_SENSOR_CHANGE_REWARD = True
TURN_SENSOR_REWARD_DISABLED_PHASE_KEYWORDS = ("stage_forward",)
TURN_REWARD_FRONT_BLOCKED_THRESHOLD = 0.35
TURN_REWARD_CLEARANCE_IMPROVEMENT = 0.15
TURN_REWARD_POSITIVE = 0.30
TURN_REWARD_CLEARANCE_WORSENING = 0.15
TURN_REWARD_WORSENING_TURN_PENALTY = 0.30
TURN_PENALTY_FRONT_CLEAR_THRESHOLD = 0.65
TURN_REWARD_UNNECESSARY_TURN_PENALTY = 0.03
ENABLE_FORWARD_SENSOR_CHANGE_REWARD = True
FORWARD_REWARD_POSITIVE = 0.30
FORWARD_REWARD_CLEAR_PATH_POSITIVE = 0.20
FORWARD_REWARD_CLEAR_PATH_THRESHOLD = 0.35
FORWARD_REWARD_CLEAR_PATH_MARGIN = 0.15
FORWARD_REWARD_VICTIM_SIGNAL_DROP = 0.10
FORWARD_REWARD_FRONT_WORSENING = 0.15
FORWARD_REWARD_BAD_FORWARD_PENALTY = 0.25
WALL_AVOID_FORWARD_POSITIVE_ENABLED = True
WALL_AVOID_FORWARD_REWARD_MAX_PER_EPISODE = 2
WALL_AVOID_FORWARD_REQUIRE_MOVED = True
WALL_AVOID_FORWARD_REQUIRE_NO_COLLISION = True
WALL_AVOID_FORWARD_MIN_FRONT_CLEARANCE = 0.45
WALL_AVOID_FORWARD_RECOVERY_REWARD = 0.10
WALL_AVOID_FORWARD_RECOVERY_MAX_PER_EPISODE = 4
WALL_AVOID_FORWARD_SIGNAL_DROP_TOLERANCE = 0.35
WALL_AVOID_FORWARD_COLLISION_PENALTY = 0.25
WALL_AVOID_COLLISION_PENALTY_SCALE = 0.8
WALL_AVOID_FALLBACK_COLLISION_PENALTY_FACTOR = 0.4
WALL_AVOID_USE_ABS_ELIGIBILITY_ON_FORWARD_COLLISION = False
WALL_AVOID_USE_SURROGATE_POST_ON_FORWARD_COLLISION = False
WALL_AVOID_USE_ABS_ELIGIBILITY_ON_TURN_PENALTY = True
WALL_AVOID_IGNORE_VICTIM_DROP_FOR_TURN_REWARD = True
WALL_AVOID_TURN_REWARD_POSITIVE = 0.12
WALL_AVOID_TURN_REWARD_TIE_POSITIVE = 0.03
WALL_AVOID_TURN_REWARD_OPEN_SIDE_POSITIVE = 0.45
WALL_AVOID_TURN_OPEN_SIDE_MARGIN = 0.20
WALL_AVOID_TURN_PENALTY_NEGATIVE = 0.45
WALL_AVOID_TURN_WORSENED_FRONT_PENALTY = 0.03
WALL_AVOID_SPIN_TURN_PENALTY = 0.03
WALL_AVOID_SPIN_TURN_THRESHOLD = 3
WALL_AVOID_SPIN_PENALTY_MAX_PER_EPISODE = 5
WALL_AVOID_TURN_REWARD_MAX_PER_ACTION_PER_EPISODE = 2
WALL_AVOID_TURN_REWARD_MAX_PER_EPISODE = 4
ENABLE_DELAYED_ACTION_CREDIT = False
ENABLE_PREVIOUS_TURN_CREDIT = False
RECENT_CREDIT_WINDOW = 3
RECENT_CREDIT_DECAY = 0.5
EVAL_FORCE_ACTION_ON_NO_SPIKE = False
RESET_HIDDEN_TRACE_ON_EVAL_EPISODE_START = True
RESET_NEURON_STATE_ON_EVAL_EPISODE_START = True
RESET_ACTION_HISTORY_ON_EVAL_EPISODE_START = True
EVAL_TRACE_FIRST_STEPS = 20
ENABLE_OUTPUT_COLUMN_BALANCE = True
OUTPUT_COLUMN_BALANCE_RATIO = 2.0

# Logging
LOG_DIR = "logs"
PRINT_STEP_DEBUG = False
PRINT_POSITIVE_REWARD_STEPS = False
SAVE_POSITIVE_REWARD_STEPS_JSON = True
SAVE_STAGE_SUMMARY_JSON = True
SAVE_FINAL_SUMMARY_JSON = True
PRINT_STAGE_COMPACT_SUMMARY = False
ENABLE_OUTPUT_SELECTION_MODE_DIAGNOSTICS = False
SAVE_PRIMITIVE_REGRESSION_SUMMARY_JSON = True
SAVE_PRIMITIVE_CORE_TRAINING_SUMMARY_JSON = True
SAVE_INTERMEDIATE_CURRICULUM_SUMMARY_JSON = True
SAVE_STABLE_WALL_AVOID_SUCCESS_SUMMARY_JSON = True
SAVE_FINAL_EVAL_TRACE_JSON = True
ENABLE_FINAL_MAP_STM_SWEEP = False
ENABLE_PRIMITIVE_REHEARSAL_AFTER_INTERMEDIATE = False
ENABLE_INTERLEAVED_PRIMITIVE_REHEARSAL = True
REHEARSAL_EPISODES_PER_PRIMITIVE = 3
REHEARSAL_EXPLORATION_EPSILON = 0.15
INTERLEAVED_REHEARSAL_STAGE_NAMES = (
    "stage_turn_right",
    "stage_turn_left",
    "stage_wall_avoid",
)
PRIMITIVE_REGRESSION_HISTORY_EVAL_EPISODES = 1
PRIMITIVE_REHEARSAL_STAGE_NAMES = (
    "stage_forward",
    "stage_turn_right",
    "stage_turn_left",
    "stage_wall_avoid",
)
PRIMITIVE_REHEARSAL_TRAIN_EPISODE_SCALE = 2.0
PRIMITIVE_REHEARSAL_CYCLES = 2
ENABLE_STAGE_ROLLBACK_ON_PRIMITIVE_FAILURE = True
DISCARD_COMPLEX_STAGE_ON_PRIMITIVE_FAILURE = True
STAGE_ROLLBACK_RETRY_SCALE = 0.5
STAGE_ROLLBACK_MIN_TRAIN_EPISODES = 1
ENABLE_PRIMITIVE_CONSOLIDATION_BEFORE_COMPLEX = True
PRIMITIVE_CONSOLIDATION_CYCLES = 5
PRIMITIVE_CONSOLIDATION_EPISODES_PER_STAGE = 2
PRIMITIVE_CONSOLIDATION_MAX_ATTEMPTS = 2
ENABLE_PRIMITIVE_CORE_TRAINING = True
PRIMITIVE_CORE_EPISODES = 100
PRIMITIVE_CORE_TASK_SAMPLING = "balanced_cycle"
PRIMITIVE_CORE_EVAL_INTERVAL = 4
ENABLE_PRIMITIVE_CORE_BLOCK_ROLLBACK = True
PRIMITIVE_CORE_BLOCK_SIZE = 4
PRIMITIVE_CORE_ACCEPT_IF_ALL_PASS = True
PRIMITIVE_CORE_ACCEPT_IF_SCORE_IMPROVES = True
PRIMITIVE_CORE_ENABLE_HIDDEN_TRACE = False
PRIMITIVE_CORE_HIDDEN_TRACE_INPUT_SCALE = 0.0
PRIMITIVE_CORE_STM_RECURRENT_CURRENT_SCALE = 0.0
PRIMITIVE_RSTDP_SCALE = {
    "stage_forward": 0.5,
    "stage_turn_right": 0.5,
    "stage_turn_left": 0.5,
    "stage_wall_avoid": 0.5,
}
ENABLE_PRIMITIVE_OBSERVATION_ALIAS_DIAGNOSTIC = True
PRIMITIVE_REFERENCE_FIRST_ACTIONS = {
    "stage_forward": 0,
    "stage_turn_right": 2,
    "stage_turn_left": 1,
    "stage_wall_avoid": 2,
}
PRIMITIVE_ALIAS_RAW_L2_THRESHOLD = 0.25
PRIMITIVE_ALIAS_ENCODED_COSINE_THRESHOLD = 0.95
PRIMITIVE_ALIAS_ACTIVE_OVERLAP_THRESHOLD = 0.85
ENABLE_PRIMITIVE_HIDDEN_REPRESENTATION_DIAGNOSTIC = True
PRIMITIVE_HIDDEN_ALIAS_COSINE_THRESHOLD = 0.8
PRIMITIVE_HIDDEN_ALIAS_OVERLAP_THRESHOLD = 0.8
ENABLE_OUTPUT_COLUMN_INTERFERENCE_DIAGNOSTIC = True
ENABLE_PRIMITIVE_CORE_SEARCH_SWEEP = False
PRIMITIVE_CORE_SEARCH_MAX_CONFIGS = 12
PRIMITIVE_CORE_SEARCH_STOP_ON_FOUND = True
PRIMITIVE_CORE_SEARCH_HIDDEN_DIMS = (16, 24, 32)
PRIMITIVE_CORE_SEARCH_HIDDEN_BASE_THRESHOLDS = (NETWORK_HIDDEN_BASE_THRESHOLD,)
PRIMITIVE_CORE_SEARCH_OUTPUT_WTA_MODES = (False, True)
PRIMITIVE_CORE_SEARCH_OUTPUT_BASE_THRESHOLDS = (NETWORK_OUTPUT_BASE_THRESHOLD,)
PRIMITIVE_CORE_SEARCH_RSTDP_SCALES = (0.25, 0.5, 1.0)
PRIMITIVE_CORE_SEARCH_OUTPUT_DEPRESSION_SCALES = (0.25, 0.5, 1.0)
PRIMITIVE_CORE_SEARCH_ANTI_TARGET_DEPRESSION = (False, True)
PRIMITIVE_CORE_SEARCH_REHEARSAL_STYLES = ("balanced_cycle",)
PRIMITIVE_CORE_SEARCH_HIDDEN_TRACE_INPUT_SCALES = (0.0, 0.25)
PRIMITIVE_CORE_SEARCH_STM_RECURRENT_CURRENT_SCALES = (0.0,)
ENABLE_FINAL_MAP_CHECKPOINT_SELECTION = True
FINAL_MAP_CHECKPOINT_PROBE_EPISODES = 1
FINAL_MAP_CHECKPOINT_REQUIRE_PRIMITIVES = True
ENABLE_BRANCH_CANDIDATE_CURRICULUM = True
COMMIT_ONLY_IF_PRIMITIVE_PASSED = True
COMMIT_ONLY_IF_SCORE_IMPROVES = True
BRANCH_ACCEPT_COLLISION_IMPROVEMENT = 10
BRANCH_ACCEPT_FOUND_IMPROVEMENT = 1
DEFAULT_COMPLEX_STAGE_ENABLED = {
    "stage_wall_avoid_open_left": False,
    "stage_wall_avoid_open_right": False,
    "stage_wall_avoid_both_open": False,
    "stage_wall_avoid_near_wall_after_forward": False,
    "stage_wall_avoid_corner_escape_left": False,
    "stage_wall_avoid_corner_escape_right": False,
    "stage_two_step_forward_easy": True,
    "stage_two_step_forward_medium": True,
    "stage_turn_then_two_forward": True,
    "stage_zigzag_single_turn": False,
    "stage_zigzag_corridor": False,
    "stage_simple_deadend_escape": False,
    "stage_two_victim_linear": True,
    "stage_two_victim_with_one_turn": True,
    "stage_two_victim_small": True,
    "stage_mini_rescue_easy": True,
    "stage_mini_rescue": True,
    "stage_rescue_map_6x9_train": False,
}
FINAL_MAP_TRAIN_EXPLORATION_EPSILON = 0.25
FINAL_MAP_EVAL_EXPLORATION_EPSILON = 0.0
ENABLE_STAGEWISE_RSTDP_SCALE = True
STAGE_RSTDP_SCALE = {
    "stage_forward": 1.0,
    "stage_turn_right": 1.0,
    "stage_turn_left": 1.0,
    "stage_wall_avoid": 1.0,
    "stage_wall_avoid_open_left": 0.7,
    "stage_wall_avoid_open_right": 0.7,
    "stage_wall_avoid_both_open": 0.7,
    "stage_wall_avoid_near_wall_after_forward": 0.65,
    "stage_wall_avoid_corner_escape_left": 0.65,
    "stage_wall_avoid_corner_escape_right": 0.65,
    "stage_two_step_forward_easy": 0.6,
    "stage_two_step_forward_medium": 0.55,
    "stage_turn_then_two_forward": 0.5,
    "stage_zigzag_single_turn": 0.45,
    "stage_zigzag_corridor": 0.35,
    "stage_simple_deadend_escape": 0.35,
    "stage_two_victim_linear": 0.35,
    "stage_two_victim_with_one_turn": 0.30,
    "stage_two_victim_small": 0.25,
    "stage_mini_rescue_easy": 0.25,
    "stage_mini_rescue": 0.20,
    "stage_rescue_map_6x9_train": 0.15,
    "final_eval": 1.0,
}
ENABLE_STAGEWISE_STM_TRACE_SCALE = True
STAGE_HIDDEN_TRACE_INPUT_SCALE = {
    "stage_wall_avoid": 2.0,
    "stage_wall_avoid_open_left": 1.8,
    "stage_wall_avoid_open_right": 1.8,
    "stage_wall_avoid_both_open": 1.8,
    "stage_wall_avoid_near_wall_after_forward": 1.6,
    "stage_wall_avoid_corner_escape_left": 1.6,
    "stage_wall_avoid_corner_escape_right": 1.6,
    "stage_two_step_forward_easy": 1.5,
    "stage_two_step_forward_medium": 1.4,
    "stage_turn_then_two_forward": 1.4,
    "stage_zigzag_single_turn": 1.2,
    "stage_zigzag_corridor": 1.0,
    "stage_simple_deadend_escape": 1.0,
    "stage_two_victim_linear": 1.0,
    "stage_two_victim_with_one_turn": 0.9,
    "stage_two_victim_small": 0.9,
    "stage_mini_rescue_easy": 0.85,
    "stage_mini_rescue": 0.8,
    "stage_rescue_map_6x9_train": 0.7,
    "final_eval": 0.7,
}
TARGET_RECURRENT_INPUT_RATIO_MAX = 0.8
FINAL_MAP_FORWARD_COLLISION_PENALTY_SCALE = 1.2
FINAL_MAP_FORWARD_COLLISION_RSTDP_SCALE = 1.0
FINAL_MAP_REPEATED_FORWARD_COLLISION_PENALTY = 0.05
FINAL_MAP_FORWARD_SENSOR_POSITIVE_ENABLED = False

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------
ENV_WIDTH = 6
ENV_HEIGHT = 9

ENV_REWARD_DANGER = -1.0
ENV_VICTIM_DETECTION_RADIUS = 1
ENV_LOW_FRONT_CLEARANCE_THRESHOLD = 0.35

ENV_USE_FIXED_MAPS = True
ENV_MAP_SELECTION_MODE = "fixed_index" # "cycle", "random", "fixed_index"
ENV_FIXED_MAP_INDEX = 0

EASY_FORWARD_1_VICTIM_MAP = {
    "name": "easy_forward_1_victim",
    "grid": [
        "...",
        "R.V",
        "...",
    ],
    "heading": 1,
}

EASY_TURN_RIGHT_1_VICTIM_MAP = {
    "name": "easy_turn_right_1_victim",
    "grid": [
        "...",
        "R.V",
        "...",
    ],
    "heading": 0,
}

EASY_TURN_LEFT_1_VICTIM_MAP = {
    "name": "easy_turn_left_1_victim",
    "grid": [
        "...",
        "R.V",
        "...",
    ],
    "heading": 2,
}

EASY_WALL_AVOID_1_VICTIM_MAP = {
    "name": "easy_wall_avoid_1_victim",
    "grid": [
        "...",
        "R#V",
        "...",
    ],
    "heading": 1,
}

WALL_AVOID_FRONT_WALL_OPEN_LEFT_MAP = {
    "name": "wall_avoid_front_wall_open_left",
    "grid": [
        "....",
        ".R#V",
        ".#..",
    ],
    "heading": 1,
}

WALL_AVOID_FRONT_WALL_OPEN_RIGHT_MAP = {
    "name": "wall_avoid_front_wall_open_right",
    "grid": [
        ".#..",
        ".R#V",
        "....",
    ],
    "heading": 1,
}

WALL_AVOID_FRONT_WALL_BOTH_OPEN_MAP = {
    "name": "wall_avoid_front_wall_both_open",
    "grid": [
        "....",
        ".R#V",
        "....",
    ],
    "heading": 1,
}

WALL_AVOID_NEAR_WALL_AFTER_FORWARD_MAP = {
    "name": "wall_avoid_near_wall_after_forward",
    "grid": [
        "....V",
        ".R.#.",
        ".....",
    ],
    "heading": 1,
}

WALL_AVOID_CORNER_ESCAPE_LEFT_MAP = {
    "name": "wall_avoid_corner_escape_left",
    "grid": [
        "....",
        "#R#V",
        "##..",
    ],
    "heading": 1,
}

WALL_AVOID_CORNER_ESCAPE_RIGHT_MAP = {
    "name": "wall_avoid_corner_escape_right",
    "grid": [
        "##..",
        "#R#V",
        "....",
    ],
    "heading": 1,
}

TWO_STEP_FORWARD_EASY_MAP = {
    "name": "two_step_forward_easy_1_victim",
    "grid": [
        "...",
        "R.V",
        "...",
    ],
    "heading": 1,
}

TWO_STEP_FORWARD_MEDIUM_MAP = {
    "name": "two_step_forward_medium_1_victim",
    "grid": [
        "....",
        "R..V",
        "....",
    ],
    "heading": 1,
}

TURN_THEN_TWO_FORWARD_MAP = {
    "name": "turn_then_two_forward_1_victim",
    "grid": [
        "....",
        "....",
        "R..V",
    ],
    "heading": 0,
}

ZIGZAG_SINGLE_TURN_MAP = {
    "name": "zigzag_single_turn_1_victim",
    "grid": [
        "R#.",
        "..V",
        "...",
    ],
    "heading": 1,
}

ZIGZAG_CORRIDOR_MAP = {
    "name": "zigzag_corridor_1_victim",
    "grid": [
        "R#..",
        "...#",
        "#..V",
    ],
    "heading": 1,
}

SIMPLE_DEADEND_ESCAPE_MAP = {
    "name": "simple_deadend_escape_1_victim",
    "grid": [
        "###",
        "#R#",
        "#.V",
    ],
    "heading": 0,
}

TWO_VICTIM_LINEAR_MAP = {
    "name": "two_victim_linear",
    "grid": [
        "R.V.V",
        ".....",
        ".....",
    ],
    "heading": 1,
}

TWO_VICTIM_WITH_ONE_TURN_MAP = {
    "name": "two_victim_with_one_turn",
    "grid": [
        "R.V",
        "..V",
        "...",
    ],
    "heading": 1,
}

TWO_VICTIM_SMALL_MAP = {
    "name": "two_victim_small",
    "grid": [
        "R..V",
        "....",
        ".#..",
        "V...",
    ],
    "heading": 1,
}

MINI_RESCUE_EASY_4X4_MAP = {
    "name": "mini_rescue_easy_4x4",
    "grid": [
        "R..V",
        ".#..",
        "..V.",
        "....",
    ],
    "heading": 1,
}

MINI_RESCUE_5X5_MAP = {
    "name": "mini_rescue_5x5",
    "grid": [
        "R...V",
        ".#.#.",
        "..V..",
        ".#...",
        "V...D",
    ],
    "heading": 1,
}

RESCUE_MAP_6X9 = {
    "name": "rescue_map_6x9",

    "grid": [
        "R.....",   # row 0
        "#V..V.",   # row 1
        "....#.",   # row 2
        ".#....",   # row 3
        "...VV.",   # row 4
        "#..VV.",   # row 5
        "...DD.",   # row 6
        "......",   # row 7
        "VD.DD.",   # row 8
    ],

    # ------------------------------------------
    # horizontal wall
    # (r,c) <-> (r+1,c)
    # ------------------------------------------
    "h_walls": [

       (0,1), (0,3), (0,4), (3,3), (3,4),
       (5,3), (5,4), (6,3), (6,4),
       (7,1), (7,3), (7,4)

    ],

    # ------------------------------------------
    # vertical wall
    # (r,c) <-> (r,c+1)
    # ------------------------------------------
    "v_walls": [

        (7,0),
        (1,1), (2,1), (4,1), (5,1), (6,1),
        (1,2), (2,2), (3,2), (4,2), (5,2),
        (1,4), (4,4)

    ],

    "heading": 1,
}

CURRICULUM_FIXED_MAPS = [
    EASY_FORWARD_1_VICTIM_MAP,
    EASY_TURN_RIGHT_1_VICTIM_MAP,
    EASY_TURN_LEFT_1_VICTIM_MAP,
    EASY_WALL_AVOID_1_VICTIM_MAP,
    WALL_AVOID_FRONT_WALL_OPEN_LEFT_MAP,
    WALL_AVOID_FRONT_WALL_OPEN_RIGHT_MAP,
    WALL_AVOID_FRONT_WALL_BOTH_OPEN_MAP,
    WALL_AVOID_NEAR_WALL_AFTER_FORWARD_MAP,
    WALL_AVOID_CORNER_ESCAPE_LEFT_MAP,
    WALL_AVOID_CORNER_ESCAPE_RIGHT_MAP,
    TWO_STEP_FORWARD_EASY_MAP,
    TWO_STEP_FORWARD_MEDIUM_MAP,
    TURN_THEN_TWO_FORWARD_MAP,
    ZIGZAG_SINGLE_TURN_MAP,
    ZIGZAG_CORRIDOR_MAP,
    SIMPLE_DEADEND_ESCAPE_MAP,
    TWO_VICTIM_LINEAR_MAP,
    TWO_VICTIM_WITH_ONE_TURN_MAP,
    TWO_VICTIM_SMALL_MAP,
    MINI_RESCUE_EASY_4X4_MAP,
    MINI_RESCUE_5X5_MAP,
]
CURRICULUM_FIXED_MAPS_BY_NAME = {
    map_spec["name"]: map_spec
    for map_spec in CURRICULUM_FIXED_MAPS
}
ALL_FIXED_MAPS_BY_NAME = {
    **CURRICULUM_FIXED_MAPS_BY_NAME,
    RESCUE_MAP_6X9["name"]: RESCUE_MAP_6X9,
}

# Use either a stage index, 0..3, or one of the curriculum map names.
ENV_CURRICULUM_STAGE = 0
CURRICULUM_MAP_NAMES = [
    "easy_forward_1_victim",
    "easy_turn_right_1_victim",
    "easy_turn_left_1_victim",
    "easy_wall_avoid_1_victim",
    "wall_avoid_front_wall_open_left",
    "wall_avoid_front_wall_open_right",
    "wall_avoid_front_wall_both_open",
    "wall_avoid_near_wall_after_forward",
    "wall_avoid_corner_escape_left",
    "wall_avoid_corner_escape_right",
    "two_step_forward_easy_1_victim",
    "two_step_forward_medium_1_victim",
    "turn_then_two_forward_1_victim",
    "zigzag_single_turn_1_victim",
    "zigzag_corridor_1_victim",
    "simple_deadend_escape_1_victim",
    "two_victim_linear",
    "two_victim_with_one_turn",
    "two_victim_small",
    "mini_rescue_easy_4x4",
    "mini_rescue_5x5",
    "rescue_map_6x9",
]

CURRICULUM_STAGE_SETTINGS = {
    "stage_forward": {
        "train_episodes": 5,
        "eval_episodes": 3,
        "exploration_epsilon": 0.10,
    },
    "stage_turn_right": {
        "train_episodes": 10,
        "eval_episodes": 3,
        "exploration_epsilon": 0.10,
    },
    "stage_turn_left": {
        "train_episodes": 10,
        "eval_episodes": 3,
        "exploration_epsilon": 0.10,
    },
    "stage_wall_avoid": {
        "train_episodes": 15,
        "eval_episodes": 3,
        "exploration_epsilon": 0.20,
    },
    "stage_wall_avoid_open_left": {
        "train_episodes": 12,
        "eval_episodes": 3,
        "exploration_epsilon": 0.20,
    },
    "stage_wall_avoid_open_right": {
        "train_episodes": 12,
        "eval_episodes": 3,
        "exploration_epsilon": 0.20,
    },
    "stage_wall_avoid_both_open": {
        "train_episodes": 12,
        "eval_episodes": 3,
        "exploration_epsilon": 0.20,
    },
    "stage_wall_avoid_near_wall_after_forward": {
        "train_episodes": 12,
        "eval_episodes": 3,
        "exploration_epsilon": 0.20,
    },
    "stage_wall_avoid_corner_escape_left": {
        "train_episodes": 12,
        "eval_episodes": 3,
        "exploration_epsilon": 0.20,
    },
    "stage_wall_avoid_corner_escape_right": {
        "train_episodes": 12,
        "eval_episodes": 3,
        "exploration_epsilon": 0.20,
    },
    "stage_two_step_forward_easy": {
        "train_episodes": 8,
        "eval_episodes": 3,
        "exploration_epsilon": 0.20,
    },
    "stage_two_step_forward_medium": {
        "train_episodes": 15,
        "eval_episodes": 3,
        "exploration_epsilon": 0.25,
    },
    "stage_turn_then_two_forward": {
        "train_episodes": 20,
        "eval_episodes": 3,
        "exploration_epsilon": 0.30,
    },
    "stage_zigzag_single_turn": {
        "train_episodes": 15,
        "eval_episodes": 3,
        "exploration_epsilon": 0.25,
    },
    "stage_zigzag_corridor": {
        "train_episodes": 20,
        "eval_episodes": 3,
        "exploration_epsilon": 0.30,
    },
    "stage_simple_deadend_escape": {
        "train_episodes": 20,
        "eval_episodes": 3,
        "exploration_epsilon": 0.30,
    },
    "stage_two_victim_linear": {
        "train_episodes": 20,
        "eval_episodes": 3,
        "exploration_epsilon": 0.30,
    },
    "stage_two_victim_with_one_turn": {
        "train_episodes": 25,
        "eval_episodes": 3,
        "exploration_epsilon": 0.32,
    },
    "stage_two_victim_small": {
        "train_episodes": 30,
        "eval_episodes": 3,
        "exploration_epsilon": 0.35,
    },
    "stage_mini_rescue_easy": {
        "train_episodes": 30,
        "eval_episodes": 3,
        "exploration_epsilon": 0.32,
    },
    "stage_mini_rescue": {
        "train_episodes": 40,
        "eval_episodes": 3,
        "exploration_epsilon": 0.35,
    },
    "stage_rescue_map_6x9_train": {
        "train_episodes": 50,
        "eval_episodes": 3,
        "exploration_epsilon": 0.35,
    },
}

CURRICULUM_STAGES = [
    {
        "name": "stage_forward",
        "map_name": "easy_forward_1_victim",
    },
    {
        "name": "stage_turn_right",
        "map_name": "easy_turn_right_1_victim",
    },
    {
        "name": "stage_turn_left",
        "map_name": "easy_turn_left_1_victim",
    },
    {
        "name": "stage_wall_avoid",
        "map_name": "easy_wall_avoid_1_victim",
    },
    {
        "name": "stage_wall_avoid_open_left",
        "map_name": "wall_avoid_front_wall_open_left",
    },
    {
        "name": "stage_wall_avoid_open_right",
        "map_name": "wall_avoid_front_wall_open_right",
    },
    {
        "name": "stage_wall_avoid_both_open",
        "map_name": "wall_avoid_front_wall_both_open",
    },
    {
        "name": "stage_wall_avoid_near_wall_after_forward",
        "map_name": "wall_avoid_near_wall_after_forward",
    },
    {
        "name": "stage_wall_avoid_corner_escape_left",
        "map_name": "wall_avoid_corner_escape_left",
    },
    {
        "name": "stage_wall_avoid_corner_escape_right",
        "map_name": "wall_avoid_corner_escape_right",
    },
    {
        "name": "stage_two_step_forward_easy",
        "map_name": "two_step_forward_easy_1_victim",
    },
    {
        "name": "stage_two_step_forward_medium",
        "map_name": "two_step_forward_medium_1_victim",
    },
    {
        "name": "stage_turn_then_two_forward",
        "map_name": "turn_then_two_forward_1_victim",
    },
    {
        "name": "stage_zigzag_single_turn",
        "map_name": "zigzag_single_turn_1_victim",
    },
    {
        "name": "stage_zigzag_corridor",
        "map_name": "zigzag_corridor_1_victim",
    },
    {
        "name": "stage_simple_deadend_escape",
        "map_name": "simple_deadend_escape_1_victim",
    },
    {
        "name": "stage_two_victim_linear",
        "map_name": "two_victim_linear",
    },
    {
        "name": "stage_two_victim_with_one_turn",
        "map_name": "two_victim_with_one_turn",
    },
    {
        "name": "stage_two_victim_small",
        "map_name": "two_victim_small",
    },
    {
        "name": "stage_mini_rescue_easy",
        "map_name": "mini_rescue_easy_4x4",
    },
    {
        "name": "stage_mini_rescue",
        "map_name": "mini_rescue_5x5",
    },
    {
        "name": "stage_rescue_map_6x9_train",
        "map_name": "rescue_map_6x9",
    },
]
CURRICULUM_FINAL_EVAL_MAP = "rescue_map_6x9"
CURRICULUM_FINAL_EVAL_EPISODES = 3

# Backward-compatible alias for older main.py snippets.
CURRICULUM_EVAL_MAP_NAME = CURRICULUM_FINAL_EVAL_MAP

if isinstance(ENV_CURRICULUM_STAGE, str):
    ENV_FIXED_MAPS = [ALL_FIXED_MAPS_BY_NAME[ENV_CURRICULUM_STAGE]]
else:
    ENV_FIXED_MAPS = [CURRICULUM_FIXED_MAPS[int(ENV_CURRICULUM_STAGE)]]
# -------------------------------------------------------------------
# ------------------------------------------------------------------
# real robot 
# ------------------------------------------------------------------
# Paste these into config.py when running on the real robot.
# This version assumes HC-06 Bluetooth serial, red/blue/green survivor markers,
# and one DFR0034 analog sound sensor on Arduino A0.

# ------------------------------------------------------------------
# Real robot serial / Bluetooth settings
# ------------------------------------------------------------------
# Windows Bluetooth serial example: "COM8", "COM9", ...
# Linux/Raspberry Pi examples: "/dev/rfcomm0", "/dev/ttyUSB0"
ROBOT_SERIAL_PORT = "COM8"

# HC-06 default baudrate is usually 9600. Arduino Serial.begin(...) must match.
ROBOT_BAUDRATE = 9600
ROBOT_SERIAL_TIMEOUT_S = 2.0

ROBOT_ENABLE_LEARNING = True
ROBOT_STEP_PAUSE_S = 0.05
ROBOT_ACTION_INTERVAL_S = 0.30

# ------------------------------------------------------------------
# ToF distance calibration
# ------------------------------------------------------------------
# These convert raw VL53L1X distance [mm] to 0..1 clearance.
ROBOT_CLEAR_MIN_MM = 10.0
ROBOT_CLEAR_MAX_MM = 200.0

# Safety interlock only. If forward is selected and front distance is below
# this value, Arduino stops instead of driving into the wall and reports
# collision=True. It must not auto-turn or choose another action.
ROBOT_OBSTACLE_STOP_MM = 10.0

# ------------------------------------------------------------------
# TCS34725 survivor color markers
# ------------------------------------------------------------------
# Any of these classified colors will produce victim_signal = 1.0.
ROBOT_VICTIM_COLOR_NAMES = ["red", "blue", "green"]

# Backward-compatible single-color option. robot_interface.py will prefer
# ROBOT_VICTIM_COLOR_NAMES when it exists.
ROBOT_VICTIM_COLOR_NAME = "red"

# Minimum clear-channel value for color classification to be trusted.
ROBOT_COLOR_MIN_C = 40

# ------------------------------------------------------------------
# DFR0034 analog sound sensor
# ------------------------------------------------------------------
# Arduino wiring:
#   DFR0034 VCC -> 5V
#   DFR0034 GND -> GND
#   DFR0034 OUT -> A0
ROBOT_ENABLE_SOUND_SENSOR = True

# Convert raw analog peak value to 0..1 sound_signal.
# Tune these after checking raw values in your actual room.
# Example: quiet floor ~= 350, sound marker/buzzer nearby ~= 800.
ROBOT_SOUND_RAW_MIN = 350
ROBOT_SOUND_RAW_MAX = 800

# If sound_signal exceeds this value, real_robot_env.py may count it as a
# sensor-based survivor detection signal.
ROBOT_SOUND_THRESHOLD = 0.55

# ------------------------------------------------------------------
# Fixed maps only for real robot
# ------------------------------------------------------------------
# Keep real-robot fixed maps under ROBOT_* names so they do not overwrite
# the simulation ENV_FIXED_MAPS above. Add robot-specific maps here only when
# a real robot runner passes ROBOT_ENV_FIXED_MAPS explicitly.
ROBOT_ENV_FIXED_MAPS = []
ROBOT_ENV_USE_FIXED_MAPS = True
ROBOT_ENV_MAP_SELECTION_MODE = "fixed_index"
ROBOT_ENV_FIXED_MAP_INDEX = 0
ROBOT_ENV_USE_RANDOM_HEADING_ON_RESET = False
ROBOT_ENV_MAX_STEPS = 50

# Keep the encoder input names compatible with robot_interface.py.
# Adding sound_signal changes encoder.output_dim and therefore the input-hidden
# crossbar row count. Do not reuse old saved weights from the 4-feature version.
ENCODER_FEATURE_NAMES = [
    "front_clearance",
    "left_clearance",
    "right_clearance",
    "victim_signal",
    "sound_signal",
]
ENCODER_VALUE_RANGES = {
    "front_clearance": (0.0, 1.0),
    "left_clearance": (0.0, 1.0),
    "right_clearance": (0.0, 1.0),
    "victim_signal": (0.0, 1.0),
    "sound_signal": (0.0, 1.0),
    "front_delta": (-1.0, 1.0),
    "left_delta": (-1.0, 1.0),
    "right_delta": (-1.0, 1.0),
    "victim_signal_delta": (-1.0, 1.0),
    "sound_signal_delta": (-1.0, 1.0),
    "last_action_forward": (0.0, 1.0),
    "last_action_left": (0.0, 1.0),
    "last_action_right": (0.0, 1.0),
    "last_collision": (0.0, 1.0),
    "last_moved": (0.0, 1.0),
}
ENABLE_TEMPORAL_INPUT_FEATURES = False
TEMPORAL_INPUT_ENABLED_PHASE_KEYWORDS = ()

# Stable wall-avoid-success STM setting selected from the STM trace sweep.
# Keep this as the regression baseline unless a new sweep result is explicitly
# chosen and compared against primitive regression.
ENABLE_STM_RECURRENT_ABLATION = True
STM_RECURRENT_CURRENT_SCALE = 0.5
STM_RECURRENT_SCALE_SWEEP_VALUES = (0.0, 0.25, 0.5, 1.0, 2.0)
STM_CONDUCTANCE_SCALE = 4.0

# Cross-decision recurrent memory.  The hidden trace is a volatile analog
# feedback signal that survives across environment decisions inside an episode.
ENABLE_CROSS_DECISION_HIDDEN_TRACE = True
CROSS_DECISION_HIDDEN_TRACE_ENABLED_PHASE_KEYWORDS = (
    "stage_wall_avoid",
    "stage_two_step_forward_easy",
    "stage_two_step_forward_medium",
    "stage_turn_then_two_forward",
    "stage_zigzag_single_turn",
    "stage_zigzag_corridor",
    "stage_simple_deadend_escape",
    "stage_two_victim_linear",
    "stage_two_victim_with_one_turn",
    "stage_two_victim_small",
    "stage_mini_rescue_easy",
    "stage_mini_rescue",
    "stage_rescue_map_6x9_train",
    "final_eval",
)
HIDDEN_TRACE_DECAY = 0.5
HIDDEN_TRACE_INPUT_SCALE = 2.0
HIDDEN_TRACE_CLIP_MAX = 1.0
HIDDEN_TRACE_RESET_EACH_EPISODE = True

# Exhaustive STM/trace sweep candidates for diagnostic mode.
STM_TRACE_SWEEP_HIDDEN_TRACE_DECAYS = (0.5, 0.7, 0.85, 0.95)
STM_TRACE_SWEEP_HIDDEN_TRACE_INPUT_SCALES = (0.5, 1.0, 2.0, 4.0)
STM_TRACE_SWEEP_RECURRENT_CURRENT_SCALES = (0.5, 1.0, 2.0, 4.0)
STM_TRACE_SWEEP_CONDUCTANCE_SCALES = (1.0, 2.0, 4.0)

# Small final-map sweep candidates.  This mode retrains normally and keeps the
# same crossbar/SNN/R-STDP path; it only changes STM/trace circuit parameters.
FINAL_MAP_STM_SWEEP_HIDDEN_TRACE_DECAYS = (0.5, 0.7)
FINAL_MAP_STM_SWEEP_HIDDEN_TRACE_INPUT_SCALES = (1.0, 2.0, 3.0)
FINAL_MAP_STM_SWEEP_RECURRENT_CURRENT_SCALES = (0.25, 0.5, 0.75)
FINAL_MAP_STM_SWEEP_CONDUCTANCE_SCALES = (2.0, 4.0)
FINAL_MAP_STM_SWEEP_MAX_CONFIGS = 0
