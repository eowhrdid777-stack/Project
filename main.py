
from __future__ import annotations

from collections import deque
import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np

import config as cfg
from encoding import SensorSpikeEncoder
from env import AbstractRescueGridEnv
from learning import RewardModulatedSTDPLearner, RSTDPConfig
from metrics import SNNMetrics
from network import MemristiveSNNNetwork
from sensor_features import (
    active_encoder_feature_names,
    active_encoder_value_ranges,
    real_robot_sensor_feasibility_report,
)


def _cfg(name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


def _log_dir() -> Path:
    path = Path(str(_cfg("LOG_DIR", "logs")))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_log_name(name: str) -> str:
    text = str(name).strip() or "unnamed"
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json_log(filename: str, data: Any) -> Path:
    path = _log_dir() / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


DELTA_OBSERVATION_FEATURES = (
    "front_delta",
    "left_delta",
    "right_delta",
    "victim_signal_delta",
    "sound_signal_delta",
)
TRACE_OBSERVATION_FEATURES = (
    "front_trace",
    "left_trace",
    "right_trace",
    "victim_signal_trace",
    "sound_signal_trace",
)
LAST_ACTION_OBSERVATION_FEATURES = (
    "last_action_forward",
    "last_action_left",
    "last_action_right",
)
LAST_OUTCOME_OBSERVATION_FEATURES = (
    "last_collision",
    "last_moved",
)
TEMPORAL_OBSERVATION_FEATURES = (
    *DELTA_OBSERVATION_FEATURES,
    *TRACE_OBSERVATION_FEATURES,
    *LAST_ACTION_OBSERVATION_FEATURES,
    *LAST_OUTCOME_OBSERVATION_FEATURES,
)


def _temporal_input_enabled_for_phase(phase_name: str) -> bool:
    if not bool(_cfg("ENABLE_TEMPORAL_INPUT_FEATURES", False)):
        return False
    keywords = _cfg(
        "TEMPORAL_INPUT_ENABLED_PHASE_KEYWORDS",
        ("stage_wall_avoid", "final_eval"),
    )
    if isinstance(keywords, str):
        keywords = (keywords,)
    phase = str(phase_name).lower()
    return any(str(keyword).lower() in phase for keyword in keywords)


def _cross_decision_hidden_trace_enabled_for_phase(phase_name: str) -> bool:
    if "primitive_core" in str(phase_name).lower() and not bool(
        _cfg("PRIMITIVE_CORE_ENABLE_HIDDEN_TRACE", False)
    ):
        return False
    if not bool(_cfg("ENABLE_CROSS_DECISION_HIDDEN_TRACE", True)):
        return False
    keywords = _cfg(
        "CROSS_DECISION_HIDDEN_TRACE_ENABLED_PHASE_KEYWORDS",
        (),
    )
    if isinstance(keywords, str):
        keywords = (keywords,)
    keywords = tuple(str(keyword).strip().lower() for keyword in keywords if str(keyword).strip())
    if not keywords:
        return True
    phase = str(phase_name).lower()
    return any(keyword in phase for keyword in keywords)


def _observation_for_phase(obs: Any, phase_name: str) -> Any:
    if not isinstance(obs, dict):
        return obs
    out = dict(obs)
    if not _temporal_input_enabled_for_phase(phase_name):
        out["_silenced_features"] = tuple(TEMPORAL_OBSERVATION_FEATURES)
    return out


def _maybe_save_json(filename: str, data: Any, enabled_config_name: str) -> Optional[Path]:
    if not bool(_cfg(enabled_config_name, True)):
        return None
    return _write_json_log(filename, data)


BASE_ENCODER_FEATURE_NAMES = (
    "front_clearance",
    "left_clearance",
    "right_clearance",
    "victim_signal",
    "sound_signal",
)


def _encoder_feature_diagnostics(encoder: SensorSpikeEncoder) -> Dict[str, Any]:
    active = [str(name) for name in getattr(encoder, "feature_names", [])]
    active_set = set(active)
    disabled_temporal = [
        str(name) for name in TEMPORAL_OBSERVATION_FEATURES if str(name) not in active_set
    ]
    return {
        "active_encoder_features": active,
        "disabled_temporal_features": disabled_temporal,
        "using_delta_features": any(name in active_set for name in DELTA_OBSERVATION_FEATURES),
        "using_trace_features": any(name in active_set for name in TRACE_OBSERVATION_FEATURES),
        "using_last_action_features": any(
            name in active_set for name in LAST_ACTION_OBSERVATION_FEATURES
        ),
        "using_last_collision_or_moved_features": any(
            name in active_set for name in LAST_OUTCOME_OBSERVATION_FEATURES
        ),
    }


def _write_active_encoder_feature_reports(encoder: SensorSpikeEncoder) -> Dict[str, Any]:
    features = [str(name) for name in getattr(encoder, "feature_names", [])]
    observed: Dict[str, list[float]] = {name: [] for name in features}
    for stage in _cfg("CURRICULUM_STAGES", []):
        if not isinstance(stage, dict) or "map_name" not in stage:
            continue
        try:
            env = build_env_for_map(str(stage["map_name"]), seed=int(_cfg("SEED", 42)) + 91000)
            obs = env.reset(map_index=0)
        except Exception:
            continue
        if not isinstance(obs, dict):
            continue
        for name in features:
            if name in obs:
                observed[name].append(_safe_float(obs.get(name), 0.0))

    observed_min_max = {
        name: {
            "min": float(min(values)) if values else None,
            "max": float(max(values)) if values else None,
        }
        for name, values in observed.items()
    }
    summary = {
        "active_encoder_features": features,
        "feature_count": int(len(features)),
        "input_neuron_count": int(getattr(encoder, "output_dim", 0)),
        "per_feature_population_size": int(
            getattr(encoder, "neurons_per_feature", 0)
        ),
        "value_ranges": dict(getattr(encoder, "value_ranges", {})),
        "observed_value_min_max": observed_min_max,
    }
    _write_json_log("active_encoder_feature_summary.json", summary)
    _write_json_log(
        "real_robot_sensor_feasibility_report.json",
        real_robot_sensor_feasibility_report(),
    )
    return summary


# ============================================================
# Experiment settings
# ============================================================
N_EPISODES_BASELINE = int(_cfg("N_EPISODES_BASELINE", _cfg("EXPERIMENT_N_EPISODES_BASELINE", 3)))
N_EPISODES_TRAIN = int(_cfg("N_EPISODES_TRAIN", _cfg("EXPERIMENT_N_EPISODES_TRAIN", 5)))
N_EPISODES_EVAL = int(_cfg("N_EPISODES_EVAL", _cfg("EXPERIMENT_N_EPISODES_EVAL", 3)))


# ============================================================
# Builders
# ============================================================
def build_encoder() -> SensorSpikeEncoder:
    """Build the population-latency encoder used by the network."""
    feature_names = active_encoder_feature_names()
    configured_ranges = dict(_cfg("ENCODER_VALUE_RANGES", {}))
    value_ranges = active_encoder_value_ranges()
    value_ranges.update(
        {
            str(name): tuple(value)
            for name, value in configured_ranges.items()
            if str(name) in feature_names
        }
    )

    return SensorSpikeEncoder(
        feature_names=feature_names,
        value_ranges=value_ranges,
        mode=_cfg("ENCODER_MODE", "population_latency"),
        neurons_per_feature=_cfg("ENCODER_NEURONS_PER_FEATURE", None),
        latency_steps=_cfg("ENCODER_LATENCY_STEPS", None),
        activation_threshold=_cfg("ENCODER_ACTIVATION_THRESHOLD", None),
        sigma_scale=_cfg("ENCODER_SIGMA_SCALE", None),
    )


def build_env(seed: Optional[int] = None) -> AbstractRescueGridEnv:
    """Build simulation environment.

    The reviewed env.py can read fixed-map settings directly from config.py.
    They are also passed here explicitly so main.py remains clear about which
    experiment settings control the map sequence.
    """
    return AbstractRescueGridEnv(
        width=int(_cfg("ENV_WIDTH", 8)),
        height=int(_cfg("ENV_HEIGHT", 8)),
        max_steps=int(_cfg("ENV_MAX_STEPS", 50)),
        obstacle_density=float(_cfg("ENV_OBSTACLE_DENSITY", 0.12)),
        seed=int(_cfg("SEED", 42) if seed is None else seed),
        n_victims=int(_cfg("ENV_N_VICTIMS", 3)),
        victim_signal_sigma=float(_cfg("ENV_VICTIM_SIGNAL_SIGMA", 2.2)),
        victim_detection_radius=float(_cfg("ENV_VICTIM_DETECTION_RADIUS", 0.0)),
        reward_step_penalty=float(_cfg("ENV_REWARD_STEP_PENALTY", 0.0)),
        reward_collision=float(_cfg("ENV_REWARD_COLLISION", -0.05)),
        reward_closer=float(_cfg("ENV_REWARD_CLOSER", 0.10)),
        reward_farther=float(_cfg("ENV_REWARD_FARTHER", -0.03)),
        reward_found_victim=float(_cfg("ENV_REWARD_FOUND_VICTIM", 3.0)),
        reward_stay=float(_cfg("ENV_REWARD_STAY", -0.02)),
        reward_turn=float(_cfg("ENV_REWARD_TURN", -0.002)),
        use_random_heading_on_reset=bool(_cfg("ENV_USE_RANDOM_HEADING_ON_RESET", True)),
        ensure_reachable=bool(_cfg("ENV_ENSURE_REACHABLE", True)),
        generation_max_attempts=int(_cfg("ENV_GENERATION_MAX_ATTEMPTS", 200)),
        use_fixed_maps=bool(_cfg("ENV_USE_FIXED_MAPS", _cfg("ENV_USE_FIXED_MAP", False))),
        fixed_maps=_cfg("ENV_FIXED_MAPS", None),
        fixed_agent_heading=int(_cfg("ENV_FIXED_AGENT_HEADING", 0)),
        map_selection_mode=str(_cfg("ENV_MAP_SELECTION_MODE", "cycle")),
        fixed_map_index=int(_cfg("ENV_FIXED_MAP_INDEX", 0)),
    )


def _fixed_map_by_name(map_name: str) -> Dict[str, Any]:
    registry = dict(_cfg("ALL_FIXED_MAPS_BY_NAME", {}))
    if not registry:
        for spec in _cfg("ENV_FIXED_MAPS", []):
            if isinstance(spec, dict) and "name" in spec:
                registry[str(spec["name"])] = spec

    if map_name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise KeyError(f"Unknown fixed map {map_name!r}. Available maps: {available}")

    return dict(registry[map_name])


def build_env_for_map(map_name: str, seed: Optional[int] = None) -> AbstractRescueGridEnv:
    """Build a fixed-map environment for one named curriculum/eval map."""
    return AbstractRescueGridEnv(
        width=int(_cfg("ENV_WIDTH", 8)),
        height=int(_cfg("ENV_HEIGHT", 8)),
        max_steps=int(_cfg("ENV_MAX_STEPS", 50)),
        obstacle_density=float(_cfg("ENV_OBSTACLE_DENSITY", 0.12)),
        seed=int(_cfg("SEED", 42) if seed is None else seed),
        n_victims=int(_cfg("ENV_N_VICTIMS", 3)),
        victim_signal_sigma=float(_cfg("ENV_VICTIM_SIGNAL_SIGMA", 2.2)),
        victim_detection_radius=float(_cfg("ENV_VICTIM_DETECTION_RADIUS", 0.0)),
        reward_step_penalty=float(_cfg("ENV_REWARD_STEP_PENALTY", 0.0)),
        reward_collision=float(_cfg("ENV_REWARD_COLLISION", -0.05)),
        reward_closer=float(_cfg("ENV_REWARD_CLOSER", 0.10)),
        reward_farther=float(_cfg("ENV_REWARD_FARTHER", -0.03)),
        reward_found_victim=float(_cfg("ENV_REWARD_FOUND_VICTIM", 3.0)),
        reward_stay=float(_cfg("ENV_REWARD_STAY", -0.02)),
        reward_turn=float(_cfg("ENV_REWARD_TURN", -0.002)),
        use_random_heading_on_reset=bool(_cfg("ENV_USE_RANDOM_HEADING_ON_RESET", True)),
        ensure_reachable=bool(_cfg("ENV_ENSURE_REACHABLE", True)),
        generation_max_attempts=int(_cfg("ENV_GENERATION_MAX_ATTEMPTS", 200)),
        use_fixed_maps=True,
        fixed_maps=[_fixed_map_by_name(str(map_name))],
        fixed_agent_heading=int(_cfg("ENV_FIXED_AGENT_HEADING", 0)),
        map_selection_mode="fixed_index",
        fixed_map_index=0,
    )


def build_network(
    encoder: SensorSpikeEncoder,
    seed: Optional[int] = None,
) -> MemristiveSNNNetwork:
    """Build the three-crossbar recurrent SNN.

    Crossbar backends are selected inside network.py through config values:
        NETWORK_INPUT_HIDDEN_CROSSBAR_TYPE
        NETWORK_RECURRENT_HIDDEN_CROSSBAR_TYPE
        NETWORK_OUTPUT_CROSSBAR_TYPE
    """
    return MemristiveSNNNetwork(
        encoder=encoder,
        n_actions=int(_cfg("NETWORK_N_ACTIONS", 3)),
        hidden_dim=int(_cfg("NETWORK_HIDDEN_DIM", 8)),
        seed=int(_cfg("SEED", 42) if seed is None else seed),
    )


def build_learner() -> RewardModulatedSTDPLearner:
    """Build external R-STDP learner.

    network.py should only decide actions and store spike timing records.
    R-STDP update is performed here through learning.py.
    """
    return RewardModulatedSTDPLearner(
        RSTDPConfig(
            tau_plus=float(_cfg("RSTDP_TAU_PLUS", 2.0)),
            tau_minus=float(_cfg("RSTDP_TAU_MINUS", 2.0)),
            a_plus=float(_cfg("RSTDP_A_PLUS", 1.0)),
            a_minus=float(_cfg("RSTDP_A_MINUS", 0.8)),
            eligibility_threshold=float(_cfg("RSTDP_ELIGIBILITY_THRESHOLD", 1e-6)),
            use_surrogate_post_on_fallback=bool(
                _cfg("RSTDP_USE_SURROGATE_POST_ON_FALLBACK", False)
            ),
            use_surrogate_post_on_target=bool(
                _cfg("RSTDP_USE_SURROGATE_POST_ON_TARGET", True)
            ),
            use_abs_eligibility_on_target_negative=bool(
                _cfg("RSTDP_USE_ABS_ELIGIBILITY_ON_TARGET_NEGATIVE", False)
            ),
            enable_hidden_rstdp=bool(_cfg("RSTDP_ENABLE_HIDDEN", False)),
            hidden_update_input_path=bool(_cfg("RSTDP_HIDDEN_UPDATE_INPUT_PATH", True)),
            hidden_update_recurrent_path=bool(
                _cfg("RSTDP_HIDDEN_UPDATE_RECURRENT_PATH", False)
            ),
            delta_w_scale=float(_cfg("RSTDP_DELTA_W_SCALE", 1.0)),
            pulse_base=int(_cfg("RSTDP_PULSE_BASE", 1)),
            pulse_max=int(_cfg("RSTDP_PULSE_MAX", 4)),
            delta_w_per_pulse=float(_cfg("RSTDP_DELTA_W_PER_PULSE", 0.05)),
            delta_w_min_abs=float(_cfg("RSTDP_DELTA_W_MIN_ABS", 0.0)),
            output_depression_scale=float(_cfg("OUTPUT_DEPRESSION_SCALE", 1.0)),
            anti_target_depression=bool(_cfg("ANTI_TARGET_DEPRESSION", False)),
            anti_target_depression_scale=float(
                _cfg("ANTI_TARGET_DEPRESSION_SCALE", 0.25)
            ),
        )
    )


# ============================================================
# Training utilities
# ============================================================
def reward_to_target(reward: float, chosen_action: int) -> Optional[int]:
    """Minimal target policy used by output R-STDP.

    Positive reward reinforces the chosen action. Negative reward is handled by
    the signed reward gate in learning.py without inventing a separate target.
    """
    if float(reward) > 0.0:
        return int(chosen_action)
    return None


def compute_learning_reward(
    info: Dict[str, Any],
    previous_distance_to_victim: float,
) -> float:
    """Return the compact reward signal used only for R-STDP learning."""
    return float(
        compute_learning_signal(
            info=info,
            previous_distance_to_victim=previous_distance_to_victim,
        )["reward"]
    )


def compute_learning_signal(
    info: Dict[str, Any],
    previous_distance_to_victim: float,
    prev_obs: Optional[Dict[str, Any]] = None,
    next_obs: Optional[Dict[str, Any]] = None,
    executed_action: Optional[int] = None,
) -> Dict[str, Any]:
    """Return the global modulatory signal used by R-STDP.

    This does not choose or overwrite actions; it only tags the reward signal
    that gates the existing pulse-based learning path.
    """
    del previous_distance_to_victim

    action = int(
        info.get(
            "executed_action",
            -1 if executed_action is None else int(executed_action),
        )
    )
    signal = {
        "reward": 0.0,
        "target": None,
        "reason": "none",
        "turn_sensor_reward_applied": False,
        "turn_sensor_penalty_applied": False,
        "wall_avoid_forward_collision": False,
        "wall_avoid_forward_after_clearance": False,
        "wall_avoid_forward_positive_applied": False,
        "wall_avoid_forward_positive_skipped": False,
        "wall_avoid_turn_reward_applied": False,
        "wall_avoid_repeated_turn_spin": False,
    }
    enable_turn_positive_reward = bool(
        info.get(
            "enable_turn_sensor_positive_reward",
            _cfg("ENABLE_TURN_SENSOR_CHANGE_REWARD", True),
        )
    )
    is_wall_avoid_phase = bool(info.get("is_wall_avoid_phase", False))

    if bool(info.get("found_victim", False)):
        signal.update({"reward": 1.0, "target": action, "reason": "found_victim"})
        return signal

    if is_wall_avoid_phase and action == 0 and bool(info.get("collision", False)):
        signal.update(
            {
                "reward": -float(_cfg("WALL_AVOID_FORWARD_COLLISION_PENALTY", 0.4)),
                "target": action,
                "reason": "wall_avoid_forward_collision",
                "wall_avoid_forward_collision": True,
            }
        )
        return signal

    if (
        bool(info.get("is_final_map_train_phase", False))
        and action == 0
        and bool(info.get("collision", False))
    ):
        penalty = 0.5 * float(_cfg("FINAL_MAP_FORWARD_COLLISION_PENALTY_SCALE", 1.2))
        if int(info.get("consecutive_forward_collision_count", 0)) >= 2:
            penalty += float(_cfg("FINAL_MAP_REPEATED_FORWARD_COLLISION_PENALTY", 0.05))
        signal.update(
            {
                "reward": -float(penalty),
                "target": action,
                "reason": "final_map_forward_collision",
                "wall_avoid_forward_collision": False,
            }
        )
        return signal

    if bool(info.get("collision", False)):
        signal.update({"reward": -0.5, "target": action, "reason": "collision"})
        return signal

    if (
        action == 0
        and bool(_cfg("ENABLE_FORWARD_SENSOR_CHANGE_REWARD", True))
        and isinstance(prev_obs, dict)
        and isinstance(next_obs, dict)
    ):
        raw_reward = _safe_float(info.get("raw_reward", 0.0), 0.0)
        prev_front = _safe_float(prev_obs.get("front_clearance", 1.0), 1.0)
        next_front = _safe_float(next_obs.get("front_clearance", prev_front), prev_front)
        prev_left = _safe_float(prev_obs.get("left_clearance", 0.0), 0.0)
        prev_right = _safe_float(prev_obs.get("right_clearance", 0.0), 0.0)
        prev_victim_signal = _safe_float(prev_obs.get("victim_signal", 0.0), 0.0)
        next_victim_signal = _safe_float(
            next_obs.get("victim_signal", prev_victim_signal),
            prev_victim_signal,
        )
        prev_sound_signal = _safe_float(prev_obs.get("sound_signal", prev_victim_signal), prev_victim_signal)
        next_sound_signal = _safe_float(next_obs.get("sound_signal", next_victim_signal), next_victim_signal)
        signal_drop = float(_cfg("FORWARD_REWARD_VICTIM_SIGNAL_DROP", 0.10))
        positive_signal_drop = (
            float(_cfg("WALL_AVOID_FORWARD_SIGNAL_DROP_TOLERANCE", signal_drop))
            if is_wall_avoid_phase
            else signal_drop
        )
        front_worsening = float(_cfg("FORWARD_REWARD_FRONT_WORSENING", 0.15))
        final_map_positive_disabled = bool(
            info.get("is_final_map_train_phase", False)
        ) and not bool(_cfg("FINAL_MAP_FORWARD_SENSOR_POSITIVE_ENABLED", True))

        if (
            is_wall_avoid_phase
            and bool(info.get("moved", False))
            and not bool(info.get("collision", False))
            and prev_front >= float(_cfg("WALL_AVOID_FORWARD_MIN_FRONT_CLEARANCE", 0.45))
        ):
            if int(info.get("wall_avoid_forward_recovery_count", 0)) < int(
                _cfg("WALL_AVOID_FORWARD_RECOVERY_MAX_PER_EPISODE", 4)
            ):
                signal.update(
                    {
                        "reward": float(_cfg("WALL_AVOID_FORWARD_RECOVERY_REWARD", 0.12)),
                        "target": action,
                        "reason": "wall_avoid_forward_after_clearance",
                        "wall_avoid_forward_after_clearance": True,
                    }
                )
                return signal

        def wall_avoid_forward_positive_allowed() -> bool:
            if not is_wall_avoid_phase:
                return True
            if not bool(_cfg("WALL_AVOID_FORWARD_POSITIVE_ENABLED", True)):
                return False
            if int(info.get("wall_avoid_forward_positive_count", 0)) >= int(
                _cfg("WALL_AVOID_FORWARD_REWARD_MAX_PER_EPISODE", 2)
            ):
                return False
            if bool(info.get("previous_step_collision", False)):
                return False
            if (
                bool(_cfg("WALL_AVOID_FORWARD_REQUIRE_MOVED", True))
                and not bool(info.get("moved", False))
            ):
                return False
            if (
                bool(_cfg("WALL_AVOID_FORWARD_REQUIRE_NO_COLLISION", True))
                and bool(info.get("collision", False))
            ):
                return False
            if prev_front < float(_cfg("WALL_AVOID_FORWARD_MIN_FRONT_CLEARANCE", 0.45)):
                return False
            if (
                next_victim_signal < prev_victim_signal - positive_signal_drop
                or next_sound_signal < prev_sound_signal - positive_signal_drop
            ):
                return False
            return True

        if raw_reward > 0.0:
            if final_map_positive_disabled:
                return signal
            if wall_avoid_forward_positive_allowed():
                signal.update(
                    {
                        "reward": float(_cfg("FORWARD_REWARD_POSITIVE", 0.30)),
                        "target": action,
                        "reason": "forward_improved_global_reward",
                        "wall_avoid_forward_positive_applied": is_wall_avoid_phase,
                    }
                )
                return signal
            signal["wall_avoid_forward_positive_skipped"] = bool(is_wall_avoid_phase)
            return signal

        clear_path_threshold = float(_cfg("FORWARD_REWARD_CLEAR_PATH_THRESHOLD", 0.35))
        clear_path_margin = float(_cfg("FORWARD_REWARD_CLEAR_PATH_MARGIN", 0.15))
        if (
            prev_front > clear_path_threshold
            and prev_front >= prev_left + clear_path_margin
            and prev_front >= prev_right + clear_path_margin
        ):
            if final_map_positive_disabled:
                return signal
            if wall_avoid_forward_positive_allowed():
                signal.update(
                    {
                        "reward": float(_cfg("FORWARD_REWARD_CLEAR_PATH_POSITIVE", 0.20)),
                        "target": action,
                        "reason": "forward_clear_path_sensor",
                        "wall_avoid_forward_positive_applied": is_wall_avoid_phase,
                    }
                )
                return signal
            signal["wall_avoid_forward_positive_skipped"] = bool(is_wall_avoid_phase)
            return signal

        victim_drop_bad_forward = bool(next_victim_signal < prev_victim_signal - signal_drop)
        if is_wall_avoid_phase:
            victim_drop_bad_forward = False

        if (
            is_wall_avoid_phase
            and bool(info.get("moved", False))
            and not bool(info.get("collision", False))
        ):
            return signal

        if raw_reward <= 0.0 and (
            victim_drop_bad_forward or next_front < prev_front - front_worsening
        ):
            signal.update(
                {
                    "reward": -float(_cfg("FORWARD_REWARD_BAD_FORWARD_PENALTY", 0.25)),
                    "target": action,
                    "reason": "forward_worsened_sensor_signal",
                }
            )
            return signal

    if (
        action in (1, 2)
        and bool(_cfg("ENABLE_TURN_SENSOR_CHANGE_REWARD", True))
        and isinstance(prev_obs, dict)
        and isinstance(next_obs, dict)
    ):
        prev_front = _safe_float(prev_obs.get("front_clearance", 1.0), 1.0)
        next_front = _safe_float(next_obs.get("front_clearance", prev_front), prev_front)
        prev_left = _safe_float(prev_obs.get("left_clearance", 0.0), 0.0)
        prev_right = _safe_float(prev_obs.get("right_clearance", 0.0), 0.0)
        chosen_side = prev_left if action == 1 else prev_right
        other_side = prev_right if action == 1 else prev_left
        blocked_threshold = float(_cfg("TURN_REWARD_FRONT_BLOCKED_THRESHOLD", 0.35))
        improvement_threshold = float(_cfg("TURN_REWARD_CLEARANCE_IMPROVEMENT", 0.15))
        worsening_threshold = float(_cfg("TURN_REWARD_CLEARANCE_WORSENING", improvement_threshold))
        clear_threshold = float(_cfg("TURN_PENALTY_FRONT_CLEAR_THRESHOLD", 0.65))
        directional_improved = bool(next_front > prev_front + improvement_threshold)
        chosen_side_advantage = bool(
            chosen_side > prev_front + improvement_threshold
            and chosen_side >= other_side
        )
        open_side_margin = float(_cfg("WALL_AVOID_TURN_OPEN_SIDE_MARGIN", 0.20))
        wall_avoid_open_side_advantage = bool(
            is_wall_avoid_phase
            and chosen_side > prev_front + improvement_threshold
            and chosen_side >= other_side + open_side_margin
        )
        directional_worsened = bool(next_front < prev_front - worsening_threshold)
        chosen_side_disadvantage = bool(chosen_side + worsening_threshold < other_side)

        if is_wall_avoid_phase and int(info.get("consecutive_same_turn_count", 0)) >= int(
            _cfg("WALL_AVOID_SPIN_TURN_THRESHOLD", 3)
        ):
            if int(info.get("wall_avoid_spin_penalty_count", 0)) < int(
                _cfg("WALL_AVOID_SPIN_PENALTY_MAX_PER_EPISODE", 5)
            ):
                signal.update(
                    {
                        "reward": -float(_cfg("WALL_AVOID_SPIN_TURN_PENALTY", 0.06)),
                        "target": action,
                        "reason": "wall_avoid_repeated_turn_spin",
                        "turn_sensor_penalty_applied": True,
                        "wall_avoid_repeated_turn_spin": True,
                    }
                )
            return signal

        if (
            enable_turn_positive_reward
            and directional_improved
            and (prev_front < blocked_threshold or chosen_side_advantage)
        ):
            if is_wall_avoid_phase:
                if int(info.get("wall_avoid_turn_reward_action_count", 0)) >= int(
                    _cfg("WALL_AVOID_TURN_REWARD_MAX_PER_ACTION_PER_EPISODE", 2)
                ):
                    return signal
                if int(info.get("wall_avoid_turn_reward_total_count", 0)) >= int(
                    _cfg("WALL_AVOID_TURN_REWARD_MAX_PER_EPISODE", 4)
                ):
                    return signal
            if is_wall_avoid_phase:
                turn_reward = float(
                    _cfg(
                        "WALL_AVOID_TURN_REWARD_OPEN_SIDE_POSITIVE",
                        _cfg("WALL_AVOID_TURN_REWARD_POSITIVE", 0.12),
                    )
                    if wall_avoid_open_side_advantage
                    else _cfg("WALL_AVOID_TURN_REWARD_TIE_POSITIVE", 0.06)
                )
            else:
                turn_reward = float(_cfg("TURN_REWARD_POSITIVE", 0.15))
            signal.update(
                {
                    "reward": turn_reward,
                    "target": action,
                    "reason": "turn_improved_front_clearance",
                    "turn_sensor_reward_applied": True,
                    "wall_avoid_turn_reward_applied": is_wall_avoid_phase,
                }
            )
            return signal

        if directional_worsened or chosen_side_disadvantage:
            signal.update(
                {
                    "reward": -float(
                        _cfg(
                            "WALL_AVOID_TURN_WORSENED_FRONT_PENALTY",
                            _cfg("WALL_AVOID_TURN_PENALTY_NEGATIVE", 0.45),
                        )
                        if is_wall_avoid_phase
                        else _cfg("TURN_REWARD_WORSENING_TURN_PENALTY", 0.15)
                    ),
                    "target": action,
                    "reason": "turn_worsened_front_clearance",
                    "turn_sensor_penalty_applied": True,
                }
            )
            return signal

        if prev_front > clear_threshold:
            signal.update(
                {
                    "reward": -float(
                        _cfg("TURN_REWARD_UNNECESSARY_TURN_PENALTY", 0.03)
                    ),
                    "target": action,
                    "reason": "unnecessary_turn_when_front_clear",
                    "turn_sensor_penalty_applied": True,
                }
            )
            return signal

    return signal


def _is_train_phase(phase_name: str) -> bool:
    name = str(phase_name).lower()
    return name == "train" or name.endswith("_train")


def _is_eval_phase(phase_name: str) -> bool:
    name = str(phase_name).lower()
    return name == "eval" or name.endswith("_eval")


def _turn_sensor_positive_reward_enabled(phase_name: str) -> bool:
    if not bool(_cfg("ENABLE_TURN_SENSOR_CHANGE_REWARD", True)):
        return False

    keywords = _cfg("TURN_SENSOR_REWARD_DISABLED_PHASE_KEYWORDS", ("stage_forward",))
    if isinstance(keywords, str):
        keywords = (keywords,)
    phase = str(phase_name).lower()
    return not any(str(keyword).lower() in phase for keyword in keywords)


def _is_wall_avoid_phase(phase_name: str) -> bool:
    return "wall_avoid" in str(phase_name).lower()


def _base_stage_name_from_phase(phase_name: str) -> str:
    name = str(phase_name).lower()
    if name.startswith("sweep_"):
        name = name[len("sweep_"):]
    for suffix in ("_train", "_eval"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _stage_name_for_lookup(phase_name: str) -> str:
    phase = str(phase_name).lower()
    candidates = set()
    for config_name in (
        "CURRICULUM_STAGE_SETTINGS",
        "STAGE_RSTDP_SCALE",
        "STAGE_HIDDEN_TRACE_INPUT_SCALE",
    ):
        raw = _cfg(config_name, {})
        if isinstance(raw, dict):
            candidates.update(str(name).lower() for name in raw.keys())
    for stage in _cfg("CURRICULUM_STAGES", []):
        if isinstance(stage, dict) and "name" in stage:
            candidates.add(str(stage["name"]).lower())
    candidates.update(_primitive_regression_stage_names() if "_primitive_regression_stage_names" in globals() else ())
    for candidate in sorted(candidates, key=len, reverse=True):
        if candidate and candidate in phase:
            return candidate
    return _base_stage_name_from_phase(phase)


def _stage_settings(stage_name_or_phase: str) -> Dict[str, Any]:
    settings = _cfg("CURRICULUM_STAGE_SETTINGS", {})
    if not isinstance(settings, dict):
        return {}
    base_name = _stage_name_for_lookup(stage_name_or_phase)
    raw = settings.get(base_name, {})
    return dict(raw) if isinstance(raw, dict) else {}


def _stage_setting(stage_name_or_phase: str, key: str, default: Any) -> Any:
    return _stage_settings(stage_name_or_phase).get(key, default)


def _hidden_rstdp_enabled_for_phase(phase_name: str) -> bool:
    keywords = _cfg("RSTDP_ENABLE_HIDDEN_PHASE_KEYWORDS", ())
    if isinstance(keywords, str):
        keywords = (keywords,)
    phase = str(phase_name).lower()
    return any(str(keyword).lower() in phase for keyword in keywords)


def _train_exploration_epsilon(phase_name: Optional[str] = None) -> float:
    if phase_name is not None:
        phase = str(phase_name).lower()
        if "rehearsal" in phase:
            return float(
                np.clip(
                    _safe_float(_cfg("REHEARSAL_EXPLORATION_EPSILON", 0.15), 0.15),
                    0.0,
                    1.0,
                )
            )
        if "stage_rescue_map_6x9_train" in phase:
            return float(
                np.clip(
                    _safe_float(_cfg("FINAL_MAP_TRAIN_EXPLORATION_EPSILON", 0.25), 0.25),
                    0.0,
                    1.0,
                )
            )
        stage_epsilon = _stage_setting(str(phase_name), "exploration_epsilon", None)
        if stage_epsilon is not None:
            return float(np.clip(_safe_float(stage_epsilon, 0.0), 0.0, 1.0))

    if phase_name is not None and _is_wall_avoid_phase(str(phase_name)):
        epsilon = _safe_float(
            _cfg("WALL_AVOID_TRAIN_EXPLORATION_EPSILON", None),
            -1.0,
        )
        if epsilon >= 0.0:
            return float(np.clip(epsilon, 0.0, 1.0))

    epsilon = _safe_float(
        _cfg("TRAIN_EXPLORATION_EPSILON", _cfg("EXPLORATION_EPSILON", 0.10)),
        0.10,
    )
    return float(np.clip(epsilon, 0.0, 1.0))


def _stage_rstdp_scale(phase_name: str) -> float:
    if not bool(_cfg("ENABLE_STAGEWISE_RSTDP_SCALE", False)):
        return 1.0
    stage_name = _stage_name_for_lookup(phase_name)
    if "primitive_core" in str(phase_name).lower():
        primitive_scales = _cfg("PRIMITIVE_RSTDP_SCALE", {})
        if isinstance(primitive_scales, dict) and stage_name in primitive_scales:
            return float(
                np.clip(
                    _safe_float(primitive_scales.get(stage_name, 1.0), 1.0),
                    0.0,
                    10.0,
                )
            )
    scales = _cfg("STAGE_RSTDP_SCALE", {})
    if not isinstance(scales, dict):
        return 1.0
    return float(np.clip(_safe_float(scales.get(stage_name, 1.0), 1.0), 0.0, 10.0))


def _stage_hidden_trace_input_scale(phase_name: str) -> float:
    default = _safe_float(_cfg("HIDDEN_TRACE_INPUT_SCALE", 1.0), 1.0)
    if "primitive_core" in str(phase_name).lower():
        return float(
            np.clip(
                _safe_float(_cfg("PRIMITIVE_CORE_HIDDEN_TRACE_INPUT_SCALE", default), default),
                0.0,
                10.0,
            )
        )
    if not bool(_cfg("ENABLE_STAGEWISE_STM_TRACE_SCALE", False)):
        return float(default)
    scales = _cfg("STAGE_HIDDEN_TRACE_INPUT_SCALE", {})
    if not isinstance(scales, dict):
        return float(default)
    stage_name = _stage_name_for_lookup(phase_name)
    return float(np.clip(_safe_float(scales.get(stage_name, default), default), 0.0, 10.0))


def _is_final_map_train_phase(phase_name: str) -> bool:
    phase = str(phase_name).lower()
    return "stage_rescue_map_6x9_train" in phase and _is_train_phase(phase)


def _maybe_apply_train_exploration(
    *,
    phase_name: str,
    learner: Optional[RewardModulatedSTDPLearner],
    snn_action: int,
    rng: Any,
) -> Dict[str, Any]:
    epsilon = _train_exploration_epsilon(phase_name)
    exploration_enabled = _is_train_phase(phase_name) and learner is not None

    if not exploration_enabled or epsilon <= 0.0:
        return {
            "executed_action": int(snn_action),
            "exploration_used": False,
            "exploration_epsilon": float(epsilon),
        }

    if float(rng.random()) >= epsilon:
        return {
            "executed_action": int(snn_action),
            "exploration_used": False,
            "exploration_epsilon": float(epsilon),
        }

    candidates = [0, 1, 2]
    replacement_candidates = [a for a in candidates if a != int(snn_action)]
    if replacement_candidates:
        candidates = replacement_candidates

    executed_action = int(rng.choice(candidates))
    return {
        "executed_action": executed_action,
        "exploration_used": True,
        "exploration_epsilon": float(epsilon),
    }


def _output_learning_event_record(
    output_event: Optional[Any],
    *,
    skip_message: Optional[str] = None,
) -> Dict[str, Any]:
    if output_event is None:
        return {
            "output_winner": None,
            "learning_event_message": (
                skip_message
                if skip_message is not None
                else "no output learning event"
            ),
            "updated_pairs": [],
            "directions": [],
            "n_pulses_plus": 0,
            "n_pulses_minus": 0,
            "used_surrogate_target_post": False,
        }

    return {
        "output_winner": getattr(output_event, "winner", None),
        "learning_event_message": getattr(output_event, "message", ""),
        "updated_pairs": list(getattr(output_event, "updated_pairs", [])),
        "directions": list(getattr(output_event, "directions", [])),
        "n_pulses_plus": int(getattr(output_event, "n_pulses_plus", 0)),
        "n_pulses_minus": int(getattr(output_event, "n_pulses_minus", 0)),
        "used_surrogate_target_post": bool(
            getattr(output_event, "used_surrogate_post", False)
            and getattr(output_event, "target", None) is not None
        ),
    }


def _learn_from_recorded_decision(
    *,
    net: MemristiveSNNNetwork,
    learner: RewardModulatedSTDPLearner,
    decision: Any,
    reward: float,
    target: Optional[int],
) -> Dict[str, Optional[Any]]:
    original_decision = net.last_decision
    net.last_decision = decision
    try:
        return learner.learn(
            net=net,
            reward=float(reward),
            target=target,
        )
    finally:
        net.last_decision = original_decision


def _print_delayed_credit_update(record: Dict[str, Any]) -> None:
    print("DELAYED CREDIT UPDATE:")
    for key in (
        "source_event",
        "reason",
        "original_step",
        "credit_step",
        "executed_action",
        "credit_reward",
        "target",
        "used_surrogate_target_post",
        "learning_event_message",
    ):
        print(f"{key}={record.get(key)}")


def _hist_increment(hist: Dict[int, int], action: int, amount: int = 1) -> None:
    action = int(action)
    hist[action] = int(hist.get(action, 0)) + int(amount)


def _output_column_balance_status(
    net: MemristiveSNNNetwork,
    target_action: Optional[int],
    reward: float,
) -> Dict[str, Any]:
    if target_action is None or float(reward) <= 0.0:
        return {"blocked": False}
    if not bool(_cfg("ENABLE_OUTPUT_COLUMN_BALANCE", True)):
        return {"blocked": False}

    means = _output_column_weight_mean(net)
    target = int(target_action)
    if target not in means or not means:
        return {"blocked": False}

    values = [float(v) for v in means.values()]
    overall_mean = float(np.mean(values)) if values else 0.0
    if overall_mean <= 0.0:
        positive_values = [v for v in values if v > 0.0]
        if not positive_values:
            return {
                "blocked": False,
                "column_mean": float(means[target]),
                "overall_mean": float(overall_mean),
            }
        overall_mean = float(np.mean(positive_values))

    ratio = max(1.0, float(_cfg("OUTPUT_COLUMN_BALANCE_RATIO", 2.0)))
    threshold = float(overall_mean * ratio)
    column_mean = float(means[target])
    blocked = bool(column_mean > threshold)
    return {
        "blocked": blocked,
        "column_mean": column_mean,
        "overall_mean": float(overall_mean),
        "balance_ratio": ratio,
        "balance_threshold": threshold,
    }


def _fixed_map_index_for_episode(env: AbstractRescueGridEnv, ep: int) -> Optional[int]:
    """Return a deterministic fixed-map index for comparable phases.

    If fixed-map mode is enabled, each phase uses the same sequence:
    map 0, map 1, ..., map N-1, then repeat.  This makes baseline/train/eval
    comparisons easier to interpret.  If random-map mode is used, returns None.
    """
    if not bool(getattr(env, "use_fixed_maps", False)):
        return None
    if not bool(_cfg("EXPERIMENT_ALIGN_FIXED_MAP_SEQUENCE", True)):
        return None
    n_maps = int(env.num_fixed_maps()) if hasattr(env, "num_fixed_maps") else 0
    if n_maps <= 0:
        return None
    return int(ep % n_maps)


def print_startup_env_summary(env: AbstractRescueGridEnv) -> None:
    """Print the active environment map loaded at program start."""
    state = env.get_env_state() if hasattr(env, "get_env_state") else {}
    map_name = state.get("map_name") or "random_generated"
    width = int(state.get("width", getattr(env, "width", 0)))
    height = int(state.get("height", getattr(env, "height", 0)))
    victim_count = int(
        state.get(
            "initial_victim_count",
            len(getattr(env, "victim_positions", [])),
        )
    )
    start_pos = state.get("agent_pos", getattr(env, "agent_pos", None))
    heading = state.get("agent_heading_name", state.get("agent_heading", None))

    print("=" * 70)
    print("ENV STARTUP")
    print("=" * 70)
    print(f"active_map: {map_name}")
    print(f"size: width={width}, height={height}")
    print(f"victim_count: {victim_count}")
    print(f"start_pos: {start_pos}")
    print(f"heading: {heading}")
    print()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _build_action_stats(env: AbstractRescueGridEnv) -> Dict[int, Dict[str, Any]]:
    action_names = getattr(env, "ACTION_NAMES", {0: "forward", 1: "turn_left", 2: "turn_right", 3: "stay"})
    try:
        action_ids = sorted(int(action) for action in action_names.keys())
    except Exception:
        action_ids = [0, 1, 2, 3]

    return {
        action: {
            "count": 0,
            "reward_sum": 0.0,
            "found_victim": 0,
            "collision": 0,
            "moved": 0,
        }
        for action in action_ids
    }


def _update_action_stats(
    stats: Dict[int, Dict[str, Any]],
    action: int,
    reward: float,
    info: Dict[str, Any],
) -> None:
    action = int(action)
    if action not in stats:
        stats[action] = {
            "count": 0,
            "reward_sum": 0.0,
            "found_victim": 0,
            "collision": 0,
            "moved": 0,
        }

    stats[action]["count"] += 1
    stats[action]["reward_sum"] += float(reward)
    stats[action]["found_victim"] += int(bool(info.get("found_victim", False)))
    stats[action]["collision"] += int(bool(info.get("collision", False)))
    stats[action]["moved"] += int(bool(info.get("moved", False)))


def _finalize_action_stats(stats: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "action_counts": {
            action: int(values["count"])
            for action, values in stats.items()
        },
        "action_mean_reward": {
            action: (
                float(values["reward_sum"] / values["count"])
                if values["count"] > 0
                else None
            )
            for action, values in stats.items()
        },
        "action_event_counts": {
            action: {
                "found_victim": int(values["found_victim"]),
                "collision": int(values["collision"]),
                "moved": int(values["moved"]),
            }
            for action, values in stats.items()
        },
    }


def _output_column_weight_mean(net: MemristiveSNNNetwork) -> Dict[int, float]:
    crossbar = getattr(net, "output_crossbar", None)
    if crossbar is None:
        return {}

    n_rows = int(getattr(crossbar, "n_rows", 0))
    n_cols = int(getattr(crossbar, "n_logical_cols", 0))
    if n_rows <= 0 or n_cols <= 0:
        return {}

    out: Dict[int, float] = {}
    for col in range(n_cols):
        weights = []
        for row in range(n_rows):
            if hasattr(crossbar, "read_pair_ideal"):
                gp, gm = crossbar.read_pair_ideal((row, col))
                weight = float(gp - gm)
            elif hasattr(crossbar, "read_weight_measured"):
                weight = float(crossbar.read_weight_measured((row, col)))
            else:
                continue
            weights.append(weight)
        out[col] = float(np.mean(weights)) if weights else 0.0

    return out


def _output_column_weight_std(net: MemristiveSNNNetwork) -> Dict[int, float]:
    crossbar = getattr(net, "output_crossbar", None)
    if crossbar is None:
        return {}

    n_rows = int(getattr(crossbar, "n_rows", 0))
    n_cols = int(getattr(crossbar, "n_logical_cols", 0))
    if n_rows <= 0 or n_cols <= 0:
        return {}

    out: Dict[int, float] = {}
    for col in range(n_cols):
        weights = []
        for row in range(n_rows):
            if hasattr(crossbar, "read_pair_ideal"):
                gp, gm = crossbar.read_pair_ideal((row, col))
                weight = float(gp - gm)
            elif hasattr(crossbar, "read_weight_measured"):
                weight = float(crossbar.read_weight_measured((row, col)))
            else:
                continue
            weights.append(weight)
        out[col] = float(np.std(weights)) if weights else 0.0

    return out


def _output_column_effective_score(net: MemristiveSNNNetwork) -> Dict[int, float]:
    # Diagnostic readout only: signed differential conductance seen by the output column.
    return _output_column_weight_mean(net)


def _output_column_status(net: MemristiveSNNNetwork) -> Dict[int, Dict[str, float]]:
    crossbar = getattr(net, "output_crossbar", None)
    if crossbar is None:
        return {}

    n_rows = int(getattr(crossbar, "n_rows", 0))
    n_cols = int(getattr(crossbar, "n_logical_cols", 0))
    if n_rows <= 0 or n_cols <= 0:
        return {}

    status: Dict[int, Dict[str, float]] = {}
    for col in range(n_cols):
        plus_values: list[float] = []
        minus_values: list[float] = []
        weights: list[float] = []
        common_modes: list[float] = []
        for row in range(n_rows):
            if hasattr(crossbar, "read_pair_ideal"):
                gp, gm = crossbar.read_pair_ideal((row, col))
            elif hasattr(crossbar, "read_pair"):
                gp, gm = crossbar.read_pair((row, col))
            else:
                continue
            gp_f = float(gp)
            gm_f = float(gm)
            plus_values.append(gp_f)
            minus_values.append(gm_f)
            weights.append(gp_f - gm_f)
            common_modes.append(0.5 * (gp_f + gm_f))

        if not weights:
            status[col] = {
                "conductance_mean": 0.0,
                "conductance_std": 0.0,
                "effective_signed_weight_mean": 0.0,
                "effective_signed_weight_norm": 0.0,
                "effective_signed_weight_std": 0.0,
                "common_mode_mean": 0.0,
            }
            continue

        g_all = np.asarray(plus_values + minus_values, dtype=float)
        w_arr = np.asarray(weights, dtype=float)
        common_arr = np.asarray(common_modes, dtype=float)
        status[col] = {
            "conductance_mean": float(g_all.mean()),
            "conductance_std": float(g_all.std()),
            "g_plus_mean": float(np.mean(plus_values)),
            "g_minus_mean": float(np.mean(minus_values)),
            "effective_signed_weight_mean": float(w_arr.mean()),
            "effective_signed_weight_norm": float(np.linalg.norm(w_arr)),
            "effective_signed_weight_abs_mean": float(np.mean(np.abs(w_arr))),
            "effective_signed_weight_std": float(w_arr.std()),
            "effective_signed_weight_min": float(w_arr.min()),
            "effective_signed_weight_max": float(w_arr.max()),
            "common_mode_mean": float(common_arr.mean()),
        }
    return status


def _output_column_status_delta(
    before: Dict[int, Dict[str, float]],
    after: Dict[int, Dict[str, float]],
) -> Dict[int, Dict[str, float]]:
    delta: Dict[int, Dict[str, float]] = {}
    for action in sorted(set(before.keys()) | set(after.keys())):
        before_item = before.get(action, {})
        after_item = after.get(action, {})
        keys = set(before_item.keys()) | set(after_item.keys())
        delta[action] = {
            key: float(after_item.get(key, 0.0) - before_item.get(key, 0.0))
            for key in sorted(keys)
        }
    return delta


def _output_column_interference_record(
    net: MemristiveSNNNetwork,
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[int, Dict[str, Any]]:
    column_status = _output_column_status(net)
    summary = summary if isinstance(summary, dict) else {}
    metrics = _metrics_from_summary(summary) if summary else {}
    action_hist = metrics.get("action_histogram", {})
    action_reward = summary.get("action_mean_reward", {}) if summary else {}
    action_events = summary.get("action_event_counts", {}) if summary else {}
    plus_hist = summary.get("potentiation_action_histogram", {}) if summary else {}
    minus_hist = summary.get("depression_action_histogram", {}) if summary else {}
    learning_hist = summary.get("learning_update_action_histogram", {}) if summary else {}

    out: Dict[int, Dict[str, Any]] = {}
    for action in sorted(column_status.keys()):
        selected_count = int(action_hist.get(action, action_hist.get(str(action), 0)) or 0)
        mean_reward = action_reward.get(action, action_reward.get(str(action), 0.0))
        event_item = action_events.get(action, action_events.get(str(action), {}))
        if not isinstance(event_item, dict):
            event_item = {}
        pulse_plus = int(plus_hist.get(action, plus_hist.get(str(action), 0)) or 0)
        pulse_minus = int(minus_hist.get(action, minus_hist.get(str(action), 0)) or 0)
        update_count = int(learning_hist.get(action, learning_hist.get(str(action), 0)) or 0)
        reward_sum = float(_safe_float(mean_reward, 0.0) * max(selected_count, 1))
        out[action] = {
            **column_status[action],
            "pulse_plus_count": pulse_plus,
            "pulse_minus_count": pulse_minus,
            "reward_sum": reward_sum,
            "positive_reward_sum": float(max(reward_sum, 0.0)),
            "negative_reward_sum": float(min(reward_sum, 0.0)),
            "positive_update_count": int(pulse_plus > 0) if update_count == 0 else int(update_count if pulse_plus > 0 else 0),
            "negative_update_count": int(pulse_minus > 0) if update_count == 0 else int(update_count if pulse_minus > 0 else 0),
            "selected_count": selected_count,
            "collision_count": int(event_item.get("collision", 0) or 0),
            "collision_penalty_count": int(event_item.get("collision", 0) or 0),
            "found_count": int(event_item.get("found_victim", 0) or 0),
        }
    return out


def _cosine_similarity(a: Any, b: Any) -> float:
    arr_a = np.asarray(a, dtype=float).reshape(-1)
    arr_b = np.asarray(b, dtype=float).reshape(-1)
    if arr_a.size == 0 or arr_b.size == 0 or arr_a.size != arr_b.size:
        return 0.0
    denom = float(np.linalg.norm(arr_a) * np.linalg.norm(arr_b))
    if denom <= 1e-30:
        return 1.0 if float(np.linalg.norm(arr_a - arr_b)) <= 1e-12 else 0.0
    return float(np.dot(arr_a, arr_b) / denom)


def _normalized_observation_vector(
    encoder: SensorSpikeEncoder,
    obs: Dict[str, Any],
) -> list[float]:
    values: list[float] = []
    ranges = getattr(encoder, "value_ranges", {})
    for name in getattr(encoder, "feature_names", []):
        raw = _safe_float(obs.get(str(name), 0.0), 0.0)
        lo, hi = ranges.get(str(name), (0.0, 1.0))
        lo_f = float(lo)
        hi_f = float(hi)
        if abs(hi_f - lo_f) <= 1e-12:
            values.append(0.0)
        else:
            values.append(float(np.clip((raw - lo_f) / (hi_f - lo_f), 0.0, 1.0)))
    return values


def _recurrent_crossbar_status(net: MemristiveSNNNetwork) -> Dict[str, float]:
    crossbar = getattr(net, "recurrent_hidden_crossbar", None)
    if crossbar is None:
        return {}

    n_rows = int(getattr(crossbar, "n_rows", 0))
    n_cols = int(getattr(crossbar, "n_logical_cols", 0))
    if n_rows <= 0 or n_cols <= 0:
        return {}

    weights = []
    common_modes = []
    for row in range(n_rows):
        for col in range(n_cols):
            if hasattr(crossbar, "read_pair_ideal"):
                gp, gm = crossbar.read_pair_ideal((row, col))
            elif hasattr(crossbar, "read_pair"):
                gp, gm = crossbar.read_pair((row, col))
            else:
                continue
            gp = float(gp)
            gm = float(gm)
            weights.append(gp - gm)
            common_modes.append(0.5 * (gp + gm))

    if not weights:
        return {}

    weight_arr = np.asarray(weights, dtype=float)
    common_arr = np.asarray(common_modes, dtype=float)
    status: Dict[str, float] = {
        "recurrent_weight_mean": float(weight_arr.mean()),
        "recurrent_weight_abs_mean": float(np.mean(np.abs(weight_arr))),
        "recurrent_weight_std": float(weight_arr.std()),
        "recurrent_weight_min": float(weight_arr.min()),
        "recurrent_weight_max": float(weight_arr.max()),
        "recurrent_common_mode_mean": float(common_arr.mean()),
    }

    summary = crossbar.summary() if callable(getattr(crossbar, "summary", None)) else {}
    if isinstance(summary, dict):
        for key in ("z_mean", "x_mean", "r_mean"):
            if key in summary:
                status[key] = float(summary[key])
    return status


def _records_matrix(decision: Any, attr: str) -> np.ndarray:
    rows = []
    for rec in getattr(decision, "step_records", []) or []:
        output_result = getattr(rec, "output_result", None)
        value = getattr(output_result, attr, None)
        if value is not None:
            rows.append(np.asarray(value, dtype=float).reshape(-1))
    if not rows:
        return np.zeros((0, 0), dtype=float)
    return np.vstack(rows)


def _decision_output_spike_counts(decision: Any) -> list[int]:
    mat = _records_matrix(decision, "spikes")
    if mat.size == 0:
        return []
    return [int(v) for v in mat.sum(axis=0).astype(int).tolist()]


def _decision_first_spike_steps(decision: Any) -> list[Optional[int]]:
    mat = _records_matrix(decision, "spikes")
    if mat.size == 0:
        return []
    first_steps: list[Optional[int]] = []
    for col in range(mat.shape[1]):
        hits = np.flatnonzero(mat[:, col] > 0)
        first_steps.append(None if hits.size == 0 else int(hits[0]))
    return first_steps


def _decision_output_mean(decision: Any, attr: str) -> list[float]:
    mat = _records_matrix(decision, attr)
    if mat.size == 0:
        return []
    return [float(v) for v in mat.mean(axis=0).tolist()]


def _decision_record_matrix(decision: Any, attr: str) -> np.ndarray:
    rows = []
    for rec in getattr(decision, "step_records", []) or []:
        value = getattr(rec, attr, None)
        if value is not None:
            rows.append(np.asarray(value, dtype=float).reshape(-1))
    if not rows:
        return np.zeros((0, 0), dtype=float)
    return np.vstack(rows)


def _decision_hidden_current_stats(decision: Any) -> Dict[str, float]:
    input_mat = _decision_record_matrix(decision, "input_hidden_current")
    recurrent_mat = _decision_record_matrix(decision, "recurrent_hidden_current")
    prev_spike_mat = _decision_record_matrix(decision, "prev_hidden_spikes_used")
    hidden_trace_mat = _decision_record_matrix(decision, "hidden_trace_used")
    if input_mat.size == 0:
        return {
            "input_hidden_current_abs_mean": 0.0,
            "recurrent_hidden_current_abs_mean": 0.0,
            "recurrent_to_input_current_ratio": 0.0,
            "input_hidden_current_norm": 0.0,
            "recurrent_hidden_current_norm": 0.0,
            "total_hidden_current_norm": 0.0,
            "prev_hidden_spike_count": 0.0,
            "hidden_trace_mean": 0.0,
            "hidden_trace_nonzero_count": 0.0,
        }

    if recurrent_mat.size == 0:
        recurrent_mat = np.zeros_like(input_mat)
    total_mat = input_mat + recurrent_mat
    input_abs = float(np.mean(np.abs(input_mat)))
    recurrent_abs = float(np.mean(np.abs(recurrent_mat)))
    denom = max(input_abs, 1e-30)
    return {
        "input_hidden_current_abs_mean": input_abs,
        "recurrent_hidden_current_abs_mean": recurrent_abs,
        "recurrent_to_input_current_ratio": float(recurrent_abs / denom),
        "input_hidden_current_norm": float(np.linalg.norm(input_mat)),
        "recurrent_hidden_current_norm": float(np.linalg.norm(recurrent_mat)),
        "total_hidden_current_norm": float(np.linalg.norm(total_mat)),
        "prev_hidden_spike_count": float(np.sum(prev_spike_mat)) if prev_spike_mat.size else 0.0,
        "hidden_trace_mean": float(np.mean(hidden_trace_mat)) if hidden_trace_mat.size else 0.0,
        "hidden_trace_nonzero_count": (
            float(np.mean(np.sum(hidden_trace_mat > 1e-12, axis=1)))
            if hidden_trace_mat.size
            else 0.0
        ),
    }


def _as_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    arr = np.asarray(value, dtype=float).reshape(-1)
    return [float(v) for v in arr.tolist()]


def _metrics_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    metrics = summary.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _phase_compact_line(label: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    metrics = _metrics_from_summary(summary)
    return {
        "success_rate": summary.get("success_rate", 0.0),
        "action_histogram": metrics.get("action_histogram", {}),
        "snn_action_histogram": metrics.get("snn_action_histogram", {}),
        "executed_action_histogram": metrics.get("executed_action_histogram", {}),
        "output_winner_histogram": summary.get("output_winner_histogram", {}),
        "spike_selected_action_histogram": summary.get("spike_selected_action_histogram", {}),
        "fallback_action_histogram": summary.get("fallback_action_histogram", {}),
        "fallback_count_by_action": summary.get("fallback_count_by_action", {}),
        "fallback_when_front_blocked_count": summary.get(
            "fallback_when_front_blocked_count", 0
        ),
        "fallback_when_front_blocked_action_histogram": summary.get(
            "fallback_when_front_blocked_action_histogram", {}
        ),
        "no_output_spike_count": summary.get("no_output_spike_count", 0),
        "output_threshold_mean_per_action": summary.get(
            "output_threshold_mean_per_action", {}
        ),
        "output_current_mean_per_action": summary.get(
            "output_current_mean_per_action", {}
        ),
        "mean_abs_input_hidden_current": summary.get("mean_abs_input_hidden_current", 0.0),
        "mean_abs_recurrent_hidden_current": summary.get(
            "mean_abs_recurrent_hidden_current", 0.0
        ),
        "mean_recurrent_to_input_current_ratio": summary.get(
            "mean_recurrent_to_input_current_ratio", 0.0
        ),
        "median_recurrent_to_input_current_ratio": summary.get(
            "median_recurrent_to_input_current_ratio", 0.0
        ),
        "max_recurrent_to_input_current_ratio": summary.get(
            "max_recurrent_to_input_current_ratio", 0.0
        ),
        "mean_prev_hidden_spike_count": summary.get("mean_prev_hidden_spike_count", 0.0),
        "mean_hidden_spike_count": summary.get("mean_hidden_spike_count", 0.0),
        "hidden_trace_mean": summary.get("hidden_trace_mean", 0.0),
        "hidden_trace_nonzero_count": summary.get("hidden_trace_nonzero_count", 0.0),
        "enable_cross_decision_hidden_trace": summary.get(
            "enable_cross_decision_hidden_trace", False
        ),
        "hidden_trace_decay": summary.get("hidden_trace_decay", 0.0),
        "hidden_trace_input_scale": summary.get("hidden_trace_input_scale", 0.0),
        "input_hidden_current_abs_mean_by_action": summary.get(
            "input_hidden_current_abs_mean_by_action", {}
        ),
        "recurrent_hidden_current_abs_mean_by_action": summary.get(
            "recurrent_hidden_current_abs_mean_by_action", {}
        ),
        "recurrent_to_input_current_ratio_by_action": summary.get(
            "recurrent_to_input_current_ratio_by_action", {}
        ),
        "active_encoder_features": summary.get("active_encoder_features", []),
        "disabled_temporal_features": summary.get("disabled_temporal_features", []),
        "using_delta_features": summary.get("using_delta_features", False),
        "using_trace_features": summary.get("using_trace_features", False),
        "using_last_action_features": summary.get("using_last_action_features", False),
        "using_last_collision_or_moved_features": summary.get(
            "using_last_collision_or_moved_features", False
        ),
        "reset_neuron_state_each_decision": summary.get(
            "reset_neuron_state_each_decision", False
        ),
        "prev_hidden_spikes_reset_each_decision": summary.get(
            "prev_hidden_spikes_reset_each_decision", False
        ),
        "recurrent_memory_scope": summary.get("recurrent_memory_scope", "unknown"),
        "output_spike_total_per_episode": summary.get(
            "output_spike_total_per_episode", []
        ),
        "no_output_spike_episode_count": summary.get(
            "no_output_spike_episode_count", 0
        ),
        "found_victim_count": metrics.get("found_victim_count", 0),
        "collision_count": summary.get("collision_count", metrics.get("collision_count", 0)),
        "moved_count": summary.get("moved_count", metrics.get("moved_count", 0)),
        "total_victim_count": summary.get("total_victim_count", 0),
        "collision_free_success_rate": summary.get("collision_free_success_rate", 0.0),
        "victims_found_per_collision": summary.get("victims_found_per_collision", 0.0),
        "victims_found_per_step": summary.get("victims_found_per_step", 0.0),
        "repeated_turn_count": summary.get("repeated_turn_count", 0),
        "repeated_forward_collision_count": summary.get("repeated_forward_collision_count", 0),
        "unique_cells_visited_count": summary.get("unique_cells_visited_count", 0.0),
        "mean_fallback_count": summary.get("mean_fallback_count", 0.0),
        "delayed_credit_update_count": summary.get("delayed_credit_update_count", 0),
        "delayed_credit_action_histogram": summary.get("delayed_credit_action_histogram", {}),
        "direct_credit_action_histogram": summary.get("direct_credit_action_histogram", {}),
        "previous_turn_credit_action_histogram": summary.get("previous_turn_credit_action_histogram", {}),
        "learning_update_action_histogram": summary.get("learning_update_action_histogram", {}),
        "potentiation_action_histogram": summary.get("potentiation_action_histogram", {}),
        "depression_action_histogram": summary.get("depression_action_histogram", {}),
        "turn_sensor_reward_action_histogram": summary.get("turn_sensor_reward_action_histogram", {}),
        "turn_sensor_penalty_action_histogram": summary.get("turn_sensor_penalty_action_histogram", {}),
        "learning_reason_histogram": summary.get("learning_reason_histogram", {}),
        "wall_avoid_forward_collision_count": summary.get("wall_avoid_forward_collision_count", 0),
        "wall_avoid_forward_after_clearance_count": summary.get(
            "wall_avoid_forward_after_clearance_count", 0
        ),
        "wall_avoid_repeated_turn_spin_count": summary.get(
            "wall_avoid_repeated_turn_spin_count", 0
        ),
        "wall_avoid_forward_positive_count": summary.get("wall_avoid_forward_positive_count", 0),
        "wall_avoid_forward_positive_skipped_count": summary.get("wall_avoid_forward_positive_skipped_count", 0),
        "wall_avoid_turn_reward_count": summary.get("wall_avoid_turn_reward_count", 0),
        "wall_avoid_action0_selected_when_front_blocked_count": summary.get(
            "wall_avoid_action0_selected_when_front_blocked_count", 0
        ),
        "wall_avoid_forward_recovery_action_histogram": summary.get(
            "wall_avoid_forward_recovery_action_histogram", {}
        ),
        "wall_avoid_spin_penalty_action_histogram": summary.get(
            "wall_avoid_spin_penalty_action_histogram", {}
        ),
        "selected_step_action_histogram": summary.get("selected_step_action_histogram", {}),
        "eval_first_episode_action_sequence": summary.get("eval_first_episode_action_sequence", []),
        "eval_first_episode_collision_sequence": summary.get("eval_first_episode_collision_sequence", []),
        "eval_first_episode_moved_sequence": summary.get("eval_first_episode_moved_sequence", []),
        "eval_first_episode_victim_signal_sequence": summary.get(
            "eval_first_episode_victim_signal_sequence", []
        ),
        "eval_first_episode_front_clearance_sequence": summary.get(
            "eval_first_episode_front_clearance_sequence", []
        ),
    }


def _mean_first_spike_time_by_action(metrics: SNNMetrics) -> Dict[int, float]:
    buckets: Dict[int, list[float]] = {0: [], 1: [], 2: []}
    for step in getattr(metrics, "steps", []):
        values = getattr(step, "first_spike_step", [])
        for action in (0, 1, 2):
            if action < len(values):
                value = int(values[action])
                if value >= 0:
                    buckets[action].append(float(value))
    return {
        action: (float(np.mean(values)) if values else -1.0)
        for action, values in buckets.items()
    }


def _print_compact_stage_summary(stage_record: Dict[str, Any]) -> None:
    train_summary = stage_record["stage_train"]
    eval_summary = stage_record["stage_eval"]
    train = _phase_compact_line("train", train_summary)
    eval_ = _phase_compact_line("eval", eval_summary)

    print("-" * 70)
    print(f"stage: {stage_record['stage_name']}")
    print(f"active_map: {stage_record['active_map_name']}")
    print(f"train success_rate: {train['success_rate']}")
    print(f"eval  success_rate: {eval_['success_rate']}")
    print(f"train action_histogram: {train['action_histogram']}")
    print(f"eval  action_histogram: {eval_['action_histogram']}")
    print(f"train snn_action_histogram: {train['snn_action_histogram']}")
    print(f"eval  snn_action_histogram: {eval_['snn_action_histogram']}")
    print(f"train output_winner_histogram: {train['output_winner_histogram']}")
    print(f"eval  output_winner_histogram: {eval_['output_winner_histogram']}")
    print(f"train spike_selected_action_histogram: {train['spike_selected_action_histogram']}")
    print(f"eval  spike_selected_action_histogram: {eval_['spike_selected_action_histogram']}")
    print(f"train fallback_action_histogram: {train['fallback_action_histogram']}")
    print(f"eval  fallback_action_histogram: {eval_['fallback_action_histogram']}")
    print(
        "fallback_when_front_blocked_count: "
        f"train={train['fallback_when_front_blocked_count']} "
        f"eval={eval_['fallback_when_front_blocked_count']}"
    )
    print(
        "eval fallback_when_front_blocked_action_histogram: "
        f"{eval_['fallback_when_front_blocked_action_histogram']}"
    )
    print(
        "output silence: "
        f"train_no_output_spike_count={train['no_output_spike_count']} "
        f"eval_no_output_spike_count={eval_['no_output_spike_count']} "
        f"eval_no_output_spike_episode_count={eval_['no_output_spike_episode_count']}"
    )
    print(
        "eval output_threshold_mean_per_action: "
        f"{eval_['output_threshold_mean_per_action']}"
    )
    print(
        "eval output_current_mean_per_action: "
        f"{eval_['output_current_mean_per_action']}"
    )
    print(
        "encoder features: "
        f"active={eval_['active_encoder_features']} "
        f"disabled_temporal={eval_['disabled_temporal_features']}"
    )
    print(
        "temporal feature flags: "
        f"delta={eval_['using_delta_features']} "
        f"trace={eval_['using_trace_features']} "
        f"last_action={eval_['using_last_action_features']} "
        f"last_collision_or_moved={eval_['using_last_collision_or_moved_features']}"
    )
    print(
        "recurrent memory scope: "
        f"reset_each_decision={eval_['reset_neuron_state_each_decision']} "
        f"prev_reset_each_decision={eval_['prev_hidden_spikes_reset_each_decision']} "
        f"scope={eval_['recurrent_memory_scope']} "
        f"hidden_trace_enabled={eval_['enable_cross_decision_hidden_trace']}"
    )
    print(
        "recurrent current diagnostics: "
        f"train_input_abs={train['mean_abs_input_hidden_current']} "
        f"train_rec_abs={train['mean_abs_recurrent_hidden_current']} "
        f"train_ratio_mean={train['mean_recurrent_to_input_current_ratio']} "
        f"train_ratio_median={train['median_recurrent_to_input_current_ratio']} "
        f"train_ratio_max={train['max_recurrent_to_input_current_ratio']} "
        f"train_prev_hidden_spikes={train['mean_prev_hidden_spike_count']} "
        f"train_hidden_spikes={train['mean_hidden_spike_count']} "
        f"train_hidden_trace_mean={train['hidden_trace_mean']} "
        f"train_hidden_trace_nonzero={train['hidden_trace_nonzero_count']}"
    )
    print(
        "eval recurrent current diagnostics: "
        f"input_abs={eval_['mean_abs_input_hidden_current']} "
        f"rec_abs={eval_['mean_abs_recurrent_hidden_current']} "
        f"ratio_mean={eval_['mean_recurrent_to_input_current_ratio']} "
        f"ratio_median={eval_['median_recurrent_to_input_current_ratio']} "
        f"ratio_max={eval_['max_recurrent_to_input_current_ratio']} "
        f"prev_hidden_spikes={eval_['mean_prev_hidden_spike_count']} "
        f"hidden_spikes={eval_['mean_hidden_spike_count']} "
        f"hidden_trace_mean={eval_['hidden_trace_mean']} "
        f"hidden_trace_nonzero={eval_['hidden_trace_nonzero_count']}"
    )
    print(
        "eval recurrent current by action: "
        f"input_abs={eval_['input_hidden_current_abs_mean_by_action']} "
        f"rec_abs={eval_['recurrent_hidden_current_abs_mean_by_action']} "
        f"ratio={eval_['recurrent_to_input_current_ratio_by_action']}"
    )
    print(
        "eval output_spike_total_per_episode: "
        f"{eval_['output_spike_total_per_episode']}"
    )
    print(
        "output_column_weight_mean: "
        f"start={stage_record['stage_start_output_column_weight_mean']} "
        f"train={stage_record['stage_train_output_column_weight_mean']} "
        f"end={stage_record['stage_end_output_column_weight_mean']}"
    )
    print(
        "recurrent_crossbar_status: "
        f"start={stage_record.get('stage_start_recurrent_crossbar_status', {})} "
        f"train={stage_record.get('stage_train_recurrent_crossbar_status', {})} "
        f"end={stage_record.get('stage_end_recurrent_crossbar_status', {})}"
    )
    print(
        "train found_victim_count/collision_count/moved_count: "
        f"{train['found_victim_count']}/{train['collision_count']}/{train['moved_count']}"
    )
    print(
        "eval  found_victim_count/collision_count/moved_count: "
        f"{eval_['found_victim_count']}/{eval_['collision_count']}/{eval_['moved_count']}"
    )
    print(
        "mean_fallback_count: "
        f"train={train['mean_fallback_count']} eval={eval_['mean_fallback_count']}"
    )
    print(
        "delayed_credit_update_count: "
        f"train={train['delayed_credit_update_count']} eval={eval_['delayed_credit_update_count']}"
    )
    print(
        "train delayed_credit_action_histogram: "
        f"{train['delayed_credit_action_histogram']}"
    )
    print(
        "train direct_credit_action_histogram: "
        f"{train['direct_credit_action_histogram']}"
    )
    print(
        "train previous_turn_credit_action_histogram: "
        f"{train['previous_turn_credit_action_histogram']}"
    )
    print(f"train learning_update_action_histogram: {train['learning_update_action_histogram']}")
    print(f"train potentiation_action_histogram: {train['potentiation_action_histogram']}")
    print(f"train depression_action_histogram: {train['depression_action_histogram']}")
    print(f"train turn_sensor_reward_action_histogram: {train['turn_sensor_reward_action_histogram']}")
    print(f"train turn_sensor_penalty_action_histogram: {train['turn_sensor_penalty_action_histogram']}")
    print(f"train learning_reason_histogram: {train['learning_reason_histogram']}")
    print(f"eval output_winner_histogram: {eval_['output_winner_histogram']}")
    print(f"eval snn_action_histogram: {eval_['snn_action_histogram']}")
    print(f"eval executed_action_histogram: {eval_['executed_action_histogram']}")
    print(
        "train wall_avoid counts: "
        f"forward_collision={train['wall_avoid_forward_collision_count']} "
        f"forward_after_clearance={train['wall_avoid_forward_after_clearance_count']} "
        f"spin_penalty={train['wall_avoid_repeated_turn_spin_count']} "
        f"forward_positive={train['wall_avoid_forward_positive_count']} "
        f"forward_positive_skipped={train['wall_avoid_forward_positive_skipped_count']} "
        f"turn_reward={train['wall_avoid_turn_reward_count']} "
        f"blocked_action0={train['wall_avoid_action0_selected_when_front_blocked_count']}"
    )
    print(
        "eval wall_avoid counts: "
        f"forward_collision={eval_['wall_avoid_forward_collision_count']} "
        f"forward_after_clearance={eval_['wall_avoid_forward_after_clearance_count']} "
        f"spin_penalty={eval_['wall_avoid_repeated_turn_spin_count']} "
        f"forward_positive={eval_['wall_avoid_forward_positive_count']} "
        f"forward_positive_skipped={eval_['wall_avoid_forward_positive_skipped_count']} "
        f"turn_reward={eval_['wall_avoid_turn_reward_count']} "
        f"blocked_action0={eval_['wall_avoid_action0_selected_when_front_blocked_count']}"
    )
    print(f"train selected_step_action_histogram: {train['selected_step_action_histogram']}")
    print(
        "train wall_avoid recovery/spin hists: "
        f"recovery={train['wall_avoid_forward_recovery_action_histogram']} "
        f"spin={train['wall_avoid_spin_penalty_action_histogram']}"
    )
    print(f"eval first_episode_action_sequence: {eval_['eval_first_episode_action_sequence']}")
    print(f"eval first_episode_collision_sequence: {eval_['eval_first_episode_collision_sequence']}")
    print(f"eval first_episode_moved_sequence: {eval_['eval_first_episode_moved_sequence']}")
    mode_diag = stage_record.get("eval_by_selection_mode", {})
    if isinstance(mode_diag, dict) and mode_diag:
        print("eval_by_selection_mode:")
        for mode, diag in mode_diag.items():
            if not isinstance(diag, dict):
                continue
            print(
                f"  {mode}: success={diag.get('success_rate')} "
                f"actions={diag.get('action_histogram')} "
                f"collisions={diag.get('collision_count')} "
                f"seq={diag.get('first_episode_action_sequence')}"
            )
    ablation_all = stage_record.get("recurrent_ablation_eval", {})
    ablation = (
        ablation_all.get(stage_record.get("stage_name"), {})
        if isinstance(ablation_all, dict)
        else {}
    )
    if isinstance(ablation, dict) and ablation:
        print("recurrent_ablation_eval:")
        for label, diag in ablation.items():
            if not isinstance(diag, dict):
                continue
            print(
                f"  {label}: success={diag.get('success_rate')} "
                f"actions={diag.get('action_histogram')} "
                f"collisions={diag.get('collision_count')} "
                f"ratio={diag.get('mean_recurrent_to_input_current_ratio')} "
                f"seq={diag.get('sequence')}"
            )
    scale_sweep = stage_record.get("stm_recurrent_scale_sweep_eval", {})
    if isinstance(scale_sweep, dict) and scale_sweep:
        print("stm_recurrent_scale_sweep_eval:")
        for scale, diag in scale_sweep.items():
            if not isinstance(diag, dict):
                continue
            print(
                f"  scale={scale}: success={diag.get('success_rate')} "
                f"actions={diag.get('action_histogram')} "
                f"collisions={diag.get('collision_count')} "
                f"ratio={diag.get('mean_recurrent_to_input_current_ratio')} "
                f"seq={diag.get('sequence')}"
            )
    interpretation = stage_record.get("stm_recurrent_interpretation", {})
    if isinstance(interpretation, dict) and interpretation:
        print(
            "stm_recurrent_interpretation: "
            f"{interpretation.get('case')} - {interpretation.get('reason')}"
        )
    print()


def _primitive_regression_stage_names() -> tuple[str, ...]:
    raw = _cfg(
        "PRIMITIVE_REGRESSION_STAGE_NAMES",
        ("stage_forward", "stage_turn_right", "stage_turn_left", "stage_wall_avoid"),
    )
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(name) for name in raw)


def _hist_count(hist: Any, action: int) -> int:
    if not isinstance(hist, dict):
        return 0
    return int(hist.get(action, hist.get(str(action), 0)) or 0)


def _compact_eval_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    metrics = _metrics_from_summary(summary)
    return {
        "eval_success": float(summary.get("success_rate", 0.0)),
        "action_sequence": summary.get("eval_first_episode_action_sequence", []),
        "raw_action_sequence": summary.get("eval_first_episode_raw_action_sequence", []),
        "valid_actions_only": bool(summary.get("valid_actions_only", True)),
        "no_action_count": int(summary.get("eval_first_episode_no_action_count", 0)),
        "no_spike_count": int(summary.get("eval_first_episode_no_spike_count", 0)),
        "early_done_before_action": bool(
            summary.get("eval_first_episode_early_done_before_action", False)
        ),
        "collision_sequence": summary.get("eval_first_episode_collision_sequence", []),
        "moved_sequence": summary.get("eval_first_episode_moved_sequence", []),
        "collision_count": int(summary.get("collision_count", metrics.get("collision_count", 0))),
        "moved_count": int(summary.get("moved_count", metrics.get("moved_count", 0))),
        "found_victim_count": int(
            summary.get("found_victim_count", metrics.get("found_victim_count", 0))
        ),
        "action_histogram": metrics.get("action_histogram", {}),
        "snn_action_histogram": metrics.get("snn_action_histogram", {}),
        "spike_selected_action_histogram": summary.get("spike_selected_action_histogram", {}),
        "fallback_action_histogram": summary.get("fallback_action_histogram", {}),
        "output_winner_histogram": summary.get("output_winner_histogram", {}),
        "output_column_weight_mean": summary.get("output_column_weight_mean", {}),
        "output_column_weight_std": summary.get("output_column_weight_std", {}),
        "output_column_effective_score": summary.get(
            "output_column_effective_score",
            summary.get("output_column_weight_mean", {}),
        ),
        "output_spike_count_by_action": summary.get("output_spike_count_by_action", {}),
        "output_first_spike_time_by_action_mean": summary.get(
            "output_first_spike_time_by_action_mean",
            {},
        ),
        "pulse_plus_count_by_action": summary.get("potentiation_action_histogram", {}),
        "pulse_minus_count_by_action": summary.get("depression_action_histogram", {}),
        "reward_by_action": summary.get("action_mean_reward", {}),
        "event_count_by_action": summary.get("action_event_counts", {}),
        "eval_state_reset_debug": summary.get("eval_state_reset_debug", {}),
        "mean_recurrent_to_input_current_ratio": float(
            summary.get("mean_recurrent_to_input_current_ratio", 0.0)
        ),
        "hidden_trace_mean": float(summary.get("hidden_trace_mean", 0.0)),
        "hidden_trace_nonzero_count": float(summary.get("hidden_trace_nonzero_count", 0.0)),
    }


def _run_primitive_regression(
    *,
    net: MemristiveSNNNetwork,
    seed: int,
    verbose: bool = False,
    save_json: bool = True,
    filename: str = "primitive_regression_summary.json",
    eval_episodes_override: Optional[int] = None,
) -> Dict[str, Any]:
    stage_names = _primitive_regression_stage_names()
    stage_lookup = {
        str(stage.get("name", "")): dict(stage)
        for stage in _cfg("CURRICULUM_STAGES", [])
        if isinstance(stage, dict)
    }
    snapshot = copy.deepcopy(net)
    stages: Dict[str, Any] = {}
    warnings: list[str] = []

    for idx, stage_name in enumerate(stage_names):
        stage = stage_lookup.get(stage_name)
        if stage is None:
            warnings.append(f"{stage_name}: missing curriculum stage")
            continue
        map_name = str(stage["map_name"])
        eval_episodes = int(
            eval_episodes_override
            if eval_episodes_override is not None
            else stage.get(
                "eval_episodes",
                _stage_setting(stage_name, "eval_episodes", N_EPISODES_EVAL),
            )
        )
        eval_env = build_env_for_map(map_name, seed=seed + 4000 + idx)
        _, eval_summary = run_phase(
            env=eval_env,
            net=copy.deepcopy(snapshot),
            learner=None,
            n_episodes=eval_episodes,
            phase_name=f"{stage_name}_regression_eval",
            verbose=verbose,
            log_name=f"{stage_name}_regression_eval",
            trace_first_steps=int(_cfg("EVAL_TRACE_FIRST_STEPS", 20)),
        )
        compact = _compact_eval_summary(eval_summary)
        stages[stage_name] = compact
        if compact["eval_success"] < 1.0:
            warnings.append(
                f"{stage_name}: eval_success={compact['eval_success']} < 1.0"
            )

    passed = len(warnings) == 0 and all(
        float(item.get("eval_success", 0.0)) >= 1.0 for item in stages.values()
    )
    summary = {
        "passed": bool(passed),
        "warnings": warnings,
        "criteria": {
            name: "eval_success == 1.0"
            for name in stage_names
        },
        "stages": stages,
        "primitive_action_confusion": _primitive_action_confusion(stages),
    }
    summary.update(stages)
    if save_json:
        _maybe_save_json(
            filename,
            summary,
            "SAVE_PRIMITIVE_REGRESSION_SUMMARY_JSON",
        )
    return summary


def _run_primitive_consolidation_before_complex(
    *,
    net: MemristiveSNNNetwork,
    learner: RewardModulatedSTDPLearner,
    seed: int,
    after_stage_name: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    if not bool(_cfg("ENABLE_PRIMITIVE_CONSOLIDATION_BEFORE_COMPLEX", True)):
        return {"enabled": False, "passed": False, "history": []}

    stage_names = _primitive_regression_stage_names()
    stage_lookup = {
        str(stage.get("name", "")): dict(stage)
        for stage in _cfg("CURRICULUM_STAGES", [])
        if isinstance(stage, dict)
    }
    cycles = max(1, int(_cfg("PRIMITIVE_CONSOLIDATION_CYCLES", 5)))
    episodes_per_stage = max(
        1,
        int(_cfg("PRIMITIVE_CONSOLIDATION_EPISODES_PER_STAGE", 2)),
    )
    attempts = max(1, int(_cfg("PRIMITIVE_CONSOLIDATION_MAX_ATTEMPTS", 1)))
    history: list[Dict[str, Any]] = []
    final_regression: Dict[str, Any] = {}

    for attempt_idx in range(attempts):
        attempt_record: Dict[str, Any] = {
            "attempt": int(attempt_idx),
            "cycles": int(cycles),
            "episodes_per_stage": int(episodes_per_stage),
            "train": {},
        }
        for cycle_idx in range(cycles):
            for stage_offset, primitive_name in enumerate(stage_names):
                stage = stage_lookup.get(primitive_name)
                if stage is None:
                    attempt_record["train"][primitive_name] = {
                        "skipped": True,
                        "reason": "missing primitive stage",
                    }
                    continue
                map_name = str(stage["map_name"])
                env = build_env_for_map(
                    map_name,
                    seed=seed + attempt_idx * 1000 + cycle_idx * 100 + stage_offset,
                )
                _, train_summary = run_phase(
                    env=env,
                    net=net,
                    learner=learner,
                    n_episodes=episodes_per_stage,
                    phase_name=f"{primitive_name}_primitive_consolidation_train",
                    verbose=verbose,
                    log_name=(
                        f"{primitive_name}_consolidation_"
                        f"{_safe_log_name(after_stage_name)}_a{attempt_idx}_c{cycle_idx}"
                    ),
                )
                attempt_record["train"][
                    f"{primitive_name}_cycle_{cycle_idx}"
                ] = _compact_eval_summary(train_summary)

        final_regression = _run_primitive_regression(
            net=net,
            seed=seed + 7000 + attempt_idx,
            verbose=False,
            save_json=False,
            eval_episodes_override=int(
                _cfg("PRIMITIVE_REGRESSION_HISTORY_EVAL_EPISODES", 1)
            ),
        )
        attempt_record["regression"] = final_regression
        attempt_record["passed"] = bool(final_regression.get("passed", False))
        history.append(attempt_record)
        if bool(final_regression.get("passed", False)):
            break

    return {
        "enabled": True,
        "after_stage_name": str(after_stage_name),
        "cycles": int(cycles),
        "episodes_per_stage": int(episodes_per_stage),
        "attempts": int(len(history)),
        "passed": bool(final_regression.get("passed", False)),
        "final_regression": final_regression,
        "history": history,
    }


def _primitive_core_action_collapse_count(regression: Dict[str, Any]) -> int:
    count = 0
    for item in regression.get("stages", {}).values():
        hist = item.get("action_histogram", {}) if isinstance(item, dict) else {}
        if not isinstance(hist, dict) or not hist:
            continue
        total = sum(int(v) for v in hist.values())
        if total <= 0:
            continue
        if max(int(v) for v in hist.values()) / float(total) >= 0.90:
            count += 1
    return int(count)


def _primitive_core_score(regression: Dict[str, Any]) -> float:
    stages = regression.get("stages", {}) if isinstance(regression, dict) else {}
    passed_count = sum(
        1
        for item in stages.values()
        if isinstance(item, dict) and float(item.get("eval_success", 0.0)) >= 1.0
    )
    total_collisions = sum(
        int(item.get("collision_count", 0))
        for item in stages.values()
        if isinstance(item, dict)
    )
    action_collapse_count = _primitive_core_action_collapse_count(regression)
    return float(100.0 * passed_count - 10.0 * total_collisions - 20.0 * action_collapse_count)


def _primitive_action_confusion(
    stages: Dict[str, Any],
) -> Dict[str, Any]:
    reference_actions_raw = _cfg("PRIMITIVE_REFERENCE_FIRST_ACTIONS", {})
    reference_actions = (
        dict(reference_actions_raw) if isinstance(reference_actions_raw, dict) else {}
    )
    action_hist_confusion: Dict[str, Dict[int, int]] = {}
    first_action_confusion: Dict[str, Dict[int, int]] = {}
    dominant_action_confusion: Dict[str, Dict[int, int]] = {}
    action_sequence_entropy: Dict[str, float] = {}
    alternating_lr_oscillation_count: Dict[str, int] = {}
    action0_suppression_score: Dict[str, float] = {}

    for stage_name, item in stages.items():
        if not isinstance(item, dict):
            continue
        action_hist = item.get("action_histogram", {})
        row = {0: 0, 1: 0, 2: 0}
        if isinstance(action_hist, dict):
            for action in (0, 1, 2):
                row[action] = int(action_hist.get(action, action_hist.get(str(action), 0)) or 0)
        sequence = [int(a) for a in item.get("action_sequence", []) if int(a) in (0, 1, 2)]
        first_row = {0: 0, 1: 0, 2: 0}
        if sequence:
            first_row[int(sequence[0])] = 1
        dominant_row = {0: 0, 1: 0, 2: 0}
        if row:
            dominant_action = max(row.keys(), key=lambda action: row[action])
            if row[dominant_action] > 0:
                dominant_row[int(dominant_action)] = 1
        total = max(sum(row.values()), 1)
        probs = [float(row[action] / total) for action in (0, 1, 2) if row[action] > 0]
        entropy = -sum(p * float(np.log2(max(p, 1e-12))) for p in probs)
        alternations = 0
        for prev_action, next_action in zip(sequence[:-1], sequence[1:]):
            if {int(prev_action), int(next_action)} == {1, 2}:
                alternations += 1
        action0_rate = float(row.get(0, 0) / total)
        action_hist_confusion[str(stage_name)] = row
        first_action_confusion[str(stage_name)] = first_row
        dominant_action_confusion[str(stage_name)] = dominant_row
        action_sequence_entropy[str(stage_name)] = float(entropy)
        alternating_lr_oscillation_count[str(stage_name)] = int(alternations)
        action0_suppression_score[str(stage_name)] = float(1.0 - action0_rate)

    return {
        "reference_actions": {
            str(stage): int(action)
            for stage, action in reference_actions.items()
            if str(stage) in stages
        },
        "action_histogram_confusion": action_hist_confusion,
        "first_action_confusion": first_action_confusion,
        "dominant_action_confusion": dominant_action_confusion,
        "action_sequence_entropy": action_sequence_entropy,
        "alternating_lr_oscillation_count": alternating_lr_oscillation_count,
        "action0_suppression_score": action0_suppression_score,
    }


def _primitive_core_output_bias_diagnostics(
    net: MemristiveSNNNetwork,
    regression: Dict[str, Any],
) -> Dict[str, Any]:
    stages = regression.get("stages", {}) if isinstance(regression, dict) else {}
    output_spike_count_by_action: Dict[int, int] = {}
    pulse_plus_count_by_action: Dict[int, int] = {}
    pulse_minus_count_by_action: Dict[int, int] = {}
    reward_sum_by_action: Dict[int, float] = {}
    reward_count_by_action: Dict[int, int] = {}
    collision_count_by_action: Dict[int, int] = {}
    found_count_by_action: Dict[int, int] = {}
    first_spike_values: Dict[int, list[float]] = {}

    for item in stages.values():
        if not isinstance(item, dict):
            continue
        for action, count in item.get("output_spike_count_by_action", {}).items():
            _hist_increment(output_spike_count_by_action, int(action), int(count))
        for action, count in item.get("pulse_plus_count_by_action", {}).items():
            _hist_increment(pulse_plus_count_by_action, int(action), int(count))
        for action, count in item.get("pulse_minus_count_by_action", {}).items():
            _hist_increment(pulse_minus_count_by_action, int(action), int(count))
        for action, value in item.get("output_first_spike_time_by_action_mean", {}).items():
            first_spike_values.setdefault(int(action), []).append(float(value))
        for action, value in item.get("reward_by_action", {}).items():
            action_i = int(action)
            event_count = int(
                item.get("event_count_by_action", {})
                .get(action, item.get("event_count_by_action", {}).get(str(action), {}))
                .get("count", 1)
            ) if isinstance(item.get("event_count_by_action", {}), dict) else 1
            reward_sum_by_action[action_i] = float(
                reward_sum_by_action.get(action_i, 0.0) + float(value) * max(event_count, 1)
            )
            reward_count_by_action[action_i] = int(
                reward_count_by_action.get(action_i, 0) + max(event_count, 1)
            )
        for action, values in item.get("event_count_by_action", {}).items():
            if isinstance(values, dict):
                _hist_increment(collision_count_by_action, int(action), int(values.get("collision", 0)))
                _hist_increment(found_count_by_action, int(action), int(values.get("found_victim", 0)))

    return {
        "output_column_weight_mean": _output_column_weight_mean(net),
        "output_column_weight_std": _output_column_weight_std(net),
        "output_column_effective_score": _output_column_effective_score(net),
        "output_spike_count_by_action": output_spike_count_by_action,
        "first_spike_time_mean_by_action": {
            action: float(np.mean(values)) for action, values in first_spike_values.items()
        },
        "pulse_plus_count_by_action": pulse_plus_count_by_action,
        "pulse_minus_count_by_action": pulse_minus_count_by_action,
        "reward_sum_by_action": reward_sum_by_action,
        "reward_mean_by_action": {
            action: float(reward_sum_by_action[action] / max(reward_count_by_action.get(action, 1), 1))
            for action in reward_sum_by_action
        },
        "collision_count_by_action": collision_count_by_action,
        "found_count_by_action": found_count_by_action,
    }


def _primitive_core_task_for_episode(
    *,
    task_names: tuple[str, ...],
    episode_index: int,
    style: str,
    rng: np.random.Generator,
) -> str:
    if not task_names:
        raise ValueError("primitive core requires at least one task")
    if str(style).lower() == "random_balanced":
        cycle = list(task_names)
        rng.shuffle(cycle)
        return str(cycle[int(episode_index) % len(cycle)])
    return str(task_names[int(episode_index) % len(task_names)])


def _run_primitive_core_training(
    *,
    net: MemristiveSNNNetwork,
    learner: RewardModulatedSTDPLearner,
    seed: int,
    verbose: bool = False,
) -> tuple[MemristiveSNNNetwork, Dict[str, Any]]:
    if not bool(_cfg("ENABLE_PRIMITIVE_CORE_TRAINING", True)):
        return net, {"enabled": False, "primitive_core_checkpoint_found": False}

    task_names = _primitive_regression_stage_names()
    stage_lookup = {
        str(stage.get("name", "")): dict(stage)
        for stage in _cfg("CURRICULUM_STAGES", [])
        if isinstance(stage, dict)
    }
    total_episodes = max(1, int(_cfg("PRIMITIVE_CORE_EPISODES", 100)))
    block_size = max(1, int(_cfg("PRIMITIVE_CORE_BLOCK_SIZE", 4)))
    eval_interval = max(1, int(_cfg("PRIMITIVE_CORE_EVAL_INTERVAL", block_size)))
    sampling = str(_cfg("PRIMITIVE_CORE_TASK_SAMPLING", "balanced_cycle"))
    rng = np.random.default_rng(seed + 61000)

    best_net = copy.deepcopy(net)
    best_regression = _run_primitive_regression(
        net=best_net,
        seed=seed + 61100,
        verbose=False,
        save_json=False,
        eval_episodes_override=int(_cfg("PRIMITIVE_REGRESSION_HISTORY_EVAL_EPISODES", 1)),
    )
    best_score = _primitive_core_score(best_regression)
    history: list[Dict[str, Any]] = [
        {
            "block_index": -1,
            "episode_end": 0,
            "accepted": True,
            "passed": bool(best_regression.get("passed", False)),
            "score": float(best_score),
            "regression": best_regression,
            "output_column_status": _output_column_status(best_net),
            "output_column_interference_by_action": _output_column_interference_record(
                best_net,
                None,
            ),
            "output_bias_diagnostics": _primitive_core_output_bias_diagnostics(
                best_net,
                best_regression,
            ),
        }
    ]
    checkpoint_found = bool(best_regression.get("passed", False))
    checkpoint_net = copy.deepcopy(best_net) if checkpoint_found else None

    episode_cursor = 0
    block_index = 0
    while episode_cursor < total_episodes and not checkpoint_found:
        block_start_snapshot = copy.deepcopy(best_net)
        net = copy.deepcopy(best_net)
        block_start_column_status = _output_column_status(net)
        block_tasks: list[str] = []
        block_train: list[Dict[str, Any]] = []
        for _ in range(min(block_size, total_episodes - episode_cursor)):
            task_name = _primitive_core_task_for_episode(
                task_names=task_names,
                episode_index=episode_cursor,
                style=sampling,
                rng=rng,
            )
            block_tasks.append(task_name)
            stage = stage_lookup.get(task_name)
            if stage is None:
                block_train.append({"task": task_name, "skipped": True, "reason": "missing stage"})
                episode_cursor += 1
                continue
            task_column_status_before = _output_column_status(net)
            env = build_env_for_map(str(stage["map_name"]), seed=seed + 62000 + episode_cursor)
            _, train_summary = run_phase(
                env=env,
                net=net,
                learner=learner,
                n_episodes=1,
                phase_name=f"{task_name}_primitive_core_train",
                verbose=verbose,
                log_name=f"{task_name}_primitive_core_ep_{episode_cursor}",
            )
            task_column_status_after = _output_column_status(net)
            block_train.append(
                {
                    "task": task_name,
                    "episode_index": int(episode_cursor),
                    "train": _compact_eval_summary(train_summary),
                    "output_column_status_before": task_column_status_before,
                    "output_column_status_after": task_column_status_after,
                    "delta_output_column_status": _output_column_status_delta(
                        task_column_status_before,
                        task_column_status_after,
                    ),
                    "output_column_interference_by_action": _output_column_interference_record(
                        net,
                        train_summary,
                    ),
                }
            )
            episode_cursor += 1

        should_eval = (
            episode_cursor % eval_interval == 0
            or episode_cursor >= total_episodes
            or bool(_cfg("ENABLE_PRIMITIVE_CORE_BLOCK_ROLLBACK", True))
        )
        if should_eval:
            candidate_column_status = _output_column_status(net)
            regression = _run_primitive_regression(
                net=net,
                seed=seed + 63000 + block_index,
                verbose=False,
                save_json=False,
                eval_episodes_override=int(_cfg("PRIMITIVE_REGRESSION_HISTORY_EVAL_EPISODES", 1)),
            )
            score = _primitive_core_score(regression)
            passed = bool(regression.get("passed", False))
            score_improved = bool(score > best_score)
            accepted_by_pass = bool(
                passed and _cfg("PRIMITIVE_CORE_ACCEPT_IF_ALL_PASS", True)
            )
            accepted_by_score = bool(
                score_improved and _cfg("PRIMITIVE_CORE_ACCEPT_IF_SCORE_IMPROVES", True)
            )
            accepted = bool(accepted_by_pass or accepted_by_score)
            reject_reasons: list[str] = []
            if not accepted:
                reject_reasons.append("primitive_core_score_not_improved")
                if not passed:
                    reject_reasons.append("primitive_full_regression_not_passed")
            if accepted:
                best_net = copy.deepcopy(net)
                best_score = float(score)
                best_regression = regression
                if passed:
                    checkpoint_found = True
                    checkpoint_net = copy.deepcopy(net)
            else:
                net = copy.deepcopy(block_start_snapshot)
            history.append(
                {
                    "block_index": int(block_index),
                    "episode_end": int(episode_cursor),
                    "tasks": block_tasks,
                    "train": block_train,
                    "score": float(score),
                    "previous_best_score": float(best_score),
                    "score_improved": bool(score_improved),
                    "passed": bool(passed),
                    "accepted": bool(accepted),
                    "accepted_by_pass": bool(accepted_by_pass),
                    "accepted_by_score": bool(accepted_by_score),
                    "reject_reasons": reject_reasons,
                    "regression": regression,
                    "output_column_status_before_block": block_start_column_status,
                    "output_column_status_after_candidate_block": candidate_column_status,
                    "delta_output_column_status_candidate_block": _output_column_status_delta(
                        block_start_column_status,
                        candidate_column_status,
                    ),
                    "output_column_status_after_accept_or_rollback": _output_column_status(
                        best_net if accepted else block_start_snapshot
                    ),
                    "output_column_interference_by_action_after_block": _output_column_interference_record(
                        net if accepted else block_start_snapshot,
                        None,
                    ),
                    "output_bias_diagnostics": _primitive_core_output_bias_diagnostics(
                        net if accepted else block_start_snapshot,
                        regression,
                    ),
                }
            )
        block_index += 1

    selected_net = checkpoint_net if checkpoint_net is not None else best_net
    final_regression = _run_primitive_regression(
        net=selected_net,
        seed=seed + 64000,
        verbose=False,
        save_json=False,
        eval_episodes_override=int(_cfg("PRIMITIVE_REGRESSION_HISTORY_EVAL_EPISODES", 1)),
    )
    final_score = _primitive_core_score(final_regression)
    output_column_interference_diagnostic = {
        "enabled": bool(_cfg("ENABLE_OUTPUT_COLUMN_INTERFERENCE_DIAGNOSTIC", True)),
        "final_output_column_status": _output_column_status(selected_net),
        "final_output_column_interference_by_action": _output_column_interference_record(
            selected_net,
            None,
        ),
        "block_history": [
            {
                "block_index": item.get("block_index"),
                "episode_end": item.get("episode_end"),
                "tasks": item.get("tasks", []),
                "accepted": item.get("accepted"),
                "passed": item.get("passed"),
                "score": item.get("score"),
                "train": [
                    {
                        "task": train_item.get("task"),
                        "episode_index": train_item.get("episode_index"),
                        "delta_output_column_status": train_item.get(
                            "delta_output_column_status",
                            {},
                        ),
                        "output_column_interference_by_action": train_item.get(
                            "output_column_interference_by_action",
                            {},
                        ),
                    }
                    for train_item in item.get("train", [])
                    if isinstance(train_item, dict)
                ],
                "delta_output_column_status_candidate_block": item.get(
                    "delta_output_column_status_candidate_block",
                    {},
                ),
                "output_column_interference_by_action_after_block": item.get(
                    "output_column_interference_by_action_after_block",
                    {},
                ),
            }
            for item in history
            if isinstance(item, dict)
        ],
    }
    if bool(_cfg("ENABLE_OUTPUT_COLUMN_INTERFERENCE_DIAGNOSTIC", True)):
        _write_json_log(
            "primitive_output_column_interference_diagnostic.json",
            output_column_interference_diagnostic,
        )
    summary = {
        "enabled": True,
        "primitive_core_checkpoint_found": bool(final_regression.get("passed", False)),
        "episodes_requested": int(total_episodes),
        "episodes_consumed": int(episode_cursor),
        "task_sampling": sampling,
        "block_size": int(block_size),
        "eval_interval": int(eval_interval),
        "block_rollback_enabled": bool(_cfg("ENABLE_PRIMITIVE_CORE_BLOCK_ROLLBACK", True)),
        "primitive_rstdp_scale": _cfg("PRIMITIVE_RSTDP_SCALE", {}),
        "primitive_core_hidden_trace_input_scale": float(
            _cfg("PRIMITIVE_CORE_HIDDEN_TRACE_INPUT_SCALE", 0.0)
        ),
        "primitive_core_enable_hidden_trace": bool(
            _cfg("PRIMITIVE_CORE_ENABLE_HIDDEN_TRACE", False)
        ),
        "primitive_core_stm_recurrent_current_scale": float(
            _cfg("PRIMITIVE_CORE_STM_RECURRENT_CURRENT_SCALE", 0.0)
        ),
        "output_depression_scale": float(_cfg("OUTPUT_DEPRESSION_SCALE", 1.0)),
        "anti_target_depression": bool(_cfg("ANTI_TARGET_DEPRESSION", False)),
        "final_score": float(final_score),
        "final_regression": final_regression,
        "output_bias_diagnostics": _primitive_core_output_bias_diagnostics(
            selected_net,
            final_regression,
        ),
        "output_column_status": _output_column_status(selected_net),
        "output_column_interference_diagnostic": output_column_interference_diagnostic,
        "history": history,
    }
    if bool(_cfg("SAVE_PRIMITIVE_CORE_TRAINING_SUMMARY_JSON", True)):
        _write_json_log("primitive_core_training_summary.json", summary)
    return selected_net, summary


def run_primitive_observation_alias_diagnostic(
    *,
    net: MemristiveSNNNetwork,
    seed: int,
    label: str = "initial",
    save_json: bool = True,
) -> Dict[str, Any]:
    if not bool(_cfg("ENABLE_PRIMITIVE_OBSERVATION_ALIAS_DIAGNOSTIC", True)):
        return {"enabled": False, "label": str(label)}

    stage_names = _primitive_regression_stage_names()
    stage_lookup = {
        str(stage.get("name", "")): dict(stage)
        for stage in _cfg("CURRICULUM_STAGES", [])
        if isinstance(stage, dict)
    }
    reference_actions_raw = _cfg("PRIMITIVE_REFERENCE_FIRST_ACTIONS", {})
    reference_actions = (
        dict(reference_actions_raw) if isinstance(reference_actions_raw, dict) else {}
    )
    encoder = net.encoder
    stage_records: Dict[str, Any] = {}

    for idx, stage_name in enumerate(stage_names):
        stage = stage_lookup.get(stage_name)
        if stage is None:
            stage_records[stage_name] = {"missing": True}
            continue
        map_name = str(stage["map_name"])
        env = build_env_for_map(map_name, seed=seed + 81000 + idx)
        obs = env.reset(map_index=0)
        obs_for_encoder = _observation_for_phase(obs, f"{stage_name}_alias_eval")
        env_state = env.get_env_state() if hasattr(env, "get_env_state") else {}
        normalized = _normalized_observation_vector(encoder, obs_for_encoder)
        encoded_window = np.asarray(
            [out.spikes for out in encoder.encode_window(obs_for_encoder)],
            dtype=int,
        )
        encoder_output = encoder.encode(obs_for_encoder, sim_step=0)
        spike_times = np.asarray(encoder_output.spike_times, dtype=float).reshape(-1)
        active_indices = np.where(encoded_window.sum(axis=0) > 0)[0].astype(int)

        decision_net = copy.deepcopy(net)
        decision_net.reset_episode()
        decision = decision_net.decide(obs_for_encoder)
        selected_action = int(getattr(decision, "action", -1))
        reference_action = reference_actions.get(stage_name, None)
        reference_action = None if reference_action is None else int(reference_action)

        selected_step_info: Dict[str, Any] = {}
        if selected_action in (0, 1, 2):
            selected_env = build_env_for_map(map_name, seed=seed + 82000 + idx)
            selected_env.reset(map_index=0)
            selected_result = selected_env.step(selected_action)
            selected_step_info = {
                "reward": float(selected_result.reward),
                "done": bool(selected_result.done),
                "info": dict(selected_result.info or {}),
            }

        reference_step_info: Dict[str, Any] = {}
        if reference_action in (0, 1, 2):
            reference_env = build_env_for_map(map_name, seed=seed + 83000 + idx)
            reference_env.reset(map_index=0)
            reference_result = reference_env.step(reference_action)
            reference_step_info = {
                "reward": float(reference_result.reward),
                "done": bool(reference_result.done),
                "info": dict(reference_result.info or {}),
            }

        raw_obs = {
            str(name): float(_safe_float(obs_for_encoder.get(str(name), 0.0), 0.0))
            for name in getattr(encoder, "feature_names", [])
            if isinstance(obs_for_encoder, dict)
        }
        stage_records[stage_name] = {
            "stage_name": stage_name,
            "map_name": map_name,
            "raw_observation": raw_obs,
            "normalized_observation_vector": normalized,
            "encoder_feature_names": list(getattr(encoder, "feature_names", [])),
            "encoder_output_feature_names": list(
                getattr(encoder, "output_feature_names", [])
            ),
            "encoder_spike_window": encoded_window.tolist(),
            "encoder_spike_pattern_flat": encoded_window.reshape(-1).tolist(),
            "encoder_spike_times": [
                None if not np.isfinite(value) else float(value)
                for value in spike_times
            ],
            "active_input_neuron_indices": active_indices.tolist(),
            "active_input_neuron_count": int(active_indices.size),
            "initial_robot_position": env_state.get("agent_pos"),
            "initial_heading": env_state.get("agent_heading"),
            "initial_heading_name": env_state.get("agent_heading_name"),
            "done_at_start": bool(getattr(env, "done", False)),
            "first_selected_action": selected_action,
            "first_selected_by_spike": bool(
                getattr(decision, "selected_by_spike", selected_action >= 0)
            ),
            "first_selected_step": int(getattr(decision, "selected_step", -1)),
            "reference_action": reference_action,
            "selected_action_first_step": selected_step_info,
            "reference_action_first_step": reference_step_info,
        }

    pairwise: Dict[str, Any] = {}
    conflicts: list[Dict[str, Any]] = []
    for stage_a, stage_b in itertools.combinations(stage_names, 2):
        rec_a = stage_records.get(stage_a, {})
        rec_b = stage_records.get(stage_b, {})
        if rec_a.get("missing") or rec_b.get("missing"):
            continue
        obs_a = np.asarray(rec_a.get("normalized_observation_vector", []), dtype=float)
        obs_b = np.asarray(rec_b.get("normalized_observation_vector", []), dtype=float)
        raw_l2 = (
            float(np.linalg.norm(obs_a - obs_b))
            if obs_a.size == obs_b.size and obs_a.size > 0
            else 0.0
        )
        raw_cosine = _cosine_similarity(obs_a, obs_b)
        enc_a = np.asarray(rec_a.get("encoder_spike_pattern_flat", []), dtype=float)
        enc_b = np.asarray(rec_b.get("encoder_spike_pattern_flat", []), dtype=float)
        encoded_cosine = _cosine_similarity(enc_a, enc_b)
        active_a = set(int(x) for x in rec_a.get("active_input_neuron_indices", []))
        active_b = set(int(x) for x in rec_b.get("active_input_neuron_indices", []))
        union = active_a | active_b
        active_overlap = (
            float(len(active_a & active_b) / max(len(union), 1))
            if union
            else 1.0
        )
        action_a = rec_a.get("reference_action", None)
        action_b = rec_b.get("reference_action", None)
        actions_differ = (
            action_a is not None and action_b is not None and int(action_a) != int(action_b)
        )
        raw_alias = raw_l2 <= float(_cfg("PRIMITIVE_ALIAS_RAW_L2_THRESHOLD", 0.25))
        encoded_alias = (
            encoded_cosine
            >= float(_cfg("PRIMITIVE_ALIAS_ENCODED_COSINE_THRESHOLD", 0.95))
            and active_overlap
            >= float(_cfg("PRIMITIVE_ALIAS_ACTIVE_OVERLAP_THRESHOLD", 0.85))
        )
        conflict = bool(actions_differ and raw_alias and encoded_alias)
        item = {
            "stage_a": stage_a,
            "stage_b": stage_b,
            "raw_observation_l2_distance": raw_l2,
            "raw_observation_cosine_similarity": raw_cosine,
            "encoded_spike_pattern_cosine_similarity": encoded_cosine,
            "active_input_neuron_overlap_ratio": active_overlap,
            "reference_action_a": action_a,
            "reference_action_b": action_b,
            "actions_differ": bool(actions_differ),
            "raw_alias": bool(raw_alias),
            "encoded_alias": bool(encoded_alias),
            "perceptual_alias_conflict": bool(conflict),
        }
        key = f"{stage_a}__vs__{stage_b}"
        pairwise[key] = item
        if conflict:
            conflicts.append(item)

    summary = {
        stage_name: {
            "obs": stage_records.get(stage_name, {}).get("normalized_observation_vector", []),
            "raw_observation": stage_records.get(stage_name, {}).get("raw_observation", {}),
            "first_action": stage_records.get(stage_name, {}).get("first_selected_action"),
            "reference_action": stage_records.get(stage_name, {}).get("reference_action"),
            "initial_heading_name": stage_records.get(stage_name, {}).get(
                "initial_heading_name"
            ),
            "active_input_neuron_count": stage_records.get(stage_name, {}).get(
                "active_input_neuron_count"
            ),
        }
        for stage_name in stage_names
    }
    result = {
        "enabled": True,
        "label": str(label),
        "active_encoder_features": list(getattr(encoder, "feature_names", [])),
        "stage_records": stage_records,
        "primitive_observation_alias_summary": summary,
        "pairwise_alias_matrix": pairwise,
        "perceptual_alias_conflicts": conflicts,
        "perceptual_alias_conflict_count": int(len(conflicts)),
        "interpretation": (
            "perceptual_alias_conflict=True means two primitive tasks have very similar "
            "raw and encoded inputs while requiring different reference first actions."
        ),
    }
    if save_json:
        _write_json_log("primitive_observation_alias_diagnostic.json", result)
    return result


def _first_spike_times_from_step_vectors(vectors: list[np.ndarray], dim: int) -> np.ndarray:
    first = np.full(int(dim), np.inf, dtype=float)
    for t, vector in enumerate(vectors):
        arr = np.asarray(vector, dtype=float).reshape(-1)
        if arr.size != int(dim):
            continue
        newly_spiked = np.flatnonzero((arr > 0.0) & ~np.isfinite(first))
        first[newly_spiked] = float(t)
    return first


def _first_spike_pattern(first_spike_times: Any) -> np.ndarray:
    times = np.asarray(first_spike_times, dtype=float).reshape(-1)
    pattern = np.zeros_like(times, dtype=float)
    finite = np.isfinite(times)
    pattern[finite] = 1.0 / (1.0 + times[finite])
    return pattern


def _none_for_inf_list(values: Any) -> list[Any]:
    out: list[Any] = []
    for value in np.asarray(values, dtype=float).reshape(-1):
        out.append(None if not np.isfinite(value) else float(value))
    return out


def run_primitive_hidden_representation_diagnostic(
    *,
    net: MemristiveSNNNetwork,
    seed: int,
    label: str = "post_primitive_core",
    save_json: bool = True,
) -> Dict[str, Any]:
    if not bool(_cfg("ENABLE_PRIMITIVE_HIDDEN_REPRESENTATION_DIAGNOSTIC", True)):
        return {"enabled": False, "label": str(label)}

    stage_names = _primitive_regression_stage_names()
    stage_lookup = {
        str(stage.get("name", "")): dict(stage)
        for stage in _cfg("CURRICULUM_STAGES", [])
        if isinstance(stage, dict)
    }
    reference_actions_raw = _cfg("PRIMITIVE_REFERENCE_FIRST_ACTIONS", {})
    reference_actions = (
        dict(reference_actions_raw) if isinstance(reference_actions_raw, dict) else {}
    )
    stage_records: Dict[str, Any] = {}

    for idx, stage_name in enumerate(stage_names):
        stage = stage_lookup.get(stage_name)
        if stage is None:
            stage_records[stage_name] = {"missing": True}
            continue

        map_name = str(stage["map_name"])
        env = build_env_for_map(map_name, seed=seed + 84000 + idx)
        obs = env.reset(map_index=0)
        obs_for_encoder = _observation_for_phase(obs, f"{stage_name}_hidden_diag")
        env_state = env.get_env_state() if hasattr(env, "get_env_state") else {}

        decision_net = copy.deepcopy(net)
        decision_net.reset_episode()
        decision = decision_net.decide(obs_for_encoder)
        step_records = list(decision.step_records)
        encoded_window = np.asarray(
            [rec.encoder_output.spikes for rec in step_records],
            dtype=int,
        )
        hidden_spike_vectors = [
            np.asarray(rec.hidden_result.spikes, dtype=float).reshape(-1)
            for rec in step_records
        ]
        output_spike_vectors = [
            np.asarray(rec.output_result.spikes, dtype=float).reshape(-1)
            for rec in step_records
        ]
        hidden_spike_counts = (
            np.sum(np.asarray(hidden_spike_vectors, dtype=float), axis=0)
            if hidden_spike_vectors
            else np.zeros(int(getattr(decision_net, "hidden_dim", 0)), dtype=float)
        )
        output_spike_counts = np.asarray(decision.output_spike_counts, dtype=float).reshape(-1)
        hidden_first_spike_times = _first_spike_times_from_step_vectors(
            hidden_spike_vectors,
            int(getattr(decision_net, "hidden_dim", hidden_spike_counts.size)),
        )
        first_output_spike_times = np.asarray(decision.first_spike_steps, dtype=float).reshape(-1)

        hidden_membranes = (
            np.asarray(
                [rec.hidden_result.membrane_potentials for rec in step_records],
                dtype=float,
            )
            if step_records
            else np.zeros((0, int(getattr(decision_net, "hidden_dim", 0))), dtype=float)
        )
        output_currents = (
            np.asarray(
                [rec.output_result.synaptic_currents for rec in step_records],
                dtype=float,
            )
            if step_records
            else np.zeros((0, int(getattr(decision_net, "n_actions", 0))), dtype=float)
        )
        output_current_mean = (
            output_currents.mean(axis=0)
            if output_currents.size
            else np.zeros(int(getattr(decision_net, "n_actions", 0)), dtype=float)
        )
        output_current_max = (
            output_currents.max(axis=0)
            if output_currents.size
            else np.zeros(int(getattr(decision_net, "n_actions", 0)), dtype=float)
        )
        finite_output = np.flatnonzero(np.isfinite(first_output_spike_times))
        first_output_spike_action = (
            int(finite_output[int(np.argmin(first_output_spike_times[finite_output]))])
            if finite_output.size
            else -1
        )
        reference_action = reference_actions.get(stage_name, None)
        reference_action = None if reference_action is None else int(reference_action)

        stage_records[stage_name] = {
            "stage_name": stage_name,
            "map_name": map_name,
            "raw_observation": {
                str(name): float(_safe_float(obs_for_encoder.get(str(name), 0.0), 0.0))
                for name in getattr(net.encoder, "feature_names", [])
                if isinstance(obs_for_encoder, dict)
            },
            "normalized_observation_vector": _normalized_observation_vector(
                net.encoder,
                obs_for_encoder,
            ),
            "initial_robot_position": env_state.get("agent_pos"),
            "initial_heading": env_state.get("agent_heading"),
            "initial_heading_name": env_state.get("agent_heading_name"),
            "encoded_input_spike_pattern": encoded_window.tolist(),
            "encoded_input_spike_pattern_flat": encoded_window.reshape(-1).tolist(),
            "hidden_spike_count_vector": hidden_spike_counts.tolist(),
            "hidden_active_neuron_indices": np.flatnonzero(
                hidden_spike_counts > 0.0
            ).astype(int).tolist(),
            "hidden_first_spike_time_vector": _none_for_inf_list(
                hidden_first_spike_times
            ),
            "hidden_first_spike_pattern": _first_spike_pattern(
                hidden_first_spike_times
            ).tolist(),
            "hidden_membrane_summary": {
                "mean": float(hidden_membranes.mean()) if hidden_membranes.size else 0.0,
                "abs_mean": float(np.mean(np.abs(hidden_membranes))) if hidden_membranes.size else 0.0,
                "std": float(hidden_membranes.std()) if hidden_membranes.size else 0.0,
                "min": float(hidden_membranes.min()) if hidden_membranes.size else 0.0,
                "max": float(hidden_membranes.max()) if hidden_membranes.size else 0.0,
                "final": (
                    hidden_membranes[-1].astype(float).tolist()
                    if hidden_membranes.size
                    else []
                ),
            },
            "output_current_by_action": output_current_mean.astype(float).tolist(),
            "output_current_max_by_action": output_current_max.astype(float).tolist(),
            "output_spike_count_by_action": output_spike_counts.astype(float).tolist(),
            "first_output_spike_action": first_output_spike_action,
            "first_output_spike_time_by_action": _none_for_inf_list(
                first_output_spike_times
            ),
            "winner_action": int(decision.action),
            "selected_step": int(decision.selected_step),
            "selected_by_spike": bool(decision.selected_by_spike),
            "used_fallback": bool(decision.used_fallback),
            "reference_action": reference_action,
        }

    pairwise: Dict[str, Any] = {}
    conflicts: list[Dict[str, Any]] = []
    for stage_a, stage_b in itertools.combinations(stage_names, 2):
        rec_a = stage_records.get(stage_a, {})
        rec_b = stage_records.get(stage_b, {})
        if rec_a.get("missing") or rec_b.get("missing"):
            continue
        hidden_a = np.asarray(rec_a.get("hidden_spike_count_vector", []), dtype=float)
        hidden_b = np.asarray(rec_b.get("hidden_spike_count_vector", []), dtype=float)
        hidden_cosine = _cosine_similarity(hidden_a, hidden_b)
        active_a = set(int(x) for x in rec_a.get("hidden_active_neuron_indices", []))
        active_b = set(int(x) for x in rec_b.get("hidden_active_neuron_indices", []))
        union = active_a | active_b
        hidden_overlap = (
            float(len(active_a & active_b) / max(len(union), 1))
            if union
            else 1.0
        )
        first_a = np.asarray(rec_a.get("hidden_first_spike_pattern", []), dtype=float)
        first_b = np.asarray(rec_b.get("hidden_first_spike_pattern", []), dtype=float)
        hidden_first_similarity = _cosine_similarity(first_a, first_b)
        current_a = np.asarray(rec_a.get("output_current_by_action", []), dtype=float)
        current_b = np.asarray(rec_b.get("output_current_by_action", []), dtype=float)
        output_current_similarity = _cosine_similarity(current_a, current_b)
        ref_a = rec_a.get("reference_action")
        ref_b = rec_b.get("reference_action")
        actions_differ = (
            ref_a is not None and ref_b is not None and int(ref_a) != int(ref_b)
        )
        hidden_alias = (
            hidden_cosine
            > float(_cfg("PRIMITIVE_HIDDEN_ALIAS_COSINE_THRESHOLD", 0.8))
            and hidden_overlap
            >= float(_cfg("PRIMITIVE_HIDDEN_ALIAS_OVERLAP_THRESHOLD", 0.8))
        )
        conflict = bool(actions_differ and hidden_alias)
        item = {
            "stage_a": stage_a,
            "stage_b": stage_b,
            "hidden_spike_cosine_similarity": float(hidden_cosine),
            "hidden_spike_overlap_ratio": float(hidden_overlap),
            "hidden_first_spike_pattern_similarity": float(hidden_first_similarity),
            "output_current_similarity": float(output_current_similarity),
            "reference_action_a": ref_a,
            "reference_action_b": ref_b,
            "ref_actions": [ref_a, ref_b],
            "actions_differ": bool(actions_differ),
            "hidden_alias": bool(hidden_alias),
            "hidden_alias_conflict": bool(conflict),
        }
        key = f"{stage_a}__vs__{stage_b}"
        pairwise[key] = item
        if conflict:
            conflicts.append(item)

    result = {
        "enabled": True,
        "label": str(label),
        "active_encoder_features": list(getattr(net.encoder, "feature_names", [])),
        "hidden_dim": int(getattr(net, "hidden_dim", 0)),
        "stage_records": stage_records,
        "pairwise_hidden_matrix": pairwise,
        "hidden_alias_conflicts": conflicts,
        "hidden_alias_conflict_count": int(len(conflicts)),
        "output_column_status": _output_column_status(net),
        "interpretation": (
            "hidden_alias_conflict=True means primitive inputs are distinguishable "
            "at the encoder but their hidden spike representations overlap while "
            "reference actions differ."
        ),
    }
    if save_json:
        _write_json_log("primitive_hidden_representation_diagnostic.json", result)
    return result


def _print_primitive_observation_alias_diagnostic(
    diagnostic: Dict[str, Any],
) -> None:
    if not isinstance(diagnostic, dict) or not diagnostic.get("enabled", False):
        return

    print("primitive_observation_alias_diagnostic:")
    print(f"  active_encoder_features={diagnostic.get('active_encoder_features', [])}")
    print(
        "  perceptual_alias_conflict_count="
        f"{diagnostic.get('perceptual_alias_conflict_count', 0)}"
    )
    summary = diagnostic.get("primitive_observation_alias_summary", {})
    if isinstance(summary, dict):
        for stage_name, item in summary.items():
            if not isinstance(item, dict):
                continue
            print(
                f"  {stage_name}: ref={item.get('reference_action')} "
                f"first={item.get('first_action')} "
                f"heading={item.get('initial_heading_name')} "
                f"obs={item.get('obs')}"
            )

    pairwise = diagnostic.get("pairwise_alias_matrix", {})
    if isinstance(pairwise, dict) and pairwise:
        print("  pairwise_alias_matrix:")
        for key, item in pairwise.items():
            if not isinstance(item, dict):
                continue
            print(
                f"    {key}: "
                f"l2={item.get('raw_observation_l2_distance')} "
                f"raw_cos={item.get('raw_observation_cosine_similarity')} "
                f"encoded_cos={item.get('encoded_spike_pattern_cosine_similarity')} "
                f"overlap={item.get('active_input_neuron_overlap_ratio')} "
                f"actions=({item.get('reference_action_a')},"
                f"{item.get('reference_action_b')}) "
                f"conflict={item.get('perceptual_alias_conflict')}"
            )

    conflicts = diagnostic.get("perceptual_alias_conflicts", [])
    if conflicts:
        print(f"  perceptual_alias_conflicts={conflicts}")
    else:
        print("  perceptual_alias_conflicts=[]")
    print("  primitive_alias_json: logs\\primitive_observation_alias_diagnostic.json")


def _print_primitive_hidden_representation_diagnostic(
    diagnostic: Dict[str, Any],
) -> None:
    if not isinstance(diagnostic, dict) or not diagnostic.get("enabled", False):
        return

    print("primitive_hidden_representation_diagnostic:")
    print(f"  label={diagnostic.get('label')}")
    print(f"  hidden_dim={diagnostic.get('hidden_dim')}")
    print(
        "  hidden_alias_conflict_count="
        f"{diagnostic.get('hidden_alias_conflict_count', 0)}"
    )
    stage_records = diagnostic.get("stage_records", {})
    if isinstance(stage_records, dict):
        for stage_name, item in stage_records.items():
            if not isinstance(item, dict):
                continue
            print(
                f"  {stage_name}: ref={item.get('reference_action')} "
                f"winner={item.get('winner_action')} "
                f"first_output={item.get('first_output_spike_action')} "
                f"hidden_active={len(item.get('hidden_active_neuron_indices', []))} "
                f"out_spikes={item.get('output_spike_count_by_action')}"
            )

    pairwise = diagnostic.get("pairwise_hidden_matrix", {})
    if isinstance(pairwise, dict) and pairwise:
        print("  pairwise_hidden_matrix:")
        for key, item in pairwise.items():
            if not isinstance(item, dict):
                continue
            print(
                f"    {key}: "
                f"hidden_cos={item.get('hidden_spike_cosine_similarity')} "
                f"overlap={item.get('hidden_spike_overlap_ratio')} "
                f"first_sim={item.get('hidden_first_spike_pattern_similarity')} "
                f"out_current_sim={item.get('output_current_similarity')} "
                f"actions={item.get('ref_actions')} "
                f"conflict={item.get('hidden_alias_conflict')}"
            )
    conflicts = diagnostic.get("hidden_alias_conflicts", [])
    if conflicts:
        print(f"  hidden_alias_conflicts={conflicts}")
    else:
        print("  hidden_alias_conflicts=[]")
    print("  hidden_diag_json: logs\\primitive_hidden_representation_diagnostic.json")


def run_primitive_core_search_sweep(verbose: bool = True) -> Dict[str, Any]:
    encoder = build_encoder()
    _write_active_encoder_feature_reports(encoder)
    seed = int(_cfg("SEED", 42))
    original_scales = copy.deepcopy(_cfg("PRIMITIVE_RSTDP_SCALE", {}))
    original_style = _cfg("PRIMITIVE_CORE_TASK_SAMPLING", "balanced_cycle")
    original_trace_scale = _cfg("PRIMITIVE_CORE_HIDDEN_TRACE_INPUT_SCALE", 0.0)
    original_recurrent_scale = _cfg("PRIMITIVE_CORE_STM_RECURRENT_CURRENT_SCALE", 0.0)
    original_hidden_dim = int(_cfg("NETWORK_HIDDEN_DIM", 8))
    original_hidden_threshold = float(
        _cfg("NETWORK_HIDDEN_BASE_THRESHOLD", _cfg("NEURON_BASE_THRESHOLD", 1.0e-7))
    )
    original_output_wta = bool(_cfg("NETWORK_OUTPUT_ENABLE_WTA", _cfg("NEURON_ENABLE_WTA", False)))
    original_output_threshold = float(
        _cfg("NETWORK_OUTPUT_BASE_THRESHOLD", _cfg("NEURON_BASE_THRESHOLD", 1.0e-7))
    )
    original_depression_scale = float(_cfg("OUTPUT_DEPRESSION_SCALE", 1.0))
    original_anti_target = bool(_cfg("ANTI_TARGET_DEPRESSION", False))
    original_primitive_core_trace_enabled = bool(
        _cfg("PRIMITIVE_CORE_ENABLE_HIDDEN_TRACE", False)
    )
    original_output_interference_diagnostic_enabled = bool(
        _cfg("ENABLE_OUTPUT_COLUMN_INTERFERENCE_DIAGNOSTIC", True)
    )
    original_save_core_summary = bool(
        _cfg("SAVE_PRIMITIVE_CORE_TRAINING_SUMMARY_JSON", True)
    )

    hidden_dims = list(_cfg("PRIMITIVE_CORE_SEARCH_HIDDEN_DIMS", (8, 12, 16)))
    hidden_thresholds = list(
        _cfg("PRIMITIVE_CORE_SEARCH_HIDDEN_BASE_THRESHOLDS", (original_hidden_threshold,))
    )
    output_wta_modes = list(_cfg("PRIMITIVE_CORE_SEARCH_OUTPUT_WTA_MODES", (False, True)))
    output_thresholds = list(
        _cfg("PRIMITIVE_CORE_SEARCH_OUTPUT_BASE_THRESHOLDS", (original_output_threshold,))
    )
    rstdp_scales = list(_cfg("PRIMITIVE_CORE_SEARCH_RSTDP_SCALES", (0.25, 0.5, 1.0)))
    depression_scales = list(
        _cfg("PRIMITIVE_CORE_SEARCH_OUTPUT_DEPRESSION_SCALES", (0.25, 0.5, 1.0))
    )
    anti_target_modes = list(
        _cfg("PRIMITIVE_CORE_SEARCH_ANTI_TARGET_DEPRESSION", (False, True))
    )
    styles = list(_cfg("PRIMITIVE_CORE_SEARCH_REHEARSAL_STYLES", ("balanced_cycle",)))
    trace_scales = list(_cfg("PRIMITIVE_CORE_SEARCH_HIDDEN_TRACE_INPUT_SCALES", (0.0, 0.25)))
    recurrent_scales = list(_cfg("PRIMITIVE_CORE_SEARCH_STM_RECURRENT_CURRENT_SCALES", (0.0,)))
    max_configs = max(1, int(_cfg("PRIMITIVE_CORE_SEARCH_MAX_CONFIGS", 12)))
    stop_on_found = bool(_cfg("PRIMITIVE_CORE_SEARCH_STOP_ON_FOUND", True))

    baseline_config = (
        int(original_hidden_dim),
        float(original_hidden_threshold),
        bool(original_output_wta),
        float(original_output_threshold),
        0.5,
        float(original_depression_scale),
        bool(original_anti_target),
        str(original_style),
        float(original_trace_scale),
        float(original_recurrent_scale),
    )
    candidates = list(
        itertools.product(
            [int(v) for v in hidden_dims],
            [float(v) for v in hidden_thresholds],
            [bool(v) for v in output_wta_modes],
            [float(v) for v in output_thresholds],
            [float(v) for v in rstdp_scales],
            [float(v) for v in depression_scales],
            [bool(v) for v in anti_target_modes],
            [str(v) for v in styles],
            [float(v) for v in trace_scales],
            [float(v) for v in recurrent_scales],
        )
    )
    rng = np.random.default_rng(seed + 70091)
    rng.shuffle(candidates)
    candidates = [baseline_config] + [
        item for item in candidates if item != baseline_config
    ]
    results: list[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    try:
        for config_idx, (
            hidden_dim,
            hidden_threshold,
            output_wta,
            output_threshold,
            scale,
            depression_scale,
            anti_target,
            style,
            trace_scale,
            recurrent_scale,
        ) in enumerate(candidates):
            if config_idx >= max_configs:
                break
            cfg.NETWORK_HIDDEN_DIM = int(hidden_dim)
            cfg.NETWORK_HIDDEN_BASE_THRESHOLD = float(hidden_threshold)
            cfg.NETWORK_OUTPUT_ENABLE_WTA = bool(output_wta)
            cfg.NETWORK_OUTPUT_BASE_THRESHOLD = float(output_threshold)
            cfg.PRIMITIVE_RSTDP_SCALE = {
                name: float(scale) for name in _primitive_regression_stage_names()
            }
            cfg.OUTPUT_DEPRESSION_SCALE = float(depression_scale)
            cfg.ANTI_TARGET_DEPRESSION = bool(anti_target)
            cfg.PRIMITIVE_CORE_TASK_SAMPLING = str(style)
            cfg.PRIMITIVE_CORE_ENABLE_HIDDEN_TRACE = bool(float(trace_scale) > 0.0)
            cfg.PRIMITIVE_CORE_HIDDEN_TRACE_INPUT_SCALE = float(trace_scale)
            cfg.PRIMITIVE_CORE_STM_RECURRENT_CURRENT_SCALE = float(recurrent_scale)
            cfg.ENABLE_OUTPUT_COLUMN_INTERFERENCE_DIAGNOSTIC = False
            cfg.SAVE_PRIMITIVE_CORE_TRAINING_SUMMARY_JSON = False

            net = build_network(encoder=copy.deepcopy(encoder), seed=seed + 71000 + config_idx)
            learner = build_learner()
            _, core_summary = _run_primitive_core_training(
                net=net,
                learner=learner,
                seed=seed + 72000 + config_idx,
                verbose=False,
            )
            regression = core_summary.get("final_regression", {})
            result = {
                "config_index": int(config_idx),
                "hidden_dim": int(hidden_dim),
                "hidden_base_threshold": float(hidden_threshold),
                "output_wta_enabled": bool(output_wta),
                "output_base_threshold": float(output_threshold),
                "primitive_rstdp_scale": float(scale),
                "output_depression_scale": float(depression_scale),
                "anti_target_depression": bool(anti_target),
                "rehearsal_style": str(style),
                "primitive_core_enable_hidden_trace": bool(float(trace_scale) > 0.0),
                "hidden_trace_input_scale": float(trace_scale),
                "stm_recurrent_current_scale": float(recurrent_scale),
                "primitive_core_checkpoint_found": bool(
                    core_summary.get("primitive_core_checkpoint_found", False)
                ),
                "score": float(core_summary.get("final_score", _primitive_core_score(regression))),
                "regression": regression,
                "primitive_action_confusion": regression.get(
                    "primitive_action_confusion",
                    {},
                ),
                "output_bias_diagnostics": core_summary.get("output_bias_diagnostics", {}),
                "output_column_status": core_summary.get("output_column_status", {}),
            }
            results.append(result)
            if best is None or float(result["score"]) > float(best.get("score", -1e18)):
                best = result
            incremental_summary = {
                "results": results,
                "best": best or {},
                "max_configs": int(max_configs),
                "candidate_count": int(len(candidates)),
                "checkpoint_found": bool(
                    any(
                        bool(item.get("primitive_core_checkpoint_found", False))
                        for item in results
                    )
                ),
            }
            _write_json_log(
                "primitive_core_sensor_feature_search_summary.json",
                incremental_summary,
            )
            if verbose:
                print(
                    "primitive_core_sweep "
                    f"idx={config_idx} hidden={hidden_dim} wta={output_wta} "
                    f"hthr={hidden_threshold} othr={output_threshold} "
                    f"scale={scale} dep={depression_scale} "
                    f"anti={anti_target} trace={trace_scale} rec={recurrent_scale} "
                    f"passed={result['primitive_core_checkpoint_found']} "
                    f"score={result['score']}",
                    flush=True,
                )
            if bool(result["primitive_core_checkpoint_found"]) and stop_on_found:
                break
    finally:
        cfg.PRIMITIVE_RSTDP_SCALE = original_scales
        cfg.PRIMITIVE_CORE_TASK_SAMPLING = original_style
        cfg.NETWORK_HIDDEN_DIM = original_hidden_dim
        cfg.NETWORK_HIDDEN_BASE_THRESHOLD = original_hidden_threshold
        cfg.NETWORK_OUTPUT_ENABLE_WTA = original_output_wta
        cfg.NETWORK_OUTPUT_BASE_THRESHOLD = original_output_threshold
        cfg.OUTPUT_DEPRESSION_SCALE = original_depression_scale
        cfg.ANTI_TARGET_DEPRESSION = original_anti_target
        cfg.PRIMITIVE_CORE_ENABLE_HIDDEN_TRACE = original_primitive_core_trace_enabled
        cfg.PRIMITIVE_CORE_HIDDEN_TRACE_INPUT_SCALE = original_trace_scale
        cfg.PRIMITIVE_CORE_STM_RECURRENT_CURRENT_SCALE = original_recurrent_scale
        cfg.ENABLE_OUTPUT_COLUMN_INTERFERENCE_DIAGNOSTIC = (
            original_output_interference_diagnostic_enabled
        )
        cfg.SAVE_PRIMITIVE_CORE_TRAINING_SUMMARY_JSON = original_save_core_summary

    summary = {
        "results": results,
        "best": best or {},
        "max_configs": int(max_configs),
        "candidate_count": int(len(candidates)),
        "checkpoint_found": bool(
            any(bool(item.get("primitive_core_checkpoint_found", False)) for item in results)
        ),
    }
    _write_json_log("primitive_core_search_summary.json", summary)
    _write_json_log("primitive_core_sensor_feature_search_summary.json", summary)
    return summary


def _primitive_regression_from_stage_records(
    stage_records: list[Dict[str, Any]],
    *,
    save_json: bool = True,
) -> Dict[str, Any]:
    stage_by_name = {record.get("stage_name"): record for record in stage_records}
    stages: Dict[str, Any] = {}
    warnings: list[str] = []
    for stage_name in _primitive_regression_stage_names():
        record = stage_by_name.get(stage_name)
        if record is None:
            warnings.append(f"{stage_name}: missing stage record")
            continue
        compact = _compact_eval_summary(record.get("stage_eval", {}))
        stages[stage_name] = compact
        if float(compact.get("eval_success", 0.0)) < 1.0:
            warnings.append(
                f"{stage_name}: eval_success={compact.get('eval_success')} < 1.0"
            )
    passed = len(warnings) == 0 and len(stages) == len(_primitive_regression_stage_names())
    summary = {
        "passed": bool(passed),
        "source": "stage_eval_immediately_after_each_primitive_training_stage",
        "warnings": warnings,
        "criteria": {
            name: "stage eval_success == 1.0 immediately after primitive training"
            for name in _primitive_regression_stage_names()
        },
        "stages": stages,
        "primitive_action_confusion": _primitive_action_confusion(stages),
    }
    summary.update(stages)
    if save_json:
        _maybe_save_json(
            "primitive_stage_immediate_eval_summary.json",
            summary,
            "SAVE_PRIMITIVE_REGRESSION_SUMMARY_JSON",
        )
    return summary


def _run_primitive_rehearsal(
    *,
    net: MemristiveSNNNetwork,
    learner: RewardModulatedSTDPLearner,
    seed: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    if not bool(_cfg("ENABLE_PRIMITIVE_REHEARSAL_AFTER_INTERMEDIATE", True)):
        return {"enabled": False, "stages": {}}

    raw_names = _cfg("PRIMITIVE_REHEARSAL_STAGE_NAMES", _primitive_regression_stage_names())
    stage_names = (raw_names,) if isinstance(raw_names, str) else tuple(str(name) for name in raw_names)
    stage_lookup = {
        str(stage.get("name", "")): dict(stage)
        for stage in _cfg("CURRICULUM_STAGES", [])
        if isinstance(stage, dict)
    }
    scale = float(_cfg("PRIMITIVE_REHEARSAL_TRAIN_EPISODE_SCALE", 1.0))
    cycles = max(1, int(_cfg("PRIMITIVE_REHEARSAL_CYCLES", 1)))
    stages: Dict[str, Any] = {}

    for cycle_idx in range(cycles):
        for idx, stage_name in enumerate(stage_names):
            stage = stage_lookup.get(stage_name)
            if stage is None:
                stages[stage_name] = {"skipped": True, "reason": "missing stage"}
                continue
            map_name = str(stage["map_name"])
            base_train = int(
                stage.get(
                    "train_episodes",
                    _stage_setting(stage_name, "train_episodes", N_EPISODES_TRAIN),
                )
            )
            train_episodes = max(1, int(round(base_train * scale)))
            eval_episodes = int(
                stage.get(
                    "eval_episodes",
                    _stage_setting(stage_name, "eval_episodes", N_EPISODES_EVAL),
                )
            )
            train_env = build_env_for_map(map_name, seed=seed + 5000 + cycle_idx * 100 + idx)
            _, train_summary = run_phase(
                env=train_env,
                net=net,
                learner=learner,
                n_episodes=train_episodes,
                phase_name=f"{stage_name}_train",
                verbose=verbose,
                log_name=f"{stage_name}_rehearsal_cycle_{cycle_idx}",
            )
            eval_env = build_env_for_map(map_name, seed=seed + 6000 + cycle_idx * 100 + idx)
            _, eval_summary = run_phase(
                env=eval_env,
                net=net,
                learner=None,
                n_episodes=eval_episodes,
                phase_name=f"{stage_name}_eval",
                verbose=verbose,
                log_name=f"{stage_name}_rehearsal_eval_cycle_{cycle_idx}",
                trace_first_steps=int(_cfg("EVAL_TRACE_FIRST_STEPS", 20)),
            )
            stages[stage_name] = {
                "map_name": map_name,
                "cycle": int(cycle_idx + 1),
                "train_episodes": int(train_episodes),
                "eval_episodes": int(eval_episodes),
                "train": _compact_eval_summary(train_summary),
                "eval": _compact_eval_summary(eval_summary),
            }

    return {
        "enabled": True,
        "cycles": int(cycles),
        "description": (
            "R-STDP rehearsal of primitive maps after complex curriculum stages. "
            "This is a training protocol only; it does not override actions or "
            "manually set conductance."
        ),
        "stages": stages,
    }


def _forgotten_primitives(
    before: Optional[Dict[str, Any]],
    after: Optional[Dict[str, Any]],
) -> list[str]:
    if not isinstance(before, dict) or not isinstance(after, dict):
        return []
    before_stages = before.get("stages", {})
    after_stages = after.get("stages", {})
    forgotten = []
    for name, before_item in before_stages.items():
        if not isinstance(before_item, dict):
            continue
        before_ok = float(before_item.get("eval_success", 0.0)) >= 1.0
        after_item = after_stages.get(name, {}) if isinstance(after_stages, dict) else {}
        after_ok = isinstance(after_item, dict) and float(after_item.get("eval_success", 0.0)) >= 1.0
        if before_ok and not after_ok:
            forgotten.append(str(name))
    return forgotten


def _interleaved_rehearsal_stage_names(final_stage: bool = False) -> tuple[str, ...]:
    key = "PRIMITIVE_REHEARSAL_STAGE_NAMES" if final_stage else "INTERLEAVED_REHEARSAL_STAGE_NAMES"
    raw = _cfg(key, _primitive_regression_stage_names())
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(name) for name in raw)


def _run_interleaved_primitive_rehearsal(
    *,
    net: MemristiveSNNNetwork,
    learner: RewardModulatedSTDPLearner,
    seed: int,
    after_stage_name: str,
    final_stage: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    if not bool(_cfg("ENABLE_INTERLEAVED_PRIMITIVE_REHEARSAL", True)):
        return {"enabled": False, "after_stage_name": after_stage_name, "stages": {}}
    stage_lookup = {
        str(stage.get("name", "")): dict(stage)
        for stage in _cfg("CURRICULUM_STAGES", [])
        if isinstance(stage, dict)
    }
    episodes = max(1, int(_cfg("REHEARSAL_EPISODES_PER_PRIMITIVE", 2)))
    stages: Dict[str, Any] = {}
    for idx, primitive_name in enumerate(_interleaved_rehearsal_stage_names(final_stage=final_stage)):
        stage = stage_lookup.get(primitive_name)
        if stage is None:
            stages[primitive_name] = {"skipped": True, "reason": "missing stage"}
            continue
        map_name = str(stage["map_name"])
        train_env = build_env_for_map(map_name, seed=seed + 7000 + idx)
        _, train_summary = run_phase(
            env=train_env,
            net=net,
            learner=learner,
            n_episodes=episodes,
            phase_name=f"{primitive_name}_interleaved_rehearsal_train",
            verbose=verbose,
            log_name=f"{primitive_name}_interleaved_after_{after_stage_name}",
        )
        stages[primitive_name] = {
            "map_name": map_name,
            "train_episodes": int(episodes),
            "train": _compact_eval_summary(train_summary),
        }
    return {
        "enabled": True,
        "after_stage_name": str(after_stage_name),
        "final_stage": bool(final_stage),
        "rehearsal_exploration_epsilon": float(
            _cfg("REHEARSAL_EXPLORATION_EPSILON", 0.15)
        ),
        "stages": stages,
    }


def _build_intermediate_curriculum_summary(
    stage_records: list[Dict[str, Any]],
) -> Dict[str, Any]:
    primitive_names = set(_primitive_regression_stage_names())
    stages = {
        record["stage_name"]: {
            "map_name": record.get("map_name"),
            "train": _compact_eval_summary(record.get("stage_train", {})),
            "eval": _compact_eval_summary(record.get("stage_eval", {})),
            "stm_case": record.get("stm_recurrent_interpretation", {}).get("case"),
            "branch_candidate": record.get("branch_candidate", {}),
        }
        for record in stage_records
        if record.get("stage_name") not in primitive_names
    }
    summary = {
        "stage_count": int(len(stages)),
        "stages": stages,
    }
    _maybe_save_json(
        "intermediate_curriculum_summary.json",
        summary,
        "SAVE_INTERMEDIATE_CURRICULUM_SUMMARY_JSON",
    )
    return summary


def _build_stable_wall_avoid_success_summary(
    stage_records: list[Dict[str, Any]],
    primitive_regression_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    primitive_names = _primitive_regression_stage_names()
    stage_by_name = {record["stage_name"]: record for record in stage_records}
    primitive_stage_eval = {
        name: _compact_eval_summary(stage_by_name[name].get("stage_eval", {}))
        for name in primitive_names
        if name in stage_by_name
    }
    immediate_passed = all(
        float(item.get("eval_success", 0.0)) >= 1.0
        for item in primitive_stage_eval.values()
    ) and len(primitive_stage_eval) == len(primitive_names)
    summary = {
        "label": "stable_wall_avoid_success",
        "description": (
            "Primitive success baseline using the selected STM hidden-trace "
            "configuration. This preserves diagnostics only; it does not "
            "hand-set or restore crossbar conductance."
        ),
        "stable_config": {
            "STM_RECURRENT_CURRENT_SCALE": float(_cfg("STM_RECURRENT_CURRENT_SCALE", 0.5)),
            "STM_CONDUCTANCE_SCALE": float(_cfg("STM_CONDUCTANCE_SCALE", 4.0)),
            "ENABLE_CROSS_DECISION_HIDDEN_TRACE": bool(
                _cfg("ENABLE_CROSS_DECISION_HIDDEN_TRACE", True)
            ),
            "HIDDEN_TRACE_DECAY": float(_cfg("HIDDEN_TRACE_DECAY", 0.5)),
            "HIDDEN_TRACE_INPUT_SCALE": float(_cfg("HIDDEN_TRACE_INPUT_SCALE", 2.0)),
        },
        "active_encoder_features": list(_cfg("ENCODER_FEATURE_NAMES", BASE_ENCODER_FEATURE_NAMES)),
        "primitive_stage_eval": primitive_stage_eval,
        "primitive_stage_eval_passed": bool(immediate_passed),
        "primitive_regression_summary": primitive_regression_summary or {},
        "interpretation_note": (
            "cross-decision hidden trace was introduced as an internal STM-like recurrent "
            "memory path. It improved the training pathway and helped break the wall-avoid "
            "failure pattern, but direct causal dependence of the final action sequence on "
            "the instantaneous recurrent current was not fully established because "
            "recurrent-off ablation could produce the same sequence in some cases."
        ),
    }
    _maybe_save_json(
        "stable_wall_avoid_success_summary.json",
        summary,
        "SAVE_STABLE_WALL_AVOID_SUCCESS_SUMMARY_JSON",
    )
    return summary


def _classify_failure_mode(eval_summary: Dict[str, Any]) -> str:
    if float(eval_summary.get("success_rate", 0.0)) > 0.0:
        return "success"
    metrics = _metrics_from_summary(eval_summary)
    action_hist = metrics.get("action_histogram", {})
    total_actions = max(int(sum(int(v) for v in action_hist.values())) if isinstance(action_hist, dict) else 0, 1)
    action0 = _hist_count(action_hist, 0)
    action1 = _hist_count(action_hist, 1)
    action2 = _hist_count(action_hist, 2)
    collision_count = int(eval_summary.get("collision_count", metrics.get("collision_count", 0)))
    moved_count = int(eval_summary.get("moved_count", metrics.get("moved_count", 0)))
    fallback_count = int(sum(int(v) for v in eval_summary.get("fallback_action_histogram", {}).values())) if isinstance(eval_summary.get("fallback_action_histogram", {}), dict) else 0
    no_output_spike_count = int(eval_summary.get("no_output_spike_count", 0))
    seq = [int(x) for x in eval_summary.get("eval_first_episode_action_sequence", []) if int(x) >= 0]
    victim_seq = [
        float(x)
        for x in eval_summary.get("eval_first_episode_victim_signal_sequence", [])
    ]
    unique_cells = float(eval_summary.get("unique_cells_visited_count", 0.0))
    found_victims = int(eval_summary.get("found_victim_count", metrics.get("found_victim_count", 0)))

    if fallback_count > 0.25 * total_actions or no_output_spike_count > 0.25 * total_actions:
        return "output_silence_or_fallback"
    if action0 > 0.5 * total_actions and collision_count > 0:
        return "forward_wall_stuck"
    if max(action1, action2) > 0.7 * total_actions and moved_count <= max(1, int(0.1 * total_actions)):
        return "turn_spin"
    if len(seq) >= 6:
        turn_pairs = sum(
            1
            for a, b in zip(seq, seq[1:])
            if {a, b} == {1, 2}
        )
        if turn_pairs >= 4:
            return "oscillation"
    if moved_count > 0 and found_victims == 0 and victim_seq:
        if max(victim_seq) - min(victim_seq) <= 0.05:
            return "no_victim_tracking"
    if found_victims == 0 and collision_count <= max(1, int(0.1 * total_actions)) and unique_cells <= 3.0:
        return "insufficient_exploration"
    return "unknown"


def _final_eval_diagnostic_summary(eval_summary: Dict[str, Any]) -> Dict[str, Any]:
    metrics = _metrics_from_summary(eval_summary)
    summary = {
        "success_rate": float(eval_summary.get("success_rate", 0.0)),
        "partial_success_rate": float(eval_summary.get("partial_success_rate", 0.0)),
        "all_victims_success_rate": float(
            eval_summary.get("all_victims_success_rate", eval_summary.get("success_rate", 0.0))
        ),
        "found_victim_count": int(eval_summary.get("found_victim_count", metrics.get("found_victim_count", 0))),
        "total_victim_count": int(eval_summary.get("total_victim_count", 0)),
        "total_victim_count_all_episodes": int(
            eval_summary.get("total_victim_count", 0)
            * eval_summary.get("n_episodes", 1)
        ),
        "collision_count": int(eval_summary.get("collision_count", metrics.get("collision_count", 0))),
        "moved_count": int(eval_summary.get("moved_count", metrics.get("moved_count", 0))),
        "mean_steps": float(eval_summary.get("mean_steps", 0.0)),
        "action_histogram": metrics.get("action_histogram", {}),
        "snn_action_histogram": metrics.get("snn_action_histogram", {}),
        "spike_selected_action_histogram": eval_summary.get("spike_selected_action_histogram", {}),
        "fallback_action_histogram": eval_summary.get("fallback_action_histogram", {}),
        "output_winner_histogram": eval_summary.get("output_winner_histogram", {}),
        "first_episode_action_sequence": eval_summary.get("eval_first_episode_action_sequence", []),
        "first_episode_raw_action_sequence": eval_summary.get(
            "eval_first_episode_raw_action_sequence", []
        ),
        "no_action_count": int(eval_summary.get("eval_first_episode_no_action_count", 0)),
        "no_spike_count": int(eval_summary.get("eval_first_episode_no_spike_count", 0)),
        "early_done_before_action": bool(
            eval_summary.get("eval_first_episode_early_done_before_action", False)
        ),
        "valid_actions_only": bool(eval_summary.get("valid_actions_only", True)),
        "eval_state_reset_debug": eval_summary.get("eval_state_reset_debug", {}),
        "first_episode_collision_sequence": eval_summary.get("eval_first_episode_collision_sequence", []),
        "first_episode_moved_sequence": eval_summary.get("eval_first_episode_moved_sequence", []),
        "first_episode_victim_signal_sequence": eval_summary.get(
            "eval_first_episode_victim_signal_sequence", []
        ),
        "first_episode_front_clearance_sequence": eval_summary.get(
            "eval_first_episode_front_clearance_sequence", []
        ),
        "collision_free_success_rate": float(eval_summary.get("collision_free_success_rate", 0.0)),
        "victims_found_per_collision": float(eval_summary.get("victims_found_per_collision", 0.0)),
        "victims_found_per_step": float(eval_summary.get("victims_found_per_step", 0.0)),
        "repeated_turn_count": int(eval_summary.get("repeated_turn_count", 0)),
        "repeated_forward_collision_count": int(
            eval_summary.get("repeated_forward_collision_count", 0)
        ),
        "unique_cells_visited_count": float(eval_summary.get("unique_cells_visited_count", 0.0)),
        "unique_cells_visited_count_per_episode": eval_summary.get(
            "unique_cells_visited_count_per_episode", []
        ),
        "last_unfound_victim_distance_if_available": eval_summary.get(
            "last_unfound_victim_distance_if_available", None
        ),
        "stopped_due_to_wall_stuck": bool(
            eval_summary.get("stopped_due_to_wall_stuck", False)
        ),
        "found_order": eval_summary.get("found_order", []),
        "found_step_indices": eval_summary.get("found_step_indices", []),
        "found_order_per_episode": eval_summary.get("found_order_per_episode", []),
        "found_step_indices_per_episode": eval_summary.get(
            "found_step_indices_per_episode", []
        ),
        "stm_trace_summary": {
            "recurrent_memory_scope": eval_summary.get("recurrent_memory_scope", "unknown"),
            "hidden_trace_enabled": bool(
                eval_summary.get("enable_cross_decision_hidden_trace", False)
            ),
            "hidden_trace_decay": float(eval_summary.get("hidden_trace_decay", 0.0)),
            "hidden_trace_input_scale": float(
                eval_summary.get("hidden_trace_input_scale", 0.0)
            ),
            "hidden_trace_mean": float(eval_summary.get("hidden_trace_mean", 0.0)),
            "hidden_trace_nonzero_count": float(
                eval_summary.get("hidden_trace_nonzero_count", 0.0)
            ),
            "recurrent_to_input_current_ratio": float(
                eval_summary.get("mean_recurrent_to_input_current_ratio", 0.0)
            ),
            "recurrent_to_input_current_ratio_by_action": eval_summary.get(
                "recurrent_to_input_current_ratio_by_action", {}
            ),
            "recurrent_crossbar_status": eval_summary.get("recurrent_crossbar_status", {}),
        },
        "output_action_bias_diagnostics": {
            "output_spike_count_by_action": eval_summary.get(
                "output_spike_count_by_action", {}
            ),
            "output_first_spike_time_by_action_mean": eval_summary.get(
                "output_first_spike_time_by_action_mean", {}
            ),
            "output_column_weight_mean": eval_summary.get(
                "output_column_weight_mean", {}
            ),
            "output_column_weight_std": eval_summary.get(
                "output_column_weight_std", {}
            ),
            "output_column_effective_score": eval_summary.get(
                "output_column_effective_score", {}
            ),
            "pulse_plus_count_by_action": eval_summary.get(
                "potentiation_action_histogram", {}
            ),
            "pulse_minus_count_by_action": eval_summary.get(
                "depression_action_histogram", {}
            ),
            "reward_by_action": eval_summary.get("action_mean_reward", {}),
            "event_count_by_action": eval_summary.get("action_event_counts", {}),
            "collision_by_action": {
                str(action): int(values.get("collision", 0))
                for action, values in eval_summary.get("action_event_counts", {}).items()
                if isinstance(values, dict)
            },
            "found_by_action": {
                str(action): int(values.get("found_victim", 0))
                for action, values in eval_summary.get("action_event_counts", {}).items()
                if isinstance(values, dict)
            },
        },
    }
    summary["failure_mode"] = _classify_failure_mode(eval_summary)
    return summary


def _score_final_checkpoint_candidate(
    final_diag: Dict[str, Any],
    primitive_passed: bool,
) -> float:
    found = float(final_diag.get("found_victim_count", 0.0))
    total = max(float(final_diag.get("total_victim_count", 0.0)), 1.0)
    success = float(final_diag.get("all_victims_success_rate", final_diag.get("success_rate", 0.0)))
    collisions = float(final_diag.get("collision_count", 0.0))
    unique_cells = float(final_diag.get("unique_cells_visited_count", 0.0))
    failure_mode = str(final_diag.get("failure_mode", "unknown"))
    primitive_bonus = 1000.0 if bool(primitive_passed) else -100000.0
    wall_stuck_penalty = 50.0 if failure_mode == "forward_wall_stuck" else 0.0
    fallback_penalty = 50.0 if failure_mode == "output_silence_or_fallback" else 0.0
    action_hist = final_diag.get("action_histogram", {})
    total_actions = max(
        int(sum(int(v) for v in action_hist.values())) if isinstance(action_hist, dict) else 0,
        1,
    )
    action0_count = _hist_count(action_hist, 0)
    action_collapse_penalty = 50.0 if action0_count / float(total_actions) > 0.85 else 0.0
    return (
        primitive_bonus
        + 200.0 * success
        + 20.0 * found
        + 2.0 * unique_cells
        - 1.0 * collisions
        - action_collapse_penalty
        - wall_stuck_penalty
        - fallback_penalty
    )


def _probe_final_checkpoint(
    *,
    net: MemristiveSNNNetwork,
    seed: int,
    label: str,
    final_eval_map_name: str,
    primitive_passed: bool,
) -> Dict[str, Any]:
    probe_env = build_env_for_map(final_eval_map_name, seed=seed)
    _, probe_summary = run_phase(
        env=probe_env,
        net=copy.deepcopy(net),
        learner=None,
        n_episodes=max(1, int(_cfg("FINAL_MAP_CHECKPOINT_PROBE_EPISODES", 1))),
        phase_name="final_eval_probe",
        verbose=False,
        log_name=f"final_eval_probe_{_safe_log_name(label)}",
        trace_first_steps=int(_cfg("EVAL_TRACE_FIRST_STEPS", 20)),
    )
    final_diag = _final_eval_diagnostic_summary(probe_summary)
    return {
        "label": str(label),
        "primitive_passed": bool(primitive_passed),
        "score": float(_score_final_checkpoint_candidate(final_diag, primitive_passed)),
        "final_eval_summary": final_diag,
    }


def _candidate_improves(
    candidate: Dict[str, Any],
    current_best: Optional[Dict[str, Any]],
) -> bool:
    if current_best is None:
        return True
    cand = candidate.get("final_eval_summary", {})
    best = current_best.get("final_eval_summary", {})
    cand_found = int(cand.get("found_victim_count", 0))
    best_found = int(best.get("found_victim_count", 0))
    if cand_found >= best_found + int(_cfg("BRANCH_ACCEPT_FOUND_IMPROVEMENT", 1)):
        return True
    cand_collision = int(cand.get("collision_count", 0))
    best_collision = int(best.get("collision_count", 0))
    if cand_found >= best_found and cand_collision <= best_collision - int(
        _cfg("BRANCH_ACCEPT_COLLISION_IMPROVEMENT", 10)
    ):
        return True
    return float(candidate.get("score", -1e18)) > float(current_best.get("score", -1e18))


def _default_complex_stage_enabled(stage_name: str) -> bool:
    raw = _cfg("DEFAULT_COMPLEX_STAGE_ENABLED", {})
    if not isinstance(raw, dict):
        return True
    return bool(raw.get(str(stage_name), True))


def _branch_reject_reasons(
    *,
    primitive_passed: bool,
    score_improved: bool,
) -> list[str]:
    reasons: list[str] = []
    if bool(_cfg("COMMIT_ONLY_IF_PRIMITIVE_PASSED", True)) and not primitive_passed:
        reasons.append("primitive_regression_failed")
    if bool(_cfg("COMMIT_ONLY_IF_SCORE_IMPROVES", True)) and not score_improved:
        reasons.append("final_probe_score_not_improved")
    return reasons


def _save_positive_reward_steps(log_name: str, steps: Any) -> Optional[Path]:
    if not bool(_cfg("SAVE_POSITIVE_REWARD_STEPS_JSON", True)):
        return None
    filename = f"positive_reward_steps_{_safe_log_name(log_name)}.json"
    return _write_json_log(filename, steps)


def _relax_stm_crossbars(net: MemristiveSNNNetwork, dt_s: float) -> None:
    """Relax volatile STM arrays between environment steps when available.

    For pure grid simulation this is optional, but when recurrent STM devices
    are modeled as volatile physical devices, a decision/action step should
    correspond to some elapsed time.  Crossbars without relax_all() are ignored.
    """
    if dt_s <= 0.0:
        return

    # By default, relax only the recurrent path because that is the intended STM path.
    names = ["recurrent_hidden_crossbar"]
    if bool(_cfg("NETWORK_RELAX_ALL_STM_CROSSBARS", False)):
        names = ["input_hidden_crossbar", "recurrent_hidden_crossbar", "output_crossbar"]

    for name in names:
        cb = getattr(net, name, None)
        relax_all = getattr(cb, "relax_all", None)
        if callable(relax_all):
            relax_all(float(dt_s))


def run_episode(
    env: AbstractRescueGridEnv,
    net: MemristiveSNNNetwork,
    learner: Optional[RewardModulatedSTDPLearner],
    metrics: Optional[SNNMetrics] = None,
    episode_idx: int = 0,
    phase_name: str = "train",
    verbose: bool = False,
    trace_first_steps: int = 0,
) -> Dict[str, Any]:
    map_index = _fixed_map_index_for_episode(env, episode_idx)
    obs = env.reset(map_index=map_index) if map_index is not None else env.reset()
    obs = _observation_for_phase(obs, phase_name)
    net.reset_episode()
    eval_phase = _is_eval_phase(phase_name)
    reset_hidden_trace = bool(
        eval_phase and _cfg("RESET_HIDDEN_TRACE_ON_EVAL_EPISODE_START", True)
    )
    reset_neuron_state = bool(
        eval_phase and _cfg("RESET_NEURON_STATE_ON_EVAL_EPISODE_START", True)
    )
    reset_action_history = bool(
        eval_phase and _cfg("RESET_ACTION_HISTORY_ON_EVAL_EPISODE_START", True)
    )
    if reset_neuron_state:
        for layer_name in ("hidden_layer", "recurrent_layer", "output_layer"):
            layer = getattr(net, layer_name, None)
            if layer is not None and hasattr(layer, "reset_state"):
                layer.reset_state(reset_threshold_devices=False)
        if hasattr(net, "prev_hidden_spikes"):
            net.prev_hidden_spikes.fill(0.0)
    if reset_hidden_trace and hasattr(net, "hidden_trace"):
        net.hidden_trace.fill(0.0)
    if reset_action_history and hasattr(net, "action_history"):
        net.action_history.clear()
    eval_state_reset_debug = {
        "hidden_trace_reset": bool(reset_hidden_trace),
        "neuron_state_reset": bool(reset_neuron_state),
        "prev_hidden_spikes_reset": bool(reset_neuron_state),
        "output_buffer_reset": bool(reset_neuron_state),
        "action_history_reset": bool(reset_action_history),
    }
    original_hidden_trace_enabled = bool(
        getattr(net, "enable_cross_decision_hidden_trace", False)
    )
    original_hidden_trace_input_scale = float(
        getattr(net, "hidden_trace_input_scale", _cfg("HIDDEN_TRACE_INPUT_SCALE", 1.0))
    )
    original_stm_recurrent_current_scale = float(
        getattr(net, "stm_recurrent_current_scale", _cfg("STM_RECURRENT_CURRENT_SCALE", 1.0))
    )
    if "primitive_core" in str(phase_name).lower():
        net.stm_recurrent_current_scale = float(
            _cfg("PRIMITIVE_CORE_STM_RECURRENT_CURRENT_SCALE", original_stm_recurrent_current_scale)
        )
    stage_hidden_trace_input_scale = _stage_hidden_trace_input_scale(phase_name)
    net.enable_cross_decision_hidden_trace = bool(
        _cross_decision_hidden_trace_enabled_for_phase(phase_name)
    )
    if hasattr(net, "set_hidden_trace_params"):
        net.set_hidden_trace_params(
            input_scale=float(stage_hidden_trace_input_scale),
            enabled=bool(net.enable_cross_decision_hidden_trace),
        )
    else:
        net.hidden_trace_input_scale = float(stage_hidden_trace_input_scale)
    original_force_action_on_no_spike = bool(
        getattr(net, "force_action_on_no_spike", True)
    )
    if eval_phase:
        net.force_action_on_no_spike = bool(
            _cfg("EVAL_FORCE_ACTION_ON_NO_SPIKE", original_force_action_on_no_spike)
        )
    initial_state = env.get_env_state() if hasattr(env, "get_env_state") else {}

    done = False
    total_reward = 0.0
    step_idx = 0
    fallback_count = 0
    found_victim_count = 0
    found_step_indices: list[int] = []
    found_order: list[Any] = []
    positive_reward_steps = []
    eval_trace_steps = []
    last_step_info: Dict[str, Any] = {}
    distance_values = [
        _safe_float(initial_state.get("last_distance_to_victim", 0.0), 0.0)
    ]
    action_stats = _build_action_stats(env)
    stm_relax_dt_s = float(_cfg("SIM_ENV_STEP_DT_S", _cfg("ENV_STEP_DT_S", 0.0)))
    relax_stm = bool(_cfg("NETWORK_RELAX_STM_BETWEEN_ENV_STEPS", True))
    step_debug = bool(verbose) or bool(_cfg("PRINT_STEP_DEBUG", False))
    enable_delayed_action_credit = bool(_cfg("ENABLE_DELAYED_ACTION_CREDIT", False))
    enable_previous_turn_credit = bool(_cfg("ENABLE_PREVIOUS_TURN_CREDIT", False))
    stage_rstdp_scale = _stage_rstdp_scale(phase_name)
    recent_decisions = None
    if enable_delayed_action_credit:
        recent_credit_window = max(1, int(_cfg("RECENT_CREDIT_WINDOW", 3)))
        recent_decisions = deque(maxlen=recent_credit_window)
    delayed_credit_updates = []
    delayed_credit_update_count = 0
    delayed_credit_action_histogram: Dict[int, int] = {}
    direct_credit_action_histogram: Dict[int, int] = {}
    previous_turn_credit_action_histogram: Dict[int, int] = {}
    is_wall_avoid_phase = _is_wall_avoid_phase(phase_name)
    previous_step_collision = False
    wall_avoid_forward_positive_count = 0
    wall_avoid_forward_recovery_count = 0
    wall_avoid_turn_reward_action_counts: Dict[int, int] = {}
    wall_avoid_turn_reward_total_count = 0
    wall_avoid_spin_penalty_count = 0
    last_executed_action: Optional[int] = None
    consecutive_same_turn_count = 0
    consecutive_forward_collision_count = 0
    episode_output_spike_total = 0
    no_output_spike_count = 0
    episode_collision_count = 0
    episode_moved_count = 0
    repeated_turn_count = 0
    repeated_forward_collision_count = 0
    visited_cells = set()
    initial_agent_pos = initial_state.get("agent_pos")
    if initial_agent_pos is not None:
        try:
            visited_cells.add(tuple(initial_agent_pos))
        except TypeError:
            pass

    if step_debug:
        print("=" * 70)
        print(f"{phase_name.upper()} EPISODE {episode_idx}")
        print("=" * 70)
        if map_index is not None:
            print(f"map_index: {map_index}")
        if initial_state:
            print("env_state:", initial_state)
        print("initial obs:", obs)
        print(env.render_ascii())
        print()

    while not done:
        prev_obs = dict(obs) if isinstance(obs, dict) else obs
        decision = net.decide(obs)
        selected_by_spike = bool(
            getattr(
                decision,
                "selected_by_spike",
                int(decision.action) >= 0 and not bool(decision.used_fallback),
            )
        )
        selected_mode = str(getattr(decision, "selection_mode", "first_spike"))
        no_action_due_to_no_spike = bool(int(decision.action) < 0)
        output_spike_counts = _decision_output_spike_counts(decision)
        first_spike_step = _decision_first_spike_steps(decision)
        output_threshold_mean = _decision_output_mean(decision, "thresholds")
        output_current_mean = _decision_output_mean(decision, "synaptic_currents")
        hidden_current_stats = _decision_hidden_current_stats(decision)
        output_spike_total = int(sum(output_spike_counts))
        episode_output_spike_total += output_spike_total
        if output_spike_total == 0:
            no_output_spike_count += 1

        snn_action = int(decision.action)
        if no_action_due_to_no_spike:
            exploration = {
                "executed_action": -1,
                "exploration_used": False,
                "exploration_epsilon": 0.0,
            }
        else:
            exploration = _maybe_apply_train_exploration(
                phase_name=phase_name,
                learner=learner,
                snn_action=snn_action,
                rng=getattr(net, "rng", getattr(env, "rng", np.random.default_rng())),
            )
        executed_action = int(exploration["executed_action"])
        action_was_explored = executed_action != snn_action
        previous_executed_action = last_executed_action
        if executed_action in (1, 2):
            if previous_executed_action == executed_action:
                consecutive_same_turn_count += 1
            else:
                consecutive_same_turn_count = 1
        else:
            consecutive_same_turn_count = 0
        previous_distance_to_victim = _safe_float(
            getattr(env, "last_distance_to_victim", distance_values[-1]),
            distance_values[-1],
        )
        if no_action_due_to_no_spike:
            env_state = env.get_env_state() if hasattr(env, "get_env_state") else {}
            step = SimpleNamespace(
                observation=prev_obs,
                reward=0.0,
                done=True,
                info={
                    "no_action_due_to_no_spike": True,
                    "found_victim": False,
                    "collision": False,
                    "moved": False,
                    "rescued_count": int(env_state.get("rescued_count", 0)),
                    "remaining_victims": int(
                        env_state.get(
                            "remaining_victims",
                            len(getattr(env, "victim_positions", [])),
                        )
                    ),
                    "distance_to_nearest_victim": _safe_float(
                        env_state.get(
                            "last_distance_to_victim",
                            previous_distance_to_victim,
                        ),
                        previous_distance_to_victim,
                    ),
                },
            )
        else:
            step = env.step(executed_action)
        next_obs = _observation_for_phase(step.observation, phase_name)
        step_info = dict(step.info or {})
        step_info["raw_reward"] = float(step.reward)
        action_names = getattr(env, "ACTION_NAMES", {})
        step_info.update(
            {
                "selected_action": snn_action,
                "selected_mode": selected_mode,
                "selected_by_spike": bool(selected_by_spike),
                "fallback_used": bool(decision.used_fallback),
                "snn_action": snn_action,
                "snn_action_name": action_names.get(snn_action, str(snn_action)),
                "executed_action": executed_action,
                "executed_action_name": action_names.get(executed_action, str(executed_action)),
                "action_was_explored": bool(action_was_explored),
                "exploration_used": bool(exploration["exploration_used"]),
                "exploration_epsilon": float(exploration["exploration_epsilon"]),
                "no_action_due_to_no_spike": bool(no_action_due_to_no_spike),
                "no_output_spike": bool(output_spike_total == 0),
                "fallback_scores": _as_float_list(decision.output_scores_fallback),
                "output_scores": (
                    _as_float_list(decision.output_scores_fallback)
                    if decision.output_scores_fallback is not None
                    else output_current_mean
                ),
                "output_spike_counts": output_spike_counts,
                "first_spike_step": first_spike_step,
                "output_threshold_mean_per_action": output_threshold_mean,
                "output_current_mean_per_action": output_current_mean,
                **hidden_current_stats,
                "output_spike_total": int(output_spike_total),
            }
        )
        step_info["front_clearance_before_action"] = _safe_float(
            prev_obs.get("front_clearance", 1.0) if isinstance(prev_obs, dict) else 1.0,
            1.0,
        )
        step_info["front_clearance"] = _safe_float(
            next_obs.get("front_clearance", 1.0) if isinstance(next_obs, dict) else 1.0,
            1.0,
        )
        step_info["post_step_distance_to_victim"] = _safe_float(
            getattr(env, "last_distance_to_victim", step_info.get("distance_to_nearest_victim", 0.0)),
            0.0,
        )
        step_info["pre_step_front_clearance"] = _safe_float(
            prev_obs.get("front_clearance", 1.0) if isinstance(prev_obs, dict) else 1.0,
            1.0,
        )
        step_info["is_wall_avoid_phase"] = bool(is_wall_avoid_phase)
        step_info["previous_step_collision"] = bool(previous_step_collision)
        step_info["wall_avoid_forward_positive_count_before"] = int(
            wall_avoid_forward_positive_count
        )
        step_info["wall_avoid_forward_positive_count"] = int(
            wall_avoid_forward_positive_count
        )
        step_info["wall_avoid_forward_recovery_count"] = int(
            wall_avoid_forward_recovery_count
        )
        step_info["wall_avoid_turn_reward_action_count_before"] = int(
            wall_avoid_turn_reward_action_counts.get(executed_action, 0)
        )
        step_info["wall_avoid_turn_reward_action_count"] = int(
            wall_avoid_turn_reward_action_counts.get(executed_action, 0)
        )
        step_info["wall_avoid_turn_reward_total_count"] = int(
            wall_avoid_turn_reward_total_count
        )
        step_info["last_executed_action"] = (
            None if previous_executed_action is None else int(previous_executed_action)
        )
        step_info["consecutive_same_turn_count"] = int(consecutive_same_turn_count)
        step_info["wall_avoid_spin_penalty_count"] = int(wall_avoid_spin_penalty_count)
        step_info["enable_turn_sensor_positive_reward"] = (
            _turn_sensor_positive_reward_enabled(phase_name)
        )
        current_pos = step_info.get("new_pos", None)
        if current_pos is None and hasattr(env, "get_env_state"):
            current_pos = env.get_env_state().get("agent_pos")
        if current_pos is not None:
            try:
                visited_cells.add(tuple(current_pos))
            except TypeError:
                pass
        step_collision = bool(step_info.get("collision", False))
        step_moved = bool(step_info.get("moved", False))
        episode_collision_count += int(step_collision)
        episode_moved_count += int(step_moved)
        if executed_action in (1, 2) and previous_executed_action == executed_action:
            repeated_turn_count += 1
        if executed_action == 0 and step_collision:
            repeated_forward_collision_count += 1
            consecutive_forward_collision_count += 1
        elif executed_action == 0:
            consecutive_forward_collision_count = 0
        step_info["stage_rstdp_scale"] = float(stage_rstdp_scale)
        step_info["stage_hidden_trace_input_scale"] = float(stage_hidden_trace_input_scale)
        step_info["consecutive_forward_collision_count"] = int(
            consecutive_forward_collision_count
        )
        step_info["is_final_map_train_phase"] = bool(_is_final_map_train_phase(phase_name))
        learning_signal = compute_learning_signal(
            info=step_info,
            previous_distance_to_victim=previous_distance_to_victim,
            prev_obs=prev_obs if isinstance(prev_obs, dict) else None,
            next_obs=next_obs if isinstance(next_obs, dict) else None,
            executed_action=executed_action,
        )
        learning_reward = float(learning_signal["reward"])
        learning_target = learning_signal.get("target")
        learning_reason = str(learning_signal.get("reason", "none"))
        if float(learning_reward) != 0.0 and _is_train_phase(phase_name):
            effective_rstdp_scale = (
                float(_cfg("FINAL_MAP_FORWARD_COLLISION_RSTDP_SCALE", 1.0))
                if learning_reason == "final_map_forward_collision"
                else float(stage_rstdp_scale)
            )
            learning_reward *= float(effective_rstdp_scale)
            learning_signal["reward"] = float(learning_reward)
        else:
            effective_rstdp_scale = float(stage_rstdp_scale)
        if learning_reason == "wall_avoid_forward_collision" and bool(decision.used_fallback):
            learning_reward *= float(
                _cfg("WALL_AVOID_FALLBACK_COLLISION_PENALTY_FACTOR", 1.0)
            )
            learning_signal["reward"] = float(learning_reward)
        step_info["learning_reward"] = float(learning_reward)
        step_info["learning_target"] = (
            None if learning_target is None else int(learning_target)
        )
        step_info["learning_reason"] = learning_reason
        step_info["turn_sensor_reward_applied"] = bool(
            learning_signal.get("turn_sensor_reward_applied", False)
        )
        step_info["effective_rstdp_scale"] = float(effective_rstdp_scale)
        step_info["turn_sensor_penalty_applied"] = bool(
            learning_signal.get("turn_sensor_penalty_applied", False)
        )
        step_info["wall_avoid_forward_collision"] = bool(
            learning_signal.get("wall_avoid_forward_collision", False)
        )
        step_info["wall_avoid_forward_after_clearance"] = bool(
            learning_signal.get("wall_avoid_forward_after_clearance", False)
        )
        step_info["wall_avoid_forward_positive_applied"] = bool(
            learning_signal.get("wall_avoid_forward_positive_applied", False)
        )
        step_info["wall_avoid_forward_positive_skipped"] = bool(
            learning_signal.get("wall_avoid_forward_positive_skipped", False)
        )
        step_info["wall_avoid_turn_reward_applied"] = bool(
            learning_signal.get("wall_avoid_turn_reward_applied", False)
        )
        step_info["wall_avoid_repeated_turn_spin"] = bool(
            learning_signal.get("wall_avoid_repeated_turn_spin", False)
        )
        if step_info["wall_avoid_forward_positive_applied"]:
            wall_avoid_forward_positive_count += 1
        if step_info["wall_avoid_forward_after_clearance"]:
            wall_avoid_forward_recovery_count += 1
        if step_info["wall_avoid_turn_reward_applied"]:
            wall_avoid_turn_reward_action_counts[executed_action] = (
                wall_avoid_turn_reward_action_counts.get(executed_action, 0) + 1
            )
            wall_avoid_turn_reward_total_count += 1
        if step_info["wall_avoid_repeated_turn_spin"]:
            wall_avoid_spin_penalty_count += 1
        last_executed_action = int(executed_action)
        last_step_info = step_info
        decision_record = {
            "decision": decision,
            "snn_action": int(snn_action),
            "executed_action": int(executed_action),
            "action_was_explored": bool(action_was_explored),
            "used_fallback": bool(decision.used_fallback),
            "observation": dict(obs) if isinstance(obs, dict) else obs,
            "step_idx": int(step_idx),
        }
        if enable_delayed_action_credit and recent_decisions is not None:
            recent_decisions.append(decision_record)

        if decision.used_fallback:
            fallback_count += 1
        if bool(step_info.get("found_victim", False)):
            found_victim_count += 1
            found_step_indices.append(int(step_idx))
            found_order.append(step_info.get("found_victim_pos", None))

        if "distance_to_nearest_victim" in step_info:
            distance_values.append(
                _safe_float(step_info.get("distance_to_nearest_victim"), 0.0)
            )
        _update_action_stats(
            stats=action_stats,
            action=executed_action,
            reward=float(step.reward),
            info=step_info,
        )

        found_victim = bool(step_info.get("found_victim", False))
        collision = bool(step_info.get("collision", False))
        learning_events = None
        output_event = None
        target = None
        step_delayed_credit_updates = []
        step_delayed_credit_action_histogram: Dict[int, int] = {}
        step_direct_credit_action_histogram: Dict[int, int] = {}
        step_previous_turn_credit_action_histogram: Dict[int, int] = {}
        current_skip_message = None

        def apply_positive_credit(
            *,
            credit_record: Dict[str, Any],
            credit_reward: float,
            credit_target: int,
            reason: str,
            credit_type: str,
        ) -> tuple[Optional[Dict[str, Optional[Any]]], Optional[Any], Dict[str, Any]]:
            balance_status = _output_column_balance_status(
                net=net,
                target_action=int(credit_target),
                reward=float(credit_reward),
            )
            skip_message = None
            credit_events = None
            credit_output_event = None
            credit_applied = False
            if bool(balance_status.get("blocked", False)):
                skip_message = (
                    "skipped by output column balance "
                    f"(target={int(credit_target)}, "
                    f"column_mean={balance_status.get('column_mean')}, "
                    f"threshold={balance_status.get('balance_threshold')})"
                )
                event_record = _output_learning_event_record(
                    None,
                    skip_message=skip_message,
                )
            else:
                credit_events = _learn_from_recorded_decision(
                    net=net,
                    learner=learner,
                    decision=credit_record["decision"],
                    reward=float(credit_reward),
                    target=int(credit_target),
                )
                credit_output_event = credit_events.get("output")
                event_record = _output_learning_event_record(credit_output_event)
                credit_applied = True

            debug_record = {
                "source_event": "found_victim",
                "reason": str(reason),
                "credit_type": str(credit_type),
                "episode_idx": int(episode_idx),
                "original_step": int(step_idx),
                "credit_step": int(credit_record["step_idx"]),
                "snn_action": int(credit_record["snn_action"]),
                "executed_action": int(credit_record["executed_action"]),
                "action_was_explored": bool(credit_record["action_was_explored"]),
                "used_fallback": bool(credit_record["used_fallback"]),
                "credit_reward": float(credit_reward),
                "target": int(credit_target),
                "credit_applied": bool(credit_applied),
                "skipped_by_output_column_balance": bool(balance_status.get("blocked", False)),
                "output_column_balance": balance_status,
                **event_record,
            }
            if skip_message is not None:
                debug_record["learning_event_message"] = skip_message
            return credit_events, credit_output_event, debug_record

        if (
            enable_delayed_action_credit
            and recent_decisions is not None
            and learner is not None
            and found_victim
            and float(learning_reward) > 0.0
        ):
            current_record = recent_decisions[-1]
            if not bool(current_record.get("used_fallback", False)) and int(executed_action) == 0:
                target = int(executed_action)
                learning_events, output_event, direct_record = apply_positive_credit(
                    credit_record=current_record,
                    credit_reward=float(learning_reward),
                    credit_target=target,
                    reason="direct_found_action",
                    credit_type="direct",
                )
                step_delayed_credit_updates.append(direct_record)
                delayed_credit_updates.append(direct_record)
                if bool(direct_record.get("credit_applied", False)):
                    delayed_credit_update_count += 1
                    _hist_increment(delayed_credit_action_histogram, target)
                    _hist_increment(step_delayed_credit_action_histogram, target)
                    _hist_increment(direct_credit_action_histogram, target)
                    _hist_increment(step_direct_credit_action_histogram, target)
                else:
                    current_skip_message = str(direct_record.get("learning_event_message", "credit skipped"))

                if step_debug or bool(_cfg("PRINT_POSITIVE_REWARD_STEPS", False)):
                    _print_delayed_credit_update(direct_record)
            else:
                current_skip_message = "skipped_recent_action_not_causal"
                skipped_record = {
                    "source_event": "found_victim",
                    "reason": "skipped_recent_action_not_causal",
                    "credit_type": "direct",
                    "episode_idx": int(episode_idx),
                    "original_step": int(step_idx),
                    "credit_step": int(current_record["step_idx"]),
                    "snn_action": int(current_record["snn_action"]),
                    "executed_action": int(current_record["executed_action"]),
                    "action_was_explored": bool(current_record["action_was_explored"]),
                    "used_fallback": bool(current_record["used_fallback"]),
                    "credit_reward": 0.0,
                    "target": int(current_record["executed_action"]),
                    "credit_applied": False,
                    "learning_event_message": current_skip_message,
                    "used_surrogate_target_post": False,
                }
                step_delayed_credit_updates.append(skipped_record)
                delayed_credit_updates.append(skipped_record)
                if step_debug or bool(_cfg("PRINT_POSITIVE_REWARD_STEPS", False)):
                    _print_delayed_credit_update(skipped_record)

            current_distance_to_victim = _safe_float(
                step_info.get("distance_to_nearest_victim", previous_distance_to_victim),
                previous_distance_to_victim,
            )
            current_helped = bool(found_victim) or bool(current_distance_to_victim < previous_distance_to_victim)
            previous_turn_record = None
            if len(recent_decisions) >= 2:
                candidate = list(recent_decisions)[-2]
                if int(candidate.get("executed_action", -1)) in (1, 2):
                    previous_turn_record = candidate

            if (
                enable_previous_turn_credit
                and previous_turn_record is not None
                and int(executed_action) == 0
                and current_helped
                and not bool(previous_turn_record.get("used_fallback", False))
            ):
                turn_target = int(previous_turn_record["executed_action"])
                _, _, turn_record = apply_positive_credit(
                    credit_record=previous_turn_record,
                    credit_reward=float(learning_reward) * 0.3,
                    credit_target=turn_target,
                    reason="previous_turn_then_forward_found",
                    credit_type="previous_turn",
                )
                step_delayed_credit_updates.append(turn_record)
                delayed_credit_updates.append(turn_record)
                if bool(turn_record.get("credit_applied", False)):
                    delayed_credit_update_count += 1
                    _hist_increment(delayed_credit_action_histogram, turn_target)
                    _hist_increment(step_delayed_credit_action_histogram, turn_target)
                    _hist_increment(previous_turn_credit_action_histogram, turn_target)
                    _hist_increment(step_previous_turn_credit_action_histogram, turn_target)

                if step_debug or bool(_cfg("PRINT_POSITIVE_REWARD_STEPS", False)):
                    _print_delayed_credit_update(turn_record)
            elif enable_previous_turn_credit and len(recent_decisions) >= 2:
                candidate = list(recent_decisions)[-2]
                skipped_turn_record = {
                    "source_event": "found_victim",
                    "reason": "skipped_recent_action_not_causal",
                    "credit_type": "previous_turn",
                    "episode_idx": int(episode_idx),
                    "original_step": int(step_idx),
                    "credit_step": int(candidate["step_idx"]),
                    "snn_action": int(candidate["snn_action"]),
                    "executed_action": int(candidate["executed_action"]),
                    "action_was_explored": bool(candidate["action_was_explored"]),
                    "used_fallback": bool(candidate["used_fallback"]),
                    "credit_reward": 0.0,
                    "target": int(candidate["executed_action"]),
                    "credit_applied": False,
                    "learning_event_message": "skipped_recent_action_not_causal",
                    "used_surrogate_target_post": False,
                }
                step_delayed_credit_updates.append(skipped_turn_record)
                delayed_credit_updates.append(skipped_turn_record)
                if step_debug or bool(_cfg("PRINT_POSITIVE_REWARD_STEPS", False)):
                    _print_delayed_credit_update(skipped_turn_record)

        elif learner is not None and float(learning_reward) != 0.0:
            target = None if learning_target is None else int(learning_target)
            if target is None:
                current_skip_message = "learner.learn skipped because learning target is None"
            else:
                balance_status = (
                    _output_column_balance_status(
                        net=net,
                        target_action=int(target),
                        reward=float(learning_reward),
                    )
                    if float(learning_reward) > 0.0
                    else {"blocked": False}
                )
                if bool(balance_status.get("blocked", False)):
                    current_skip_message = (
                        "skipped by output column balance "
                        f"(target={int(target)}, "
                        f"column_mean={balance_status.get('column_mean')}, "
                        f"threshold={balance_status.get('balance_threshold')})"
                    )
                else:
                    use_negative_abs_eligibility = False
                    use_negative_surrogate_post = False
                    if learning_reason == "wall_avoid_forward_collision":
                        use_negative_abs_eligibility = bool(
                            _cfg(
                                "WALL_AVOID_USE_ABS_ELIGIBILITY_ON_FORWARD_COLLISION",
                                True,
                            )
                        )
                        use_negative_surrogate_post = bool(
                            _cfg(
                                "WALL_AVOID_USE_SURROGATE_POST_ON_FORWARD_COLLISION",
                                True,
                            )
                        )
                    elif is_wall_avoid_phase and learning_reason in {
                        "turn_worsened_front_clearance",
                        "unnecessary_turn_when_front_clear",
                    }:
                        use_negative_abs_eligibility = bool(
                            int(target) == 1
                            and
                            _cfg(
                                "WALL_AVOID_USE_ABS_ELIGIBILITY_ON_TURN_PENALTY",
                                True,
                            )
                        )
                    elif (
                        is_wall_avoid_phase
                        and learning_reason == "wall_avoid_repeated_turn_spin"
                    ):
                        use_negative_abs_eligibility = bool(
                            _cfg(
                                "WALL_AVOID_USE_ABS_ELIGIBILITY_ON_TURN_PENALTY",
                                True,
                            )
                        )
                    original_hidden_rstdp = bool(learner.cfg.enable_hidden_rstdp)
                    learner.cfg.enable_hidden_rstdp = bool(
                        original_hidden_rstdp
                        or _hidden_rstdp_enabled_for_phase(phase_name)
                    )
                    try:
                        learning_events = learner.learn(
                            net=net,
                            reward=float(learning_reward),
                            target=int(target),
                            use_abs_eligibility_on_target_negative=use_negative_abs_eligibility,
                            use_surrogate_post_on_target_negative=use_negative_surrogate_post,
                        )
                    finally:
                        learner.cfg.enable_hidden_rstdp = original_hidden_rstdp
                    output_event = None if learning_events is None else learning_events.get("output")

        step_info["delayed_credit_update_count"] = int(len(step_delayed_credit_updates))
        step_info["delayed_credit_action_histogram"] = {
            int(action): int(count)
            for action, count in step_delayed_credit_action_histogram.items()
        }
        step_info["direct_credit_action_histogram"] = {
            int(action): int(count)
            for action, count in step_direct_credit_action_histogram.items()
        }
        step_info["previous_turn_credit_action_histogram"] = {
            int(action): int(count)
            for action, count in step_previous_turn_credit_action_histogram.items()
        }

        if _is_train_phase(phase_name) and float(step.reward) > 0.0:
            skip_message = None
            if learner is None:
                skip_message = "learner.learn skipped because learner is None"
            elif current_skip_message is not None:
                skip_message = current_skip_message
            event_record = _output_learning_event_record(
                output_event,
                skip_message=skip_message,
            )
            positive_reward_record = {
                "episode_idx": int(episode_idx),
                "step_idx": int(step_idx),
                "snn_action": int(snn_action),
                "executed_action": int(executed_action),
                "reward": float(step.reward),
                "raw_reward": float(step.reward),
                "learning_reward": float(learning_reward),
                "target": None if target is None else int(target),
                "used_fallback": bool(decision.used_fallback),
                "delayed_credit_updates": step_delayed_credit_updates,
                **event_record,
            }
            positive_reward_steps.append(positive_reward_record)
            if bool(_cfg("PRINT_POSITIVE_REWARD_STEPS", False)):
                print("TRAIN POSITIVE REWARD STEP:", positive_reward_record)

        if len(eval_trace_steps) < int(trace_first_steps):
            output_winner = None
            selected_step = int(getattr(decision, "selected_step", -1))
            if 0 <= selected_step < len(decision.step_records):
                output_winner = int(
                    decision.step_records[selected_step].output_result.winner
                )
            eval_trace_steps.append(
                {
                    "step_idx": int(step_idx),
                    "obs": prev_obs,
                    "obs_before_action": prev_obs,
                    **{
                        name: _safe_float(
                            prev_obs.get(name, 0.0) if isinstance(prev_obs, dict) else 0.0,
                            0.0,
                        )
                        for name in TEMPORAL_OBSERVATION_FEATURES
                    },
                    "selected_action": int(snn_action),
                    "selected_mode": selected_mode,
                    "selected_by_spike": bool(selected_by_spike),
                    "fallback_used": bool(decision.used_fallback),
                    "fallback_scores": _as_float_list(decision.output_scores_fallback),
                    "output_scores": (
                        _as_float_list(decision.output_scores_fallback)
                        if decision.output_scores_fallback is not None
                        else output_current_mean
                    ),
                    "output_spike_counts": output_spike_counts,
                    "first_spike_step": first_spike_step,
                    "snn_action": int(snn_action),
                    "executed_action": int(executed_action),
                    "reward": float(step.reward),
                    "next_obs": next_obs,
                    "found_victim": bool(found_victim),
                    "collision": bool(collision),
                    "moved": bool(step_info.get("moved", False)),
                    "output_winner": output_winner,
                    "used_fallback": bool(decision.used_fallback),
                    "selected_step": selected_step,
                    "front_clearance": float(step_info["front_clearance_before_action"]),
                    "victim_signal": _safe_float(
                        prev_obs.get("victim_signal", 0.0) if isinstance(prev_obs, dict) else 0.0,
                        0.0,
                    ),
                    "sound_signal": _safe_float(
                        prev_obs.get("sound_signal", 0.0) if isinstance(prev_obs, dict) else 0.0,
                        0.0,
                    ),
                    "agent_pos": step_info.get("old_pos", None),
                    "next_agent_pos": step_info.get("new_pos", None),
                    "output_threshold_mean_per_action": output_threshold_mean,
                    "output_current_mean_per_action": output_current_mean,
                    **hidden_current_stats,
                    "no_action_due_to_no_spike": bool(no_action_due_to_no_spike),
                    "output_winners": [
                        int(rec.output_result.winner)
                        for rec in decision.step_records
                    ],
                }
            )

        if relax_stm and stm_relax_dt_s > 0.0:
            _relax_stm_crossbars(net, stm_relax_dt_s)

        if metrics is not None:
            metrics.add_episode(
                rollout_info={
                    "used_fallback": decision.used_fallback,
                    "selected_by_spike": bool(selected_by_spike),
                    "selected_mode": selected_mode,
                    "selected_step": decision.selected_step,
                    "action": executed_action,
                    "snn_action": snn_action,
                    "executed_action": executed_action,
                    "fallback_action": snn_action if bool(decision.used_fallback) else -1,
                    "fallback_scores": _as_float_list(decision.output_scores_fallback),
                    "output_scores": (
                        _as_float_list(decision.output_scores_fallback)
                        if decision.output_scores_fallback is not None
                        else output_current_mean
                    ),
                    "output_spike_counts": output_spike_counts,
                    "first_spike_step": first_spike_step,
                    "output_threshold_mean": output_threshold_mean,
                    "output_current_mean": output_current_mean,
                    **hidden_current_stats,
                    "no_action_due_to_no_spike": bool(no_action_due_to_no_spike),
                    "no_output_spike": bool(output_spike_total == 0),
                    "hidden_spikes": [rec.hidden_result.spikes for rec in decision.step_records],
                    "output_spikes": [rec.output_result.spikes for rec in decision.step_records],
                    "output_winners": [rec.output_result.winner for rec in decision.step_records],
                    "reward": float(step.reward),
                    "env_info": step_info,
                },
                learning_event=output_event,
            )

        total_reward += float(step.reward)

        if step_debug:
            print(
                f"[step {step_idx}] snn_action={snn_action} executed_action={executed_action} "
                f"env_reward={step.reward} learning_reward={learning_reward}"
            )
            print(
                f"selected_step={decision.selected_step} "
                f"used_fallback={decision.used_fallback} "
                f"action_was_explored={action_was_explored} target={target}"
            )
            if learning_events is not None:
                print("learning output:", learning_events.get("output"))
                print("learning hidden:", learning_events.get("hidden"))
            print(env.render_ascii())
            print()

        obs = next_obs
        previous_step_collision = bool(collision)
        done = bool(step.done)
        step_idx += 1

    success = bool(len(env.victim_positions) == 0)
    final_state = env.get_env_state() if hasattr(env, "get_env_state") else {}
    final_distance_to_victim = _safe_float(
        final_state.get(
            "last_distance_to_victim",
            last_step_info.get(
                "post_step_distance_to_victim",
                last_step_info.get("distance_to_nearest_victim", 0.0),
            ),
        ),
        0.0,
    )
    last_info_distance_to_victim = _safe_float(
        last_step_info.get(
            "distance_to_nearest_victim",
            final_distance_to_victim,
        ),
        0.0,
    )
    action_summary = _finalize_action_stats(action_stats)

    if step_debug:
        print(f"episode reward: {total_reward}")
        print(
            f"success: {success}, rescued_count: {final_state.get('rescued_count', 0)}, "
            f"remaining_victims: {final_state.get('remaining_victims', len(env.victim_positions))}, "
            f"fallback_count: {fallback_count}"
        )
        print(
            f"found_victim_count: {found_victim_count}, "
            f"min_distance_to_victim: {min(distance_values)}, "
            f"final_distance_to_victim: {final_distance_to_victim}"
        )
        print("action_mean_reward:", action_summary["action_mean_reward"])
        print("action_event_counts:", action_summary["action_event_counts"])
        print()

    net.force_action_on_no_spike = original_force_action_on_no_spike
    net.stm_recurrent_current_scale = float(original_stm_recurrent_current_scale)
    if hasattr(net, "set_hidden_trace_params"):
        net.set_hidden_trace_params(
            input_scale=float(original_hidden_trace_input_scale),
            enabled=bool(original_hidden_trace_enabled),
        )
    else:
        net.enable_cross_decision_hidden_trace = original_hidden_trace_enabled
        net.hidden_trace_input_scale = float(original_hidden_trace_input_scale)

    return {
        "episode_reward": float(total_reward),
        "success": bool(success),
        "fallback_count": int(fallback_count),
        "output_spike_total": int(episode_output_spike_total),
        "no_output_spike_count": int(no_output_spike_count),
        "steps": int(step_idx),
        "map_index": -1 if map_index is None else int(map_index),
        "initial_victim_count": int(
            initial_state.get(
                "initial_victim_count",
                final_state.get(
                    "initial_victim_count",
                    final_state.get("rescued_count", found_victim_count)
                    + final_state.get("remaining_victims", len(env.victim_positions)),
                ),
            )
        ),
        "rescued_count": int(final_state.get("rescued_count", found_victim_count)),
        "remaining_victims": int(
            final_state.get("remaining_victims", len(env.victim_positions))
        ),
        "found_victim_count": int(found_victim_count),
        "found_step_indices": [int(idx) for idx in found_step_indices],
        "found_order": found_order,
        "collision_count": int(episode_collision_count),
        "moved_count": int(episode_moved_count),
        "repeated_turn_count": int(repeated_turn_count),
        "repeated_forward_collision_count": int(repeated_forward_collision_count),
        "unique_cells_visited_count": int(len(visited_cells)),
        "visited_cells": sorted([list(pos) for pos in visited_cells]),
        "min_distance_to_victim": float(min(distance_values)),
        "final_distance_to_victim": float(final_distance_to_victim),
        "last_step_distance_to_victim": float(last_info_distance_to_victim),
        "action_counts": action_summary["action_counts"],
        "action_mean_reward": action_summary["action_mean_reward"],
        "action_event_counts": action_summary["action_event_counts"],
        "positive_reward_steps": positive_reward_steps,
        "eval_trace": eval_trace_steps,
        "delayed_credit_updates": delayed_credit_updates,
        "delayed_credit_update_count": int(delayed_credit_update_count),
        "delayed_credit_action_histogram": {
            int(action): int(count)
            for action, count in delayed_credit_action_histogram.items()
        },
        "direct_credit_action_histogram": {
            int(action): int(count)
            for action, count in direct_credit_action_histogram.items()
        },
        "previous_turn_credit_action_histogram": {
            int(action): int(count)
            for action, count in previous_turn_credit_action_histogram.items()
        },
        "eval_state_reset_debug": eval_state_reset_debug,
    }


def run_phase(
    env: AbstractRescueGridEnv,
    net: MemristiveSNNNetwork,
    learner: Optional[RewardModulatedSTDPLearner],
    n_episodes: int,
    phase_name: str,
    verbose: bool = False,
    log_name: Optional[str] = None,
    trace_log_name: Optional[str] = None,
    trace_first_steps: int = 0,
):
    metrics = SNNMetrics()
    results = []

    for ep in range(int(n_episodes)):
        result = run_episode(
            env=env,
            net=net,
            learner=learner,
            metrics=metrics,
            episode_idx=ep,
            phase_name=phase_name,
            verbose=verbose,
            trace_first_steps=int(trace_first_steps) if ep == 0 else 0,
        )
        results.append(result)

    rewards = np.array([r["episode_reward"] for r in results], dtype=float)
    successes = np.array([r["success"] for r in results], dtype=float)
    fallbacks = np.array([r["fallback_count"] for r in results], dtype=float)
    output_spike_totals = np.array([r.get("output_spike_total", 0) for r in results], dtype=float)
    no_output_spike_counts = np.array([r.get("no_output_spike_count", 0) for r in results], dtype=float)
    steps = np.array([r["steps"] for r in results], dtype=float)
    rescued_counts = np.array([r["rescued_count"] for r in results], dtype=float)
    remaining_victims = np.array([r["remaining_victims"] for r in results], dtype=float)
    found_victim_counts = np.array([r["found_victim_count"] for r in results], dtype=float)
    initial_victim_counts = np.array([r.get("initial_victim_count", 0) for r in results], dtype=float)
    episode_collision_counts = np.array([r.get("collision_count", 0) for r in results], dtype=float)
    episode_moved_counts = np.array([r.get("moved_count", 0) for r in results], dtype=float)
    repeated_turn_counts = np.array([r.get("repeated_turn_count", 0) for r in results], dtype=float)
    repeated_forward_collision_counts = np.array(
        [r.get("repeated_forward_collision_count", 0) for r in results],
        dtype=float,
    )
    unique_cells_visited_counts = np.array(
        [r.get("unique_cells_visited_count", 0) for r in results],
        dtype=float,
    )
    min_distances = np.array([r["min_distance_to_victim"] for r in results], dtype=float)
    final_distances = np.array([r["final_distance_to_victim"] for r in results], dtype=float)
    positive_reward_steps = [
        step
        for result in results
        for step in result.get("positive_reward_steps", [])
    ]
    delayed_credit_updates = [
        update
        for result in results
        for update in result.get("delayed_credit_updates", [])
    ]
    delayed_credit_update_count = int(
        sum(int(result.get("delayed_credit_update_count", 0)) for result in results)
    )
    delayed_credit_action_histogram: Dict[int, int] = {}
    direct_credit_action_histogram: Dict[int, int] = {}
    previous_turn_credit_action_histogram: Dict[int, int] = {}
    for result in results:
        for action, count in result.get("delayed_credit_action_histogram", {}).items():
            _hist_increment(delayed_credit_action_histogram, int(action), int(count))
        for action, count in result.get("direct_credit_action_histogram", {}).items():
            _hist_increment(direct_credit_action_histogram, int(action), int(count))
        for action, count in result.get("previous_turn_credit_action_histogram", {}).items():
            _hist_increment(previous_turn_credit_action_histogram, int(action), int(count))
    metrics_summary = metrics.summary_dict()
    compact_metrics_summary = metrics.compact_summary_dict()
    trace_steps = results[0].get("eval_trace", []) if results else []
    if trace_log_name:
        _write_json_log(trace_log_name, trace_steps)
    raw_eval_first_episode_action_sequence = [
        int(step.get("executed_action", -1))
        for step in trace_steps
    ]
    eval_first_episode_action_sequence = [
        int(action)
        for action in raw_eval_first_episode_action_sequence
        if int(action) in (0, 1, 2)
    ]
    eval_first_episode_no_action_count = int(
        sum(1 for action in raw_eval_first_episode_action_sequence if int(action) not in (0, 1, 2))
    )
    eval_first_episode_no_spike_count = int(
        sum(1 for step in trace_steps if bool(step.get("no_action_due_to_no_spike", False)))
    )
    eval_first_episode_early_done_before_action = bool(
        trace_steps
        and not eval_first_episode_action_sequence
        and eval_first_episode_no_action_count > 0
    )
    eval_first_episode_collision_sequence = [
        bool(step.get("collision", False))
        for step in trace_steps
    ]
    eval_first_episode_moved_sequence = [
        bool(step.get("moved", False))
        for step in trace_steps
    ]
    eval_first_episode_victim_signal_sequence = [
        _safe_float(step.get("victim_signal", 0.0), 0.0)
        for step in trace_steps
    ]
    eval_first_episode_front_clearance_sequence = [
        _safe_float(step.get("front_clearance", 0.0), 0.0)
        for step in trace_steps
    ]
    found_step_indices_per_episode = [
        [int(idx) for idx in result.get("found_step_indices", [])]
        for result in results
    ]
    found_order_per_episode = [
        result.get("found_order", [])
        for result in results
    ]
    feature_diagnostics = _encoder_feature_diagnostics(net.encoder)
    eval_state_reset_debug = (
        results[0].get("eval_state_reset_debug", {}) if results else {}
    )
    reset_each_decision = bool(getattr(net, "reset_neuron_state_each_decision", True))
    enable_hidden_trace = bool(_cross_decision_hidden_trace_enabled_for_phase(phase_name))
    recurrent_memory_scope = (
        "across_env_steps_hidden_trace"
        if enable_hidden_trace
        else (
            "within_decision_window"
            if reset_each_decision
            else "across_env_steps"
        )
    )

    summary = {
        "phase": str(phase_name),
        "n_episodes": int(n_episodes),
        **feature_diagnostics,
        "eval_state_reset_debug": eval_state_reset_debug,
        "reset_neuron_state_each_decision": bool(reset_each_decision),
        "prev_hidden_spikes_reset_each_decision": bool(reset_each_decision),
        "recurrent_memory_scope": recurrent_memory_scope,
        "stm_recurrent_current_scale": float(
            getattr(net, "stm_recurrent_current_scale", _cfg("STM_RECURRENT_CURRENT_SCALE", 1.0))
        ),
        "stage_name_for_lookup": _stage_name_for_lookup(phase_name),
        "stage_rstdp_scale": float(_stage_rstdp_scale(phase_name)),
        "train_exploration_epsilon": (
            float(_train_exploration_epsilon(phase_name))
            if _is_train_phase(phase_name)
            else 0.0
        ),
        "eval_exploration_epsilon": 0.0,
        "target_recurrent_input_ratio_max": float(
            _cfg("TARGET_RECURRENT_INPUT_RATIO_MAX", 0.8)
        ),
        "enable_cross_decision_hidden_trace": bool(enable_hidden_trace),
        "hidden_trace_decay": float(getattr(net, "hidden_trace_decay", _cfg("HIDDEN_TRACE_DECAY", 0.0))),
        "hidden_trace_input_scale": float(
            getattr(net, "hidden_trace_input_scale", _cfg("HIDDEN_TRACE_INPUT_SCALE", 0.0))
        ),
        "recurrent_hidden_current_masked": bool(
            getattr(net, "mask_recurrent_hidden_current", False)
        ),
        "mean_reward": float(rewards.mean()) if rewards.size else 0.0,
        "success_rate": float(successes.mean()) if successes.size else 0.0,
        "all_victims_success_rate": float(successes.mean()) if successes.size else 0.0,
        "partial_success_rate": (
            float(found_victim_counts.sum() / max(float(initial_victim_counts.sum()), 1.0))
            if found_victim_counts.size and initial_victim_counts.size
            else 0.0
        ),
        "collision_free_success_rate": (
            float(np.mean((successes > 0.0) & (episode_collision_counts == 0.0)))
            if successes.size and episode_collision_counts.size
            else 0.0
        ),
        "total_victim_count": int(initial_victim_counts.max()) if initial_victim_counts.size else 0,
        "mean_rescued_count": float(rescued_counts.mean()) if rescued_counts.size else 0.0,
        "mean_remaining_victims": float(remaining_victims.mean()) if remaining_victims.size else 0.0,
        "mean_found_victim_count": float(found_victim_counts.mean()) if found_victim_counts.size else 0.0,
        "mean_min_distance_to_victim": float(min_distances.mean()) if min_distances.size else 0.0,
        "best_min_distance_to_victim": float(min_distances.min()) if min_distances.size else 0.0,
        "mean_final_distance_to_victim": float(final_distances.mean()) if final_distances.size else 0.0,
        "mean_fallback_count": float(fallbacks.mean()) if fallbacks.size else 0.0,
        "output_spike_total_per_episode": [
            int(r.get("output_spike_total", 0)) for r in results
        ],
        "mean_output_spike_total_per_episode": (
            float(output_spike_totals.mean()) if output_spike_totals.size else 0.0
        ),
        "no_output_spike_episode_count": int(
            np.sum(output_spike_totals == 0.0)
        ) if output_spike_totals.size else 0,
        "mean_no_output_spike_count": (
            float(no_output_spike_counts.mean()) if no_output_spike_counts.size else 0.0
        ),
        "mean_steps": float(steps.mean()) if steps.size else 0.0,
        "found_victim_count": int(found_victim_counts.sum()) if found_victim_counts.size else 0,
        "collision_count": int(episode_collision_counts.sum()) if episode_collision_counts.size else 0,
        "moved_count": int(episode_moved_counts.sum()) if episode_moved_counts.size else 0,
        "victims_found_per_collision": (
            float(found_victim_counts.sum() / max(float(episode_collision_counts.sum()), 1.0))
            if found_victim_counts.size
            else 0.0
        ),
        "victims_found_per_step": (
            float(found_victim_counts.sum() / max(float(steps.sum()), 1.0))
            if found_victim_counts.size and steps.size
            else 0.0
        ),
        "repeated_turn_count": int(repeated_turn_counts.sum()) if repeated_turn_counts.size else 0,
        "repeated_forward_collision_count": (
            int(repeated_forward_collision_counts.sum())
            if repeated_forward_collision_counts.size
            else 0
        ),
        "unique_cells_visited_count": (
            float(unique_cells_visited_counts.mean())
            if unique_cells_visited_counts.size
            else 0.0
        ),
        "unique_cells_visited_count_per_episode": [
            int(r.get("unique_cells_visited_count", 0)) for r in results
        ],
        "found_step_indices_per_episode": found_step_indices_per_episode,
        "found_order_per_episode": found_order_per_episode,
        "found_step_indices": found_step_indices_per_episode[0] if found_step_indices_per_episode else [],
        "found_order": found_order_per_episode[0] if found_order_per_episode else [],
        "last_unfound_victim_distance_if_available": (
            float(final_distances[-1]) if final_distances.size else None
        ),
        "stopped_due_to_wall_stuck": bool(
            int(repeated_forward_collision_counts.sum()) > 0
            and metrics_summary.get("action_histogram", {}).get(0, metrics_summary.get("action_histogram", {}).get("0", 0))
            if repeated_forward_collision_counts.size
            else False
        ),
        "action_mean_reward": metrics_summary.get("action_mean_reward", {}),
        "action_event_counts": metrics_summary.get("action_event_counts", {}),
        "output_neuron_spike_histogram": metrics_summary.get("output_neuron_spike_histogram", {}),
        "output_winner_histogram": metrics_summary.get("output_winner_histogram", {}),
        "spike_selected_action_histogram": metrics_summary.get("spike_selected_action_histogram", {}),
        "fallback_action_histogram": metrics_summary.get("fallback_action_histogram", {}),
        "fallback_count_by_action": metrics_summary.get("fallback_count_by_action", {}),
        "fallback_when_front_blocked_count": metrics_summary.get("fallback_when_front_blocked_count", 0),
        "fallback_when_front_blocked_action_histogram": metrics_summary.get(
            "fallback_when_front_blocked_action_histogram", {}
        ),
        "no_output_spike_count": metrics_summary.get("no_output_spike_count", 0),
        "output_threshold_mean_per_action": metrics_summary.get(
            "output_threshold_mean_per_action", {}
        ),
        "output_current_mean_per_action": metrics_summary.get(
            "output_current_mean_per_action", {}
        ),
        "mean_abs_input_hidden_current": metrics_summary.get(
            "mean_abs_input_hidden_current", 0.0
        ),
        "mean_abs_recurrent_hidden_current": metrics_summary.get(
            "mean_abs_recurrent_hidden_current", 0.0
        ),
        "mean_recurrent_to_input_current_ratio": metrics_summary.get(
            "mean_recurrent_to_input_current_ratio", 0.0
        ),
        "median_recurrent_to_input_current_ratio": metrics_summary.get(
            "median_recurrent_to_input_current_ratio", 0.0
        ),
        "max_recurrent_to_input_current_ratio": metrics_summary.get(
            "max_recurrent_to_input_current_ratio", 0.0
        ),
        "mean_prev_hidden_spike_count": metrics_summary.get(
            "mean_prev_hidden_spike_count", 0.0
        ),
        "mean_hidden_spike_count": metrics_summary.get(
            "mean_hidden_spike_count",
            metrics_summary.get("mean_hidden_spikes", 0.0),
        ),
        "hidden_trace_mean": metrics_summary.get("hidden_trace_mean", 0.0),
        "hidden_trace_nonzero_count": metrics_summary.get(
            "hidden_trace_nonzero_count", 0.0
        ),
        "input_hidden_current_abs_mean_by_action": metrics_summary.get(
            "input_hidden_current_abs_mean_by_action", {}
        ),
        "recurrent_hidden_current_abs_mean_by_action": metrics_summary.get(
            "recurrent_hidden_current_abs_mean_by_action", {}
        ),
        "recurrent_to_input_current_ratio_by_action": metrics_summary.get(
            "recurrent_to_input_current_ratio_by_action", {}
        ),
        "output_column_weight_mean": _output_column_weight_mean(net),
        "output_column_weight_std": _output_column_weight_std(net),
        "output_column_effective_score": _output_column_effective_score(net),
        "recurrent_crossbar_status": _recurrent_crossbar_status(net),
        "positive_reward_steps": positive_reward_steps,
        "delayed_credit_updates": delayed_credit_updates,
        "delayed_credit_update_count": int(delayed_credit_update_count),
        "delayed_credit_action_histogram": delayed_credit_action_histogram,
        "direct_credit_action_histogram": direct_credit_action_histogram,
        "previous_turn_credit_action_histogram": previous_turn_credit_action_histogram,
        "learning_update_action_histogram": metrics_summary.get("learning_update_action_histogram", {}),
        "potentiation_action_histogram": metrics_summary.get("potentiation_action_histogram", {}),
        "depression_action_histogram": metrics_summary.get("depression_action_histogram", {}),
        "turn_sensor_reward_count": metrics_summary.get("turn_sensor_reward_count", 0),
        "turn_sensor_reward_action_histogram": metrics_summary.get("turn_sensor_reward_action_histogram", {}),
        "turn_sensor_penalty_count": metrics_summary.get("turn_sensor_penalty_count", 0),
        "turn_sensor_penalty_action_histogram": metrics_summary.get("turn_sensor_penalty_action_histogram", {}),
        "learning_reason_histogram": metrics_summary.get("learning_reason_histogram", {}),
        "wall_avoid_forward_collision_count": metrics_summary.get("wall_avoid_forward_collision_count", 0),
        "wall_avoid_forward_after_clearance_count": metrics_summary.get("wall_avoid_forward_after_clearance_count", 0),
        "wall_avoid_repeated_turn_spin_count": metrics_summary.get("wall_avoid_repeated_turn_spin_count", 0),
        "wall_avoid_forward_positive_count": metrics_summary.get("wall_avoid_forward_positive_count", 0),
        "wall_avoid_forward_positive_skipped_count": metrics_summary.get("wall_avoid_forward_positive_skipped_count", 0),
        "wall_avoid_turn_reward_count": metrics_summary.get("wall_avoid_turn_reward_count", 0),
        "wall_avoid_action0_selected_when_front_blocked_count": metrics_summary.get("wall_avoid_action0_selected_when_front_blocked_count", 0),
        "wall_avoid_forward_recovery_action_histogram": metrics_summary.get("wall_avoid_forward_recovery_action_histogram", {}),
        "wall_avoid_spin_penalty_action_histogram": metrics_summary.get("wall_avoid_spin_penalty_action_histogram", {}),
        "selected_step_action_histogram": metrics_summary.get("selected_step_action_histogram", {}),
        "output_spike_count_by_action": metrics_summary.get("output_neuron_spike_histogram", {}),
        "output_first_spike_time_by_action_mean": _mean_first_spike_time_by_action(metrics),
        "eval_first_episode_action_sequence": eval_first_episode_action_sequence,
        "eval_first_episode_raw_action_sequence": raw_eval_first_episode_action_sequence,
        "eval_first_episode_no_action_count": int(eval_first_episode_no_action_count),
        "eval_first_episode_no_spike_count": int(eval_first_episode_no_spike_count),
        "eval_first_episode_early_done_before_action": bool(
            eval_first_episode_early_done_before_action
        ),
        "valid_actions_only": bool(
            all(action in (0, 1, 2) for action in eval_first_episode_action_sequence)
        ),
        "eval_first_episode_collision_sequence": eval_first_episode_collision_sequence,
        "eval_first_episode_moved_sequence": eval_first_episode_moved_sequence,
        "eval_first_episode_victim_signal_sequence": eval_first_episode_victim_signal_sequence,
        "eval_first_episode_front_clearance_sequence": eval_first_episode_front_clearance_sequence,
        "first_episode_action_sequence": eval_first_episode_action_sequence,
        "first_episode_collision_sequence": eval_first_episode_collision_sequence,
        "first_episode_moved_sequence": eval_first_episode_moved_sequence,
        "first_episode_victim_signal_sequence": eval_first_episode_victim_signal_sequence,
        "first_episode_front_clearance_sequence": eval_first_episode_front_clearance_sequence,
        "metrics": metrics_summary,
        "compact_metrics": compact_metrics_summary,
    }

    if _is_train_phase(phase_name):
        _save_positive_reward_steps(log_name or phase_name, positive_reward_steps)

    if verbose:
        print("-" * 70)
        print(f"{phase_name.upper()} SUMMARY")
        print("-" * 70)
        for k, v in summary.items():
            if k not in {"metrics", "positive_reward_steps"}:
                print(f"{k}: {v}")
        if bool(_cfg("PRINT_POSITIVE_REWARD_STEPS", False)) and positive_reward_steps:
            print("positive_reward_steps:", positive_reward_steps)
        print("metrics:", summary["metrics"])
        print()

    return results, summary


def _selection_mode_suffix(mode: str) -> str:
    return "hybrid" if str(mode) == "hybrid_spike_count_latency" else str(mode)


def _diagnostic_selection_modes() -> list[str]:
    raw_modes = _cfg(
        "OUTPUT_ACTION_SELECTION_DIAGNOSTIC_MODES",
        ("first_spike", "spike_count", "hybrid_spike_count_latency"),
    )
    modes = [raw_modes] if isinstance(raw_modes, str) else list(raw_modes)
    out: list[str] = []
    for mode in modes:
        canonical = MemristiveSNNNetwork._normalize_action_selection_mode(mode)
        if canonical not in out:
            out.append(canonical)
    return out


def _set_output_selection_mode(net: MemristiveSNNNetwork, mode: str) -> None:
    if hasattr(net, "set_output_action_selection_mode"):
        net.set_output_action_selection_mode(mode)
    else:
        net.output_action_selection_mode = str(mode)


def _selection_mode_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    metrics = _metrics_from_summary(summary)
    return {
        "success_rate": float(summary.get("success_rate", 0.0)),
        "action_histogram": metrics.get("action_histogram", {}),
        "snn_action_histogram": metrics.get("snn_action_histogram", {}),
        "spike_selected_action_histogram": summary.get(
            "spike_selected_action_histogram", {}
        ),
        "output_winner_histogram": summary.get("output_winner_histogram", {}),
        "fallback_action_histogram": summary.get("fallback_action_histogram", {}),
        "collision_count": int(metrics.get("collision_count", 0)),
        "moved_count": int(metrics.get("moved_count", 0)),
        "no_output_spike_count": int(summary.get("no_output_spike_count", 0)),
        "mean_fallback_count": float(summary.get("mean_fallback_count", 0.0)),
        "output_spike_total_per_episode": summary.get(
            "output_spike_total_per_episode", []
        ),
        "first_episode_action_sequence": summary.get(
            "eval_first_episode_action_sequence", []
        ),
        "first_episode_collision_sequence": summary.get(
            "eval_first_episode_collision_sequence", []
        ),
        "first_episode_moved_sequence": summary.get(
            "eval_first_episode_moved_sequence", []
        ),
    }


def run_eval_selection_mode_diagnostics(
    *,
    stage_name: str,
    map_name: str,
    net: MemristiveSNNNetwork,
    eval_episodes: int,
    seed: int,
    stage_idx: int,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Compare output-spike readout modes using cloned network state only."""
    diagnostics: Dict[str, Dict[str, Any]] = {}
    for mode in _diagnostic_selection_modes():
        suffix = _selection_mode_suffix(mode)
        diag_net = copy.deepcopy(net)
        _set_output_selection_mode(diag_net, mode)
        diag_env = build_env_for_map(map_name, seed=seed + 1000 + int(stage_idx))
        _, summary = run_phase(
            env=diag_env,
            net=diag_net,
            learner=None,
            n_episodes=int(eval_episodes),
            phase_name=f"{stage_name}_{suffix}_eval",
            verbose=verbose,
            log_name=f"{stage_name}_{suffix}_eval",
            trace_log_name=f"eval_trace_{_safe_log_name(stage_name)}_{suffix}.json",
            trace_first_steps=int(_cfg("EVAL_TRACE_FIRST_STEPS", 20)),
        )
        diagnostics[mode] = _selection_mode_summary(summary)
    return diagnostics


def _set_recurrent_current_mask(net: MemristiveSNNNetwork, masked: bool) -> None:
    if hasattr(net, "set_recurrent_current_diagnostic_mask"):
        net.set_recurrent_current_diagnostic_mask(bool(masked))
    else:
        net.mask_recurrent_hidden_current = bool(masked)


def _set_recurrent_current_scale(net: MemristiveSNNNetwork, scale: float) -> None:
    if hasattr(net, "set_recurrent_current_scale"):
        net.set_recurrent_current_scale(float(scale))
    else:
        net.stm_recurrent_current_scale = float(scale)


def _recurrent_diagnostic_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    metrics = _metrics_from_summary(summary)
    return {
        "success_rate": float(summary.get("success_rate", 0.0)),
        "action_histogram": metrics.get("action_histogram", {}),
        "snn_action_histogram": metrics.get("snn_action_histogram", {}),
        "collision_count": int(metrics.get("collision_count", 0)),
        "moved_count": int(metrics.get("moved_count", 0)),
        "sequence": summary.get("eval_first_episode_action_sequence", []),
        "action_sequence": summary.get("eval_first_episode_action_sequence", []),
        "collision_sequence": summary.get("eval_first_episode_collision_sequence", []),
        "mean_abs_input_hidden_current": summary.get(
            "mean_abs_input_hidden_current", 0.0
        ),
        "mean_abs_recurrent_hidden_current": summary.get(
            "mean_abs_recurrent_hidden_current", 0.0
        ),
        "mean_recurrent_to_input_current_ratio": summary.get(
            "mean_recurrent_to_input_current_ratio", 0.0
        ),
        "median_recurrent_to_input_current_ratio": summary.get(
            "median_recurrent_to_input_current_ratio", 0.0
        ),
        "max_recurrent_to_input_current_ratio": summary.get(
            "max_recurrent_to_input_current_ratio", 0.0
        ),
        "mean_prev_hidden_spike_count": summary.get("mean_prev_hidden_spike_count", 0.0),
        "mean_hidden_spike_count": summary.get("mean_hidden_spike_count", 0.0),
        "hidden_trace_mean": summary.get("hidden_trace_mean", 0.0),
        "hidden_trace_nonzero_count": summary.get("hidden_trace_nonzero_count", 0.0),
        "recurrent_to_input_current_ratio_by_action": summary.get(
            "recurrent_to_input_current_ratio_by_action", {}
        ),
    }


def run_eval_recurrent_ablation(
    *,
    stage_name: str,
    map_name: str,
    net: MemristiveSNNNetwork,
    eval_episodes: int,
    seed: int,
    stage_idx: int,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    if not bool(_cfg("ENABLE_STM_RECURRENT_ABLATION", True)):
        return {}

    diagnostics: Dict[str, Dict[str, Any]] = {}
    for label, masked in (("on", False), ("off", True)):
        diag_net = copy.deepcopy(net)
        _set_recurrent_current_mask(diag_net, masked)
        diag_env = build_env_for_map(map_name, seed=seed + 2000 + int(stage_idx))
        _, summary = run_phase(
            env=diag_env,
            net=diag_net,
            learner=None,
            n_episodes=int(eval_episodes),
            phase_name=f"{stage_name}_recurrent_{label}_eval",
            verbose=verbose,
            log_name=f"{stage_name}_recurrent_{label}_eval",
            trace_first_steps=int(_cfg("EVAL_TRACE_FIRST_STEPS", 20)),
        )
        diagnostics[label] = _recurrent_diagnostic_summary(summary)
    return diagnostics


def run_stm_recurrent_scale_sweep(
    *,
    stage_name: str,
    map_name: str,
    net: MemristiveSNNNetwork,
    eval_episodes: int,
    seed: int,
    stage_idx: int,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    raw_values = _cfg("STM_RECURRENT_SCALE_SWEEP_VALUES", (0.0, 0.25, 0.5, 1.0, 2.0))
    values = [raw_values] if isinstance(raw_values, (int, float, str)) else list(raw_values)
    diagnostics: Dict[str, Dict[str, Any]] = {}
    for raw_scale in values:
        scale = float(raw_scale)
        diag_net = copy.deepcopy(net)
        _set_recurrent_current_mask(diag_net, False)
        _set_recurrent_current_scale(diag_net, scale)
        diag_env = build_env_for_map(map_name, seed=seed + 3000 + int(stage_idx))
        _, summary = run_phase(
            env=diag_env,
            net=diag_net,
            learner=None,
            n_episodes=int(eval_episodes),
            phase_name=f"{stage_name}_stm_scale_{scale:g}_eval",
            verbose=verbose,
            log_name=f"{stage_name}_stm_scale_{scale:g}_eval",
            trace_first_steps=int(_cfg("EVAL_TRACE_FIRST_STEPS", 20)),
        )
        diagnostics[f"{scale:g}"] = _recurrent_diagnostic_summary(summary)
    return diagnostics


def _stm_recurrent_interpretation(stage_record: Dict[str, Any]) -> Dict[str, Any]:
    eval_summary = stage_record.get("stage_eval", {})
    ratio = float(eval_summary.get("mean_recurrent_to_input_current_ratio", 0.0))
    reset_each = bool(eval_summary.get("reset_neuron_state_each_decision", True))
    memory_scope = str(eval_summary.get("recurrent_memory_scope", "unknown"))
    ablation_all = stage_record.get("recurrent_ablation_eval", {})
    stage_name = str(stage_record.get("stage_name", ""))
    ablation = ablation_all.get(stage_name, ablation_all) if isinstance(ablation_all, dict) else {}
    on = ablation.get("on", {}) if isinstance(ablation, dict) else {}
    off = ablation.get("off", {}) if isinstance(ablation, dict) else {}
    on_off_differs = bool(
        on.get("success_rate") != off.get("success_rate")
        or on.get("collision_count") != off.get("collision_count")
        or on.get("action_histogram") != off.get("action_histogram")
    )

    if reset_each and memory_scope == "within_decision_window":
        case = "Case D"
        reason = (
            "reset_neuron_state_each_decision=True, so prev_hidden_spikes is reset "
            "at each observation decision; recurrent memory is within the decision window."
        )
    elif ratio > 1.0:
        case = "Case C"
        reason = "recurrent current is larger than the input current and may dominate dynamics."
    elif ratio < 0.05 and not on_off_differs:
        case = "Case B"
        reason = "recurrent current is weak and recurrent ON/OFF eval is nearly unchanged."
    elif on_off_differs:
        case = "Case A"
        reason = "recurrent ON/OFF eval changes behavior, indicating temporal dynamics contribution."
    else:
        case = "Case B"
        reason = "recurrent ON/OFF eval is nearly unchanged."

    return {
        "case": case,
        "reason": reason,
        "mean_recurrent_to_input_current_ratio": ratio,
        "on_off_eval_differs": bool(on_off_differs),
        "recurrent_memory_scope": memory_scope,
    }


def _sweep_values(config_name: str, default: tuple[float, ...]) -> list[float]:
    raw = _cfg(config_name, default)
    values = [raw] if isinstance(raw, (int, float, str)) else list(raw)
    return [float(value) for value in values]


def _bad_wall_avoid_sequence(seq: list[int]) -> bool:
    seq = [int(x) for x in seq if int(x) >= 0]
    if len(seq) < 4:
        return False
    if len(set(seq[:10])) <= 1:
        return True
    if seq[0] in (1, 2) and all(action == 0 for action in seq[1:10]):
        return True
    return False


def _sweep_stage_eval_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    metrics = _metrics_from_summary(summary)
    return {
        "eval_success": float(summary.get("success_rate", 0.0)),
        "action_histogram": metrics.get("action_histogram", {}),
        "collision_count": int(metrics.get("collision_count", 0)),
        "recurrent_to_input_current_ratio": float(
            summary.get("mean_recurrent_to_input_current_ratio", 0.0)
        ),
        "hidden_trace_mean": float(summary.get("hidden_trace_mean", 0.0)),
        "hidden_trace_nonzero_count": float(
            summary.get("hidden_trace_nonzero_count", 0.0)
        ),
        "action_sequence": summary.get("eval_first_episode_action_sequence", []),
    }


def _run_curriculum_once_for_sweep(
    *,
    seed: int,
    curriculum_stages: list[Dict[str, Any]],
) -> Dict[str, Any]:
    encoder = build_encoder()
    active_encoder_feature_summary = _write_active_encoder_feature_reports(encoder)
    net = build_network(encoder=encoder, seed=seed)
    learner = build_learner()

    stage_eval: Dict[str, Dict[str, Any]] = {}
    for stage_idx, stage in enumerate(curriculum_stages):
        stage_name = str(stage.get("name", f"stage_{stage_idx}"))
        map_name = str(stage["map_name"])
        train_episodes = int(
            stage.get(
                "train_episodes",
                _stage_setting(stage_name, "train_episodes", N_EPISODES_TRAIN),
            )
        )
        eval_episodes = int(
            stage.get(
                "eval_episodes",
                _stage_setting(stage_name, "eval_episodes", N_EPISODES_EVAL),
            )
        )

        train_env = build_env_for_map(map_name, seed=seed + stage_idx)
        run_phase(
            env=train_env,
            net=net,
            learner=learner,
            n_episodes=train_episodes,
            phase_name=f"sweep_{stage_name}_train",
            verbose=False,
            log_name=f"{stage_name}_sweep",
        )
        eval_env = build_env_for_map(map_name, seed=seed + 1000 + stage_idx)
        _, eval_summary = run_phase(
            env=eval_env,
            net=net,
            learner=None,
            n_episodes=eval_episodes,
            phase_name=f"sweep_{stage_name}_eval",
            verbose=False,
            log_name=f"{stage_name}_eval_sweep",
            trace_first_steps=int(_cfg("EVAL_TRACE_FIRST_STEPS", 20)),
        )
        stage_eval[stage_name] = _sweep_stage_eval_summary(eval_summary)

    return {"stage_eval": stage_eval}


def _score_stm_trace_sweep_result(result: Dict[str, Any]) -> float:
    stage_eval = result.get("stage_eval", {})
    primitive_names = ("stage_forward", "stage_turn_right", "stage_turn_left")
    primitive_success = [
        float(stage_eval.get(name, {}).get("eval_success", 0.0))
        for name in primitive_names
    ]
    wall = stage_eval.get("stage_wall_avoid", {})
    wall_success = float(wall.get("eval_success", 0.0))
    wall_collisions = float(wall.get("collision_count", 0))
    wall_seq = [int(x) for x in wall.get("action_sequence", [])]
    ratio = float(wall.get("recurrent_to_input_current_ratio", 0.0))
    bad_seq = _bad_wall_avoid_sequence(wall_seq)
    primitive_ok = all(value >= 1.0 for value in primitive_success)
    ratio_penalty = 0.0
    if ratio < 0.1:
        ratio_penalty = 0.2 - ratio
    elif ratio > 1.0:
        ratio_penalty = ratio - 1.0
    ratio_bonus = min(max(ratio - 0.1, 0.0), 0.5)
    return (
        100.0 * float(primitive_ok)
        + 20.0 * sum(primitive_success)
        + 60.0 * wall_success
        + 5.0 * ratio_bonus
        - 0.2 * wall_collisions
        - 30.0 * float(bad_seq)
        - 10.0 * ratio_penalty
    )


def run_stm_trace_parameter_sweep(verbose: bool = False) -> Dict[str, Any]:
    """Run hardware-compatible STM/hidden-trace parameter sweep.

    This rebuilds and retrains a fresh crossbar SNN per candidate.  It only
    changes physical/trace parameters before construction; it never writes an
    action-specific policy table or hand-sets conductance from env knowledge.
    """
    seed = int(_cfg("SEED", 42))
    curriculum_stages = list(_cfg("CURRICULUM_STAGES", []))
    decay_values = _sweep_values(
        "STM_TRACE_SWEEP_HIDDEN_TRACE_DECAYS",
        (0.5, 0.7, 0.85, 0.95),
    )
    trace_scale_values = _sweep_values(
        "STM_TRACE_SWEEP_HIDDEN_TRACE_INPUT_SCALES",
        (0.5, 1.0, 2.0, 4.0),
    )
    recurrent_scale_values = _sweep_values(
        "STM_TRACE_SWEEP_RECURRENT_CURRENT_SCALES",
        (0.5, 1.0, 2.0, 4.0),
    )
    conductance_scale_values = _sweep_values(
        "STM_TRACE_SWEEP_CONDUCTANCE_SCALES",
        (1.0, 2.0, 4.0),
    )

    candidate_grid = list(
        itertools.product(
            decay_values,
            trace_scale_values,
            recurrent_scale_values,
            conductance_scale_values,
        )
    )
    max_configs = int(_cfg("STM_TRACE_SWEEP_MAX_CONFIGS", 0))
    if max_configs > 0:
        candidate_grid = candidate_grid[:max_configs]

    config_names = (
        "ENABLE_CROSS_DECISION_HIDDEN_TRACE",
        "HIDDEN_TRACE_DECAY",
        "HIDDEN_TRACE_INPUT_SCALE",
        "STM_RECURRENT_CURRENT_SCALE",
        "STM_CONDUCTANCE_SCALE",
        "SAVE_POSITIVE_REWARD_STEPS_JSON",
    )
    saved = {name: getattr(cfg, name, None) for name in config_names}
    had_attr = {name: hasattr(cfg, name) for name in config_names}

    results = []
    try:
        cfg.ENABLE_CROSS_DECISION_HIDDEN_TRACE = True
        cfg.SAVE_POSITIVE_REWARD_STEPS_JSON = False
        for idx, (decay, trace_scale, recurrent_scale, conductance_scale) in enumerate(candidate_grid):
            cfg.HIDDEN_TRACE_DECAY = float(decay)
            cfg.HIDDEN_TRACE_INPUT_SCALE = float(trace_scale)
            cfg.STM_RECURRENT_CURRENT_SCALE = float(recurrent_scale)
            cfg.STM_CONDUCTANCE_SCALE = float(conductance_scale)
            run_result = _run_curriculum_once_for_sweep(
                seed=seed,
                curriculum_stages=curriculum_stages,
            )
            result = {
                "index": int(idx),
                "params": {
                    "HIDDEN_TRACE_DECAY": float(decay),
                    "HIDDEN_TRACE_INPUT_SCALE": float(trace_scale),
                    "STM_RECURRENT_CURRENT_SCALE": float(recurrent_scale),
                    "STM_CONDUCTANCE_SCALE": float(conductance_scale),
                },
                **run_result,
            }
            result["score"] = float(_score_stm_trace_sweep_result(result))
            wall = result["stage_eval"].get("stage_wall_avoid", {})
            result["wall_avoid_bad_fixed_sequence"] = bool(
                _bad_wall_avoid_sequence(wall.get("action_sequence", []))
            )
            results.append(result)
            if verbose:
                print(
                    f"sweep {idx + 1}/{len(candidate_grid)} params={result['params']} "
                    f"primitive=({result['stage_eval'].get('stage_forward', {}).get('eval_success')}, "
                    f"{result['stage_eval'].get('stage_turn_right', {}).get('eval_success')}, "
                    f"{result['stage_eval'].get('stage_turn_left', {}).get('eval_success')}) "
                    f"wall={wall.get('eval_success')} collisions={wall.get('collision_count')} "
                    f"ratio={wall.get('recurrent_to_input_current_ratio')} "
                    f"seq={wall.get('action_sequence')}"
                )
    finally:
        for name, value in saved.items():
            if had_attr[name]:
                setattr(cfg, name, value)
            elif hasattr(cfg, name):
                delattr(cfg, name)

    selected = max(results, key=lambda item: float(item.get("score", 0.0))) if results else None
    summary = {
        "num_candidates": int(len(results)),
        "selection_criteria": [
            "primitive eval_success must stay high",
            "wall_avoid fixed forward-only or turn-only sequence should break",
            "wall recurrent/input current ratio should not be too small or too dominant",
            "wall_avoid collision_count should decrease",
        ],
        "selected_config": selected,
        "results": results,
    }
    path = _write_json_log("stm_trace_parameter_sweep.json", summary)
    print(f"stm_trace_parameter_sweep_json: {path}")
    if selected is not None:
        wall = selected["stage_eval"].get("stage_wall_avoid", {})
        print(
            "stm_trace_sweep_selected: "
            f"params={selected['params']} score={selected['score']} "
            f"wall_success={wall.get('eval_success')} "
            f"wall_collisions={wall.get('collision_count')} "
            f"wall_ratio={wall.get('recurrent_to_input_current_ratio')} "
            f"wall_seq={wall.get('action_sequence')}"
        )
    return summary


def _run_curriculum_final_map_once_for_sweep(
    *,
    seed: int,
    curriculum_stages: list[Dict[str, Any]],
    final_eval_map_name: str,
    final_eval_episodes: int,
) -> Dict[str, Any]:
    encoder = build_encoder()
    net = build_network(encoder=encoder, seed=seed)
    learner = build_learner()

    stage_eval: Dict[str, Dict[str, Any]] = {}
    for stage_idx, stage in enumerate(curriculum_stages):
        stage_name = str(stage.get("name", f"stage_{stage_idx}"))
        map_name = str(stage["map_name"])
        train_episodes = int(
            stage.get(
                "train_episodes",
                _stage_setting(stage_name, "train_episodes", N_EPISODES_TRAIN),
            )
        )
        eval_episodes = int(
            stage.get(
                "eval_episodes",
                _stage_setting(stage_name, "eval_episodes", N_EPISODES_EVAL),
            )
        )
        train_env = build_env_for_map(map_name, seed=seed + stage_idx)
        run_phase(
            env=train_env,
            net=net,
            learner=learner,
            n_episodes=train_episodes,
            phase_name=f"sweep_{stage_name}_train",
            verbose=False,
            log_name=f"{stage_name}_final_map_sweep",
        )
        eval_env = build_env_for_map(map_name, seed=seed + 1000 + stage_idx)
        _, eval_summary = run_phase(
            env=eval_env,
            net=net,
            learner=None,
            n_episodes=eval_episodes,
            phase_name=f"sweep_{stage_name}_eval",
            verbose=False,
            log_name=f"{stage_name}_final_map_sweep_eval",
            trace_first_steps=int(_cfg("EVAL_TRACE_FIRST_STEPS", 20)),
        )
        stage_eval[stage_name] = _sweep_stage_eval_summary(eval_summary)

    primitive_regression = _run_primitive_regression(
        net=net,
        seed=seed,
        verbose=False,
        save_json=False,
    )
    final_env = build_env_for_map(final_eval_map_name, seed=seed + 9000)
    _, final_eval_summary = run_phase(
        env=final_env,
        net=net,
        learner=None,
        n_episodes=final_eval_episodes,
        phase_name="final_eval",
        verbose=False,
        log_name="final_map_sweep_final_eval",
        trace_first_steps=int(_cfg("EVAL_TRACE_FIRST_STEPS", 20)),
    )
    final_diag = _final_eval_diagnostic_summary(final_eval_summary)
    return {
        "stage_eval": stage_eval,
        "primitive_regression_summary": primitive_regression,
        "final_eval_summary": final_diag,
    }


def _score_final_map_stm_sweep_result(result: Dict[str, Any]) -> float:
    primitive = result.get("primitive_regression_summary", {})
    primitive_stages = primitive.get("stages", {}) if isinstance(primitive, dict) else {}
    primitive_success = [
        float(item.get("eval_success", 0.0))
        for item in primitive_stages.values()
        if isinstance(item, dict)
    ]
    primitive_ok = bool(primitive_success) and all(value >= 1.0 for value in primitive_success)
    final_eval = result.get("final_eval_summary", {})
    success = float(final_eval.get("success_rate", 0.0))
    found = float(final_eval.get("found_victim_count", 0.0))
    collisions = float(final_eval.get("collision_count", 0.0))
    unique_cells = float(final_eval.get("unique_cells_visited_count", 0.0))
    stm = final_eval.get("stm_trace_summary", {}) if isinstance(final_eval, dict) else {}
    ratio = float(stm.get("recurrent_to_input_current_ratio", 0.0)) if isinstance(stm, dict) else 0.0
    action_hist = final_eval.get("action_histogram", {}) if isinstance(final_eval, dict) else {}
    total_actions = max(
        int(sum(int(v) for v in action_hist.values())) if isinstance(action_hist, dict) else 0,
        1,
    )
    action_collapse = (
        max((_hist_count(action_hist, action) for action in (0, 1, 2)), default=0)
        / float(total_actions)
    )
    ratio_penalty = 0.0
    if ratio < 0.05:
        ratio_penalty = 0.1 - ratio
    elif ratio > 1.5:
        ratio_penalty = ratio - 1.5
    return (
        200.0 * float(primitive_ok)
        + 50.0 * sum(primitive_success)
        + 120.0 * success
        + 15.0 * found
        + 0.5 * unique_cells
        - 0.2 * collisions
        - 25.0 * max(action_collapse - 0.85, 0.0)
        - 10.0 * ratio_penalty
    )


def run_final_map_stm_sweep(verbose: bool = False) -> Dict[str, Any]:
    seed = int(_cfg("SEED", 42))
    curriculum_stages = list(_cfg("CURRICULUM_STAGES", []))
    final_eval_map_name = str(
        _cfg("CURRICULUM_FINAL_EVAL_MAP", _cfg("CURRICULUM_EVAL_MAP_NAME", "rescue_map_6x9"))
    )
    final_eval_episodes = int(_cfg("CURRICULUM_FINAL_EVAL_EPISODES", N_EPISODES_EVAL))
    decay_values = _sweep_values("FINAL_MAP_STM_SWEEP_HIDDEN_TRACE_DECAYS", (0.5, 0.7))
    trace_scale_values = _sweep_values(
        "FINAL_MAP_STM_SWEEP_HIDDEN_TRACE_INPUT_SCALES",
        (1.0, 2.0, 3.0),
    )
    recurrent_scale_values = _sweep_values(
        "FINAL_MAP_STM_SWEEP_RECURRENT_CURRENT_SCALES",
        (0.25, 0.5, 0.75),
    )
    conductance_scale_values = _sweep_values(
        "FINAL_MAP_STM_SWEEP_CONDUCTANCE_SCALES",
        (2.0, 4.0),
    )
    candidate_grid = list(
        itertools.product(
            decay_values,
            trace_scale_values,
            recurrent_scale_values,
            conductance_scale_values,
        )
    )
    max_configs = int(_cfg("FINAL_MAP_STM_SWEEP_MAX_CONFIGS", 0))
    if max_configs > 0:
        candidate_grid = candidate_grid[:max_configs]

    config_names = (
        "ENABLE_CROSS_DECISION_HIDDEN_TRACE",
        "HIDDEN_TRACE_DECAY",
        "HIDDEN_TRACE_INPUT_SCALE",
        "STM_RECURRENT_CURRENT_SCALE",
        "STM_CONDUCTANCE_SCALE",
        "SAVE_POSITIVE_REWARD_STEPS_JSON",
    )
    saved = {name: getattr(cfg, name, None) for name in config_names}
    had_attr = {name: hasattr(cfg, name) for name in config_names}

    results = []
    try:
        cfg.ENABLE_CROSS_DECISION_HIDDEN_TRACE = True
        cfg.SAVE_POSITIVE_REWARD_STEPS_JSON = False
        for idx, (decay, trace_scale, recurrent_scale, conductance_scale) in enumerate(candidate_grid):
            cfg.HIDDEN_TRACE_DECAY = float(decay)
            cfg.HIDDEN_TRACE_INPUT_SCALE = float(trace_scale)
            cfg.STM_RECURRENT_CURRENT_SCALE = float(recurrent_scale)
            cfg.STM_CONDUCTANCE_SCALE = float(conductance_scale)
            run_result = _run_curriculum_final_map_once_for_sweep(
                seed=seed,
                curriculum_stages=curriculum_stages,
                final_eval_map_name=final_eval_map_name,
                final_eval_episodes=final_eval_episodes,
            )
            result = {
                "index": int(idx),
                "params": {
                    "HIDDEN_TRACE_DECAY": float(decay),
                    "HIDDEN_TRACE_INPUT_SCALE": float(trace_scale),
                    "STM_RECURRENT_CURRENT_SCALE": float(recurrent_scale),
                    "STM_CONDUCTANCE_SCALE": float(conductance_scale),
                },
                **run_result,
            }
            result["score"] = float(_score_final_map_stm_sweep_result(result))
            results.append(result)
            if verbose:
                final_eval = result.get("final_eval_summary", {})
                primitive = result.get("primitive_regression_summary", {})
                print(
                    f"final_map_sweep {idx + 1}/{len(candidate_grid)} "
                    f"params={result['params']} "
                    f"primitive_passed={primitive.get('passed')} "
                    f"final_success={final_eval.get('success_rate')} "
                    f"found={final_eval.get('found_victim_count')}/"
                    f"{final_eval.get('total_victim_count')} "
                    f"collisions={final_eval.get('collision_count')} "
                    f"failure={final_eval.get('failure_mode')} "
                    f"seq={final_eval.get('first_episode_action_sequence')}"
                )
    finally:
        for name, value in saved.items():
            if had_attr[name]:
                setattr(cfg, name, value)
            elif hasattr(cfg, name):
                delattr(cfg, name)

    selected = max(results, key=lambda item: float(item.get("score", 0.0))) if results else None
    summary = {
        "num_candidates": int(len(results)),
        "selection_criteria": [
            "primitive 4-stage regression must remain at eval_success=1.0",
            "final_eval victim_found_count or success_rate should improve",
            "collision_count should not grow excessively",
            "recurrent/input ratio should not dominate",
            "action histogram should not collapse to one action",
        ],
        "selected_config": selected,
        "results": results,
    }
    path = _write_json_log("final_map_stm_sweep.json", summary)
    print(f"final_map_stm_sweep_json: {path}")
    if selected is not None:
        final_eval = selected.get("final_eval_summary", {})
        print(
            "final_map_stm_sweep_selected: "
            f"params={selected.get('params')} score={selected.get('score')} "
            f"primitive_passed={selected.get('primitive_regression_summary', {}).get('passed')} "
            f"final_success={final_eval.get('success_rate')} "
            f"found={final_eval.get('found_victim_count')}/"
            f"{final_eval.get('total_victim_count')} "
            f"collisions={final_eval.get('collision_count')} "
            f"failure={final_eval.get('failure_mode')}"
        )
    return summary


# ============================================================
# Experiment entry point
# ============================================================
def run_experiment(verbose: bool = False) -> Dict[str, Any]:
    seed = int(_cfg("SEED", 42))

    encoder = build_encoder()
    env = build_env(seed=seed)
    net = build_network(encoder=encoder, seed=seed)
    learner = build_learner()

    if verbose:
        print_startup_env_summary(env)

    if verbose and bool(getattr(env, "use_fixed_maps", False)) and hasattr(env, "list_fixed_maps"):
        print("Fixed maps:", env.list_fixed_maps())
        print()

    baseline_results, baseline_summary = run_phase(
        env=env,
        net=net,
        learner=None,
        n_episodes=N_EPISODES_BASELINE,
        phase_name="baseline",
        verbose=verbose,
    )

    train_results, train_summary = run_phase(
        env=env,
        net=net,
        learner=learner,
        n_episodes=N_EPISODES_TRAIN,
        phase_name="train",
        verbose=verbose,
    )

    eval_results, eval_summary = run_phase(
        env=env,
        net=net,
        learner=None,
        n_episodes=N_EPISODES_EVAL,
        phase_name="eval",
        verbose=verbose,
    )

    final_summary = {
        "baseline": baseline_summary,
        "train": train_summary,
        "eval": eval_summary,
        "baseline_results": baseline_results,
        "train_results": train_results,
        "eval_results": eval_results,
    }
    if bool(_cfg("SAVE_FINAL_SUMMARY_JSON", True)):
        _write_json_log("single_final_summary.json", final_summary)

    if verbose:
        print("=" * 70)
        print("FINAL COMPARISON")
        print("=" * 70)
        print("baseline mean_reward:", baseline_summary["mean_reward"])
        print("train    mean_reward:", train_summary["mean_reward"])
        print("eval     mean_reward:", eval_summary["mean_reward"])
        print("baseline success_rate:", baseline_summary["success_rate"])
        print("eval     success_rate:", eval_summary["success_rate"])
        print("baseline mean_rescued_count:", baseline_summary["mean_rescued_count"])
        print("train    mean_rescued_count:", train_summary["mean_rescued_count"])
        print("eval     mean_rescued_count:", eval_summary["mean_rescued_count"])
        print("baseline mean_remaining_victims:", baseline_summary["mean_remaining_victims"])
        print("train    mean_remaining_victims:", train_summary["mean_remaining_victims"])
        print("eval     mean_remaining_victims:", eval_summary["mean_remaining_victims"])
        print("baseline mean_found_victim_count:", baseline_summary["mean_found_victim_count"])
        print("train    mean_found_victim_count:", train_summary["mean_found_victim_count"])
        print("eval     mean_found_victim_count:", eval_summary["mean_found_victim_count"])
        print("baseline mean_min_distance_to_victim:", baseline_summary["mean_min_distance_to_victim"])
        print("train    mean_min_distance_to_victim:", train_summary["mean_min_distance_to_victim"])
        print("eval     mean_min_distance_to_victim:", eval_summary["mean_min_distance_to_victim"])
        print("baseline mean_final_distance_to_victim:", baseline_summary["mean_final_distance_to_victim"])
        print("train    mean_final_distance_to_victim:", train_summary["mean_final_distance_to_victim"])
        print("eval     mean_final_distance_to_victim:", eval_summary["mean_final_distance_to_victim"])
        print("baseline mean_fallback_count:", baseline_summary["mean_fallback_count"])
        print("eval     mean_fallback_count:", eval_summary["mean_fallback_count"])
        print("baseline action_mean_reward:", baseline_summary["action_mean_reward"])
        print("train    action_mean_reward:", train_summary["action_mean_reward"])
        print("eval     action_mean_reward:", eval_summary["action_mean_reward"])
        print("baseline action_event_counts:", baseline_summary["action_event_counts"])
        print("train    action_event_counts:", train_summary["action_event_counts"])
        print("eval     action_event_counts:", eval_summary["action_event_counts"])
        print("baseline output_neuron_spike_histogram:", baseline_summary["output_neuron_spike_histogram"])
        print("train    output_neuron_spike_histogram:", train_summary["output_neuron_spike_histogram"])
        print("eval     output_neuron_spike_histogram:", eval_summary["output_neuron_spike_histogram"])
        print("baseline output_winner_histogram:", baseline_summary["output_winner_histogram"])
        print("train    output_winner_histogram:", train_summary["output_winner_histogram"])
        print("eval     output_winner_histogram:", eval_summary["output_winner_histogram"])
        print("baseline output_column_weight_mean:", baseline_summary["output_column_weight_mean"])
        print("train    output_column_weight_mean:", train_summary["output_column_weight_mean"])
        print("eval     output_column_weight_mean:", eval_summary["output_column_weight_mean"])
        print()
    else:
        print("single run summary")
        print(f"baseline success_rate: {baseline_summary['success_rate']}")
        print(f"train success_rate: {train_summary['success_rate']}")
        print(f"eval success_rate: {eval_summary['success_rate']}")
        print(f"eval action_histogram: {_metrics_from_summary(eval_summary).get('action_histogram', {})}")
        print(f"logs: {_log_dir()}")
        print()

    return final_summary


def run_curriculum(verbose: bool = False) -> Dict[str, Any]:
    seed = int(_cfg("SEED", 42))
    curriculum_stages = list(
        _cfg(
            "CURRICULUM_STAGES",
            [
                {
                    "name": "stage_forward",
                    "map_name": "easy_forward_1_victim",
                    "train_episodes": N_EPISODES_TRAIN,
                    "eval_episodes": N_EPISODES_EVAL,
                },
            ],
        )
    )
    final_eval_map_name = str(
        _cfg("CURRICULUM_FINAL_EVAL_MAP", _cfg("CURRICULUM_EVAL_MAP_NAME", "rescue_map_6x9"))
    )
    final_eval_episodes = int(_cfg("CURRICULUM_FINAL_EVAL_EPISODES", N_EPISODES_EVAL))

    encoder = build_encoder()
    active_encoder_feature_summary = _write_active_encoder_feature_reports(encoder)
    net = build_network(encoder=encoder, seed=seed)
    learner = build_learner()

    stage_records = []
    stage_summaries = []
    skipped_complex_stage_records: list[Dict[str, Any]] = []
    stable_wall_avoid_net_snapshot: Optional[MemristiveSNNNetwork] = None
    stable_wall_avoid_stage_index: Optional[int] = None
    primitive_regression_history: list[Dict[str, Any]] = []
    primitive_regression_before_complex: Optional[Dict[str, Any]] = None
    curriculum_stage_regression: Dict[str, Any] = {}
    primitive_full_regression_after_each_stage: Dict[str, Any] = {}
    primitive_core_training_summary: Dict[str, Any] = {
        "enabled": bool(_cfg("ENABLE_PRIMITIVE_CORE_TRAINING", True)),
        "primitive_core_checkpoint_found": False,
    }
    primitive_core_attempted = False
    primitive_core_checkpoint_found = False
    final_checkpoint_candidates: list[Dict[str, Any]] = []
    best_final_checkpoint_net: Optional[MemristiveSNNNetwork] = None
    best_final_checkpoint_summary: Optional[Dict[str, Any]] = None
    primitive_observation_alias_diagnostic = run_primitive_observation_alias_diagnostic(
        net=net,
        seed=seed,
        label="pre_curriculum_initial",
        save_json=True,
    )
    primitive_hidden_representation_diagnostic: Dict[str, Any] = {
        "enabled": False,
        "reason": "primitive core training not attempted yet",
    }

    print("=" * 70)
    print("CURRICULUM START")
    print("=" * 70)
    print("stages:", [str(stage.get("name", f"stage_{idx}")) for idx, stage in enumerate(curriculum_stages)])
    print("final_eval_map:", final_eval_map_name)
    print("active_encoder_features:", active_encoder_feature_summary.get("active_encoder_features", []))
    print("logs:", _log_dir())
    print()

    for stage_idx, stage in enumerate(curriculum_stages):
        stage_name = str(stage.get("name", f"stage_{stage_idx}"))
        map_name = str(stage["map_name"])
        primitive_names = set(_primitive_regression_stage_names())
        is_complex_stage = stage_name not in primitive_names
        default_complex_enabled = (
            True if not is_complex_stage else _default_complex_stage_enabled(stage_name)
        )
        if (
            is_complex_stage
            and bool(_cfg("ENABLE_PRIMITIVE_CORE_TRAINING", True))
            and not primitive_core_attempted
        ):
            primitive_core_attempted = True
            net, primitive_core_training_summary = _run_primitive_core_training(
                net=net,
                learner=learner,
                seed=seed + 60000,
                verbose=False,
            )
            primitive_core_checkpoint_found = bool(
                primitive_core_training_summary.get("primitive_core_checkpoint_found", False)
            )
            primitive_hidden_representation_diagnostic = (
                run_primitive_hidden_representation_diagnostic(
                    net=net,
                    seed=seed + 66000,
                    label="post_primitive_core_training",
                    save_json=True,
                )
            )
            if primitive_core_checkpoint_found and bool(
                _cfg("ENABLE_FINAL_MAP_CHECKPOINT_SELECTION", True)
            ):
                best_final_checkpoint_net = copy.deepcopy(net)
                best_final_checkpoint_summary = _probe_final_checkpoint(
                    net=net,
                    seed=seed + 65000,
                    label="primitive_core_checkpoint",
                    final_eval_map_name=final_eval_map_name,
                    primitive_passed=True,
                )
                final_checkpoint_candidates.append(best_final_checkpoint_summary)
        if (
            is_complex_stage
            and bool(_cfg("ENABLE_PRIMITIVE_CORE_TRAINING", True))
            and not primitive_core_checkpoint_found
        ):
            skipped = {
                "stage_name": stage_name,
                "map_name": map_name,
                "reason": "no primitive core checkpoint passed full regression",
                "primitive_core_checkpoint_found": False,
            }
            skipped_complex_stage_records.append(skipped)
            if verbose:
                print(
                    f"SKIP {stage_name}: no primitive core checkpoint passed "
                    "full regression"
                )
            continue
        branch_candidate_mode = bool(
            _cfg("ENABLE_BRANCH_CANDIDATE_CURRICULUM", True)
        ) and is_complex_stage and best_final_checkpoint_net is not None
        if is_complex_stage and not default_complex_enabled and not branch_candidate_mode:
            skipped = {
                "stage_name": stage_name,
                "map_name": map_name,
                "reason": "disabled_default_complex_stage_without_accepted_checkpoint",
                "default_complex_enabled": False,
                "branch_candidate_available": False,
            }
            skipped_complex_stage_records.append(skipped)
            if verbose:
                print(
                    f"SKIP {stage_name}: default complex stage disabled and no "
                    "accepted checkpoint is available for branch-candidate testing"
                )
            continue
        if branch_candidate_mode:
            net = copy.deepcopy(best_final_checkpoint_net)
        if is_complex_stage and primitive_regression_before_complex is None and stage_records:
            primitive_regression_before_complex = _primitive_regression_from_stage_records(
                stage_records,
                save_json=False,
            )
            primitive_regression_history.append(
                {
                    "label": "before_complex",
                    "after_stage_name": None,
                    "regression": primitive_regression_before_complex,
                    "forgotten_primitives": [],
                }
            )
        train_episodes = int(
            stage.get(
                "train_episodes",
                _stage_setting(stage_name, "train_episodes", N_EPISODES_TRAIN),
            )
        )
        eval_episodes = int(
            stage.get(
                "eval_episodes",
                _stage_setting(stage_name, "eval_episodes", N_EPISODES_EVAL),
            )
        )
        env = build_env_for_map(map_name, seed=seed + stage_idx)
        active_map_name = str(getattr(env, "current_map_name", None) or map_name)
        stage_start_net_snapshot = copy.deepcopy(net)
        stage_start_weight_mean = _output_column_weight_mean(net)
        stage_start_recurrent_status = _recurrent_crossbar_status(net)

        if verbose:
            print("=" * 70)
            print(f"CURRICULUM STAGE {stage_idx}: {stage_name}")
            print("=" * 70)
            print("active_map_name:", active_map_name)
            print("stage_start output_column_weight_mean:", stage_start_weight_mean)
            print("stage_start recurrent_crossbar_status:", stage_start_recurrent_status)
            print_startup_env_summary(env)

        train_results, train_summary = run_phase(
            env=env,
            net=net,
            learner=learner,
            n_episodes=train_episodes,
            phase_name=f"{stage_name}_train",
            verbose=verbose,
            log_name=stage_name,
        )
        stage_post_train_net_snapshot = copy.deepcopy(net)
        stage_train_weight_mean = _output_column_weight_mean(net)
        stage_train_recurrent_status = _recurrent_crossbar_status(net)
        selection_diag_source = copy.deepcopy(stage_post_train_net_snapshot)
        eval_env = build_env_for_map(map_name, seed=seed + 1000 + stage_idx)

        eval_results, eval_summary = run_phase(
            env=eval_env,
            net=copy.deepcopy(stage_post_train_net_snapshot),
            learner=None,
            n_episodes=eval_episodes,
            phase_name=f"{stage_name}_eval",
            verbose=verbose,
            log_name=f"{stage_name}_eval",
            trace_log_name=f"eval_trace_{_safe_log_name(stage_name)}.json",
            trace_first_steps=int(_cfg("EVAL_TRACE_FIRST_STEPS", 20)),
        )
        stage_end_weight_mean = _output_column_weight_mean(net)
        stage_end_recurrent_status = _recurrent_crossbar_status(net)
        eval_by_selection_mode = (
            run_eval_selection_mode_diagnostics(
                stage_name=stage_name,
                map_name=map_name,
                net=selection_diag_source,
                eval_episodes=eval_episodes,
                seed=seed,
                stage_idx=stage_idx,
                verbose=False,
            )
            if bool(_cfg("ENABLE_OUTPUT_SELECTION_MODE_DIAGNOSTICS", False))
            else {}
        )
        recurrent_ablation = run_eval_recurrent_ablation(
            stage_name=stage_name,
            map_name=map_name,
            net=selection_diag_source,
            eval_episodes=eval_episodes,
            seed=seed,
            stage_idx=stage_idx,
            verbose=False,
        )
        stm_scale_sweep = (
            run_stm_recurrent_scale_sweep(
                stage_name=stage_name,
                map_name=map_name,
                net=selection_diag_source,
                eval_episodes=eval_episodes,
                seed=seed,
                stage_idx=stage_idx,
                verbose=False,
            )
            if _is_wall_avoid_phase(stage_name)
            else {}
        )

        stage_record = {
            "stage_name": stage_name,
            "map_name": map_name,
            "active_map_name": active_map_name,
            "stage_start_output_column_weight_mean": stage_start_weight_mean,
            "stage_train_output_column_weight_mean": stage_train_weight_mean,
            "stage_end_output_column_weight_mean": stage_end_weight_mean,
            "stage_start_recurrent_crossbar_status": stage_start_recurrent_status,
            "stage_train_recurrent_crossbar_status": stage_train_recurrent_status,
            "stage_end_recurrent_crossbar_status": stage_end_recurrent_status,
            "stage_train": train_summary,
            "stage_eval": eval_summary,
            "eval_by_selection_mode": eval_by_selection_mode,
            "recurrent_ablation_eval": {stage_name: recurrent_ablation},
            "stm_recurrent_scale_sweep_eval": stm_scale_sweep,
            "train_results": train_results,
            "eval_results": eval_results,
            "interleaved_rehearsal": {},
            "primitive_regression_after_stage": {},
            "rollback": {"rolled_back": False},
            "branch_candidate": {
                "enabled": bool(branch_candidate_mode),
                "default_complex_enabled": bool(default_complex_enabled),
                "accepted": None if branch_candidate_mode else True,
            },
        }
        stage_record["stm_recurrent_interpretation"] = _stm_recurrent_interpretation(
            stage_record
        )
        if _is_wall_avoid_phase(stage_name):
            stage_record["wall_avoid_eval_by_selection_mode"] = eval_by_selection_mode

        if stage_name in primitive_names:
            full_regression = _run_primitive_regression(
                net=net,
                seed=seed + 76000 + stage_idx,
                verbose=False,
                save_json=False,
                eval_episodes_override=int(
                    _cfg("PRIMITIVE_REGRESSION_HISTORY_EVAL_EPISODES", 1)
                ),
            )
            primitive_full_regression_after_each_stage[stage_name] = full_regression
            stage_record["primitive_full_regression_after_stage"] = full_regression
            curriculum_stage_regression.setdefault(stage_name, {})[
                "primitive_full_regression_after_stage"
            ] = full_regression

        if is_complex_stage:
            rehearsal_summary = _run_interleaved_primitive_rehearsal(
                net=net,
                learner=learner,
                seed=seed + stage_idx * 37,
                after_stage_name=stage_name,
                final_stage=("stage_rescue_map_6x9_train" in stage_name),
                verbose=False,
            )
            stage_record["interleaved_rehearsal"] = rehearsal_summary
            regression_after = _run_primitive_regression(
                net=net,
                seed=seed + 8000 + stage_idx,
                verbose=False,
                save_json=False,
                eval_episodes_override=int(
                    _cfg("PRIMITIVE_REGRESSION_HISTORY_EVAL_EPISODES", 1)
                ),
            )
            forgotten_after = _forgotten_primitives(
                primitive_regression_before_complex,
                regression_after,
            )
            stage_record["primitive_regression_after_stage"] = regression_after
            stage_record["forgotten_primitives_after_stage"] = forgotten_after
            primitive_regression_history.append(
                {
                    "label": "after_complex_stage",
                    "after_stage_name": stage_name,
                    "regression": regression_after,
                    "forgotten_primitives": forgotten_after,
                }
            )
            curriculum_stage_regression[stage_name] = {
                "after_first_try": regression_after,
                "forgotten_primitives": forgotten_after,
                "interleaved_rehearsal": rehearsal_summary,
            }

            if (
                forgotten_after
                and bool(_cfg("ENABLE_STAGE_ROLLBACK_ON_PRIMITIVE_FAILURE", True))
            ):
                retry_scale = float(_cfg("STAGE_ROLLBACK_RETRY_SCALE", 0.5))
                retry_episodes = max(
                    int(_cfg("STAGE_ROLLBACK_MIN_TRAIN_EPISODES", 1)),
                    int(round(float(train_episodes) * retry_scale)),
                )
                saved_stage_scales = copy.deepcopy(_cfg("STAGE_RSTDP_SCALE", {}))
                if bool(_cfg("ENABLE_STAGEWISE_RSTDP_SCALE", False)) and isinstance(
                    getattr(cfg, "STAGE_RSTDP_SCALE", None),
                    dict,
                ):
                    current_scale = _safe_float(
                        cfg.STAGE_RSTDP_SCALE.get(stage_name, 1.0),
                        1.0,
                    )
                    cfg.STAGE_RSTDP_SCALE[stage_name] = float(current_scale * retry_scale)
                net = copy.deepcopy(stage_start_net_snapshot)
                retry_env = build_env_for_map(map_name, seed=seed + 9000 + stage_idx)
                retry_train_results, retry_train_summary = run_phase(
                    env=retry_env,
                    net=net,
                    learner=learner,
                    n_episodes=retry_episodes,
                    phase_name=f"{stage_name}_train",
                    verbose=verbose,
                    log_name=f"{stage_name}_rollback_retry",
                )
                retry_eval_env = build_env_for_map(map_name, seed=seed + 9500 + stage_idx)
                retry_eval_results, retry_eval_summary = run_phase(
                    env=retry_eval_env,
                    net=copy.deepcopy(net),
                    learner=None,
                    n_episodes=eval_episodes,
                    phase_name=f"{stage_name}_eval",
                    verbose=verbose,
                    log_name=f"{stage_name}_rollback_retry_eval",
                    trace_log_name=f"eval_trace_{_safe_log_name(stage_name)}_rollback_retry.json",
                    trace_first_steps=int(_cfg("EVAL_TRACE_FIRST_STEPS", 20)),
                )
                retry_rehearsal = _run_interleaved_primitive_rehearsal(
                    net=net,
                    learner=learner,
                    seed=seed + 9700 + stage_idx,
                    after_stage_name=f"{stage_name}_rollback_retry",
                    final_stage=("stage_rescue_map_6x9_train" in stage_name),
                    verbose=False,
                )
                retry_regression = _run_primitive_regression(
                    net=net,
                    seed=seed + 9800 + stage_idx,
                    verbose=False,
                    save_json=False,
                    eval_episodes_override=int(
                        _cfg("PRIMITIVE_REGRESSION_HISTORY_EVAL_EPISODES", 1)
                    ),
                )
                retry_forgotten = _forgotten_primitives(
                    primitive_regression_before_complex,
                    retry_regression,
                )
                if isinstance(getattr(cfg, "STAGE_RSTDP_SCALE", None), dict):
                    cfg.STAGE_RSTDP_SCALE.clear()
                    cfg.STAGE_RSTDP_SCALE.update(saved_stage_scales)

                stage_record.update(
                    {
                        "stage_train": retry_train_summary,
                        "stage_eval": retry_eval_summary,
                        "train_results": retry_train_results,
                        "eval_results": retry_eval_results,
                        "stage_train_output_column_weight_mean": _output_column_weight_mean(net),
                        "stage_end_output_column_weight_mean": _output_column_weight_mean(net),
                        "stage_train_recurrent_crossbar_status": _recurrent_crossbar_status(net),
                        "stage_end_recurrent_crossbar_status": _recurrent_crossbar_status(net),
                        "interleaved_rehearsal": retry_rehearsal,
                        "primitive_regression_after_stage": retry_regression,
                        "forgotten_primitives_after_stage": retry_forgotten,
                        "rollback": {
                            "rolled_back": True,
                            "reason": "primitive_regression_failed_after_stage",
                            "forgotten_primitives_before_retry": forgotten_after,
                            "forgotten_primitives_after_retry": retry_forgotten,
                            "retry_train_episodes": int(retry_episodes),
                            "retry_stage_rstdp_scale_factor": float(retry_scale),
                        },
                    }
                )
                curriculum_stage_regression[stage_name]["after_rollback_retry"] = retry_regression
                curriculum_stage_regression[stage_name]["forgotten_primitives_after_retry"] = retry_forgotten
                primitive_regression_history.append(
                    {
                        "label": "after_rollback_retry",
                        "after_stage_name": stage_name,
                        "regression": retry_regression,
                        "forgotten_primitives": retry_forgotten,
                    }
                )

                if retry_forgotten and bool(
                    _cfg("DISCARD_COMPLEX_STAGE_ON_PRIMITIVE_FAILURE", True)
                ):
                    net = copy.deepcopy(stage_start_net_snapshot)
                    discard_regression = _run_primitive_regression(
                        net=net,
                        seed=seed + 9900 + stage_idx,
                        verbose=False,
                        save_json=False,
                        eval_episodes_override=int(
                            _cfg("PRIMITIVE_REGRESSION_HISTORY_EVAL_EPISODES", 1)
                        ),
                    )
                    stage_record["primitive_regression_after_discard"] = discard_regression
                    stage_record["forgotten_primitives_after_stage"] = _forgotten_primitives(
                        primitive_regression_before_complex,
                        discard_regression,
                    )
                    stage_record["rollback"]["discarded_stage_update"] = True
                    stage_record["rollback"]["discard_reason"] = (
                        "rollback retry still failed primitive regression"
                    )
                    stage_record["stage_train_output_column_weight_mean"] = stage_start_weight_mean
                    stage_record["stage_end_output_column_weight_mean"] = _output_column_weight_mean(net)
                    stage_record["stage_train_recurrent_crossbar_status"] = stage_start_recurrent_status
                    stage_record["stage_end_recurrent_crossbar_status"] = _recurrent_crossbar_status(net)
                    curriculum_stage_regression[stage_name]["after_discard"] = discard_regression
                    primitive_regression_history.append(
                        {
                            "label": "after_discard_to_stage_start",
                            "after_stage_name": stage_name,
                            "regression": discard_regression,
                            "forgotten_primitives": stage_record[
                                "forgotten_primitives_after_stage"
                            ],
                        }
                    )

        if is_complex_stage and bool(_cfg("ENABLE_FINAL_MAP_CHECKPOINT_SELECTION", True)):
            current_regression = (
                stage_record.get("primitive_regression_after_discard")
                or stage_record.get("primitive_regression_after_stage")
                or {}
            )
            primitive_passed_current = bool(current_regression.get("passed", False))
            candidate = _probe_final_checkpoint(
                net=net,
                seed=seed + 13000 + stage_idx,
                label=f"{stage_name}_checkpoint",
                final_eval_map_name=final_eval_map_name,
                primitive_passed=primitive_passed_current,
            )
            final_checkpoint_candidates.append(candidate)
            score_improved = _candidate_improves(candidate, best_final_checkpoint_summary)
            reject_reasons = _branch_reject_reasons(
                primitive_passed=primitive_passed_current,
                score_improved=score_improved,
            )
            accepted = len(reject_reasons) == 0
            candidate["score_improved"] = bool(score_improved)
            candidate["accepted"] = bool(accepted)
            candidate["reject_reasons"] = list(reject_reasons)
            candidate_diag = candidate.get("final_eval_summary", {})
            stage_record["branch_candidate"].update(
                {
                    "enabled": bool(branch_candidate_mode),
                    "default_complex_enabled": bool(default_complex_enabled),
                    "accepted": bool(accepted),
                    "reject_reasons": reject_reasons,
                    "primitive_passed": bool(primitive_passed_current),
                    "score_improved": bool(score_improved),
                    "candidate_label": candidate.get("label"),
                    "candidate_score": float(candidate.get("score", -1e18)),
                    "previous_best_label": (
                        None
                        if best_final_checkpoint_summary is None
                        else best_final_checkpoint_summary.get("label")
                    ),
                    "previous_best_score": (
                        None
                        if best_final_checkpoint_summary is None
                        else float(best_final_checkpoint_summary.get("score", -1e18))
                    ),
                    "probe_found_victim_count": int(
                        candidate_diag.get("found_victim_count", 0)
                    ),
                    "probe_collision_count": int(
                        candidate_diag.get("collision_count", 0)
                    ),
                    "probe_failure_mode": candidate_diag.get("failure_mode", "unknown"),
                }
            )
            curriculum_stage_regression.setdefault(stage_name, {})[
                "branch_candidate"
            ] = stage_record["branch_candidate"]
            if accepted:
                stage_record["branch_candidate"]["committed_to_accepted_best"] = True
                best_final_checkpoint_summary = candidate
                best_final_checkpoint_net = copy.deepcopy(net)
            elif branch_candidate_mode and best_final_checkpoint_net is not None:
                stage_record["branch_candidate"]["reverted_to_previous_accepted"] = True
                net = copy.deepcopy(best_final_checkpoint_net)
            else:
                stage_record["branch_candidate"]["reverted_to_stage_start"] = True
                net = copy.deepcopy(stage_start_net_snapshot)
        stage_records.append(stage_record)
        stage_summaries.append(stage_record)

        if stable_wall_avoid_net_snapshot is None:
            primitive_names = set(_primitive_regression_stage_names())
            stage_by_name = {record["stage_name"]: record for record in stage_records}
            if primitive_names.issubset(stage_by_name.keys()):
                primitive_eval_ok = all(
                    float(stage_by_name[name]["stage_eval"].get("success_rate", 0.0)) >= 1.0
                    for name in primitive_names
                )
                if primitive_eval_ok:
                    stable_wall_avoid_net_snapshot = copy.deepcopy(stage_post_train_net_snapshot)
                    stable_wall_avoid_stage_index = int(stage_idx)
                    if bool(_cfg("ENABLE_FINAL_MAP_CHECKPOINT_SELECTION", True)):
                        primitive_candidate_net = copy.deepcopy(stage_post_train_net_snapshot)
                        stable_regression = _run_primitive_regression(
                            net=primitive_candidate_net,
                            seed=seed + 12100 + stage_idx,
                            verbose=False,
                            save_json=False,
                            eval_episodes_override=int(
                                _cfg("PRIMITIVE_REGRESSION_HISTORY_EVAL_EPISODES", 1)
                            ),
                        )
                        consolidation_summary = {
                            "enabled": False,
                            "passed": bool(stable_regression.get("passed", False)),
                            "final_regression": stable_regression,
                        }
                        if not bool(stable_regression.get("passed", False)):
                            consolidation_summary = (
                                _run_primitive_consolidation_before_complex(
                                    net=primitive_candidate_net,
                                    learner=learner,
                                    seed=seed + 12200 + stage_idx,
                                    after_stage_name=stage_name,
                                    verbose=False,
                                )
                            )
                            stable_regression = consolidation_summary.get(
                                "final_regression",
                                stable_regression,
                            )
                        stable_primitive_passed = bool(stable_regression.get("passed", False))
                        candidate = _probe_final_checkpoint(
                            net=primitive_candidate_net,
                            seed=seed + 12000 + stage_idx,
                            label=f"{stage_name}_primitive_checkpoint",
                            final_eval_map_name=final_eval_map_name,
                            primitive_passed=stable_primitive_passed,
                        )
                        final_checkpoint_candidates.append(candidate)
                        score_improved = _candidate_improves(
                            candidate,
                            best_final_checkpoint_summary,
                        )
                        reject_reasons = _branch_reject_reasons(
                            primitive_passed=stable_primitive_passed,
                            score_improved=score_improved,
                        )
                        candidate["score_improved"] = bool(score_improved)
                        candidate["accepted"] = len(reject_reasons) == 0
                        candidate["reject_reasons"] = list(reject_reasons)
                        stage_record["stable_checkpoint_candidate"] = {
                            "label": candidate.get("label"),
                            "primitive_passed": bool(stable_primitive_passed),
                            "primitive_regression": stable_regression,
                            "primitive_consolidation": consolidation_summary,
                            "score": float(candidate.get("score", -1e18)),
                            "score_improved": bool(score_improved),
                            "accepted": len(reject_reasons) == 0,
                            "reject_reasons": reject_reasons,
                        }
                        if not reject_reasons:
                            best_final_checkpoint_net = copy.deepcopy(primitive_candidate_net)
                            best_final_checkpoint_summary = candidate
                            net = copy.deepcopy(primitive_candidate_net)

        _maybe_save_json(
            f"stage_summary_{_safe_log_name(stage_name)}.json",
            stage_record,
            "SAVE_STAGE_SUMMARY_JSON",
        )
        if verbose or bool(_cfg("PRINT_STAGE_COMPACT_SUMMARY", False)):
            _print_compact_stage_summary(stage_record)

    intermediate_curriculum_summary = _build_intermediate_curriculum_summary(
        stage_records
    )
    primitive_rehearsal_summary = _run_primitive_rehearsal(
        net=net,
        learner=learner,
        seed=seed,
        verbose=False,
    )
    if bool(_cfg("ENABLE_PRIMITIVE_CORE_TRAINING", True)) and not primitive_core_checkpoint_found:
        best_final_checkpoint_net = None
        best_final_checkpoint_summary = None
    deployment_uses_checkpoint = (
        bool(_cfg("ENABLE_FINAL_MAP_CHECKPOINT_SELECTION", True))
        and best_final_checkpoint_net is not None
    )
    deployment_net = copy.deepcopy(
        best_final_checkpoint_net if deployment_uses_checkpoint else net
    )
    live_post_curriculum_primitive_regression_summary = _run_primitive_regression(
        net=net,
        seed=seed,
        verbose=False,
        filename="live_post_curriculum_primitive_regression_summary.json",
    )
    selected_checkpoint_primitive_regression_summary = _run_primitive_regression(
        net=deployment_net,
        seed=seed,
        verbose=False,
        filename="selected_checkpoint_primitive_regression_summary.json",
    )
    post_curriculum_primitive_regression_summary = (
        selected_checkpoint_primitive_regression_summary
    )
    primitive_stage_immediate_eval_summary = _primitive_regression_from_stage_records(
        stage_records,
        save_json=True,
    )
    primitive_regression_summary = primitive_stage_immediate_eval_summary
    forgotten_primitives = _forgotten_primitives(
        primitive_stage_immediate_eval_summary,
        post_curriculum_primitive_regression_summary,
    )
    _write_json_log("primitive_regression_history.json", primitive_regression_history)
    _write_json_log("curriculum_stage_regression.json", curriculum_stage_regression)
    _write_json_log(
        "primitive_full_regression_after_each_stage.json",
        primitive_full_regression_after_each_stage,
    )
    stable_wall_avoid_success_summary = _build_stable_wall_avoid_success_summary(
        stage_records,
        primitive_regression_summary=primitive_stage_immediate_eval_summary,
    )
    final_eval_net = copy.deepcopy(deployment_net)
    eval_env = build_env_for_map(final_eval_map_name, seed=seed + len(curriculum_stages))

    if verbose:
        print("=" * 70)
        print(f"CURRICULUM FINAL EVAL: {final_eval_map_name}")
        print("=" * 70)
        print_startup_env_summary(eval_env)

    eval_results, eval_summary = run_phase(
        env=eval_env,
        net=final_eval_net,
        learner=None,
        n_episodes=final_eval_episodes,
        phase_name="final_eval",
        verbose=verbose,
        log_name="final_eval",
        trace_log_name=(
            f"final_eval_trace_{_safe_log_name(final_eval_map_name)}.json"
            if bool(_cfg("SAVE_FINAL_EVAL_TRACE_JSON", True))
            else None
        ),
        trace_first_steps=int(_cfg("EVAL_TRACE_FIRST_STEPS", 20)),
    )
    final_eval_diagnostic_summary = _final_eval_diagnostic_summary(eval_summary)

    final_summary = {
        "curriculum_stages": curriculum_stages,
        "stage_summaries": stage_summaries,
        "stage_records": stage_records,
        "primitive_stage_immediate_eval_summary": primitive_stage_immediate_eval_summary,
        "primitive_regression_summary": primitive_regression_summary,
        "primitive_full_regression_after_each_stage": primitive_full_regression_after_each_stage,
        "primitive_observation_alias_diagnostic": primitive_observation_alias_diagnostic,
        "primitive_hidden_representation_diagnostic": primitive_hidden_representation_diagnostic,
        "active_encoder_feature_summary": active_encoder_feature_summary,
        "real_robot_sensor_feasibility_report": real_robot_sensor_feasibility_report(),
        "primitive_core_checkpoint_found": bool(primitive_core_checkpoint_found),
        "primitive_core_training_summary": primitive_core_training_summary,
        "primitive_core_action_confusion": (
            primitive_core_training_summary.get("final_regression", {})
            .get("primitive_action_confusion", {})
            if isinstance(primitive_core_training_summary, dict)
            else {}
        ),
        "complex_training_skipped_reason": (
            "no primitive core checkpoint passed full regression"
            if bool(_cfg("ENABLE_PRIMITIVE_CORE_TRAINING", True))
            and not primitive_core_checkpoint_found
            else None
        ),
        "primitive_regression_before_complex": primitive_regression_before_complex,
        "primitive_regression_history": primitive_regression_history,
        "post_curriculum_primitive_regression_summary": post_curriculum_primitive_regression_summary,
        "selected_checkpoint_primitive_regression_summary": selected_checkpoint_primitive_regression_summary,
        "live_post_curriculum_primitive_regression_summary": live_post_curriculum_primitive_regression_summary,
        "forgotten_primitives": forgotten_primitives,
        "curriculum_stage_regression": curriculum_stage_regression,
        "skipped_complex_stages": skipped_complex_stage_records,
        "default_complex_stage_enabled": _cfg("DEFAULT_COMPLEX_STAGE_ENABLED", {}),
        "final_checkpoint_candidates": final_checkpoint_candidates,
        "best_final_checkpoint": best_final_checkpoint_summary,
        "selected_checkpoint_label": (
            None
            if best_final_checkpoint_summary is None
            else best_final_checkpoint_summary.get("label")
        ),
        "final_eval_uses_checkpoint_selection": bool(deployment_uses_checkpoint),
        "final_eval_diagnostic_only": bool(
            bool(_cfg("ENABLE_PRIMITIVE_CORE_TRAINING", True))
            and not primitive_core_checkpoint_found
        ),
        "intermediate_curriculum_summary": intermediate_curriculum_summary,
        "primitive_rehearsal_summary": primitive_rehearsal_summary,
        "stable_wall_avoid_success_summary": stable_wall_avoid_success_summary,
        "stable_wall_avoid_checkpoint_stage_index": (
            None if stable_wall_avoid_stage_index is None else int(stable_wall_avoid_stage_index)
        ),
        "no_action_debug_by_stage": {
            record["stage_name"]: {
                "valid_actions_only": bool(
                    record.get("stage_eval", {}).get("valid_actions_only", True)
                ),
                "no_action_count": int(
                    record.get("stage_eval", {}).get(
                        "eval_first_episode_no_action_count",
                        0,
                    )
                ),
                "early_done_before_action": bool(
                    record.get("stage_eval", {}).get(
                        "eval_first_episode_early_done_before_action",
                        False,
                    )
                ),
                "no_spike_count": int(
                    record.get("stage_eval", {}).get(
                        "eval_first_episode_no_spike_count",
                        0,
                    )
                ),
                "fallback_count": int(
                    sum(
                        int(v)
                        for v in record.get("stage_eval", {})
                        .get("fallback_action_histogram", {})
                        .values()
                    )
                    if isinstance(
                        record.get("stage_eval", {}).get("fallback_action_histogram", {}),
                        dict,
                    )
                    else 0
                ),
                "action_sequence": record.get("stage_eval", {}).get(
                    "eval_first_episode_action_sequence",
                    [],
                ),
                "raw_action_sequence": record.get("stage_eval", {}).get(
                    "eval_first_episode_raw_action_sequence",
                    [],
                ),
            }
            for record in stage_records
        },
        "stage_mini_rescue_easy_debug": next(
            (
                {
                    "valid_actions_only": bool(
                        record.get("stage_eval", {}).get("valid_actions_only", True)
                    ),
                    "no_action_count": int(
                        record.get("stage_eval", {}).get(
                            "eval_first_episode_no_action_count",
                            0,
                        )
                    ),
                    "early_done_before_action": bool(
                        record.get("stage_eval", {}).get(
                            "eval_first_episode_early_done_before_action",
                            False,
                        )
                    ),
                    "no_spike_count": int(
                        record.get("stage_eval", {}).get(
                            "eval_first_episode_no_spike_count",
                            0,
                        )
                    ),
                    "action_sequence": record.get("stage_eval", {}).get(
                        "eval_first_episode_action_sequence",
                        [],
                    ),
                    "raw_action_sequence": record.get("stage_eval", {}).get(
                        "eval_first_episode_raw_action_sequence",
                        [],
                    ),
                }
                for record in stage_records
                if record.get("stage_name") == "stage_mini_rescue_easy"
            ),
            {},
        ),
        "final_eval_summary": final_eval_diagnostic_summary,
        "selected_checkpoint_final_eval_summary": final_eval_diagnostic_summary,
        "failure_mode": final_eval_diagnostic_summary.get("failure_mode", "unknown"),
        "output_column_weight_mean_changes": [
            {
                "stage_name": record["stage_name"],
                "map_name": record["map_name"],
                "start": record["stage_start_output_column_weight_mean"],
                "after_train": record["stage_train_output_column_weight_mean"],
                "end": record["stage_end_output_column_weight_mean"],
            }
            for record in stage_records
        ],
        "recurrent_crossbar_status_changes": [
            {
                "stage_name": record["stage_name"],
                "map_name": record["map_name"],
                "start": record.get("stage_start_recurrent_crossbar_status", {}),
                "after_train": record.get("stage_train_recurrent_crossbar_status", {}),
                "end": record.get("stage_end_recurrent_crossbar_status", {}),
            }
            for record in stage_records
        ],
        "recurrent_ablation_eval": {
            record["stage_name"]: record.get("recurrent_ablation_eval", {}).get(
                record["stage_name"],
                {},
            )
            for record in stage_records
        },
        "stm_recurrent_scale_sweep_eval": {
            record["stage_name"]: record.get("stm_recurrent_scale_sweep_eval", {})
            for record in stage_records
            if record.get("stm_recurrent_scale_sweep_eval", {})
        },
        "stm_recurrent_interpretation": {
            record["stage_name"]: record.get("stm_recurrent_interpretation", {})
            for record in stage_records
        },
        "eval_map": final_eval_map_name,
        "eval": eval_summary,
        "eval_results": eval_results,
    }
    final_summary_path = _maybe_save_json(
        "final_summary.json",
        final_summary,
        "SAVE_FINAL_SUMMARY_JSON",
    )

    print("=" * 70)
    print("CURRICULUM FINAL SUMMARY")
    print("=" * 70)
    print("primitive_stage_immediate_eval_summary:")
    for name, item in primitive_stage_immediate_eval_summary.get("stages", {}).items():
        print(
            f"  {name}: eval_success={item.get('eval_success')} "
            f"collisions={item.get('collision_count')} "
            f"seq={item.get('action_sequence')}"
        )
    if primitive_stage_immediate_eval_summary.get("warnings"):
        print(
            "primitive_stage_immediate_eval_warning: "
            f"{primitive_stage_immediate_eval_summary['warnings']}"
        )
    _print_primitive_observation_alias_diagnostic(
        primitive_observation_alias_diagnostic
    )
    _print_primitive_hidden_representation_diagnostic(
        primitive_hidden_representation_diagnostic
    )
    print(
        "primitive_core_checkpoint_found: "
        f"{primitive_core_training_summary.get('primitive_core_checkpoint_found', False)}"
    )
    core_confusion = (
        primitive_core_training_summary.get("final_regression", {})
        .get("primitive_action_confusion", {})
        if isinstance(primitive_core_training_summary, dict)
        else {}
    )
    if core_confusion:
        print("primitive_action_confusion:")
        for name, row in core_confusion.get("action_histogram_confusion", {}).items():
            print(f"  {name}: {row}")
    if bool(_cfg("ENABLE_PRIMITIVE_CORE_TRAINING", True)) and not bool(
        primitive_core_training_summary.get("primitive_core_checkpoint_found", False)
    ):
        print(
            "complex_training_skipped_reason: "
            "no primitive core checkpoint passed full regression"
        )
    print("primitive_full_regression_after_each_stage:")
    for name, regression in primitive_full_regression_after_each_stage.items():
        warnings = regression.get("warnings", []) if isinstance(regression, dict) else []
        print(
            f"  after {name}: passed={regression.get('passed') if isinstance(regression, dict) else None} "
            f"warnings={warnings}"
        )
    print("selected_checkpoint_regression:")
    for name, item in selected_checkpoint_primitive_regression_summary.get("stages", {}).items():
        print(
            f"  {name}: eval_success={item.get('eval_success')} "
            f"collisions={item.get('collision_count')} "
            f"seq={item.get('action_sequence')}"
        )
    selected_warnings = selected_checkpoint_primitive_regression_summary.get("warnings", [])
    if selected_warnings:
        print(
            "selected_checkpoint_regression_warning: "
            f"{selected_warnings}"
        )
    live_post_warnings = live_post_curriculum_primitive_regression_summary.get(
        "warnings",
        [],
    )
    print(
        "live_post_curriculum_regression: "
        f"diagnostic_passed={live_post_curriculum_primitive_regression_summary.get('passed')}"
    )
    if live_post_warnings:
        print(
            "live_post_curriculum_regression_warning: "
            f"{live_post_warnings}"
        )
    print(f"forgotten_primitives: {forgotten_primitives}")
    if skipped_complex_stage_records:
        print(
            "skipped_complex_stages: "
            f"{[item.get('stage_name') for item in skipped_complex_stage_records]}"
        )

    print("intermediate_curriculum_summary:")
    for name, item in intermediate_curriculum_summary.get("stages", {}).items():
        eval_item = item.get("eval", {})
        print(
            f"  {name}: eval_success={eval_item.get('eval_success')} "
            f"found={eval_item.get('found_victim_count')} "
            f"collisions={eval_item.get('collision_count')} "
            f"seq={eval_item.get('action_sequence')}"
        )

    if primitive_rehearsal_summary.get("enabled"):
        print("primitive_rehearsal_summary:")
        for name, item in primitive_rehearsal_summary.get("stages", {}).items():
            eval_item = item.get("eval", {}) if isinstance(item, dict) else {}
            print(
                f"  {name}: eval_success={eval_item.get('eval_success')} "
                f"collisions={eval_item.get('collision_count')} "
                f"seq={eval_item.get('action_sequence')}"
            )

    if best_final_checkpoint_summary is not None:
        best_diag = best_final_checkpoint_summary.get("final_eval_summary", {})
        print(
            "best_final_checkpoint: "
            f"label={best_final_checkpoint_summary.get('label')} "
            f"score={best_final_checkpoint_summary.get('score')} "
            f"primitive_passed={best_final_checkpoint_summary.get('primitive_passed')} "
            f"probe_found={best_diag.get('found_victim_count')}/"
            f"{best_diag.get('total_victim_count')} "
            f"probe_collisions={best_diag.get('collision_count')} "
            f"probe_failure={best_diag.get('failure_mode')}"
        )

    print("selected_checkpoint_final_eval_summary:")
    print(
        f"  map={final_eval_map_name} "
        f"deployment_source={'selected_checkpoint' if deployment_uses_checkpoint else 'live_net'} "
        f"success_rate={final_eval_diagnostic_summary['success_rate']} "
        f"partial={final_eval_diagnostic_summary['partial_success_rate']} "
        f"all_success={final_eval_diagnostic_summary['all_victims_success_rate']} "
        f"found={final_eval_diagnostic_summary['found_victim_count']}/"
        f"{final_eval_diagnostic_summary['total_victim_count_all_episodes']} "
        f"collisions={final_eval_diagnostic_summary['collision_count']} "
        f"moved={final_eval_diagnostic_summary['moved_count']} "
        f"unique_cells={final_eval_diagnostic_summary['unique_cells_visited_count']} "
        f"failure_mode={final_eval_diagnostic_summary['failure_mode']}"
    )
    print(
        f"  action_histogram={final_eval_diagnostic_summary['action_histogram']} "
        f"seq={final_eval_diagnostic_summary['first_episode_action_sequence']}"
    )
    stm_trace_summary = final_eval_diagnostic_summary.get("stm_trace_summary", {})
    print(
        "STM trace summary: "
        f"scope={stm_trace_summary.get('recurrent_memory_scope')} "
        f"hidden_trace_enabled={stm_trace_summary.get('hidden_trace_enabled')} "
        f"ratio={stm_trace_summary.get('recurrent_to_input_current_ratio')} "
        f"hidden_trace_mean={stm_trace_summary.get('hidden_trace_mean')} "
        f"hidden_trace_nonzero={stm_trace_summary.get('hidden_trace_nonzero_count')}"
    )
    if final_summary_path is not None:
        print(f"final_summary_json: {final_summary_path}")
    print()

    return final_summary

if __name__ == "__main__":
    run_mode = str(_cfg("RUN_MODE", "single")).lower()
    if run_mode in {"stm_trace_sweep", "stm_sweep", "trace_sweep"}:
        run_stm_trace_parameter_sweep(verbose=True)
    elif run_mode in {"final_map_stm_sweep", "rescue_map_stm_sweep"}:
        run_final_map_stm_sweep(verbose=True)
    elif run_mode in {"primitive_core_sweep", "primitive_core_search"}:
        run_primitive_core_search_sweep(verbose=True)
    elif run_mode == "curriculum":
        run_curriculum(verbose=False)
    else:
        run_experiment(verbose=False)
