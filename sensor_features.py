from __future__ import annotations

"""Real-robot-feasible observation feature helpers.

The functions here only derive additional sensor channels. They do not select
or override actions. All downstream action choices still go through the
encoder -> crossbar read -> neuron spike -> output winner path.
"""

from typing import Any, Dict, Iterable, List, Optional

import math

import config as cfg


def _cfg(name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


BASIC_FEATURES = (
    "front_clearance",
    "left_clearance",
    "right_clearance",
    "victim_signal",
    "sound_signal",
)
OBSTACLE_BINARY_FEATURES = (
    "front_open",
    "front_blocked",
    "left_open",
    "left_blocked",
    "right_open",
    "right_blocked",
)
SIDE_COMPARISON_FEATURES = (
    "right_more_open_than_left",
    "left_more_open_than_right",
    "front_more_open_than_sides",
    "sides_more_open_than_front",
)
COLOR_RISK_FEATURES = (
    "risk_signal",
    "safe_floor_signal",
)
HEADING_FEATURES = (
    "heading_sin",
    "heading_cos",
)
PROHIBITED_ENCODER_FEATURES = (
    "ground_truth_victim_position",
    "grid_x",
    "grid_y",
    "shortest_path_direction",
    "map_cell_type_as_action_hint",
    "last_action",
    "last_action_forward",
    "last_action_left",
    "last_action_right",
    "last_collision",
    "last_moved",
    "found_victim_count_as_input",
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clip01(value: Any) -> float:
    value_f = _as_float(value, 0.0)
    if not math.isfinite(value_f):
        value_f = 0.0
    return float(min(1.0, max(0.0, value_f)))


def _list_cfg(name: str, default: Iterable[str]) -> List[str]:
    raw = _cfg(name, tuple(default))
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in raw]


def active_encoder_feature_names() -> List[str]:
    if bool(_cfg("ENABLE_REAL_ROBOT_FEASIBLE_SENSOR_EXPANSION", True)):
        names = list(
            _cfg(
                "ACTIVE_ENCODER_FEATURES",
                list(BASIC_FEATURES)
                + list(OBSTACLE_BINARY_FEATURES)
                + list(SIDE_COMPARISON_FEATURES),
            )
        )
    else:
        names = _list_cfg("ENCODER_FEATURE_NAMES", BASIC_FEATURES)

    if bool(_cfg("ENABLE_COLOR_RISK_FEATURES", False)):
        names.extend(_list_cfg("COLOR_RISK_FEATURES", COLOR_RISK_FEATURES))
    if bool(_cfg("ENABLE_HEADING_FEATURES", False)):
        names.extend(_list_cfg("HEADING_FEATURES", HEADING_FEATURES))

    deduped: List[str] = []
    for name in names:
        name = str(name)
        if name not in deduped:
            deduped.append(name)
    validate_active_encoder_features(deduped)
    return deduped


def active_encoder_value_ranges() -> Dict[str, tuple[float, float]]:
    return {name: (0.0, 1.0) for name in active_encoder_feature_names()}


def validate_active_encoder_features(feature_names: Iterable[str]) -> None:
    active = {str(name) for name in feature_names}
    banned = sorted(active & set(PROHIBITED_ENCODER_FEATURES))
    if banned:
        raise ValueError(
            "Prohibited encoder features are active: "
            f"{banned}. These would leak policy memory, map coordinates, or "
            "ground-truth planning hints into action selection."
        )

    sound_directional = {"sound_left", "sound_right", "sound_front", "sound_back"}
    directional_sound = sorted(active & sound_directional)
    if directional_sound and not bool(_cfg("ENABLE_DIRECTIONAL_SOUND_FEATURES", False)):
        raise ValueError(
            "Directional sound features require multiple microphones or a scan "
            f"mechanism and are disabled by default: {directional_sound}"
        )


def augment_observation_with_sensor_features(
    obs: Dict[str, Any],
    *,
    danger_zone: Optional[bool] = None,
    heading: Optional[int] = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {str(key): _clip01(value) for key, value in dict(obs).items()}

    front = _clip01(out.get("front_clearance", 0.0))
    left = _clip01(out.get("left_clearance", 0.0))
    right = _clip01(out.get("right_clearance", 0.0))
    out["front_clearance"] = front
    out["left_clearance"] = left
    out["right_clearance"] = right
    out["victim_signal"] = _clip01(out.get("victim_signal", 0.0))
    out["sound_signal"] = _clip01(out.get("sound_signal", out["victim_signal"]))

    if bool(_cfg("ENABLE_REAL_ROBOT_FEASIBLE_SENSOR_EXPANSION", True)):
        open_th = _as_float(_cfg("CLEARANCE_OPEN_THRESHOLD", 0.6), 0.6)
        blocked_th = _as_float(_cfg("CLEARANCE_BLOCKED_THRESHOLD", 0.25), 0.25)

        out["front_open"] = float(front >= open_th)
        out["front_blocked"] = float(front <= blocked_th)
        out["left_open"] = float(left >= open_th)
        out["left_blocked"] = float(left <= blocked_th)
        out["right_open"] = float(right >= open_th)
        out["right_blocked"] = float(right <= blocked_th)

        side_max = max(left, right)
        out["right_more_open_than_left"] = _clip01(max(right - left, 0.0))
        out["left_more_open_than_right"] = _clip01(max(left - right, 0.0))
        out["front_more_open_than_sides"] = _clip01(max(front - side_max, 0.0))
        out["sides_more_open_than_front"] = _clip01(max(side_max - front, 0.0))

    if bool(_cfg("ENABLE_COLOR_RISK_FEATURES", False)):
        risk = float(bool(danger_zone)) if danger_zone is not None else _clip01(out.get("risk_signal", 0.0))
        out["risk_signal"] = _clip01(risk)
        out["safe_floor_signal"] = _clip01(1.0 - out["risk_signal"])

    if bool(_cfg("ENABLE_HEADING_FEATURES", False)) and heading is not None:
        angle = (int(heading) % 4) * (math.pi / 2.0)
        out["heading_sin"] = _clip01(0.5 + 0.5 * math.sin(angle))
        out["heading_cos"] = _clip01(0.5 + 0.5 * math.cos(angle))

    return out


def real_robot_sensor_feasibility_report() -> Dict[str, Any]:
    active = set(active_encoder_feature_names())
    rows: Dict[str, Dict[str, Any]] = {
        "front_clearance": {
            "source": "front distance sensor",
            "real_robot_feasible": True,
        },
        "left_clearance": {
            "source": "left distance sensor",
            "real_robot_feasible": True,
        },
        "right_clearance": {
            "source": "right distance sensor",
            "real_robot_feasible": True,
        },
        "victim_signal": {
            "source": "color sensor or victim marker detector",
            "real_robot_feasible": True,
        },
        "sound_signal": {
            "source": "single DFR0034 sound intensity sensor",
            "real_robot_feasible": True,
        },
        "front_open": {
            "source": "derived from front distance sensor",
            "real_robot_feasible": True,
        },
        "front_blocked": {
            "source": "derived from front distance sensor",
            "real_robot_feasible": True,
        },
        "left_open": {
            "source": "derived from left distance sensor",
            "real_robot_feasible": True,
        },
        "left_blocked": {
            "source": "derived from left distance sensor",
            "real_robot_feasible": True,
        },
        "right_open": {
            "source": "derived from right distance sensor",
            "real_robot_feasible": True,
        },
        "right_blocked": {
            "source": "derived from right distance sensor",
            "real_robot_feasible": True,
        },
        "right_more_open_than_left": {
            "source": "derived from left/right distance sensors",
            "real_robot_feasible": True,
        },
        "left_more_open_than_right": {
            "source": "derived from left/right distance sensors",
            "real_robot_feasible": True,
        },
        "front_more_open_than_sides": {
            "source": "derived from front/left/right distance sensors",
            "real_robot_feasible": True,
        },
        "sides_more_open_than_front": {
            "source": "derived from front/left/right distance sensors",
            "real_robot_feasible": True,
        },
        "risk_signal": {
            "source": "color sensor danger-zone color detector",
            "real_robot_feasible": True,
            "optional": True,
        },
        "safe_floor_signal": {
            "source": "inverse of color sensor danger-zone signal",
            "real_robot_feasible": True,
            "optional": True,
        },
        "heading_sin": {
            "source": "IMU or encoder odometry",
            "real_robot_feasible": "optional",
            "optional": True,
        },
        "heading_cos": {
            "source": "IMU or encoder odometry",
            "real_robot_feasible": "optional",
            "optional": True,
        },
    }
    for name, item in rows.items():
        item["enabled"] = bool(name in active)
    return {
        "active_encoder_features": active_encoder_feature_names(),
        "prohibited_encoder_features": list(PROHIBITED_ENCODER_FEATURES),
        "features": rows,
    }
