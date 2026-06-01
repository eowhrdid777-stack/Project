from __future__ import annotations

"""Real-robot environment wrapper for the rescue SNN.

This class keeps the same high-level API as env.AbstractRescueGridEnv:
    reset() -> observation dict
    step(action) -> EnvStepResult

It does not generate random maps.  It assumes the physical map is one of your
predefined fixed maps and that the robot starts exactly on the map's start cell
with the heading encoded by ^ > v < or by fixed_agent_heading.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from robot_interface import ArduinoRobotInterface, RobotCalibration
from env import EnvStepResult, MapSpec, AbstractRescueGridEnv
from sensor_features import augment_observation_with_sensor_features

GridPos = Tuple[int, int]


@dataclass
class RealRobotEnvConfig:
    max_steps: int = 50
    reward_step_penalty: float = 0.0
    reward_collision: float = -0.05
    reward_closer: float = 0.10
    reward_farther: float = -0.03
    reward_found_victim: float = 3.0
    reward_stay: float = -0.02
    reward_turn: float = -0.002
    fixed_agent_heading: int = 0
    map_selection_mode: str = "fixed_index"
    fixed_map_index: int = 0
    # A real robot can drift.  If True, we still trust the known map state for
    # reward/position; if False, only sensor-derived rewards are used.
    trust_dead_reckoning: bool = True
    # DFR0034 sound_signal threshold for sensor-based survivor detection.
    sound_threshold: float = 0.55


class RealRescueRobotEnv:
    ACTION_NAMES = AbstractRescueGridEnv.ACTION_NAMES
    HEADING_NAMES = AbstractRescueGridEnv.HEADING_NAMES
    HEADING_FROM_CHAR = AbstractRescueGridEnv.HEADING_FROM_CHAR
    SENSOR_TRACE_KEYS = AbstractRescueGridEnv.SENSOR_TRACE_KEYS

    def __init__(
        self,
        interface: ArduinoRobotInterface,
        fixed_maps: Sequence[MapSpec],
        cfg: Optional[RealRobotEnvConfig] = None,
    ) -> None:
        self.interface = interface
        self.cfg = RealRobotEnvConfig() if cfg is None else cfg
        self.max_steps = int(self.cfg.max_steps)
        self.map_selection_mode = self.cfg.map_selection_mode
        self.fixed_map_index = int(self.cfg.fixed_map_index)

        # Reuse the simulation env parser/validator for map specs, but never use
        # its random reset path for real hardware.
        self._map_env = AbstractRescueGridEnv(
            max_steps=self.max_steps,
            use_fixed_maps=True,
            fixed_maps=fixed_maps,
            map_selection_mode="fixed_index",
            fixed_map_index=self.fixed_map_index,
            fixed_agent_heading=self.cfg.fixed_agent_heading,
        )
        self.fixed_maps = self._map_env.fixed_maps
        self.current_map_index: Optional[int] = None
        self.current_map_name: Optional[str] = None
        self.agent_pos: GridPos = (0, 0)
        self.agent_heading: int = 0
        self.victim_positions: Set[GridPos] = set()
        self.initial_victim_count: int = 0
        self.rescued_count: int = 0
        self.step_count: int = 0
        self.done: bool = False
        self.last_distance_to_victim: float = 0.0
        self._previous_sensor_obs_for_delta: Dict[str, float] = {}
        self._last_action_for_observation: Optional[int] = None
        self._last_collision_for_observation: bool = False
        self._last_moved_for_observation: bool = False

    def reset(self, map_index: Optional[int] = None) -> Dict[str, float]:
        idx = self.fixed_map_index if map_index is None else int(map_index)
        self._map_env.reset(map_index=idx)
        st = self._map_env.get_env_state()
        self.current_map_index = idx
        self.current_map_name = st["map_name"]
        self.agent_pos = tuple(st["agent_pos"])  # type: ignore[assignment]
        self.agent_heading = int(st["agent_heading"])
        self.victim_positions = set(tuple(p) for p in st["victim_positions"])
        self.initial_victim_count = int(st["initial_victim_count"])
        self.rescued_count = 0
        self.step_count = 0
        self.done = False
        self.last_distance_to_victim = self._nearest_victim_distance(self.agent_pos)
        base_obs = self._base_observation()
        self._reset_temporal_observation_state(base_obs)
        return self._with_temporal_features(base_obs)

    def _base_observation(self) -> Dict[str, float]:
        return augment_observation_with_sensor_features(
            self.interface.get_observation(),
            heading=int(self.agent_heading),
        )

    def _reset_temporal_observation_state(self, base_obs: Dict[str, float]) -> None:
        self._previous_sensor_obs_for_delta = dict(base_obs)
        self._last_action_for_observation = None
        self._last_collision_for_observation = False
        self._last_moved_for_observation = False

    def _with_temporal_features(self, obs: Dict[str, float]) -> Dict[str, float]:
        out = dict(obs)
        prev = self._previous_sensor_obs_for_delta or obs
        out["front_delta"] = float(
            out.get("front_clearance", 0.0) - prev.get("front_clearance", out.get("front_clearance", 0.0))
        )
        out["left_delta"] = float(
            out.get("left_clearance", 0.0) - prev.get("left_clearance", out.get("left_clearance", 0.0))
        )
        out["right_delta"] = float(
            out.get("right_clearance", 0.0) - prev.get("right_clearance", out.get("right_clearance", 0.0))
        )
        out["victim_signal_delta"] = float(
            out.get("victim_signal", 0.0) - prev.get("victim_signal", out.get("victim_signal", 0.0))
        )
        out["sound_signal_delta"] = float(
            out.get("sound_signal", 0.0) - prev.get("sound_signal", out.get("sound_signal", 0.0))
        )
        action = self._last_action_for_observation
        out["last_action_forward"] = float(action == 0)
        out["last_action_left"] = float(action == 1)
        out["last_action_right"] = float(action == 2)
        out["last_collision"] = float(bool(self._last_collision_for_observation))
        out["last_moved"] = float(bool(self._last_moved_for_observation))
        return out

    def get_observation(self) -> Dict[str, float]:
        return self._with_temporal_features(self._base_observation())

    def step(self, action: int) -> EnvStepResult:
        if self.done:
            return EnvStepResult(self.get_observation(), 0.0, True, {"warning": "Episode already done."})
        action = int(action)
        if action not in self.ACTION_NAMES:
            raise ValueError(f"Unsupported action {action}")

        old_pos = self.agent_pos
        old_heading = self.agent_heading
        old_distance = self._nearest_victim_distance(old_pos)

        # Pre-read for collision-risk logging; Arduino firmware should also block
        # unsafe forward movement based on its front ToF sensor.
        before_sensor_obs = self._base_observation()
        before_obs = self._with_temporal_features(before_sensor_obs)
        ack = self.interface.execute_action(action)
        after_sensor_obs = self._base_observation()

        collision = bool(ack.get("collision", False))
        moved = False
        if action == 1:
            self.agent_heading = (self.agent_heading - 1) % 4
        elif action == 2:
            self.agent_heading = (self.agent_heading + 1) % 4
        elif action == 0:
            next_pos = self._forward_pos(self.agent_pos, self.agent_heading)
            if self._is_blocked(next_pos) or collision:
                collision = True
            else:
                self.agent_pos = next_pos
                moved = True

        self._previous_sensor_obs_for_delta = dict(before_sensor_obs)
        self._last_action_for_observation = int(action)
        self._last_collision_for_observation = bool(collision)
        self._last_moved_for_observation = bool(moved)
        after_obs = self._with_temporal_features(after_sensor_obs)

        self.step_count += 1
        found_victim = False
        if self.agent_pos in self.victim_positions:
            # Map-based success signal.  This is more reliable than color alone if
            # your physical map is aligned and predefined.
            found_victim = True
        elif float(after_obs.get("victim_signal", 0.0)) >= 0.5:
            # Sensor-based fallback if a victim marker is detected by TCS34725.
            found_victim = True
        elif float(after_obs.get("sound_signal", 0.0)) >= float(self.cfg.sound_threshold):
            # Optional DFR0034 sound-intensity fallback. This should be used as
            # a coarse survivor cue, not as speech recognition.
            found_victim = True

        new_distance = self._nearest_victim_distance(self.agent_pos)
        reward = float(self.cfg.reward_step_penalty)
        if collision:
            reward += float(self.cfg.reward_collision)
        else:
            if new_distance < old_distance:
                reward += float(self.cfg.reward_closer)
            elif new_distance > old_distance:
                reward += float(self.cfg.reward_farther)
        if action == 3:
            reward += float(self.cfg.reward_stay)
        elif action in (1, 2):
            reward += float(self.cfg.reward_turn)

        if found_victim:
            if self.agent_pos in self.victim_positions:
                self.victim_positions.remove(self.agent_pos)
            self.rescued_count += 1
            reward += float(self.cfg.reward_found_victim)
            try:
                self.interface.beep(120)
            except Exception:
                pass
            if not self.victim_positions:
                self.done = True

        if self.step_count >= self.max_steps:
            self.done = True
        self.last_distance_to_victim = new_distance

        info = {
            "action_name": self.ACTION_NAMES[action],
            "old_pos": old_pos,
            "new_pos": self.agent_pos,
            "old_heading": self.HEADING_NAMES[old_heading],
            "new_heading": self.HEADING_NAMES[self.agent_heading],
            "collision": collision,
            "moved": moved,
            "found_victim": found_victim,
            "remaining_victims": len(self.victim_positions),
            "rescued_count": self.rescued_count,
            "step_count": self.step_count,
            "map_index": self.current_map_index,
            "map_name": self.current_map_name,
            "arduino_ack": ack,
            "obs_before": before_obs,
            "obs_after": after_obs,
        }
        return EnvStepResult(after_obs, float(reward), bool(self.done), info)

    def render_ascii(self) -> str:
        # Use the simulation renderer by copying tracked state into it.
        self._map_env.agent_pos = self.agent_pos
        self._map_env.agent_heading = self.agent_heading
        self._map_env.victim_positions = set(self.victim_positions)
        return self._map_env.render_ascii()

    def _forward_pos(self, pos: GridPos, heading: int) -> GridPos:
        return self._map_env._forward_pos(pos, heading)

    def _is_blocked(self, pos: GridPos) -> bool:
        return self._map_env._is_blocked(pos)

    def _nearest_victim_distance(self, pos: GridPos) -> float:
        if not self.victim_positions:
            return 0.0
        return min(abs(pos[0] - v[0]) + abs(pos[1] - v[1]) for v in self.victim_positions)
