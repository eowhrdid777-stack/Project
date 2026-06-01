from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from sensor_features import augment_observation_with_sensor_features


Action = int
GridPos = Tuple[int, int]
GridRows = Sequence[str]
MapSpec = Union[GridRows, Dict[str, Any]]


@dataclass
class EnvStepResult:
    observation: Dict[str, float]
    reward: float
    done: bool
    info: Dict[str, Any]


class AbstractRescueGridEnv:
    """
    구조 로봇용 grid-world 환경

    기본 문자:
        # : 칸 자체가 벽
        . or space : 빈칸
        A/S/R : 로봇 시작 위치
        ^ > v < : 로봇 시작 위치 + 방향
        V : 생존자
        D : 위험구역

    edge wall:
        h_walls : (r,c)와 (r+1,c) 사이 벽
        v_walls : (r,c)와 (r,c+1) 사이 벽
    """

    ACTION_NAMES = {
        0: "forward",
        1: "turn_left",
        2: "turn_right",
        3: "stay",
    }

    HEADING_NAMES = {
        0: "up",
        1: "right",
        2: "down",
        3: "left",
    }

    HEADING_FROM_CHAR = {
        "^": 0,
        ">": 1,
        "v": 2,
        "<": 3,
    }

    SENSOR_TRACE_KEYS = (
        "front_clearance",
        "left_clearance",
        "right_clearance",
        "victim_signal",
        "sound_signal",
    )

    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        max_steps: int = 50,
        obstacle_density: float = 0.12,
        seed: Optional[int] = None,
        victim_signal_sigma: float = 2.2,
        victim_detection_radius: Optional[float] = None,
        reward_step_penalty: float = 0.0,
        reward_collision: float = -0.05,
        reward_closer: float = 0.10,
        reward_farther: float = -0.03,
        reward_found_victim: float = 3.0,
        reward_danger: Optional[float] = None,
        reward_stay: float = -0.02,
        reward_turn: float = -0.002,
        use_random_heading_on_reset: bool = True,
        n_victims: int = 3,
        ensure_reachable: bool = True,
        generation_max_attempts: int = 200,
        use_fixed_maps: Optional[bool] = None,
        fixed_maps: Optional[Sequence[MapSpec]] = None,
        fixed_agent_heading: Optional[int] = None,
        map_selection_mode: Optional[str] = None,
        fixed_map_index: Optional[int] = None,
    ) -> None:
        cfg = self._maybe_config()

        if fixed_maps is None:
            fixed_maps = self._cfg(cfg, "ENV_FIXED_MAPS", None)
            if fixed_maps is None:
                single_map = self._cfg(cfg, "ENV_FIXED_MAP", None)
                if single_map is not None:
                    fixed_maps = [single_map]

        if use_fixed_maps is None:
            use_fixed_maps = bool(self._cfg(cfg, "ENV_USE_FIXED_MAPS", False))
            if not use_fixed_maps:
                use_fixed_maps = bool(self._cfg(cfg, "ENV_USE_FIXED_MAP", False))

        if fixed_agent_heading is None:
            fixed_agent_heading = int(self._cfg(cfg, "ENV_FIXED_AGENT_HEADING", 0))

        if map_selection_mode is None:
            map_selection_mode = str(self._cfg(cfg, "ENV_MAP_SELECTION_MODE", "cycle"))

        if fixed_map_index is None:
            fixed_map_index = int(self._cfg(cfg, "ENV_FIXED_MAP_INDEX", 0))

        if reward_danger is None:
            reward_danger = float(self._cfg(cfg, "ENV_REWARD_DANGER", -1.0))

        if victim_detection_radius is None:
            victim_detection_radius = float(self._cfg(cfg, "ENV_VICTIM_DETECTION_RADIUS", 0.0))

        if width < 1 or height < 1:
            raise ValueError("width and height must be >= 1")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if not (0.0 <= obstacle_density < 0.45):
            raise ValueError("obstacle_density must be in [0, 0.45)")
        if n_victims < 1:
            raise ValueError("n_victims must be >= 1")
        if victim_signal_sigma <= 0.0:
            raise ValueError("victim_signal_sigma must be > 0")
        if float(victim_detection_radius) < 0.0:
            raise ValueError("victim_detection_radius must be >= 0")
        if generation_max_attempts < 1:
            raise ValueError("generation_max_attempts must be >= 1")

        self.width = int(width)
        self.height = int(height)
        self.max_steps = int(max_steps)
        self.obstacle_density = float(obstacle_density)
        self.victim_signal_sigma = float(victim_signal_sigma)
        self.victim_detection_radius = float(victim_detection_radius)

        self.reward_step_penalty = float(reward_step_penalty)
        self.reward_collision = float(reward_collision)
        self.reward_closer = float(reward_closer)
        self.reward_farther = float(reward_farther)
        self.reward_found_victim = float(reward_found_victim)
        self.reward_danger = float(reward_danger)
        self.reward_stay = float(reward_stay)
        self.reward_turn = float(reward_turn)

        self.use_random_heading_on_reset = bool(use_random_heading_on_reset)
        self.n_victims = int(n_victims)
        self.ensure_reachable = bool(ensure_reachable)
        self.generation_max_attempts = int(generation_max_attempts)

        self.use_fixed_maps = bool(use_fixed_maps)
        self.fixed_agent_heading = int(fixed_agent_heading) % 4
        self.map_selection_mode = str(map_selection_mode).lower()
        self.fixed_map_index = int(fixed_map_index)
        self.fixed_maps: List[Dict[str, Any]] = self._normalize_fixed_maps(fixed_maps)

        self.current_map_index: Optional[int] = None
        self.current_map_name: Optional[str] = None
        self._map_cursor: int = 0

        if self.use_fixed_maps and not self.fixed_maps:
            raise ValueError("use_fixed_maps=True but no fixed maps were provided.")
        if self.use_fixed_maps and self.map_selection_mode not in {"cycle", "random", "fixed", "fixed_index"}:
            raise ValueError("map_selection_mode must be 'cycle', 'random', 'fixed', or 'fixed_index'")

        self.rng = np.random.default_rng(seed)

        self.grid: np.ndarray = np.zeros((self.height, self.width), dtype=np.int8)
        self.agent_pos: GridPos = (0, 0)
        self.agent_heading: int = 0

        self.victim_positions: Set[GridPos] = set()
        self.danger_positions: Set[GridPos] = set()

        # edge wall
        self.h_walls: Set[GridPos] = set()
        self.v_walls: Set[GridPos] = set()

        self.step_count: int = 0
        self.done: bool = False

        self.last_distance_to_victim: float = 0.0
        self.episode_index: int = 0
        self.rescued_count: int = 0
        self.initial_victim_count: int = self.n_victims
        self._previous_sensor_obs_for_delta: Dict[str, float] = {}
        self._last_action_for_observation: Optional[int] = None
        self._last_collision_for_observation: bool = False
        self._last_moved_for_observation: bool = False

        self.reset()
        self.episode_index = 0
        if self.use_fixed_maps and self.map_selection_mode == "cycle":
            self._map_cursor = 0

    # ============================================================
    # reset
    # ============================================================
    def reset(
        self,
        seed: Optional[int] = None,
        map_index: Optional[int] = None,
    ) -> Dict[str, float]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.episode_index += 1
        self.step_count = 0
        self.done = False
        self.rescued_count = 0

        if self.use_fixed_maps:
            idx = self._select_fixed_map_index(map_index)
            self._reset_from_fixed_map(idx)

            if self.ensure_reachable and not self._all_victims_reachable_from_agent():
                raise RuntimeError(
                    f"Fixed map index {idx} is not reachable: at least one victim cannot be reached."
                )

            self.last_distance_to_victim = self._nearest_victim_distance(self.agent_pos)
            self._reset_temporal_observation_state()
            return self.get_observation()

        last_error: Optional[Exception] = None

        for _ in range(self.generation_max_attempts):
            try:
                self._generate_random_episode_layout()
                if (not self.ensure_reachable) or self._all_victims_reachable_from_agent():
                    self.agent_heading = (
                        int(self.rng.integers(0, 4))
                        if self.use_random_heading_on_reset
                        else 0
                    )
                    self.last_distance_to_victim = self._nearest_victim_distance(self.agent_pos)
                    self._reset_temporal_observation_state()
                    return self.get_observation()
            except Exception as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise RuntimeError("Failed to generate a valid random episode layout.") from last_error

        raise RuntimeError("Failed to generate a reachable random episode layout.")

    def _select_fixed_map_index(self, requested_index: Optional[int]) -> int:
        if not self.fixed_maps:
            raise RuntimeError("No fixed maps are available.")

        n_maps = len(self.fixed_maps)

        if requested_index is not None:
            idx = int(requested_index)
        elif self.map_selection_mode == "random":
            idx = int(self.rng.integers(0, n_maps))
        elif self.map_selection_mode in {"fixed", "fixed_index"}:
            idx = int(self.fixed_map_index)
        else:
            idx = self._map_cursor
            self._map_cursor = (self._map_cursor + 1) % n_maps

        if not (0 <= idx < n_maps):
            raise IndexError(f"Fixed map index out of range: {idx}")

        return idx

    def _reset_from_fixed_map(self, map_index: int) -> None:
        spec = self.fixed_maps[int(map_index)]
        rows = spec["rows"]
        heading_from_spec = spec.get("heading", None)
        name = spec.get("name", f"fixed_map_{map_index}")

        height = len(rows)
        width = len(rows[0])

        self.height = int(height)
        self.width = int(width)

        self.grid = np.zeros((self.height, self.width), dtype=np.int8)
        self.victim_positions = set()
        self.danger_positions = set()
        self.h_walls = set()
        self.v_walls = set()

        agent_positions: List[Tuple[GridPos, Optional[int]]] = []

        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                pos = (r, c)

                if ch == "#":
                    self.grid[r, c] = 1

                elif ch in {".", " "}:
                    self.grid[r, c] = 0

                elif ch in {"A", "S", "R"}:
                    self.grid[r, c] = 0
                    agent_positions.append((pos, None))

                elif ch in self.HEADING_FROM_CHAR:
                    self.grid[r, c] = 0
                    agent_positions.append((pos, self.HEADING_FROM_CHAR[ch]))

                elif ch == "V":
                    self.grid[r, c] = 0
                    self.victim_positions.add(pos)

                elif ch == "D":
                    self.grid[r, c] = 0
                    self.danger_positions.add(pos)

                else:
                    raise ValueError(f"Unknown map character {ch!r} at row={r}, col={c}")

        if len(agent_positions) != 1:
            raise ValueError(
                f"Fixed map {name!r} must contain exactly one agent start symbol."
            )

        if not self.victim_positions:
            raise ValueError(f"Fixed map {name!r} must contain at least one victim symbol V.")

        self.h_walls = self._normalize_h_walls(spec.get("h_walls", []), self.height, self.width)
        self.v_walls = self._normalize_v_walls(spec.get("v_walls", []), self.height, self.width)

        self.agent_pos = agent_positions[0][0]
        heading_from_symbol = agent_positions[0][1]

        if heading_from_symbol is not None:
            self.agent_heading = int(heading_from_symbol) % 4
        elif heading_from_spec is not None:
            self.agent_heading = int(heading_from_spec) % 4
        else:
            self.agent_heading = self.fixed_agent_heading

        self.current_map_index = int(map_index)
        self.current_map_name = str(name)
        self.n_victims = int(len(self.victim_positions))
        self.initial_victim_count = int(len(self.victim_positions))

    def _generate_random_episode_layout(self) -> None:
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)
        self.victim_positions = set()
        self.danger_positions = set()
        self.h_walls = set()
        self.v_walls = set()

        self.current_map_index = None
        self.current_map_name = None

        n_cells = self.width * self.height
        n_obstacles = int(round(self.obstacle_density * n_cells))

        obstacle_coords: Set[GridPos] = set()
        while len(obstacle_coords) < n_obstacles:
            r = int(self.rng.integers(0, self.height))
            c = int(self.rng.integers(0, self.width))
            obstacle_coords.add((r, c))

        for r, c in obstacle_coords:
            self.grid[r, c] = 1

        free = self._free_cells()
        required_cells = 1 + self.n_victims

        if len(free) < required_cells:
            raise RuntimeError("Not enough free cells.")

        agent_idx = int(self.rng.integers(0, len(free)))
        self.agent_pos = free.pop(agent_idx)

        for _ in range(self.n_victims):
            victim_idx = int(self.rng.integers(0, len(free)))
            self.victim_positions.add(free.pop(victim_idx))

        self.initial_victim_count = len(self.victim_positions)

    # ============================================================
    # step
    # ============================================================
    def step(self, action: Action) -> EnvStepResult:
        if self.done:
            return EnvStepResult(
                observation=self.get_observation(),
                reward=0.0,
                done=True,
                info={"warning": "Episode already done."},
            )

        action = int(action)
        if action not in self.ACTION_NAMES:
            raise ValueError(f"Unsupported action {action}.")

        previous_sensor_obs = self._base_observation()
        old_pos = self.agent_pos
        old_heading = self.agent_heading
        old_distance = self._nearest_victim_distance(old_pos)

        collision = False
        moved = False

        if action == 1:
            self.agent_heading = (self.agent_heading - 1) % 4

        elif action == 2:
            self.agent_heading = (self.agent_heading + 1) % 4

        elif action == 0:
            next_pos = self._forward_pos(old_pos, old_heading)

            if self._is_move_blocked(old_pos, next_pos):
                collision = True
            else:
                self.agent_pos = next_pos
                moved = True

        elif action == 3:
            pass

        self.step_count += 1

        nearest_victim_pos = None
        new_distance = self._nearest_victim_distance(self.agent_pos)
        if self.victim_positions:
            nearest_victim_pos = min(
                self.victim_positions,
                key=lambda victim_pos: self._distance(self.agent_pos, victim_pos),
            )
        found_victim = (
            nearest_victim_pos is not None
            and new_distance <= self.victim_detection_radius
        )
        danger_zone = self.agent_pos in self.danger_positions

        reward = float(self.reward_step_penalty)

        if collision:
            reward += self.reward_collision
        else:
            if new_distance < old_distance:
                reward += self.reward_closer
            elif new_distance > old_distance:
                reward += self.reward_farther

        if danger_zone:
            reward += self.reward_danger

        if action == 3:
            reward += self.reward_stay
        elif action in (1, 2):
            reward += self.reward_turn

        if found_victim:
            self.victim_positions.remove(nearest_victim_pos)
            self.rescued_count += 1
            reward += self.reward_found_victim

            if not self.victim_positions:
                self.done = True

        if self.step_count >= self.max_steps:
            self.done = True

        self.last_distance_to_victim = self._nearest_victim_distance(self.agent_pos)
        self._previous_sensor_obs_for_delta = dict(previous_sensor_obs)
        self._last_action_for_observation = int(action)
        self._last_collision_for_observation = bool(collision)
        self._last_moved_for_observation = bool(moved)

        info = {
            "action_name": self.ACTION_NAMES[action],
            "old_pos": old_pos,
            "new_pos": self.agent_pos,
            "old_heading": self.HEADING_NAMES[old_heading],
            "new_heading": self.HEADING_NAMES[self.agent_heading],
            "collision": bool(collision),
            "moved": bool(moved),
            "danger_zone": bool(danger_zone),
            "distance_to_nearest_victim": float(new_distance),
            "found_victim": bool(found_victim),
            "found_victim_pos": nearest_victim_pos if found_victim else None,
            "victim_detection_radius": float(self.victim_detection_radius),
            "remaining_victims": int(len(self.victim_positions)),
            "rescued_count": int(self.rescued_count),
            "step_count": int(self.step_count),
            "map_index": self.current_map_index,
            "map_name": self.current_map_name,
        }

        return EnvStepResult(
            observation=self.get_observation(),
            reward=float(reward),
            done=bool(self.done),
            info=info,
        )

    # ============================================================
    # observation
    # ============================================================
    def _base_observation(self) -> Dict[str, float]:
        front_clearance = self._directional_clearance(self.agent_heading)
        left_clearance = self._directional_clearance((self.agent_heading - 1) % 4)
        right_clearance = self._directional_clearance((self.agent_heading + 1) % 4)
        victim_signal = self._victim_signal_strength()

        obs = {
            "front_clearance": float(front_clearance),
            "left_clearance": float(left_clearance),
            "right_clearance": float(right_clearance),
            "victim_signal": float(victim_signal),
            # 현재 시뮬레이션에서는 victim_signal을 생존자 소리 신호처럼 사용
            "sound_signal": float(victim_signal),
        }
        obs = augment_observation_with_sensor_features(
            obs,
            danger_zone=bool(self.agent_pos in self.danger_positions),
            heading=int(self.agent_heading),
        )

        self._validate_observation(obs)
        return obs

    def _reset_temporal_observation_state(self) -> None:
        base_obs = self._base_observation()
        self._previous_sensor_obs_for_delta = dict(base_obs)
        self._last_action_for_observation = None
        self._last_collision_for_observation = False
        self._last_moved_for_observation = False

    @classmethod
    def _with_temporal_features(
        cls,
        obs: Dict[str, float],
        previous_obs: Optional[Dict[str, float]],
        last_action: Optional[int],
        last_collision: bool,
        last_moved: bool,
    ) -> Dict[str, float]:
        out = dict(obs)
        prev = previous_obs if previous_obs is not None else obs
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
        action = None if last_action is None else int(last_action)
        out["last_action_forward"] = float(action == 0)
        out["last_action_left"] = float(action == 1)
        out["last_action_right"] = float(action == 2)
        out["last_collision"] = float(bool(last_collision))
        out["last_moved"] = float(bool(last_moved))
        return out

    def get_observation(self) -> Dict[str, float]:
        base_obs = self._base_observation()
        return self._with_temporal_features(
            base_obs,
            self._previous_sensor_obs_for_delta,
            self._last_action_for_observation,
            self._last_collision_for_observation,
            self._last_moved_for_observation,
        )

    # ============================================================
    # fixed-map utility
    # ============================================================
    def num_fixed_maps(self) -> int:
        return len(self.fixed_maps)

    def set_fixed_map_index(self, map_index: int) -> None:
        if not (0 <= int(map_index) < len(self.fixed_maps)):
            raise IndexError(f"Fixed map index out of range: {map_index}")
        self.fixed_map_index = int(map_index)
        self.map_selection_mode = "fixed_index"

    def list_fixed_maps(self) -> List[Dict[str, Any]]:
        return [
            {
                "index": i,
                "name": spec.get("name", f"fixed_map_{i}"),
                "height": len(spec["rows"]),
                "width": len(spec["rows"][0]),
                "n_victims": sum(row.count("V") for row in spec["rows"]),
                "n_dangers": sum(row.count("D") for row in spec["rows"]),
                "n_h_walls": len(spec.get("h_walls", [])),
                "n_v_walls": len(spec.get("v_walls", [])),
            }
            for i, spec in enumerate(self.fixed_maps)
        ]

    # ============================================================
    # rendering
    # ============================================================
    def render_ascii(self) -> str:
        chars: List[str] = []
        heading_char = {0: "^", 1: ">", 2: "v", 3: "<"}[self.agent_heading]

        for r in range(self.height):
            row_chars: List[str] = []

            for c in range(self.width):
                pos = (r, c)

                if pos == self.agent_pos:
                    row_chars.append(heading_char)
                elif pos in self.victim_positions:
                    row_chars.append("V")
                elif pos in self.danger_positions:
                    row_chars.append("D")
                elif self.grid[r, c] == 1:
                    row_chars.append("#")
                else:
                    row_chars.append(".")

            chars.append(" ".join(row_chars))

        return "\n".join(chars)

    def get_env_state(self) -> Dict[str, Any]:
        return {
            "episode_index": int(self.episode_index),
            "width": int(self.width),
            "height": int(self.height),
            "step_count": int(self.step_count),
            "max_steps": int(self.max_steps),
            "agent_pos": self.agent_pos,
            "agent_heading": int(self.agent_heading),
            "agent_heading_name": self.HEADING_NAMES[self.agent_heading],
            "victim_positions": sorted(list(self.victim_positions)),
            "danger_positions": sorted(list(self.danger_positions)),
            "h_walls": sorted(list(self.h_walls)),
            "v_walls": sorted(list(self.v_walls)),
            "remaining_victims": int(len(self.victim_positions)),
            "rescued_count": int(self.rescued_count),
            "initial_victim_count": int(self.initial_victim_count),
            "done": bool(self.done),
            "last_distance_to_victim": float(self.last_distance_to_victim),
            "use_fixed_maps": bool(self.use_fixed_maps),
            "map_index": self.current_map_index,
            "map_name": self.current_map_name,
            "map_selection_mode": self.map_selection_mode,
        }

    # ============================================================
    # config helpers
    # ============================================================
    @staticmethod
    def _maybe_config() -> Optional[Any]:
        try:
            import config as cfg  # type: ignore
            return cfg
        except Exception:
            return None

    @staticmethod
    def _cfg(cfg: Optional[Any], name: str, default: Any) -> Any:
        if cfg is None:
            return default
        return getattr(cfg, name, default)

    # ============================================================
    # fixed map parsing
    # ============================================================
    def _normalize_fixed_maps(
        self,
        fixed_maps: Optional[Sequence[MapSpec]],
    ) -> List[Dict[str, Any]]:
        if fixed_maps is None:
            return []

        if isinstance(fixed_maps, (str, bytes)):
            raise TypeError("fixed_maps must be a sequence of map specs, not a single string")

        normalized: List[Dict[str, Any]] = []

        for i, spec in enumerate(fixed_maps):
            if isinstance(spec, dict):
                rows_raw = spec.get("grid", spec.get("rows", None))
                if rows_raw is None:
                    raise ValueError(f"Fixed map spec at index {i} must contain 'grid' or 'rows'.")

                name = str(spec.get("name", f"fixed_map_{i}"))
                heading = spec.get("heading", None)
                h_walls = spec.get("h_walls", [])
                v_walls = spec.get("v_walls", [])

            else:
                rows_raw = spec
                name = f"fixed_map_{i}"
                heading = None
                h_walls = []
                v_walls = []

            rows = [str(row) for row in rows_raw]
            self._validate_fixed_map_rows(rows, name=name)

            height = len(rows)
            width = len(rows[0])
            h_walls_norm = self._normalize_h_walls(h_walls, height, width)
            v_walls_norm = self._normalize_v_walls(v_walls, height, width)

            entry: Dict[str, Any] = {
                "name": name,
                "rows": rows,
                "h_walls": sorted(list(h_walls_norm)),
                "v_walls": sorted(list(v_walls_norm)),
            }

            if heading is not None:
                entry["heading"] = int(heading) % 4

            normalized.append(entry)

        return normalized

    def _validate_fixed_map_rows(self, rows: List[str], name: str) -> None:
        if not rows:
            raise ValueError(f"Fixed map {name!r} must not be empty.")

        width = len(rows[0])
        if width == 0:
            raise ValueError(f"Fixed map {name!r} must not contain empty rows.")

        allowed = set("#. ASRV D^>v<".replace(" ", "")) | {" "}

        n_agent = 0
        n_victim = 0

        for r, row in enumerate(rows):
            if len(row) != width:
                raise ValueError(f"All rows in fixed map {name!r} must have the same width.")

            for c, ch in enumerate(row):
                if ch not in allowed:
                    raise ValueError(
                        f"Unknown map character {ch!r} in fixed map {name!r} at {(r, c)}"
                    )

                if ch in {"A", "S", "R", "^", ">", "v", "<"}:
                    n_agent += 1
                elif ch == "V":
                    n_victim += 1

        if n_agent != 1:
            raise ValueError(f"Fixed map {name!r} must contain exactly one agent start symbol.")

        if n_victim < 1:
            raise ValueError(f"Fixed map {name!r} must contain at least one victim symbol V.")

    @staticmethod
    def _normalize_h_walls(
        h_walls: Sequence[Sequence[int]],
        height: int,
        width: int,
    ) -> Set[GridPos]:
        out: Set[GridPos] = set()

        for item in h_walls:
            if len(item) != 2:
                raise ValueError("Each h_wall must be (r, c).")

            r, c = int(item[0]), int(item[1])

            if not (0 <= r < height - 1 and 0 <= c < width):
                raise ValueError(
                    f"h_wall {(r, c)} out of range. "
                    f"Valid: 0<=r<{height - 1}, 0<=c<{width}"
                )

            out.add((r, c))

        return out

    @staticmethod
    def _normalize_v_walls(
        v_walls: Sequence[Sequence[int]],
        height: int,
        width: int,
    ) -> Set[GridPos]:
        out: Set[GridPos] = set()

        for item in v_walls:
            if len(item) != 2:
                raise ValueError("Each v_wall must be (r, c).")

            r, c = int(item[0]), int(item[1])

            if not (0 <= r < height and 0 <= c < width - 1):
                raise ValueError(
                    f"v_wall {(r, c)} out of range. "
                    f"Valid: 0<=r<{height}, 0<=c<{width - 1}"
                )

            out.add((r, c))

        return out

    # ============================================================
    # movement helpers
    # ============================================================
    def _free_cells(self) -> List[GridPos]:
        return [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if self.grid[r, c] == 0
        ]

    @staticmethod
    def _distance(a: GridPos, b: GridPos) -> float:
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    def _nearest_victim_distance(self, pos: GridPos) -> float:
        if not self.victim_positions:
            return 0.0
        return float(min(self._distance(pos, victim_pos) for victim_pos in self.victim_positions))

    def _forward_pos(self, pos: GridPos, heading: int) -> GridPos:
        r, c = pos
        h = int(heading) % 4

        if h == 0:
            return (r - 1, c)
        if h == 1:
            return (r, c + 1)
        if h == 2:
            return (r + 1, c)
        return (r, c - 1)

    def _is_outside(self, pos: GridPos) -> bool:
        r, c = pos
        return r < 0 or r >= self.height or c < 0 or c >= self.width

    def _is_blocked(self, pos: GridPos) -> bool:
        if self._is_outside(pos):
            return True
        return bool(self.grid[pos[0], pos[1]] == 1)

    def _is_edge_blocked(self, from_pos: GridPos, to_pos: GridPos) -> bool:
        r1, c1 = from_pos
        r2, c2 = to_pos

        if c1 == c2 and abs(r1 - r2) == 1:
            r = min(r1, r2)
            c = c1
            return (r, c) in self.h_walls

        if r1 == r2 and abs(c1 - c2) == 1:
            r = r1
            c = min(c1, c2)
            return (r, c) in self.v_walls

        return False

    def _is_move_blocked(self, from_pos: GridPos, to_pos: GridPos) -> bool:
        if self._is_blocked(to_pos):
            return True
        if self._is_edge_blocked(from_pos, to_pos):
            return True
        return False

    def _directional_clearance(self, heading: int) -> float:
        pos = self.agent_pos
        d = 0

        while True:
            nxt = self._forward_pos(pos, heading)

            if self._is_move_blocked(pos, nxt):
                break

            d += 1
            pos = nxt

        max_possible = max(self.width, self.height) - 1
        return float(np.clip(d / max(max_possible, 1), 0.0, 1.0))

    def _victim_signal_strength(self) -> float:
        if not self.victim_positions:
            return 0.0

        d = self._nearest_victim_distance(self.agent_pos)
        sigma = max(self.victim_signal_sigma, 1e-12)
        strength = np.exp(-(d ** 2) / (2.0 * sigma ** 2))

        return float(np.clip(strength, 0.0, 1.0))

    def _all_victims_reachable_from_agent(self) -> bool:
        reachable = self._reachable_free_cells(self.agent_pos)
        return all(v in reachable for v in self.victim_positions)

    def _reachable_free_cells(self, start: GridPos) -> Set[GridPos]:
        if self._is_blocked(start):
            return set()

        visited: Set[GridPos] = {start}
        q: deque[GridPos] = deque([start])

        while q:
            pos = q.popleft()

            for heading in range(4):
                nxt = self._forward_pos(pos, heading)

                if nxt in visited:
                    continue

                if self._is_move_blocked(pos, nxt):
                    continue

                visited.add(nxt)
                q.append(nxt)

        return visited

    @staticmethod
    def _validate_observation(obs: Dict[str, float]) -> None:
        required = (
            "front_clearance",
            "left_clearance",
            "right_clearance",
            "victim_signal",
        )

        for key in required:
            if key not in obs:
                raise KeyError(f"Missing observation key: {key}")

            val = float(obs[key])

            if not np.isfinite(val):
                raise ValueError(f"Observation {key} is not finite: {val}")

            if not (0.0 <= val <= 1.0):
                raise ValueError(f"Observation {key} must be in [0, 1], got {val}")


if __name__ == "__main__":
    demo_maps = [
        {
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
        },
    ]

    env = AbstractRescueGridEnv(
        max_steps=30,
        use_fixed_maps=True,
        fixed_maps=demo_maps,
        map_selection_mode="fixed_index",
        ensure_reachable=True,
    )

    obs = env.reset()
    print(env.get_env_state())
    print("obs:", obs)
    print(env.render_ascii())
