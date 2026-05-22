from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np


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
    """Grid-world rescue environment for the SNN robot simulation.

    This environment supports two layout modes.

    1) Fixed-map mode
       A predefined list of maps is used. This is recommended when the experiment
       must be reproducible and when the number/position of victims is decided in
       advance.

    2) Random-map mode
       Obstacles, agent, and victims are generated randomly. This is kept for
       stress tests or data augmentation, but it should not be mixed with a fixed
       benchmark unless that is intentional.

    Observation keys are intentionally kept compatible with the existing encoder:
        front_clearance, left_clearance, right_clearance, victim_signal

    Map symbols for fixed maps:
        # : obstacle/wall
        . or space : empty cell
        A or S : agent start position, heading supplied by fixed_agent_heading
        ^ > v < : agent start position with heading encoded in the map
        V : victim position
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

    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        max_steps: int = 50,
        obstacle_density: float = 0.12,
        seed: Optional[int] = None,
        victim_signal_sigma: float = 2.2,
        reward_step_penalty: float = 0.0,
        reward_collision: float = -0.05,
        reward_closer: float = 0.10,
        reward_farther: float = -0.03,
        reward_found_victim: float = 3.0,
        reward_stay: float = -0.02,
        reward_turn: float = -0.002,
        use_random_heading_on_reset: bool = True,
        n_victims: int = 3,
        ensure_reachable: bool = True,
        generation_max_attempts: int = 200,
        # Fixed-map options. If None, the constructor tries to read them from config.py.
        use_fixed_maps: Optional[bool] = None,
        fixed_maps: Optional[Sequence[MapSpec]] = None,
        fixed_agent_heading: Optional[int] = None,
        map_selection_mode: Optional[str] = None,
        fixed_map_index: Optional[int] = None,
    ) -> None:
        cfg = self._maybe_config()

        # Read optional fixed-map settings from config.py only when not explicitly supplied.
        if fixed_maps is None:
            fixed_maps = self._cfg(cfg, "ENV_FIXED_MAPS", None)
            if fixed_maps is None:
                single_map = self._cfg(cfg, "ENV_FIXED_MAP", None)
                if single_map is not None:
                    fixed_maps = [single_map]

        if use_fixed_maps is None:
            use_fixed_maps = bool(self._cfg(cfg, "ENV_USE_FIXED_MAPS", False))
            if not use_fixed_maps:
                # Backward-compatible alias.
                use_fixed_maps = bool(self._cfg(cfg, "ENV_USE_FIXED_MAP", False))

        if fixed_agent_heading is None:
            fixed_agent_heading = int(self._cfg(cfg, "ENV_FIXED_AGENT_HEADING", 0))

        if map_selection_mode is None:
            map_selection_mode = str(self._cfg(cfg, "ENV_MAP_SELECTION_MODE", "cycle"))

        if fixed_map_index is None:
            fixed_map_index = int(self._cfg(cfg, "ENV_FIXED_MAP_INDEX", 0))

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
        if generation_max_attempts < 1:
            raise ValueError("generation_max_attempts must be >= 1")

        self.width = int(width)
        self.height = int(height)
        self.max_steps = int(max_steps)
        self.obstacle_density = float(obstacle_density)
        self.victim_signal_sigma = float(victim_signal_sigma)

        self.reward_step_penalty = float(reward_step_penalty)
        self.reward_collision = float(reward_collision)
        self.reward_closer = float(reward_closer)
        self.reward_farther = float(reward_farther)
        self.reward_found_victim = float(reward_found_victim)
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
            raise ValueError(
                "use_fixed_maps=True but no fixed maps were provided. "
                "Set ENV_FIXED_MAPS in config.py or pass fixed_maps=[...]."
            )
        if self.use_fixed_maps and self.map_selection_mode not in {"cycle", "random", "fixed", "fixed_index"}:
            raise ValueError("map_selection_mode must be 'cycle', 'random', or 'fixed_index'")

        self.rng = np.random.default_rng(seed)

        self.grid: np.ndarray = np.zeros((self.height, self.width), dtype=np.int8)
        self.agent_pos: GridPos = (0, 0)
        self.agent_heading: int = 0
        self.victim_positions: Set[GridPos] = set()
        self.step_count: int = 0
        self.done: bool = False

        self.last_distance_to_victim: float = 0.0
        self.episode_index: int = 0
        self.rescued_count: int = 0
        self.initial_victim_count: int = self.n_victims

        # Build an initial valid state so that render/get_observation are safe
        # immediately after construction. Then reset counters/cursor so that the
        # first external reset() still starts from fixed map index 0 in cycle mode.
        self.reset()
        self.episode_index = 0
        if self.use_fixed_maps and self.map_selection_mode == "cycle":
            self._map_cursor = 0

    # ------------------------------------------------------------------
    # Reset / episode creation
    # ------------------------------------------------------------------
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
            return self.get_observation()

        last_error: Optional[Exception] = None
        for _ in range(self.generation_max_attempts):
            try:
                self._generate_random_episode_layout()
                if (not self.ensure_reachable) or self._all_victims_reachable_from_agent():
                    self.agent_heading = (
                        int(self.rng.integers(0, 4)) if self.use_random_heading_on_reset else 0
                    )
                    self.last_distance_to_victim = self._nearest_victim_distance(self.agent_pos)
                    return self.get_observation()
            except Exception as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise RuntimeError("Failed to generate a valid random episode layout.") from last_error
        raise RuntimeError(
            "Failed to generate a reachable random episode layout. "
            "Try lowering obstacle_density or increasing generation_max_attempts."
        )

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
        else:  # cycle
            idx = self._map_cursor
            self._map_cursor = (self._map_cursor + 1) % n_maps

        if not (0 <= idx < n_maps):
            raise IndexError(f"Fixed map index out of range: {idx}. Valid range: 0..{n_maps - 1}")
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

        agent_positions: List[Tuple[GridPos, Optional[int]]] = []

        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                pos = (r, c)
                if ch == "#":
                    self.grid[r, c] = 1
                elif ch in {".", " "}:
                    self.grid[r, c] = 0
                elif ch in {"A", "S"}:
                    self.grid[r, c] = 0
                    agent_positions.append((pos, None))
                elif ch in self.HEADING_FROM_CHAR:
                    self.grid[r, c] = 0
                    agent_positions.append((pos, self.HEADING_FROM_CHAR[ch]))
                elif ch == "V":
                    self.grid[r, c] = 0
                    self.victim_positions.add(pos)
                else:
                    raise ValueError(f"Unknown map character {ch!r} at row={r}, col={c}")

        if len(agent_positions) != 1:
            raise ValueError(
                f"Fixed map {name!r} must contain exactly one agent start symbol "
                "among A, S, ^, >, v, <."
            )
        if not self.victim_positions:
            raise ValueError(f"Fixed map {name!r} must contain at least one victim symbol V.")

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
            raise RuntimeError(
                f"Not enough free cells to place agent and {self.n_victims} victims."
            )

        agent_idx = int(self.rng.integers(0, len(free)))
        self.agent_pos = free.pop(agent_idx)

        for _ in range(self.n_victims):
            victim_idx = int(self.rng.integers(0, len(free)))
            self.victim_positions.add(free.pop(victim_idx))

        self.initial_victim_count = len(self.victim_positions)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
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
            raise ValueError(f"Unsupported action {action}. Valid actions: {list(self.ACTION_NAMES)}")

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
            if self._is_blocked(next_pos):
                collision = True
            else:
                self.agent_pos = next_pos
                moved = True
        elif action == 3:
            pass

        self.step_count += 1

        new_distance = self._nearest_victim_distance(self.agent_pos)
        found_victim = self.agent_pos in self.victim_positions

        reward = float(self.reward_step_penalty)
        if collision:
            reward += self.reward_collision
        else:
            if new_distance < old_distance:
                reward += self.reward_closer
            elif new_distance > old_distance:
                reward += self.reward_farther

        if action == 3:
            reward += self.reward_stay
        elif action in (1, 2):
            reward += self.reward_turn

        if found_victim:
            self.victim_positions.remove(self.agent_pos)
            self.rescued_count += 1
            reward += self.reward_found_victim
            if not self.victim_positions:
                self.done = True

        if self.step_count >= self.max_steps:
            self.done = True

        self.last_distance_to_victim = self._nearest_victim_distance(self.agent_pos)

        info = {
            "action_name": self.ACTION_NAMES[action],
            "old_pos": old_pos,
            "new_pos": self.agent_pos,
            "old_heading": self.HEADING_NAMES[old_heading],
            "new_heading": self.HEADING_NAMES[self.agent_heading],
            "collision": bool(collision),
            "moved": bool(moved),
            "distance_to_nearest_victim": float(new_distance),
            "found_victim": bool(found_victim),
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

    def get_observation(self) -> Dict[str, float]:
        front_clearance = self._directional_clearance(self.agent_heading)
        left_clearance = self._directional_clearance((self.agent_heading - 1) % 4)
        right_clearance = self._directional_clearance((self.agent_heading + 1) % 4)
        victim_signal = self._victim_signal_strength()

        obs = {
            "front_clearance": float(front_clearance),
            "left_clearance": float(left_clearance),
            "right_clearance": float(right_clearance),
            "victim_signal": float(victim_signal),
        }
        self._validate_observation(obs)
        return obs

    # ------------------------------------------------------------------
    # Fixed-map utility API
    # ------------------------------------------------------------------
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
            }
            for i, spec in enumerate(self.fixed_maps)
        ]

    # ------------------------------------------------------------------
    # Rendering / inspection
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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

    def _normalize_fixed_maps(self, fixed_maps: Optional[Sequence[MapSpec]]) -> List[Dict[str, Any]]:
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
            else:
                rows_raw = spec
                name = f"fixed_map_{i}"
                heading = None

            rows = [str(row) for row in rows_raw]
            self._validate_fixed_map_rows(rows, name=name)
            entry: Dict[str, Any] = {"name": name, "rows": rows}
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
        if len(rows) < 1 or width < 1:
            raise ValueError(f"Fixed map {name!r} has invalid dimensions.")

        allowed = set("#. ASV^>v<")
        n_agent = 0
        n_victim = 0
        for r, row in enumerate(rows):
            if len(row) != width:
                raise ValueError(f"All rows in fixed map {name!r} must have the same width.")
            for c, ch in enumerate(row):
                if ch not in allowed:
                    raise ValueError(f"Unknown map character {ch!r} in fixed map {name!r} at {(r, c)}")
                if ch in {"A", "S", "^", ">", "v", "<"}:
                    n_agent += 1
                elif ch == "V":
                    n_victim += 1
        if n_agent != 1:
            raise ValueError(f"Fixed map {name!r} must contain exactly one agent start symbol.")
        if n_victim < 1:
            raise ValueError(f"Fixed map {name!r} must contain at least one victim symbol V.")

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

    def _directional_clearance(self, heading: int) -> float:
        pos = self.agent_pos
        d = 0
        while True:
            pos = self._forward_pos(pos, heading)
            if self._is_blocked(pos):
                break
            d += 1

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
                if nxt in visited or self._is_blocked(nxt):
                    continue
                visited.add(nxt)
                q.append(nxt)
        return visited

    @staticmethod
    def _validate_observation(obs: Dict[str, float]) -> None:
        required = ("front_clearance", "left_clearance", "right_clearance", "victim_signal")
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
            "name": "demo_train_0",
            "grid": [
                "########",
                "#>.....#",
                "#..#...#",
                "#..#V..#",
                "#......#",
                "#.V.#..#",
                "#......#",
                "########",
            ],
        },
        {
            "name": "demo_train_1",
            "grid": [
                "########",
                "#>..#..#",
                "#...#V.#",
                "#......#",
                "#.##...#",
                "#...V..#",
                "#......#",
                "########",
            ],
        },
    ]

    env = AbstractRescueGridEnv(
        max_steps=30,
        use_fixed_maps=True,
        fixed_maps=demo_maps,
        map_selection_mode="cycle",
        ensure_reachable=True,
    )

    for ep in range(2):
        obs = env.reset()
        print("episode", ep, env.get_env_state())
        print("obs:", obs)
        print(env.render_ascii())
        print()
