from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


Action = int


@dataclass
class EnvStepResult:
    observation: Dict[str, float]
    reward: float
    done: bool
    info: Dict[str, Any]


class AbstractRescueGridEnv:
    """
    Abstract grid environment for the current recurrent memristive SNN project.

    This is intentionally *not* tied to final hardware choices yet.
    It gives you a lightweight environment to validate the loop:

        observation -> network action -> env transition -> reward -> learning

    Core ideas
    ----------
    - Grid world with obstacles, one robot, one victim.
    - Observation is returned as a dictionary so it plugs naturally into the
      current ``SensorSpikeEncoder`` design.
    - Observation fields are abstract / sensor-like:
        * front_clearance
        * left_clearance
        * right_clearance
        * victim_signal
    - Later you can replace each field with a more realistic sensor model
      without changing the outer RL / SNN loop too much.

    Action convention
    -----------------
    0: move forward
    1: turn left
    2: turn right
    3: stay

    Heading convention
    ------------------
    0: up
    1: right
    2: down
    3: left
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

    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        max_steps: int = 50,
        obstacle_density: float = 0.12,
        seed: Optional[int] = None,
        victim_signal_sigma: float = 2.2,
        reward_step_penalty = 0.0,
        reward_collision = -0.05,
        reward_closer = 0.10,
        reward_farther = -0.03,
        reward_found_victim = 3.0,
        use_random_heading_on_reset: bool = True,
    ) -> None:
        if width < 4 or height < 4:
            raise ValueError("width and height should both be >= 4")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if not (0.0 <= obstacle_density < 0.45):
            raise ValueError("obstacle_density should be in [0, 0.45)")

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

        self.use_random_heading_on_reset = bool(use_random_heading_on_reset)

        self.rng = np.random.default_rng(seed)

        self.grid: np.ndarray = np.zeros((self.height, self.width), dtype=np.int8)
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.agent_heading: int = 0
        self.victim_pos: Tuple[int, int] = (0, 0)
        self.step_count: int = 0
        self.done: bool = False

        self.last_distance_to_victim: float = 0.0
        self.episode_index: int = 0

        self.reset()

    # ------------------------------------------------------------------
    # Reset / episode creation
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, float]:
        self.episode_index += 1
        self.step_count = 0
        self.done = False

        self.grid.fill(0)

        # Build obstacles first.
        n_cells = self.width * self.height
        n_obstacles = int(round(self.obstacle_density * n_cells))

        obstacle_coords = set()
        while len(obstacle_coords) < n_obstacles:
            r = int(self.rng.integers(0, self.height))
            c = int(self.rng.integers(0, self.width))
            obstacle_coords.add((r, c))

        for r, c in obstacle_coords:
            self.grid[r, c] = 1

        # Pick free cells for agent and victim.
        free = self._free_cells()
        if len(free) < 2:
            raise RuntimeError("Not enough free cells to place agent and victim.")

        agent_idx = int(self.rng.integers(0, len(free)))
        self.agent_pos = free.pop(agent_idx)

        victim_idx = int(self.rng.integers(0, len(free)))
        self.victim_pos = free[victim_idx]

        # Make sure agent/victim positions are not obstacles.
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 0
        self.grid[self.victim_pos[0], self.victim_pos[1]] = 0

        self.agent_heading = int(self.rng.integers(0, 4)) if self.use_random_heading_on_reset else 0
        self.last_distance_to_victim = self._distance(self.agent_pos, self.victim_pos)

        return self.get_observation()

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
        old_distance = self._distance(old_pos, self.victim_pos)

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

        new_distance = self._distance(self.agent_pos, self.victim_pos)
        found_victim = self.agent_pos == self.victim_pos

        reward = self.reward_step_penalty

        if collision:
            reward += self.reward_collision
        else:
            if new_distance < old_distance:
                reward += self.reward_closer
            elif new_distance > old_distance:
                reward += self.reward_farther

        if action == 3:          # stay
            reward -= 0.02
        elif action in (1, 2):   # turn
            reward -= 0.002

        if found_victim:
            reward += self.reward_found_victim
            self.done = True
        elif self.step_count >= self.max_steps:
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
            "distance_to_victim": float(new_distance),
            "found_victim": found_victim,
            "step_count": self.step_count,
        }

        return EnvStepResult(
            observation=self.get_observation(),
            reward=float(reward),
            done=bool(self.done),
            info=info,
        )

    def get_observation(self) -> Dict[str, float]:
        """
        Return abstract sensor-style observation for the current state.

        Values are normalized to roughly [0, 1] so they work nicely with the
        current encoder defaults.
        """
        front_clearance = self._directional_clearance(self.agent_heading)
        left_clearance = self._directional_clearance((self.agent_heading - 1) % 4)
        right_clearance = self._directional_clearance((self.agent_heading + 1) % 4)

        victim_signal = self._victim_signal_strength()

        return {
            "front_clearance": float(front_clearance),
            "left_clearance": float(left_clearance),
            "right_clearance": float(right_clearance),
            "victim_signal": float(victim_signal),
        }

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
                elif pos == self.victim_pos:
                    row_chars.append("V")
                elif self.grid[r, c] == 1:
                    row_chars.append("#")
                else:
                    row_chars.append(".")
            chars.append(" ".join(row_chars))
        return "\n".join(chars)

    def get_env_state(self) -> Dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "width": self.width,
            "height": self.height,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "agent_pos": self.agent_pos,
            "agent_heading": self.agent_heading,
            "victim_pos": self.victim_pos,
            "done": self.done,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _free_cells(self) -> List[Tuple[int, int]]:
        free: List[Tuple[int, int]] = []
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r, c] == 0:
                    free.append((r, c))
        return free

    @staticmethod
    def _distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    def _forward_pos(self, pos: Tuple[int, int], heading: int) -> Tuple[int, int]:
        r, c = pos
        if heading == 0:
            return (r - 1, c)
        if heading == 1:
            return (r, c + 1)
        if heading == 2:
            return (r + 1, c)
        return (r, c - 1)

    def _is_outside(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return r < 0 or r >= self.height or c < 0 or c >= self.width

    def _is_blocked(self, pos: Tuple[int, int]) -> bool:
        if self._is_outside(pos):
            return True
        return bool(self.grid[pos[0], pos[1]] == 1)

    def _directional_clearance(self, heading: int) -> float:
        """
        Distance to nearest obstacle/wall along one ray, normalized to [0, 1].
        """
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
        """
        Abstract victim signal in [0, 1].

        You can later reinterpret this as:
        - thermal intensity
        - CO2 concentration
        - sound level
        - combined confidence score
        """
        d = self._distance(self.agent_pos, self.victim_pos)
        sigma = max(self.victim_signal_sigma, 1e-6)
        strength = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
        return float(np.clip(strength, 0.0, 1.0))


if __name__ == "__main__":
    env = AbstractRescueGridEnv(seed=42)
    obs = env.reset()

    print("Initial observation:", obs)
    print(env.render_ascii())
    print()

    # Tiny smoke run
    for action in [0, 1, 0, 2, 0, 3]:
        out = env.step(action)
        print(f"action={action} ({env.ACTION_NAMES[action]})")
        print("obs   =", out.observation)
        print("reward=", out.reward, "done=", out.done)
        print("info  =", out.info)
        print(env.render_ascii())
        print()
        if out.done:
            break
