from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np 


Action = int 


@dataclass
class EnvStepResult:
    observation: Dict[str, float]  # 관측값
    reward: float  # 보상
    done: bool  # 종료 여부
    info: Dict[str, Any]  # 추가 정보


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
        max_steps: int = 50,  # 최대 step 수
        obstacle_density: float = 0.12,  # 장애물 비율
        seed: Optional[int] = None,  # 난수 시드
        victim_signal_sigma: float = 2.2,  # 피해자 신호 퍼짐 정도
        reward_step_penalty = 0.0,  # 기본 step 패널티
        reward_collision = -0.05,  # 충돌 패널티
        reward_closer = 0.10,  # 가까워질 때 보상
        reward_farther = -0.03,  # 멀어질 때 패널티
        reward_found_victim = 3.0,  # 피해자 발견 보상
        use_random_heading_on_reset: bool = True,  # reset 시 방향 랜덤 여부
    ) -> None:
        if width < 4 or height < 4:  # 맵 최소 크기 검사
            raise ValueError("width and height should both be >= 4")
        if max_steps < 1:  # 최대 step 검사
            raise ValueError("max_steps must be >= 1")
        if not (0.0 <= obstacle_density < 0.45):  # 장애물 비율 검사
            raise ValueError("obstacle_density should be in [0, 0.45)")

        self.width = int(width)  # 너비 저장
        self.height = int(height)  # 높이 저장
        self.max_steps = int(max_steps)  # 최대 step 저장
        self.obstacle_density = float(obstacle_density)  # 장애물 비율 저장
        self.victim_signal_sigma = float(victim_signal_sigma)  # 신호 sigma 저장

        self.reward_step_penalty = float(reward_step_penalty)  # 기본 패널티 저장
        self.reward_collision = float(reward_collision)  # 충돌 패널티 저장
        self.reward_closer = float(reward_closer)  # 접근 보상 저장
        self.reward_farther = float(reward_farther)  # 이탈 패널티 저장
        self.reward_found_victim = float(reward_found_victim)  # 발견 보상 저장

        self.use_random_heading_on_reset = bool(use_random_heading_on_reset)  # 랜덤 방향 옵션 저장

        self.rng = np.random.default_rng(seed)  # 난수 생성기

        self.grid: np.ndarray = np.zeros((self.height, self.width), dtype=np.int8)  # 격자 맵
        self.agent_pos: Tuple[int, int] = (0, 0)  # 에이전트 위치
        self.agent_heading: int = 0  # 에이전트 방향
        self.victim_pos: Tuple[int, int] = (0, 0)  # 피해자 위치
        self.step_count: int = 0  # 현재 step 수
        self.done: bool = False  # 종료 여부

        self.last_distance_to_victim: float = 0.0  # 마지막 거리
        self.episode_index: int = 0  # 에피소드 번호

        self.reset()  # 초기 reset

    # ------------------------------------------------------------------
    # Reset / episode creation
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, float]:
        self.episode_index += 1  # 에피소드 증가
        self.step_count = 0  # step 초기화
        self.done = False  # 종료 해제

        self.grid.fill(0)  # 맵 초기화

        # Build obstacles first.
        n_cells = self.width * self.height  # 전체 칸 수
        n_obstacles = int(round(self.obstacle_density * n_cells))  # 장애물 개수

        obstacle_coords = set()  # 장애물 좌표 집합
        while len(obstacle_coords) < n_obstacles:  # 필요한 개수까지 반복
            r = int(self.rng.integers(0, self.height))  # 랜덤 행
            c = int(self.rng.integers(0, self.width))  # 랜덤 열
            obstacle_coords.add((r, c))  # 장애물 좌표 추가

        for r, c in obstacle_coords:  # 장애물 배치
            self.grid[r, c] = 1  # 1은 장애물

        # Pick free cells for agent and victim.
        free = self._free_cells()  # 빈 칸 목록
        if len(free) < 2:  # 두 칸 이상 필요
            raise RuntimeError("Not enough free cells to place agent and victim.")

        agent_idx = int(self.rng.integers(0, len(free)))  # 에이전트 위치 인덱스
        self.agent_pos = free.pop(agent_idx)  # 에이전트 위치 선택

        victim_idx = int(self.rng.integers(0, len(free)))  # 피해자 위치 인덱스
        self.victim_pos = free[victim_idx]  # 피해자 위치 선택

        # Make sure agent/victim positions are not obstacles.
        self.grid[self.agent_pos[0], self.agent_pos[1]] = 0  # 에이전트 칸 보정
        self.grid[self.victim_pos[0], self.victim_pos[1]] = 0  # 피해자 칸 보정

        self.agent_heading = int(self.rng.integers(0, 4)) if self.use_random_heading_on_reset else 0  # 초기 방향 설정
        self.last_distance_to_victim = self._distance(self.agent_pos, self.victim_pos)  # 초기 거리 저장

        return self.get_observation()  # 초기 관측 반환

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def step(self, action: Action) -> EnvStepResult:
        if self.done:  # 이미 종료됐으면
            return EnvStepResult(
                observation=self.get_observation(),  # 현재 관측 반환
                reward=0.0,  # 보상 없음
                done=True,  # 종료 유지
                info={"warning": "Episode already done."},  # 경고 정보
            )

        action = int(action)  # 행동 정수화
        if action not in self.ACTION_NAMES:  # 유효 행동 검사
            raise ValueError(f"Unsupported action {action}. Valid actions: {list(self.ACTION_NAMES)}")

        old_pos = self.agent_pos  # 이전 위치
        old_heading = self.agent_heading  # 이전 방향
        old_distance = self._distance(old_pos, self.victim_pos)  # 이전 거리

        collision = False  # 충돌 여부
        moved = False  # 이동 여부

        if action == 1:  # 좌회전
            self.agent_heading = (self.agent_heading - 1) % 4  # 방향 변경
        elif action == 2:  # 우회전
            self.agent_heading = (self.agent_heading + 1) % 4  # 방향 변경
        elif action == 0:  # 전진
            next_pos = self._forward_pos(old_pos, old_heading)  # 다음 위치 계산
            if self._is_blocked(next_pos):  
                collision = True  # 막혔으면 충돌 처리
            else:
                self.agent_pos = next_pos  # 위치 이동
                moved = True  # 이동 표시
        elif action == 3:  # 정지
            pass  # 그대로 유지

        self.step_count += 1  # step 증가

        new_distance = self._distance(self.agent_pos, self.victim_pos)  # 새 거리
        found_victim = self.agent_pos == self.victim_pos  # 피해자 발견 여부

        reward = self.reward_step_penalty  # 기본 보상

        if collision:  
            reward += self.reward_collision  # 충돌 시 충돌 패널티 추가
        else:
            if new_distance < old_distance:
                reward += self.reward_closer  # 더 가까워졌으면 보상 추가
            elif new_distance > old_distance:   
                reward += self.reward_farther  # 더 멀어졌으면 패널티 추가

        if action == 3:          # stay
            reward -= 0.02  # 정지 패널티
        elif action in (1, 2):   # turn
            reward -= 0.002  # 회전 패널티

        if found_victim:   
            reward += self.reward_found_victim  # 피해자를 찾았으면 큰 보상 추가
            self.done = True  # 종료
        elif self.step_count >= self.max_steps:  # 최대 step 도달하면 종료
            self.done = True  # 종료
            
            self.last_distance_to_victim = new_distance  # 마지막 거리 저장

        info = {
            "action_name": self.ACTION_NAMES[action],  # 행동 이름
            "old_pos": old_pos,  # 이전 위치
            "new_pos": self.agent_pos,  # 새 위치
            "old_heading": self.HEADING_NAMES[old_heading],  # 이전 방향 이름
            "new_heading": self.HEADING_NAMES[self.agent_heading],  # 새 방향 이름
            "collision": collision,  # 충돌 여부
            "moved": moved,  # 이동 여부
            "distance_to_victim": float(new_distance),  # 현재 거리
            "found_victim": found_victim,  # 발견 여부
            "step_count": self.step_count,  # 현재 step 수
        }

        return EnvStepResult(
            observation=self.get_observation(),  # 다음 관측
            reward=float(reward),  # 보상 반환
            done=bool(self.done),  # 종료 여부 반환
            info=info,  # 추가 정보 반환
        )

    def get_observation(self) -> Dict[str, float]:
        """
        Return abstract sensor-style observation for the current state.

        Values are normalized to roughly [0, 1] so they work nicely with the
        current encoder defaults.
        """
        front_clearance = self._directional_clearance(self.agent_heading)  # 앞쪽 여유 거리
        left_clearance = self._directional_clearance((self.agent_heading - 1) % 4)  # 왼쪽 여유 거리
        right_clearance = self._directional_clearance((self.agent_heading + 1) % 4)  # 오른쪽 여유 거리

        victim_signal = self._victim_signal_strength()  # 피해자 신호 세기

        return {
            "front_clearance": float(front_clearance),  # 앞 센서값
            "left_clearance": float(left_clearance),  # 왼 센서값
            "right_clearance": float(right_clearance),  # 오른 센서값
            "victim_signal": float(victim_signal),  # 피해자 신호값
        }

    # ------------------------------------------------------------------
    # Rendering / inspection
    # ------------------------------------------------------------------
    def render_ascii(self) -> str:
        chars: List[str] = []  # 출력 줄 저장
        heading_char = {0: "^", 1: ">", 2: "v", 3: "<"}[self.agent_heading]  # 방향 문자

        for r in range(self.height):  # 각 행 순회
            row_chars: List[str] = []  # 한 줄 문자 저장
            for c in range(self.width):  # 각 열 순회
                pos = (r, c)  # 현재 좌표
                if pos == self.agent_pos:  # 에이전트 위치면
                    row_chars.append(heading_char)  # 방향 문자 표시
                elif pos == self.victim_pos:  # 피해자 위치면
                    row_chars.append("V")  # 피해자 표시
                elif self.grid[r, c] == 1:  # 장애물이면
                    row_chars.append("#")  # 장애물 표시
                else:
                    row_chars.append(".")  # 빈 칸 표시
            chars.append(" ".join(row_chars))  # 한 줄 추가
        return "\n".join(chars)  # 문자열로 반환

    def get_env_state(self) -> Dict[str, Any]:
        return {
            "episode_index": self.episode_index,  # 에피소드 번호
            "width": self.width,  # 너비
            "height": self.height,  # 높이
            "step_count": self.step_count,  # 현재 step 수
            "max_steps": self.max_steps,  # 최대 step 수
            "agent_pos": self.agent_pos,  # 에이전트 위치
            "agent_heading": self.agent_heading,  # 에이전트 방향
            "victim_pos": self.victim_pos,  # 피해자 위치
            "done": self.done,  # 종료 여부
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _free_cells(self) -> List[Tuple[int, int]]:
        free: List[Tuple[int, int]] = []  # 빈 칸 리스트
        for r in range(self.height):  # 행 순회
            for c in range(self.width):  # 열 순회
                if self.grid[r, c] == 0:  # 빈 칸이면
                    free.append((r, c))  # 추가
        return free  # 반환

    @staticmethod
    def _distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))  # 맨해튼 거리

    def _forward_pos(self, pos: Tuple[int, int], heading: int) -> Tuple[int, int]:
        r, c = pos  # 현재 좌표
        if heading == 0:  # 위
            return (r - 1, c)  # 위로 이동
        if heading == 1:  # 오른쪽
            return (r, c + 1)  # 오른쪽 이동
        if heading == 2:  # 아래
            return (r + 1, c)  # 아래 이동
        return (r, c - 1)  # 왼쪽 이동

    def _is_outside(self, pos: Tuple[int, int]) -> bool:
        r, c = pos  # 좌표 분리
        return r < 0 or r >= self.height or c < 0 or c >= self.width  # 맵 밖 여부

    def _is_blocked(self, pos: Tuple[int, int]) -> bool:
        if self._is_outside(pos):  
            return True  # 맵 밖이면 막힘 처리
        return bool(self.grid[pos[0], pos[1]] == 1)  # 장애물 여부 반환

    def _directional_clearance(self, heading: int) -> float:
        """
        Distance to nearest obstacle/wall along one ray, normalized to [0, 1].
        """
        pos = self.agent_pos  # 시작 위치
        d = 0  # 거리 누적
        while True:  # 막힐 때까지 반복
            pos = self._forward_pos(pos, heading)  # 한 칸 전진
            if self._is_blocked(pos): 
                break  # 막히면 종료
            d += 1  # 거리 증가

        max_possible = max(self.width, self.height) - 1  # 최대 가능 거리
        return float(np.clip(d / max(max_possible, 1), 0.0, 1.0))  # 0~1 정규화

    def _victim_signal_strength(self) -> float:
        """
        Abstract victim signal in [0, 1].

        You can later reinterpret this as:
        - thermal intensity
        - CO2 concentration
        - sound level
        - combined confidence score
        """
        d = self._distance(self.agent_pos, self.victim_pos)  # 피해자까지 거리
        sigma = max(self.victim_signal_sigma, 1e-6)  # sigma 최소 보정
        strength = np.exp(-(d ** 2) / (2.0 * sigma ** 2))  # 가우시안 신호 계산
        return float(np.clip(strength, 0.0, 1.0))  # 0~1 범위로 제한


if __name__ == "__main__":
    env = AbstractRescueGridEnv(seed=42)  # 환경 생성
    obs = env.reset()  # 초기화

    print("Initial observation:", obs)  # 초기 관측 출력
    print(env.render_ascii())  # 맵 출력
    print() 

    # Tiny smoke run
    for action in [0, 1, 0, 2, 0, 3]:  # 간단 테스트 행동들
        out = env.step(action)  # 한 step 실행
        print(f"action={action} ({env.ACTION_NAMES[action]})")  # 행동 출력
        print("obs   =", out.observation)  # 관측 출력
        print("reward=", out.reward, "done=", out.done)  # 보상과 종료 여부 출력
        print("info  =", out.info)  # 상세 정보 출력
        print(env.render_ascii())  # 현재 맵 출력
        print()  # 줄바꿈
        if out.done:  # 종료되면
            break  # 반복 종료
