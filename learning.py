from __future__ import annotations  

from dataclasses import dataclass  
from typing import Dict, List, Optional, Sequence, Tuple  

import numpy as np  

from conductance_modulation import ProgrammingResult  
from network import MemristiveSNNNetwork  


@dataclass
class RSTDPConfig:
    """
    Reward-modulated STDP parameters.

    delta_t = t_post - t_pre

    If delta_t > 0:
        delta_w ~ +A_plus * exp(-delta_t / tau_plus)

    If delta_t < 0:
        delta_w ~ -A_minus * exp(+delta_t / tau_minus)

    Final physical programming direction is determined by:
        sign(reward * eligibility)
    """
    tau_plus: float = 2.0  # pre가 먼저일 때 시간 상수
    tau_minus: float = 2.0  # post가 먼저일 때 시간 상수
    a_plus: float = 1.0  # 강화 크기
    a_minus: float = 0.8  # 약화 크기
    eligibility_threshold: float = 1e-6  # 최소 eligibility 임계값

    # Conservative default:
    # if output decision came only from fallback, do not fabricate a fake post spike.
    use_surrogate_post_on_fallback: bool = False  # fallback일 때 가짜 post spike 사용 여부

    # Hidden-layer R-STDP is optional because it is harder to stabilize.
    enable_hidden_rstdp: bool = False  # hidden layer 학습 여부


@dataclass
class RSTDPUpdateEvent:
    layer_name: str  # 레이어 이름
    updated_pairs: List[Tuple[int, int]]  # 업데이트된 시냅스 쌍
    delta_t_records: List[float]  # delta_t 기록
    eligibility_values: List[float]  # eligibility 값들
    directions: List[int]  # 업데이트 방향
    actions: List[str]  # 실제 액션 이름
    n_pulses_plus: int  # plus 펄스 수
    n_pulses_minus: int  # minus 펄스 수
    n_refresh: int  # refresh 횟수
    reward: float  # 보상
    winner: int  # winner 뉴런
    target: Optional[int]  # target 뉴런
    message: str  # 설명 메시지


class RewardModulatedSTDPLearner:
    """
    Hardware-aware R-STDP learner.

    Important:
    - Spike timing -> eligibility
    - Reward -> global gate
    - Final update is NOT an ideal floating-point write
    - Actual programming goes through ConductanceModulationController.update_weight()
    """

    def __init__(self, config: Optional[RSTDPConfig] = None) -> None:
        self.cfg = RSTDPConfig() if config is None else config  # 설정 저장

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def learn(
        self,
        net: MemristiveSNNNetwork,
        reward: float,
        target: Optional[int] = None,
    ) -> Dict[str, Optional[RSTDPUpdateEvent]]:
        if net.last_decision is None:  # 이전 decision 없으면
            raise RuntimeError("No decision available. Call net.decide(obs) before learner.learn(...).")

        output_event = self._learn_output(net=net, reward=reward, target=target)  # output layer 학습
        hidden_event = None  # hidden 결과 기본값
        if self.cfg.enable_hidden_rstdp:  # hidden 학습이 켜져 있으면
            hidden_event = self._learn_hidden(net=net, reward=reward)  # hidden layer 학습

        return {
            "output": output_event,  # output 결과 반환
            "hidden": hidden_event,  # hidden 결과 반환
        }

    # ------------------------------------------------------------------
    # Output-layer R-STDP
    # ------------------------------------------------------------------
    def _learn_output(
        self,
        net: MemristiveSNNNetwork,
        reward: float,
        target: Optional[int],
    ) -> RSTDPUpdateEvent:
        decision = net.last_decision  # 마지막 decision 가져오기
        assert decision is not None  # None 아님 보장

        step_records = decision.step_records  # step 기록들
        n_pre = net.hidden_dim  # pre 뉴런 수
        n_post = net.n_actions  # post 뉴런 수

        pre_spikes = np.array(
            [np.asarray(rec.hidden_result.spikes, dtype=int) for rec in step_records],  # hidden spike 모음
            dtype=int,
        )  # [T, hidden_dim]

        post_spikes = np.array(
            [np.asarray(rec.output_result.spikes, dtype=int) for rec in step_records],  # output spike 모음
            dtype=int,
        )  # [T, n_actions]

        winner = int(decision.action)  # 선택된 action
        post_col = winner if target is None else int(target)  # 학습할 post 컬럼
        if not (0 <= post_col < n_post):  # 범위 검사
            raise ValueError(f"Target/post column out of range: {post_col}")

        post_times = np.flatnonzero(post_spikes[:, post_col] > 0).astype(int).tolist()  # post spike 시각들

        if (not post_times) and bool(decision.used_fallback) and self.cfg.use_surrogate_post_on_fallback:  # fallback이고 surrogate 허용이면
            post_times = [int(decision.selected_step)]  # 선택 step을 가짜 post spike로 사용

        if not post_times:  # post spike가 없으면
            return RSTDPUpdateEvent(
                layer_name="output",  # output layer
                updated_pairs=[],  # 업데이트 없음
                delta_t_records=[],  # 기록 없음
                eligibility_values=[],  # 기록 없음
                directions=[],  # 방향 없음
                actions=[],  # 액션 없음
                n_pulses_plus=0,  # plus 펄스 없음
                n_pulses_minus=0,  # minus 펄스 없음
                n_refresh=0,  # refresh 없음
                reward=float(reward),  # 현재 보상
                winner=winner,  # winner 기록
                target=target,  # target 기록
                message="No postsynaptic output spike available; skipped output R-STDP.",   
            )

        reward_sign = self._reward_sign(reward)  # 보상 부호 계산
        if reward_sign == 0:  # 보상이 0이면
            return RSTDPUpdateEvent(
                layer_name="output",  # output layer
                updated_pairs=[],  # 업데이트 없음
                delta_t_records=[],  # 기록 없음
                eligibility_values=[],  # 기록 없음
                directions=[],  # 방향 없음
                actions=[],  # 액션 없음
                n_pulses_plus=0,  # plus 펄스 없음
                n_pulses_minus=0,  # minus 펄스 없음
                n_refresh=0,  # refresh 없음
                reward=float(reward),  # 현재 보상
                winner=winner,  # winner 기록
                target=target,  # target 기록
                message="Reward is zero; no gated output R-STDP update applied.",   
            )

        updated_pairs: List[Tuple[int, int]] = []  # 업데이트된 pair들
        delta_t_records: List[float] = []  # delta_t 기록
        eligibility_values: List[float] = []  # eligibility 기록
        directions: List[int] = []  # 방향 기록
        actions: List[str] = []  # 액션 기록
        n_pulses_plus = 0  # plus 펄스 누적
        n_pulses_minus = 0  # minus 펄스 누적
        n_refresh = 0  # refresh 누적

        for row in range(n_pre):  # 모든 pre 뉴런 순회
            pre_times = np.flatnonzero(pre_spikes[:, row] > 0).astype(int).tolist()  # 해당 pre spike 시각들
            if not pre_times:  # spike 없으면 건너뜀
                continue   

            best_dt, elig = self._pair_eligibility(pre_times, post_times)  # eligibility 계산
            if abs(elig) < self.cfg.eligibility_threshold:  # 너무 작으면 건너뜀
                continue   

            direction = self._eligibility_to_direction(elig=elig, reward=reward)  # 물리적 방향 계산
            if direction == 0:  # 방향 없으면 건너뜀
                continue   

            result: ProgrammingResult = net.output_layer.controller.update_weight(
                (int(row), int(post_col)),  # 시냅스 위치
                int(direction),  # 방향
                int(net.global_step),  # 현재 global step
            )

            updated_pairs.append((int(row), int(post_col)))  # pair 기록
            delta_t_records.append(float(best_dt))  # dt 기록
            eligibility_values.append(float(elig))  # eligibility 기록
            directions.append(int(direction))  # 방향 기록
            actions.append(str(result.chosen_action))  # 액션 기록
            n_pulses_plus += int(result.n_pulses_plus)  # plus 펄스 누적
            n_pulses_minus += int(result.n_pulses_minus)  # minus 펄스 누적
            if result.did_refresh:  # refresh 발생 시
                n_refresh += 1  # refresh 증가

        return RSTDPUpdateEvent(
            layer_name="output",  # output layer
            updated_pairs=updated_pairs,  # 업데이트 pair들
            delta_t_records=delta_t_records,  # dt 기록
            eligibility_values=eligibility_values,  # eligibility 기록
            directions=directions,  # 방향 기록
            actions=actions,  # 액션 기록
            n_pulses_plus=n_pulses_plus,  # plus 펄스 총합
            n_pulses_minus=n_pulses_minus,  # minus 펄스 총합
            n_refresh=n_refresh,  # refresh 총합
            reward=float(reward),  # 보상 기록
            winner=winner,  # winner 기록
            target=target,  # target 기록
            message="Output R-STDP pulse update executed." if updated_pairs else "No output pair crossed eligibility threshold.",   
        )

    # ------------------------------------------------------------------
    # Optional hidden-layer R-STDP
    # ------------------------------------------------------------------
    def _learn_hidden(
        self,
        net: MemristiveSNNNetwork,
        reward: float,
    ) -> RSTDPUpdateEvent:
        decision = net.last_decision  # 마지막 decision
        assert decision is not None  # None 아님 보장

        step_records = decision.step_records  # step 기록들
        selected_step = int(decision.selected_step)  # 선택된 step
        if not (0 <= selected_step < len(step_records)):  # step 범위 검사
            return RSTDPUpdateEvent(
                layer_name="hidden",  # hidden layer
                updated_pairs=[],  # 업데이트 없음
                delta_t_records=[],  # 기록 없음
                eligibility_values=[],  # 기록 없음
                directions=[],  # 방향 없음
                actions=[],  # 액션 없음
                n_pulses_plus=0,  # plus 없음
                n_pulses_minus=0,  # minus 없음
                n_refresh=0,  # refresh 없음
                reward=float(reward),  # 보상 기록
                winner=-1,  # winner 없음
                target=None,  # target 없음
                message="Selected step out of range; skipped hidden R-STDP.",  
            )

        hidden_winner = int(step_records[selected_step].hidden_result.winner)  # hidden winner
        if hidden_winner < 0:  # winner 없으면
            return RSTDPUpdateEvent(
                layer_name="hidden",  # hidden layer
                updated_pairs=[],  # 업데이트 없음
                delta_t_records=[],  # 기록 없음
                eligibility_values=[],  # 기록 없음
                directions=[],  # 방향 없음
                actions=[],  # 액션 없음
                n_pulses_plus=0,  # plus 없음
                n_pulses_minus=0,  # minus 없음
                n_refresh=0,  # refresh 없음
                reward=float(reward),  # 보상 기록
                winner=-1,  # winner 없음
                target=None,  # target 없음
                message="No hidden winner at selected step; skipped hidden R-STDP.",   
            )

        pre_spikes = np.array(
            [np.asarray(rec.hidden_input_vector, dtype=int) for rec in step_records],  # hidden 입력 spike들
            dtype=int,
        )  # [T, hidden_input_dim]

        post_spikes = np.array(
            [np.asarray(rec.hidden_result.spikes, dtype=int) for rec in step_records],  # hidden 출력 spike들
            dtype=int,
        )  # [T, hidden_dim]

        post_times = np.flatnonzero(post_spikes[:, hidden_winner] > 0).astype(int).tolist()  # hidden post 시각
        if not post_times:  # hidden post spike 없으면
            return RSTDPUpdateEvent(
                layer_name="hidden",  # hidden layer
                updated_pairs=[],  # 업데이트 없음
                delta_t_records=[],  # 기록 없음
                eligibility_values=[],  # 기록 없음
                directions=[],  # 방향 없음
                actions=[],  # 액션 없음
                n_pulses_plus=0,  # plus 없음
                n_pulses_minus=0,  # minus 없음
                n_refresh=0,  # refresh 없음
                reward=float(reward),  # 보상 기록
                winner=hidden_winner,  # winner 기록
                target=None,  # target 없음
                message="No hidden postsynaptic spike available; skipped hidden R-STDP.",  
            )

        reward_sign = self._reward_sign(reward)  # 보상 부호
        if reward_sign == 0:  # 보상이 0이면
            return RSTDPUpdateEvent(
                layer_name="hidden",  # hidden layer
                updated_pairs=[],  # 업데이트 없음
                delta_t_records=[],  # 기록 없음
                eligibility_values=[],  # 기록 없음
                directions=[],  # 방향 없음
                actions=[],  # 액션 없음
                n_pulses_plus=0,  # plus 없음
                n_pulses_minus=0,  # minus 없음
                n_refresh=0,  # refresh 없음
                reward=float(reward),  # 보상 기록
                winner=hidden_winner,  # winner 기록
                target=None,  # target 없음
                message="Reward is zero; no gated hidden R-STDP update applied.",  
            )

        updated_pairs: List[Tuple[int, int]] = []  # 업데이트 pair들
        delta_t_records: List[float] = []  # dt 기록
        eligibility_values: List[float] = []  # eligibility 기록
        directions: List[int] = []  # 방향 기록
        actions: List[str] = []  # 액션 기록
        n_pulses_plus = 0  # plus 펄스 누적
        n_pulses_minus = 0  # minus 펄스 누적
        n_refresh = 0  # refresh 누적

        n_pre = pre_spikes.shape[1]  # pre 입력 수
        for row in range(n_pre):  # 모든 pre 입력 순회
            pre_times = np.flatnonzero(pre_spikes[:, row] > 0).astype(int).tolist()  # pre spike 시각들
            if not pre_times:  # spike 없으면 건너뜀
                continue   

            best_dt, elig = self._pair_eligibility(pre_times, post_times)  # eligibility 계산
            if abs(elig) < self.cfg.eligibility_threshold:  # 너무 작으면 건너뜀
                continue   

            direction = self._eligibility_to_direction(elig=elig, reward=reward)  # 방향 계산
            if direction == 0:  # 방향 없으면 건너뜀
                continue   

            result: ProgrammingResult = net.hidden_layer.controller.update_weight(
                (int(row), int(hidden_winner)),  # hidden 시냅스 위치
                int(direction),  # 방향
                int(net.global_step),  # 현재 global step
            )

            updated_pairs.append((int(row), int(hidden_winner)))  # pair 기록
            delta_t_records.append(float(best_dt))  # dt 기록
            eligibility_values.append(float(elig))  # eligibility 기록
            directions.append(int(direction))  # 방향 기록
            actions.append(str(result.chosen_action))  # 액션 기록
            n_pulses_plus += int(result.n_pulses_plus)  # plus 펄스 누적
            n_pulses_minus += int(result.n_pulses_minus)  # minus 펄스 누적
            if result.did_refresh:  # refresh 발생 시
                n_refresh += 1  # refresh 증가

        return RSTDPUpdateEvent(
            layer_name="hidden",  # hidden layer
            updated_pairs=updated_pairs,  # 업데이트 pair들
            delta_t_records=delta_t_records,  # dt 기록
            eligibility_values=eligibility_values,  # eligibility 기록
            directions=directions,  # 방향 기록
            actions=actions,  # 액션 기록
            n_pulses_plus=n_pulses_plus,  # plus 펄스 총합
            n_pulses_minus=n_pulses_minus,  # minus 펄스 총합
            n_refresh=n_refresh,  # refresh 총합
            reward=float(reward),  # 보상 기록
            winner=hidden_winner,  # winner 기록
            target=None,  # target 없음
            message="Hidden R-STDP pulse update executed." if updated_pairs else "No hidden pair crossed eligibility threshold.",   
        )

    # ------------------------------------------------------------------
    # STDP core
    # ------------------------------------------------------------------
    def _pair_eligibility(self, pre_times: Sequence[int], post_times: Sequence[int]) -> Tuple[float, float]:
        """
        Returns:
            best_dt: delta_t of the strongest pair contribution
            elig: summed STDP eligibility over all pre/post pairs
        """
        elig = 0.0  # 총 eligibility
        best_dt = 0.0  # 가장 강한 dt
        best_mag = -1.0  # 가장 큰 기여도 크기

        for t_pre in pre_times:  # 모든 pre 시각 순회
            for t_post in post_times:  # 모든 post 시각 순회
                dt = float(t_post - t_pre)  # 시간차 계산
                contrib = self._stdp_kernel(dt)  # STDP 기여도 계산
                elig += contrib  # 총합 누적

                if abs(contrib) > best_mag:  # 가장 큰 기여도면
                    best_mag = abs(contrib)  # 최대 크기 갱신
                    best_dt = dt  # 해당 dt 저장

        return float(best_dt), float(elig)  # 결과 반환

    def _stdp_kernel(self, dt: float) -> float:
        if dt > 0.0:  # pre가 먼저면
            return float(self.cfg.a_plus * np.exp(-dt / max(self.cfg.tau_plus, 1e-12)))  # 강화
        if dt < 0.0:  # post가 먼저면
            return float(-self.cfg.a_minus * np.exp(-(-dt) / max(self.cfg.tau_minus, 1e-12)))  # 약화
        return 0.0  # 동시에 발생하면 0

    @staticmethod
    def _reward_sign(reward: float) -> int:
        if reward > 0.0:  # 양의 보상
            return 1  # +1
        if reward < 0.0:  # 음의 보상
            return -1  # -1
        return 0  # 0 보상

    def _eligibility_to_direction(self, elig: float, reward: float) -> int:
        """
        Convert sign(reward * eligibility) to physical programming direction.

        +1: strengthen effective synaptic influence
        -1: weaken effective synaptic influence
         0: no update
        """
        value = float(elig) * float(reward)  # reward와 eligibility 곱
        if value > 0.0:  # 양수면
            return +1  # 강화 방향
        if value < 0.0:  # 음수면
            return -1  # 약화 방향
        return 0  # 업데이트 없음


if __name__ == "__main__":
    print("learning.py ready: explicit delta_t -> eligibility -> pulse direction path enabled.")  # 준비 메시지
