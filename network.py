from __future__ import annotations

# dataclass: 여러 값을 하나의 묶음으로 저장하기 위해 사용
from dataclasses import dataclass
# 타입 힌트를 위한 모듈들
from typing import Any, Dict, List, Optional, Sequence, Union

# 수치 계산용 라이브러리
import numpy as np

# 프로젝트 전체 설정값
import config as cfg
# differential pair 기반 crossbar 구조
from crossbar import DifferentialCrossbar
# 입력 observation을 spike로 바꾸는 encoder
from encoding import EncoderOutput, SensorSpikeEncoder
# neuron layer 및 learning 결과 구조체
from neuron import LearningEvent, MemristiveLIFOutputLayer, NeuronStepResult

# observation 입력 형식: dict / list / numpy array 모두 허용
ObsType = Union[Dict[str, float], Sequence[float], np.ndarray]


@dataclass
class RecurrentStepRecord:
    """recurrent network의 timestep별 상세 기록용 구조체."""

    t: int                                   # 현재 local timestep
    encoder_output: EncoderOutput            # 이 timestep의 encoder 출력
    hidden_input_vector: np.ndarray          # hidden layer에 실제로 넣은 입력 벡터
    hidden_result: NeuronStepResult          # hidden layer 실행 결과
    output_result: NeuronStepResult          # output layer 실행 결과
    recurrent_feedback_used: np.ndarray      # hidden에 넣을 때 사용한 이전 hidden spike


@dataclass
class NetworkDecision:
    """observation 하나를 action 하나로 바꾸는 전체 decision 결과."""

    action: int                                  # 최종 선택된 action
    selected_step: int                           # action이 결정된 timestep
    used_fallback: bool                          # fallback 사용 여부
    hidden_pre_spikes_for_learning: np.ndarray   # hidden layer learning용 pre-spikes
    output_pre_spikes_for_learning: np.ndarray   # output layer learning용 pre-spikes
    integrated_input: np.ndarray                 # 전체 window 동안 누적된 input spikes
    integrated_hidden_spikes: np.ndarray         # 전체 window 동안 누적된 hidden spikes
    hidden_scores_fallback: Optional[np.ndarray] # fallback 시 hidden score (디버깅용)
    output_scores_fallback: Optional[np.ndarray] # fallback 시 output score
    step_records: List[RecurrentStepRecord]      # timestep별 상세 실행 기록
    encoder_outputs: List[EncoderOutput]         # encoder가 만든 window 전체 출력


class MemristiveSNNNetwork:
    """Hardware-aware recurrent SNN 전체 래퍼 클래스.

    전체 구조
    ---------
    observation
        -> encoder
        -> input spikes over time
        -> hidden recurrent layer
        -> output layer
        -> action

    recurrence 방식
    ----------------
    hidden layer 입력을 만들 때
    [현재 input spikes, 이전 hidden spikes] 를 concatenate 해서 넣는다.
    즉 recurrent 구조는 neuron layer 내부가 아니라,
    network가 presynaptic vector를 어떻게 구성하느냐로 구현된다.
    """

    def __init__(
        self,
        encoder: SensorSpikeEncoder,
        n_actions: int,
        hidden_dim: Optional[int] = None,
        seed: Optional[int] = None,
        hidden_input_crossbar: Optional[DifferentialCrossbar] = None,
        hidden_layer: Optional[MemristiveLIFOutputLayer] = None,
        output_crossbar: Optional[DifferentialCrossbar] = None,
        output_layer: Optional[MemristiveLIFOutputLayer] = None,
        reset_neuron_state_each_decision: Optional[bool] = None,
        reset_neuron_state_each_episode: Optional[bool] = None,
        force_action_on_no_spike: Optional[bool] = None,
        learn_hidden_layer: Optional[bool] = None,
    ) -> None:
        # 입력 encoder 저장
        self.encoder = encoder
        # action 개수
        self.n_actions = int(n_actions)
        # 재현성을 위한 seed
        self.seed = int(getattr(cfg, "SEED", 42) if seed is None else seed)
        self.rng = np.random.default_rng(self.seed)

        # observation 하나(decision 단위) 처리할 때마다 neuron state를 reset할지 여부
        self.reset_neuron_state_each_decision = bool(
            getattr(cfg, "NETWORK_RESET_NEURON_STATE_EACH_DECISION", True)
            if reset_neuron_state_each_decision is None
            else reset_neuron_state_each_decision
        )
        # episode 시작 시 neuron state를 reset할지 여부
        self.reset_neuron_state_each_episode = bool(
            getattr(cfg, "NETWORK_RESET_NEURON_STATE_EACH_EPISODE", True)
            if reset_neuron_state_each_episode is None
            else reset_neuron_state_each_episode
        )
        # output spike가 하나도 안 나와도 fallback으로 action을 강제 선택할지 여부
        self.force_action_on_no_spike = bool(
            getattr(cfg, "NETWORK_FORCE_ACTION_ON_NO_SPIKE", True)
            if force_action_on_no_spike is None
            else force_action_on_no_spike
        )
        # hidden layer까지 learning할지 여부
        self.learn_hidden_layer = bool(
            getattr(cfg, "NETWORK_LEARN_HIDDEN_LAYER", False)
            if learn_hidden_layer is None
            else learn_hidden_layer
        )

        # hidden layer 차원 수
        default_hidden_dim = int(getattr(cfg, "NETWORK_HIDDEN_DIM", 8))
        self.hidden_dim = int(default_hidden_dim if hidden_dim is None else hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be >= 1 for the recurrent network.")

        # encoder 출력 차원 = network input 차원
        self.input_dim = int(self.encoder.output_dim)
        # hidden layer 입력은 [input, prev_hidden]를 붙여 쓰므로 차원이 더 커짐
        self.hidden_input_dim = self.input_dim + self.hidden_dim

        # ------------------------------------------------------------
        # hidden layer용 crossbar / layer 생성
        # ------------------------------------------------------------
        # crossbar와 hidden layer 둘 다 안 주어지면 자동 생성
        if hidden_input_crossbar is None and hidden_layer is None:
            hidden_input_crossbar = DifferentialCrossbar(
                # hidden layer 입력 차원 = input_dim + hidden_dim
                n_rows=self.hidden_input_dim,
                # hidden layer 뉴런 수 = hidden_dim
                n_cols=self.hidden_dim,
                seed=self.seed + 101,
            )
        # hidden layer 객체가 없으면, 위에서 만든 crossbar로 neuron layer 생성
        if hidden_layer is None:
            if hidden_input_crossbar is None:
                raise ValueError("hidden_input_crossbar must be provided when hidden_layer is None")
            hidden_layer = MemristiveLIFOutputLayer(crossbar=hidden_input_crossbar, seed=self.seed + 201)

        # ------------------------------------------------------------
        # output layer용 crossbar / layer 생성
        # ------------------------------------------------------------
        if output_crossbar is None and output_layer is None:
            output_crossbar = DifferentialCrossbar(
                # output layer 입력은 hidden spikes
                n_rows=self.hidden_dim,
                # output layer 뉴런 수 = action 개수
                n_cols=self.n_actions,
                seed=self.seed + 301,
            )
        if output_layer is None:
            if output_crossbar is None:
                raise ValueError("output_crossbar must be provided when output_layer is None")
            output_layer = MemristiveLIFOutputLayer(crossbar=output_crossbar, seed=self.seed + 401)

        # 최종 hidden/output layer 저장
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        # 실제로 hidden/output layer가 사용하는 crossbar도 저장
        self.hidden_input_crossbar = self.hidden_layer.crossbar
        self.output_crossbar = self.output_layer.crossbar

        # ------------------------------------------------------------
        # crossbar 차원 검증
        # ------------------------------------------------------------
        # hidden crossbar row 수는 input_dim + hidden_dim 이어야 함
        if int(self.hidden_input_crossbar.n_rows) != self.hidden_input_dim:
            raise ValueError(
                f"Hidden crossbar rows ({self.hidden_input_crossbar.n_rows}) must equal input_dim + hidden_dim ({self.hidden_input_dim})."
            )
        # hidden crossbar 논리적 column 수는 hidden_dim 이어야 함
        if int(self.hidden_input_crossbar.n_logical_cols) != self.hidden_dim:
            raise ValueError(
                f"Hidden crossbar logical cols ({self.hidden_input_crossbar.n_logical_cols}) must equal hidden_dim ({self.hidden_dim})."
            )
        # output crossbar row 수는 hidden_dim 이어야 함
        if int(self.output_crossbar.n_rows) != self.hidden_dim:
            raise ValueError(
                f"Output crossbar rows ({self.output_crossbar.n_rows}) must equal hidden_dim ({self.hidden_dim})."
            )
        # output crossbar 논리적 column 수는 action 수와 같아야 함
        if int(self.output_crossbar.n_logical_cols) != self.n_actions:
            raise ValueError(
                f"Output crossbar logical cols ({self.output_crossbar.n_logical_cols}) must equal n_actions ({self.n_actions})."
            )

        # ------------------------------------------------------------
        # recurrent network 상태 변수
        # ------------------------------------------------------------
        # 이전 timestep의 hidden spikes (recurrent feedback 용)
        self.prev_hidden_spikes = np.zeros(self.hidden_dim, dtype=float)
        # 현재 episode 번호
        self.episode_index = 0
        # 전체 simulation 기준 절대 timestep
        self.global_step = 0
        # 최근 decision / observation / reward / learning 결과 저장
        self.last_decision: Optional[NetworkDecision] = None
        self.last_observation: Optional[ObsType] = None
        self.last_reward: Optional[float] = None
        self.last_learning_event_output: Optional[LearningEvent] = None
        self.last_learning_event_hidden: Optional[LearningEvent] = None
        # action / reward history 저장
        self.action_history: List[int] = []
        self.reward_history: List[float] = []

    # ------------------------------------------------------------------
    # State control
    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        """episode 단위로 network 상태를 초기화."""
        self.episode_index += 1
        self.last_decision = None
        self.last_observation = None
        self.last_reward = None
        self.last_learning_event_output = None
        self.last_learning_event_hidden = None
        self.prev_hidden_spikes.fill(0.0)
        # 옵션이 켜져 있으면 neuron layer 내부 state도 reset
        if self.reset_neuron_state_each_episode:
            self.hidden_layer.reset_state()
            self.output_layer.reset_state()

    def reset_network_state(self) -> None:
        """episode와 무관하게 network 전체 내부 상태를 강제로 reset."""
        self.hidden_layer.reset_state()
        self.output_layer.reset_state()
        self.prev_hidden_spikes.fill(0.0)
        self.last_decision = None
        self.last_observation = None
        self.last_reward = None
        self.last_learning_event_output = None
        self.last_learning_event_hidden = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_window(self, observation: ObsType) -> List[EncoderOutput]:
        """observation 하나를 encoder에 넣어 전체 time window 생성."""
        window = self.encoder.encode_window(observation)
        if len(window) == 0:
            raise RuntimeError("Encoder produced an empty window.")
        return window

    def _hidden_input_vector(self, input_spikes: Sequence[float]) -> np.ndarray:
        """hidden layer 입력 벡터를 생성.

        핵심:
        현재 input spikes와 이전 hidden spikes를 이어붙여(concat)
        recurrent input vector를 만든다.
        """
        inp = np.asarray(input_spikes, dtype=float).reshape(-1)
        if inp.size != self.input_dim:
            raise ValueError(f"Expected input spike length {self.input_dim}, got {inp.size}")
        return np.concatenate([inp, self.prev_hidden_spikes.astype(float)], axis=0)

    def _fallback_action(self, integrated_hidden_spikes: np.ndarray) -> tuple[int, np.ndarray]:
        """output spike가 안 나왔을 때 fallback action 선택.

        전체 window 동안 누적된 hidden spikes를 output layer에 넣어
        measured score argmax로 action을 정한다.
        """
        scores = self.output_layer._measured_vmm(integrated_hidden_spikes)
        return int(np.argmax(scores)), scores

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def decide(self, observation: ObsType) -> NetworkDecision:
        """observation 하나를 받아 최종 action 하나를 결정.

        기본 정책
        ---------
        encoding window 동안 가장 먼저 나온 output winner를 action으로 사용

        fallback 정책
        ---------------
        output spike가 전혀 없으면,
        integrated hidden spikes를 바탕으로 argmax action 선택
        """
        # decision마다 neuron state를 reset하도록 설정되어 있으면 초기화
        if self.reset_neuron_state_each_decision:
            self.hidden_layer.reset_state()
            self.output_layer.reset_state()
            self.prev_hidden_spikes.fill(0.0)

        # 최근 observation 저장
        self.last_observation = observation
        # observation -> encoder -> spike window 생성
        window = self._prepare_window(observation)

        # timestep별 실행 기록 저장용 리스트
        step_records: List[RecurrentStepRecord] = []
        # 아직 action이 선택되지 않은 상태를 -1로 표시
        selected_action = -1
        selected_step = -1
        used_fallback = False
        # learning에 사용할 presynaptic vector 저장용
        selected_hidden_pre: Optional[np.ndarray] = None
        selected_output_pre: Optional[np.ndarray] = None

        # 전체 window 동안 input / hidden spikes 누적
        integrated_input = np.zeros(self.input_dim, dtype=float)
        integrated_hidden_spikes = np.zeros(self.hidden_dim, dtype=float)
        # fallback 시 디버깅용 score 저장
        hidden_scores_fallback: Optional[np.ndarray] = None
        output_scores_fallback: Optional[np.ndarray] = None

        # ------------------------------------------------------------
        # encoding window를 timestep별로 순회
        # ------------------------------------------------------------
        for local_t, enc_out in enumerate(window):
            # hidden layer 입력 = [현재 input spikes, 이전 hidden spikes]
            hidden_pre = self._hidden_input_vector(enc_out.spikes)
            # 현재 timestep에서 사용한 recurrent feedback 기록
            feedback_used = self.prev_hidden_spikes.copy()

            # hidden layer 실행
            hidden_out = self.hidden_layer.step(hidden_pre, step_idx=self.global_step + local_t)
            hidden_spikes = np.asarray(hidden_out.spikes, dtype=float)

            # output layer 실행
            output_out = self.output_layer.step(hidden_spikes, step_idx=self.global_step + local_t)

            # 이번 timestep 결과를 상세 기록
            step_records.append(
                RecurrentStepRecord(
                    t=int(local_t),
                    encoder_output=enc_out,
                    hidden_input_vector=hidden_pre.copy(),
                    hidden_result=hidden_out,
                    output_result=output_out,
                    recurrent_feedback_used=feedback_used,
                )
            )

            # fallback을 위해 전체 window 동안 spike를 누적
            integrated_input += np.asarray(enc_out.spikes, dtype=float)
            integrated_hidden_spikes += hidden_spikes

            # 아직 action이 정해지지 않았고, 이번 timestep에 output winner가 있다면
            # 그 첫 winner를 최종 action으로 채택
            if selected_action < 0 and output_out.winner >= 0:
                selected_action = int(output_out.winner)
                selected_step = int(local_t)
                # learning에 사용할 pre-spikes 저장
                selected_hidden_pre = hidden_pre.copy()
                selected_output_pre = hidden_spikes.copy()

            # recurrence 구현:
            # 현재 hidden spikes를 다음 timestep의 prev_hidden_spikes로 넘김
            self.prev_hidden_spikes = hidden_spikes.copy()

        # ------------------------------------------------------------
        # fallback 처리
        # ------------------------------------------------------------
        # window 전체에서 output winner가 한 번도 없었다면
        if selected_action < 0:
            if not self.force_action_on_no_spike:
                raise RuntimeError("No output spike winner found in current window and fallback is disabled.")

            # 디버깅용 hidden score 계산 (필요 시)
            if selected_hidden_pre is None:
                last_hidden_pre = self._hidden_input_vector(np.zeros(self.input_dim, dtype=float))
                hidden_scores_fallback = self.hidden_layer._measured_vmm(last_hidden_pre)

            # integrated hidden spikes 기반 fallback action 선택
            selected_action, output_scores_fallback = self._fallback_action(integrated_hidden_spikes)
            selected_step = len(window) - 1
            # learning용 presynaptic vector 저장
            selected_hidden_pre = step_records[-1].hidden_input_vector.copy()
            selected_output_pre = integrated_hidden_spikes.copy()
            used_fallback = True

        # learning용 pre-spikes가 반드시 있어야 함
        if selected_hidden_pre is None or selected_output_pre is None:
            raise RuntimeError("Internal error: no presynaptic vectors selected for learning.")

        # global_step은 이번 window 길이만큼 증가
        self.global_step += len(window)

        # 최종 decision 결과 구조체 생성
        decision = NetworkDecision(
            action=int(selected_action),
            selected_step=int(selected_step),
            used_fallback=bool(used_fallback),
            hidden_pre_spikes_for_learning=np.asarray(selected_hidden_pre, dtype=float),
            output_pre_spikes_for_learning=np.asarray(selected_output_pre, dtype=float),
            integrated_input=np.asarray(integrated_input, dtype=float),
            integrated_hidden_spikes=np.asarray(integrated_hidden_spikes, dtype=float),
            hidden_scores_fallback=None if hidden_scores_fallback is None else np.asarray(hidden_scores_fallback, dtype=float),
            output_scores_fallback=None if output_scores_fallback is None else np.asarray(output_scores_fallback, dtype=float),
            step_records=step_records,
            encoder_outputs=window,
        )
        # 최근 decision 저장
        self.last_decision = decision
        # action history에 추가
        self.action_history.append(int(selected_action))
        return decision

    def act(self, observation: ObsType) -> int:
        """action 값만 간단히 얻고 싶을 때 사용하는 wrapper."""
        return int(self.decide(observation).action)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def learn(
        self,
        reward: float,
        target: Optional[int] = None,
        update_all_active_to_target: bool = True,
        punish_wrong_winner: bool = True,
    ) -> Dict[str, Optional[LearningEvent]]:
        """pulse-based learning 적용.

        기본 동작
        ---------
        hidden -> output projection만 학습한다.
        hidden layer 학습은 선택 사항이며,
        BPTT 없이 local reward reinforcement만 사용한다.
        """
        # 먼저 decide()/act()가 실행되어 있어야 learning 가능
        if self.last_decision is None:
            raise RuntimeError("No prior decision available. Call decide()/act() before learn().")

        # output layer learning 수행
        output_event = self.output_layer.apply_reward_modulated_update(
            pre_spikes=self.last_decision.output_pre_spikes_for_learning,
            winner=int(self.last_decision.action),
            reward=float(reward),
            step_idx=int(self.global_step),
            target=target,
            update_all_active_to_target=bool(update_all_active_to_target),
            punish_wrong_winner=bool(punish_wrong_winner),
        )
        self.last_learning_event_output = output_event

        # hidden layer learning은 optional
        hidden_event: Optional[LearningEvent] = None
        if self.learn_hidden_layer:
            # action이 선택된 timestep의 hidden result를 가져옴
            selected_record = self.last_decision.step_records[self.last_decision.selected_step]
            hidden_winner = int(selected_record.hidden_result.winner)
            if hidden_winner >= 0:
                hidden_event = self.hidden_layer.apply_reward_modulated_update(
                    pre_spikes=self.last_decision.hidden_pre_spikes_for_learning,
                    winner=hidden_winner,
                    reward=float(reward),
                    step_idx=int(self.global_step),
                    target=None,
                    update_all_active_to_target=False,
                    punish_wrong_winner=False,
                )
        self.last_learning_event_hidden = hidden_event

        # 최근 reward 기록
        self.last_reward = float(reward)
        self.reward_history.append(float(reward))
        return {"output": output_event, "hidden": hidden_event}

    def act_and_learn(
        self,
        observation: ObsType,
        reward: float,
        target: Optional[int] = None,
        update_all_active_to_target: bool = True,
        punish_wrong_winner: bool = True,
    ) -> tuple[int, Dict[str, Optional[LearningEvent]]]:
        """observation으로 action을 결정하고, 바로 reward로 learning까지 수행."""
        decision = self.decide(observation)
        events = self.learn(
            reward=reward,
            target=target,
            update_all_active_to_target=update_all_active_to_target,
            punish_wrong_winner=punish_wrong_winner,
        )
        return int(decision.action), events

    # ------------------------------------------------------------------
    # Debug / inspection
    # ------------------------------------------------------------------
    def get_debug_state(self) -> Dict[str, Any]:
        """현재 network 내부 상태를 디버깅용 dict로 반환."""
        state: Dict[str, Any] = {
            "episode_index": int(self.episode_index),
            "global_step": int(self.global_step),
            "input_dim": int(self.input_dim),
            "hidden_dim": int(self.hidden_dim),
            "n_actions": int(self.n_actions),
            "prev_hidden_spikes": self.prev_hidden_spikes.copy(),
            "action_history": list(self.action_history),
            "reward_history": list(self.reward_history),
            "last_reward": self.last_reward,
        }
        # 최근 decision이 있으면 관련 정보 추가
        if self.last_decision is not None:
            state["last_action"] = int(self.last_decision.action)
            state["last_selected_step"] = int(self.last_decision.selected_step)
            state["last_used_fallback"] = bool(self.last_decision.used_fallback)
            state["last_integrated_input"] = self.last_decision.integrated_input.copy()
            state["last_integrated_hidden_spikes"] = self.last_decision.integrated_hidden_spikes.copy()
        # 최근 learning event가 있으면 같이 포함
        if self.last_learning_event_output is not None:
            state["last_learning_event_output"] = self.last_learning_event_output
        if self.last_learning_event_hidden is not None:
            state["last_learning_event_hidden"] = self.last_learning_event_hidden
        return state
