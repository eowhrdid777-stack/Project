

# from __future__ import annotations

# from typing import Optional

# import numpy as np

# import config as cfg
# from encoding import SensorSpikeEncoder
# from env import AbstractRescueGridEnv
# from learning import RewardModulatedSTDPLearner, RSTDPConfig
# from metrics import SNNMetrics
# from network import MemristiveSNNNetwork


# # ============================================================
# # Basic experiment settings
# # ============================================================
# N_EPISODES_BASELINE = 3
# N_EPISODES_TRAIN = 5
# N_EPISODES_EVAL = 3


# def build_encoder() -> SensorSpikeEncoder:
#     feature_names = [
#         "front_clearance",
#         "left_clearance",
#         "right_clearance",
#         "victim_signal",
#     ]

#     value_ranges = {
#         "front_clearance": (0.0, 1.0),
#         "left_clearance": (0.0, 1.0),
#         "right_clearance": (0.0, 1.0),
#         "victim_signal": (0.0, 1.0),
#     }

#     return SensorSpikeEncoder(
#         feature_names=feature_names,
#         value_ranges=value_ranges,
#         mode=getattr(cfg, "ENCODER_MODE", "population_latency"),
#     )


# def build_env(seed: Optional[int] = None) -> AbstractRescueGridEnv:
#     return AbstractRescueGridEnv(
#         width=getattr(cfg, "ENV_WIDTH", 8),
#         height=getattr(cfg, "ENV_HEIGHT", 8),
#         max_steps=getattr(cfg, "ENV_MAX_STEPS", 10),
#         obstacle_density=getattr(cfg, "ENV_OBSTACLE_DENSITY", 0.12),
#         seed=getattr(cfg, "SEED", 42) if seed is None else seed,
#         n_victims=getattr(cfg, "ENV_N_VICTIMS", 3),
#     )


# def build_network(
#     encoder: SensorSpikeEncoder,
#     seed: Optional[int] = None,
# ) -> MemristiveSNNNetwork:
#     return MemristiveSNNNetwork(
#         encoder=encoder,
#         n_actions=4,
#         hidden_dim=getattr(cfg, "NETWORK_HIDDEN_DIM", 8),
#         seed=getattr(cfg, "SEED", 42) if seed is None else seed,
#     )


# def build_learner() -> RewardModulatedSTDPLearner:
#     return RewardModulatedSTDPLearner(
#         RSTDPConfig(
#             tau_plus=getattr(cfg, "RSTDP_TAU_PLUS", 2.0),
#             tau_minus=getattr(cfg, "RSTDP_TAU_MINUS", 2.0),
#             a_plus=getattr(cfg, "RSTDP_A_PLUS", 1.0),
#             a_minus=getattr(cfg, "RSTDP_A_MINUS", 0.8),
#             eligibility_threshold=getattr(cfg, "RSTDP_ELIGIBILITY_THRESHOLD", 1e-6),
#             use_surrogate_post_on_fallback=getattr(
#                 cfg,
#                 "RSTDP_USE_SURROGATE_POST_ON_FALLBACK",
#                 False,
#             ),
#             enable_hidden_rstdp=getattr(cfg, "RSTDP_ENABLE_HIDDEN", False),

#             # delta_w = reward * eligibility -> pulse count mapping
#             delta_w_scale=getattr(cfg, "RSTDP_DELTA_W_SCALE", 1.0),
#             pulse_base=getattr(cfg, "RSTDP_PULSE_BASE", 1),
#             pulse_max=getattr(cfg, "RSTDP_PULSE_MAX", 4),
#             delta_w_per_pulse=getattr(cfg, "RSTDP_DELTA_W_PER_PULSE", 0.05),
#         )
#     )


# def reward_to_target(reward: float, chosen_action: int) -> Optional[int]:
#     """
#     Minimal target policy.
#     For now, only positive reward reinforces the chosen action explicitly.
#     """
#     if reward > 0.0:
#         return int(chosen_action)
#     return None


# def run_episode(
#     env: AbstractRescueGridEnv,
#     net: MemristiveSNNNetwork,
#     learner: Optional[RewardModulatedSTDPLearner],
#     metrics: Optional[SNNMetrics] = None,
#     episode_idx: int = 0,
#     phase_name: str = "train",
#     verbose: bool = True,
# ):
#     obs = env.reset()
#     net.reset_episode()

#     done = False
#     total_reward = 0.0
#     step_idx = 0
#     fallback_count = 0

#     if verbose:
#         print("=" * 70)
#         print(f"{phase_name.upper()} EPISODE {episode_idx}")
#         print("=" * 70)
#         print("initial obs:", obs)
#         print(env.render_ascii())
#         print()

#     while not done:
#         decision = net.decide(obs)
#         step = env.step(decision.action)

#         if decision.used_fallback:
#             fallback_count += 1

#         target = reward_to_target(step.reward, decision.action)

#         learning_events = None
#         if learner is not None:
#             learning_events = learner.learn(
#                 net=net,
#                 reward=step.reward,
#                 target=target,
#             )

#         if metrics is not None:
#             metrics.add_episode(
#                 rollout_info={
#                     "used_fallback": decision.used_fallback,
#                     "selected_step": decision.selected_step,
#                     "action": decision.action,
#                     "hidden_spikes": [
#                         rec.hidden_result.spikes for rec in decision.step_records
#                     ],
#                     "output_spikes": [
#                         rec.output_result.spikes for rec in decision.step_records
#                     ],
#                 },
#                 learning_event=None if learning_events is None else learning_events["output"],
#             )

#         total_reward += float(step.reward)

#         if verbose:
#             print(f"[step {step_idx}] action={decision.action} reward={step.reward}")
#             print(
#                 f"selected_step={decision.selected_step} "
#                 f"used_fallback={decision.used_fallback} target={target}"
#             )
#             if learning_events is not None:
#                 print("learning output:", learning_events["output"])
#                 print("learning hidden:", learning_events["hidden"])
#             print(env.render_ascii())
#             print()

#         obs = step.observation
#         done = bool(step.done)
#         step_idx += 1

#     # 다중 생존자 기준:
#     # 모든 생존자를 구조해서 남은 victim_positions가 0개이면 성공
#     success = bool(len(env.victim_positions) == 0)

#     if verbose:
#         print(f"episode reward: {total_reward}")
#         print(f"success: {success}, fallback_count: {fallback_count}")
#         print()

#     return {
#         "episode_reward": float(total_reward),
#         "success": success,
#         "fallback_count": int(fallback_count),
#         "steps": int(step_idx),
#     }


# def run_phase(
#     env: AbstractRescueGridEnv,
#     net: MemristiveSNNNetwork,
#     learner: Optional[RewardModulatedSTDPLearner],
#     n_episodes: int,
#     phase_name: str,
#     verbose: bool = True,
# ):
#     metrics = SNNMetrics()
#     results = []

#     for ep in range(n_episodes):
#         result = run_episode(
#             env=env,
#             net=net,
#             learner=learner,
#             metrics=metrics,
#             episode_idx=ep,
#             phase_name=phase_name,
#             verbose=verbose,
#         )
#         results.append(result)

#     rewards = np.array([r["episode_reward"] for r in results], dtype=float)
#     successes = np.array([r["success"] for r in results], dtype=float)
#     fallbacks = np.array([r["fallback_count"] for r in results], dtype=float)
#     steps = np.array([r["steps"] for r in results], dtype=float)

#     summary = {
#         "phase": phase_name,
#         "mean_reward": float(rewards.mean()) if rewards.size else 0.0,
#         "success_rate": float(successes.mean()) if successes.size else 0.0,
#         "mean_fallback_count": float(fallbacks.mean()) if fallbacks.size else 0.0,
#         "mean_steps": float(steps.mean()) if steps.size else 0.0,
#         "metrics": metrics.summary_dict(),
#     }

#     print("-" * 70)
#     print(f"{phase_name.upper()} SUMMARY")
#     print("-" * 70)
#     for k, v in summary.items():
#         if k != "metrics":
#             print(f"{k}: {v}")
#     print("metrics:", summary["metrics"])
#     print()

#     return results, summary


# def run_experiment(verbose: bool = True):
#     seed = getattr(cfg, "SEED", 42)

#     encoder = build_encoder()
#     env = build_env(seed=seed)
#     net = build_network(encoder=encoder, seed=seed)
#     learner = build_learner()

#     # 1) Baseline: no learning
#     baseline_results, baseline_summary = run_phase(
#         env=env,
#         net=net,
#         learner=None,
#         n_episodes=N_EPISODES_BASELINE,
#         phase_name="baseline",
#         verbose=verbose,
#     )

#     # 2) Train: R-STDP learning on
#     train_results, train_summary = run_phase(
#         env=env,
#         net=net,
#         learner=learner,
#         n_episodes=N_EPISODES_TRAIN,
#         phase_name="train",
#         verbose=verbose,
#     )

#     # 3) Eval: learning off, learned weights kept
#     eval_results, eval_summary = run_phase(
#         env=env,
#         net=net,
#         learner=None,
#         n_episodes=N_EPISODES_EVAL,
#         phase_name="eval",
#         verbose=verbose,
#     )

#     print("=" * 70)
#     print("FINAL COMPARISON")
#     print("=" * 70)
#     print("baseline mean_reward:", baseline_summary["mean_reward"])
#     print("train    mean_reward:", train_summary["mean_reward"])
#     print("eval     mean_reward:", eval_summary["mean_reward"])
#     print("baseline success_rate:", baseline_summary["success_rate"])
#     print("eval     success_rate:", eval_summary["success_rate"])
#     print("baseline mean_fallback_count:", baseline_summary["mean_fallback_count"])
#     print("eval     mean_fallback_count:", eval_summary["mean_fallback_count"])
#     print()

#     return {
#         "baseline": baseline_summary,
#         "train": train_summary,
#         "eval": eval_summary,
#     }


# if __name__ == "__main__":
#     run_experiment(verbose=True)

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

import config as cfg
from encoding import SensorSpikeEncoder
from env import AbstractRescueGridEnv
from learning import RewardModulatedSTDPLearner, RSTDPConfig
from metrics import SNNMetrics
from network import MemristiveSNNNetwork


def _cfg(name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


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
    feature_names = list(
        _cfg(
            "ENCODER_FEATURE_NAMES",
            [
                "front_clearance",
                "left_clearance",
                "right_clearance",
                "victim_signal",
            ],
        )
    )

    value_ranges = dict(
        _cfg(
            "ENCODER_VALUE_RANGES",
            {
                "front_clearance": (0.0, 1.0),
                "left_clearance": (0.0, 1.0),
                "right_clearance": (0.0, 1.0),
                "victim_signal": (0.0, 1.0),
            },
        )
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
        n_actions=int(_cfg("NETWORK_N_ACTIONS", 4)),
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
    verbose: bool = True,
) -> Dict[str, Any]:
    map_index = _fixed_map_index_for_episode(env, episode_idx)
    obs = env.reset(map_index=map_index) if map_index is not None else env.reset()
    net.reset_episode()

    done = False
    total_reward = 0.0
    step_idx = 0
    fallback_count = 0
    stm_relax_dt_s = float(_cfg("SIM_ENV_STEP_DT_S", _cfg("ENV_STEP_DT_S", 0.0)))
    relax_stm = bool(_cfg("NETWORK_RELAX_STM_BETWEEN_ENV_STEPS", True))

    if verbose:
        print("=" * 70)
        print(f"{phase_name.upper()} EPISODE {episode_idx}")
        print("=" * 70)
        if map_index is not None:
            print(f"map_index: {map_index}")
        state = env.get_env_state() if hasattr(env, "get_env_state") else {}
        if state:
            print("env_state:", state)
        print("initial obs:", obs)
        print(env.render_ascii())
        print()

    while not done:
        decision = net.decide(obs)
        step = env.step(decision.action)

        if decision.used_fallback:
            fallback_count += 1

        target = reward_to_target(step.reward, decision.action)

        learning_events = None
        if learner is not None:
            learning_events = learner.learn(
                net=net,
                reward=float(step.reward),
                target=target,
            )

        if relax_stm and stm_relax_dt_s > 0.0:
            _relax_stm_crossbars(net, stm_relax_dt_s)

        if metrics is not None:
            metrics.add_episode(
                rollout_info={
                    "used_fallback": decision.used_fallback,
                    "selected_step": decision.selected_step,
                    "action": decision.action,
                    "hidden_spikes": [rec.hidden_result.spikes for rec in decision.step_records],
                    "output_spikes": [rec.output_result.spikes for rec in decision.step_records],
                },
                learning_event=None if learning_events is None else learning_events.get("output"),
            )

        total_reward += float(step.reward)

        if verbose:
            print(f"[step {step_idx}] action={decision.action} reward={step.reward}")
            print(
                f"selected_step={decision.selected_step} "
                f"used_fallback={decision.used_fallback} target={target}"
            )
            if learning_events is not None:
                print("learning output:", learning_events.get("output"))
                print("learning hidden:", learning_events.get("hidden"))
            print(env.render_ascii())
            print()

        obs = step.observation
        done = bool(step.done)
        step_idx += 1

    success = bool(len(env.victim_positions) == 0)

    if verbose:
        print(f"episode reward: {total_reward}")
        print(f"success: {success}, fallback_count: {fallback_count}")
        print()

    return {
        "episode_reward": float(total_reward),
        "success": bool(success),
        "fallback_count": int(fallback_count),
        "steps": int(step_idx),
        "map_index": -1 if map_index is None else int(map_index),
    }


def run_phase(
    env: AbstractRescueGridEnv,
    net: MemristiveSNNNetwork,
    learner: Optional[RewardModulatedSTDPLearner],
    n_episodes: int,
    phase_name: str,
    verbose: bool = True,
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
        )
        results.append(result)

    rewards = np.array([r["episode_reward"] for r in results], dtype=float)
    successes = np.array([r["success"] for r in results], dtype=float)
    fallbacks = np.array([r["fallback_count"] for r in results], dtype=float)
    steps = np.array([r["steps"] for r in results], dtype=float)

    summary = {
        "phase": str(phase_name),
        "n_episodes": int(n_episodes),
        "mean_reward": float(rewards.mean()) if rewards.size else 0.0,
        "success_rate": float(successes.mean()) if successes.size else 0.0,
        "mean_fallback_count": float(fallbacks.mean()) if fallbacks.size else 0.0,
        "mean_steps": float(steps.mean()) if steps.size else 0.0,
        "metrics": metrics.summary_dict(),
    }

    print("-" * 70)
    print(f"{phase_name.upper()} SUMMARY")
    print("-" * 70)
    for k, v in summary.items():
        if k != "metrics":
            print(f"{k}: {v}")
    print("metrics:", summary["metrics"])
    print()

    return results, summary


# ============================================================
# Experiment entry point
# ============================================================
def run_experiment(verbose: bool = True) -> Dict[str, Any]:
    seed = int(_cfg("SEED", 42))

    encoder = build_encoder()
    env = build_env(seed=seed)
    net = build_network(encoder=encoder, seed=seed)
    learner = build_learner()

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

    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print("baseline mean_reward:", baseline_summary["mean_reward"])
    print("train    mean_reward:", train_summary["mean_reward"])
    print("eval     mean_reward:", eval_summary["mean_reward"])
    print("baseline success_rate:", baseline_summary["success_rate"])
    print("eval     success_rate:", eval_summary["success_rate"])
    print("baseline mean_fallback_count:", baseline_summary["mean_fallback_count"])
    print("eval     mean_fallback_count:", eval_summary["mean_fallback_count"])
    print()

    return {
        "baseline": baseline_summary,
        "train": train_summary,
        "eval": eval_summary,
        "baseline_results": baseline_results,
        "train_results": train_results,
        "eval_results": eval_results,
    }

