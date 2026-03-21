# from __future__ import annotations

# from typing import Optional

# import config as cfg
# from encoding import SensorSpikeEncoder
# from env import AbstractRescueGridEnv
# from metrics import SNNMetrics
# from network import MemristiveSNNNetwork
# from learning import RewardModulatedSTDPLearner

# learner = RewardModulatedSTDPLearner()

# def build_encoder():
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


# def build_env(seed: Optional[int] = None):
#     return AbstractRescueGridEnv(
#         width=getattr(cfg, "ENV_WIDTH", 8),
#         height=getattr(cfg, "ENV_HEIGHT", 8),
#         max_steps=getattr(cfg, "ENV_MAX_STEPS", 30),
#         obstacle_density=getattr(cfg, "ENV_OBSTACLE_DENSITY", 0.12),
#         seed=getattr(cfg, "SEED", 42) if seed is None else seed,
#     )


# def build_network(encoder, seed: Optional[int] = None):
#     return MemristiveSNNNetwork(
#         encoder=encoder,
#         n_actions=4,
#         hidden_dim=getattr(cfg, "NETWORK_HIDDEN_DIM", 8),
#         seed=getattr(cfg, "SEED", 42) if seed is None else seed,
#     )


# def reward_to_target(reward, action):
#     if reward >= 0:
#         return action
#     return None


# def run_one_episode(env, net, episode_idx=0):
#     obs = env.reset()
#     net.reset_episode()

#     done = False
#     total_reward = 0.0
#     step_idx = 0

#     print("=" * 70)
#     print("EPISODE", episode_idx)
#     print("=" * 70)
#     print("initial obs:", obs)
#     print(env.render_ascii())
#     print()

#     while not done:
#         decision = net.decide(obs)
#         step = env.step(decision.action)

#         target = reward_to_target(step.reward, decision.action)

#         events = learner.learn(
#         net=net,
#         reward=step.reward,
#         target=target
# )

#         total_reward += step.reward

#         print(f"[step {step_idx}] action={decision.action} reward={step.reward}")
#         print(env.render_ascii())
#         print()

#         obs = step.observation
#         done = step.done
#         step_idx += 1

#     print("episode reward:", total_reward)
#     print()

#     return total_reward


# def main():
#     encoder = build_encoder()
#     env = build_env()
#     net = build_network(encoder)
#     metrics = SNNMetrics()

#     for ep in range(3):
#         run_one_episode(env, net, ep)


# if __name__ == "__main__":
#     main()

from __future__ import annotations

from typing import Optional

import numpy as np

import config as cfg
from encoding import SensorSpikeEncoder
from env import AbstractRescueGridEnv
from learning import RewardModulatedSTDPLearner, RSTDPConfig
from metrics import SNNMetrics
from network import MemristiveSNNNetwork


# ============================================================
# Basic experiment settings
# ============================================================
N_EPISODES_BASELINE = 3
N_EPISODES_TRAIN = 5
N_EPISODES_EVAL = 3


def build_encoder() -> SensorSpikeEncoder:
    feature_names = [
        "front_clearance",
        "left_clearance",
        "right_clearance",
        "victim_signal",
    ]

    value_ranges = {
        "front_clearance": (0.0, 1.0),
        "left_clearance": (0.0, 1.0),
        "right_clearance": (0.0, 1.0),
        "victim_signal": (0.0, 1.0),
    }

    return SensorSpikeEncoder(
        feature_names=feature_names,
        value_ranges=value_ranges,
        mode=getattr(cfg, "ENCODER_MODE", "population_latency"),
    )


def build_env(seed: Optional[int] = None) -> AbstractRescueGridEnv:
    return AbstractRescueGridEnv(
        width=getattr(cfg, "ENV_WIDTH", 8),
        height=getattr(cfg, "ENV_HEIGHT", 8),
        max_steps=getattr(cfg, "ENV_MAX_STEPS", 10),
        obstacle_density=getattr(cfg, "ENV_OBSTACLE_DENSITY", 0.12),
        seed=getattr(cfg, "SEED", 42) if seed is None else seed,
    )


def build_network(encoder: SensorSpikeEncoder, seed: Optional[int] = None) -> MemristiveSNNNetwork:
    return MemristiveSNNNetwork(
        encoder=encoder,
        n_actions=4,
        hidden_dim=getattr(cfg, "NETWORK_HIDDEN_DIM", 8),
        seed=getattr(cfg, "SEED", 42) if seed is None else seed,
    )


def build_learner() -> RewardModulatedSTDPLearner:
    return RewardModulatedSTDPLearner(
        RSTDPConfig(
            tau_plus=getattr(cfg, "RSTDP_TAU_PLUS", 2.0),
            tau_minus=getattr(cfg, "RSTDP_TAU_MINUS", 2.0),
            a_plus=getattr(cfg, "RSTDP_A_PLUS", 1.0),
            a_minus=getattr(cfg, "RSTDP_A_MINUS", 0.8),
            eligibility_threshold=getattr(cfg, "RSTDP_ELIGIBILITY_THRESHOLD", 1e-6),
            use_surrogate_post_on_fallback=getattr(cfg, "RSTDP_USE_SURROGATE_POST_ON_FALLBACK", False),
            enable_hidden_rstdp=getattr(cfg, "RSTDP_ENABLE_HIDDEN", False),
        )
    )


def reward_to_target(reward: float, chosen_action: int) -> Optional[int]:
    """
    Minimal target policy.
    For now, only positive reward reinforces the chosen action explicitly.
    """
    if reward > 0.0:
        return int(chosen_action)
    return None


def run_episode(
    env: AbstractRescueGridEnv,
    net: MemristiveSNNNetwork,
    learner: Optional[RewardModulatedSTDPLearner],
    metrics: Optional[SNNMetrics] = None,
    episode_idx: int = 0,
    phase_name: str = "train",
    verbose: bool = True,
):
    obs = env.reset()
    net.reset_episode()

    done = False
    total_reward = 0.0
    step_idx = 0
    fallback_count = 0

    if verbose:
        print("=" * 70)
        print(f"{phase_name.upper()} EPISODE {episode_idx}")
        print("=" * 70)
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
                reward=step.reward,
                target=target,
            )

        if metrics is not None:
            metrics.add_episode(
                rollout_info={
                    "used_fallback": decision.used_fallback,
                    "selected_step": decision.selected_step,
                    "action": decision.action,
                    "hidden_spikes": [rec.hidden_result.spikes for rec in decision.step_records],
                    "output_spikes": [rec.output_result.spikes for rec in decision.step_records],
                },
                learning_event=None if learning_events is None else learning_events["output"],
            )

        total_reward += float(step.reward)

        if verbose:
            print(f"[step {step_idx}] action={decision.action} reward={step.reward}")
            print(f"selected_step={decision.selected_step} used_fallback={decision.used_fallback} target={target}")
            if learning_events is not None:
                print("learning output:", learning_events["output"])
                print("learning hidden:", learning_events["hidden"])
            print(env.render_ascii())
            print()

        obs = step.observation
        done = bool(step.done)
        step_idx += 1

    success = bool(env.agent_pos == env.victim_pos)

    if verbose:
        print(f"episode reward: {total_reward}")
        print(f"success: {success}, fallback_count: {fallback_count}")
        print()

    return {
        "episode_reward": float(total_reward),
        "success": success,
        "fallback_count": int(fallback_count),
        "steps": int(step_idx),
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

    for ep in range(n_episodes):
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
        "phase": phase_name,
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


def run_experiment(verbose: bool = True):
    seed = getattr(cfg, "SEED", 42)

    encoder = build_encoder()
    env = build_env(seed=seed)
    net = build_network(encoder=encoder, seed=seed)
    learner = build_learner()

    # 1) Baseline: no learning
    baseline_results, baseline_summary = run_phase(
        env=env,
        net=net,
        learner=None,
        n_episodes=N_EPISODES_BASELINE,
        phase_name="baseline",
        verbose=verbose,
    )

    # 2) Train: R-STDP learning on
    train_results, train_summary = run_phase(
        env=env,
        net=net,
        learner=learner,
        n_episodes=N_EPISODES_TRAIN,
        phase_name="train",
        verbose=verbose,
    )

    # 3) Eval: learning off, learned weights kept
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
    }

if __name__ == "__main__":
    run_experiment(verbose=True)
