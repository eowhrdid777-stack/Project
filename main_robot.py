from __future__ import annotations

"""Run the trained/simulated SNN loop on the real Arduino robot.

Place this file next to your existing main.py, config.py, network.py, env.py, etc.
Run on the Raspberry Pi/PC connected to Arduino Nano:
    python main_robot.py
"""

from typing import Any, Dict, Optional
import time

import config as cfg
from learning import RewardModulatedSTDPLearner
from metrics import SNNMetrics
from main import build_encoder, build_network, build_learner, reward_to_target, _relax_stm_crossbars
from robot_interface import ArduinoRobotInterface, RobotCalibration
from real_robot_env import RealRescueRobotEnv, RealRobotEnvConfig


def _cfg(name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


def build_robot_interface() -> ArduinoRobotInterface:
    cal = RobotCalibration(
        port=str(_cfg("ROBOT_SERIAL_PORT", "/dev/ttyUSB0")),
        baudrate=int(_cfg("ROBOT_BAUDRATE", 9600)),
        clear_min_mm=float(_cfg("ROBOT_CLEAR_MIN_MM", 40.0)),
        clear_max_mm=float(_cfg("ROBOT_CLEAR_MAX_MM", 450.0)),
        obstacle_stop_mm=float(_cfg("ROBOT_OBSTACLE_STOP_MM", 90.0)),
        victim_color_names=tuple(
            str(x).lower().strip()
            for x in _cfg(
                "ROBOT_VICTIM_COLOR_NAMES",
                [str(_cfg("ROBOT_VICTIM_COLOR_NAME", "red"))],
            )
        ),
        min_clear_channel=int(_cfg("ROBOT_COLOR_MIN_C", 40)),
        enable_sound_sensor=bool(_cfg("ROBOT_ENABLE_SOUND_SENSOR", True)),
        sound_raw_min=int(_cfg("ROBOT_SOUND_RAW_MIN", 350)),
        sound_raw_max=int(_cfg("ROBOT_SOUND_RAW_MAX", 800)),
    )
    return ArduinoRobotInterface(cal)


def build_real_env(interface: ArduinoRobotInterface) -> RealRescueRobotEnv:
    fixed_maps = _cfg("ENV_FIXED_MAPS", None)
    if not fixed_maps:
        raise RuntimeError("Real robot run requires ENV_FIXED_MAPS in config.py")
    rcfg = RealRobotEnvConfig(
        max_steps=int(_cfg("ENV_MAX_STEPS", 50)),
        reward_step_penalty=float(_cfg("ENV_REWARD_STEP_PENALTY", 0.0)),
        reward_collision=float(_cfg("ENV_REWARD_COLLISION", -0.05)),
        reward_closer=float(_cfg("ENV_REWARD_CLOSER", 0.10)),
        reward_farther=float(_cfg("ENV_REWARD_FARTHER", -0.03)),
        reward_found_victim=float(_cfg("ENV_REWARD_FOUND_VICTIM", 3.0)),
        reward_stay=float(_cfg("ENV_REWARD_STAY", -0.02)),
        reward_turn=float(_cfg("ENV_REWARD_TURN", -0.002)),
        fixed_agent_heading=int(_cfg("ENV_FIXED_AGENT_HEADING", 0)),
        fixed_map_index=int(_cfg("ENV_FIXED_MAP_INDEX", 0)),
        map_selection_mode="fixed_index",
        sound_threshold=float(_cfg("ROBOT_SOUND_THRESHOLD", 0.55)),
    )
    return RealRescueRobotEnv(interface=interface, fixed_maps=fixed_maps, cfg=rcfg)


def run_real_episode(map_index: int = 0, learn: bool = True, verbose: bool = True) -> Dict[str, Any]:
    encoder = build_encoder()
    net = build_network(encoder=encoder, seed=int(_cfg("SEED", 42)))
    learner: Optional[RewardModulatedSTDPLearner] = build_learner() if learn else None
    metrics = SNNMetrics()

    interface = build_robot_interface()
    try:
        env = build_real_env(interface)
        obs = env.reset(map_index=map_index)
        net.reset_episode()

        done = False
        total_reward = 0.0
        step_idx = 0
        fallback_count = 0

        if verbose:
            print("REAL ROBOT EPISODE")
            print("initial obs:", obs)
            print(env.render_ascii())

        while not done:
            decision = net.decide(obs)
            if decision.used_fallback:
                fallback_count += 1

            step = env.step(decision.action)
            target = reward_to_target(step.reward, decision.action)

            learning_events = None
            if learner is not None:
                learning_events = learner.learn(net=net, reward=step.reward, target=target)

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

            dt_s = float(_cfg("ROBOT_ACTION_INTERVAL_S", _cfg("STM_RELAX_PER_DECISION_S", 0.005)))
            _relax_stm_crossbars(net, dt_s=dt_s)

            total_reward += float(step.reward)
            obs = step.observation
            done = bool(step.done)

            if verbose:
                print(f"[step {step_idx}] action={decision.action} reward={step.reward:.3f} info={step.info}")
                print(env.render_ascii())

            step_idx += 1
            time.sleep(float(_cfg("ROBOT_STEP_PAUSE_S", 0.05)))

        success = bool(step.info.get("remaining_victims", 1) == 0) if 'step' in locals() else False
        return {
            "episode_reward": total_reward,
            "success": success,
            "fallback_count": fallback_count,
            "steps": step_idx,
            "metrics": metrics.summary_dict(),
        }
    finally:
        interface.close()


if __name__ == "__main__":
    result = run_real_episode(map_index=int(_cfg("ENV_FIXED_MAP_INDEX", 0)), learn=bool(_cfg("ROBOT_ENABLE_LEARNING", True)))
    print("REAL ROBOT RESULT:", result)
