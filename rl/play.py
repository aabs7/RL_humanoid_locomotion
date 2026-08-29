import tyro
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from rl.checkpoint import load, config_of, restore
from rl.experiment import set_seed
from rl.envs import make_eval_env
from rl.policies import make_actor_critic


@dataclass
class PlayConfig:
    checkpoint: str
    episodes: int = 10
    seed: int = 0
    render: bool = True
    deterministic: bool = True   # Argmax instead of sampling.
    record: bool = False

def rollout(agent, envs, episodes: int, deterministic: bool):
    returns, lengths, terminated_count = [], [], 0
    obs, _ = envs.reset()
    while len(returns) < episodes:
        with torch.no_grad():
            t_obs = torch.as_tensor(obs, dtype=torch.float32)
            action = (agent.act_deterministic(t_obs) if deterministic else
                      agent.act(t_obs)[0])
        obs, _, term, trunc, infos = envs.step(action.numpy())
        if "episode" in infos:
            returns.append(float(infos["episode"]["r"][0]))
            lengths.append(float(infos["episode"]["l"][0]))
            terminated_count += int(term[0])
    return np.array(returns), np.array(lengths), terminated_count

def play(cfg: PlayConfig) -> None:
    ckpt = load(cfg.checkpoint)
    train_cfg = config_of(ckpt)
    env_id = train_cfg["env_id"]
    print(f"{Path(cfg.checkpoint).name}: {env_id}, iteration {ckpt['iteration']} ,"
          f"step {ckpt['global_step']:,}")
    if ckpt.get("extra", {}).get("return") is not None:
        print(f"  recorded training return: {ckpt['extra']['return']:.1f}")

    set_seed(cfg.seed)
    render_mode = "rgb_array" if cfg.record else ("human" if cfg.render else None)
    env = make_eval_env(env_id, seed=cfg.seed, render_mode=render_mode)
    if cfg.record:
        import gymnasium as gym
        env = gym.wrappers.vector.RecordVideo(env, str(Path(cfg.checkpoint).parent.parent /"videos_play"), name_prefix="play")

    agent = make_actor_critic(env.single_observation_space, env.single_action_space)
    restore(ckpt, agent, optimizer=None, restore_rng=False)
    agent.eval()

    returns, lengths, n_term = rollout(agent, env, cfg.episodes, cfg.deterministic)
    env.close()

    mode = "greedy" if cfg.deterministic else "sampled"
    print(f"\n{cfg.episodes} episodes, {mode}")
    print(f"  return {returns.mean():8.1f} +/- {returns.std():.1f}    "
          f"min {returns.min():.1f}  max {returns.max():.1f}")
    print(f"  length {lengths.mean():8.1f}")
    print(f"  ended: {n_term} terminated, {cfg.episodes - n_term} truncated (time limit)")


if __name__ == "__main__":
    play(tyro.cli(PlayConfig))
