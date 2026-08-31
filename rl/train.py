import torch
import time
import tyro
import numpy as np

from rl.config import Config
from rl.experiment import create_run_dir, set_seed
from rl.envs import make_envs
from rl.policies import make_actor_critic
from rl.buffers import RolloutBuffer
from rl.logger import Logger, EpisodeTracker
from rl.algorithms.ppo import ppo_update
from rl.checkpoint import save
from rl.normalizers import ObsNormalizer, RewardNormalizer

def train(cfg: Config) -> None:
    run_dir = create_run_dir(cfg)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'run {run_dir}\ndevice: {device} iterations {cfg.num_iterations} batch {cfg.batch_size}')

    envs = make_envs(cfg.env_id, cfg.num_envs, cfg.seed,
                     capture_video=cfg.capture_video,
                     video_dir=run_dir / "videos")

    agent = make_actor_critic(envs.single_observation_space, envs.single_action_space).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.ppo.lr, eps=1e-5)
    buf = RolloutBuffer(cfg.num_steps, cfg.num_envs, envs.single_observation_space, envs.single_action_space, device)
    logger = Logger(run_dir, print_every=10)
    tracker = EpisodeTracker(window=100)
    obs_norm = ObsNormalizer(shape=envs.single_observation_space.shape) if cfg.obs_norm else None
    reward_norm = RewardNormalizer(num_envs=cfg.num_envs, gamma=cfg.ppo.gamma) if cfg.reward_norm else None

    normalize_obs = lambda obs: obs_norm(obs) if obs_norm else obs
    normalize_reward = lambda reward, dones: reward_norm(reward, dones) if reward_norm else reward

    obs_np, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.as_tensor(normalize_obs(obs_np), device=device, dtype=torch.float32)
    next_done = torch.zeros(cfg.num_envs, device=device, dtype=torch.bool)
    global_step, best_return, t0 = 0, -float("inf"), time.time()

    for iteration in range(1, cfg.num_iterations + 1):

        # Learning rate annealing
        if cfg.ppo.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            for g in optimizer.param_groups:
                g["lr"] = frac * cfg.ppo.lr

        for t in range(cfg.num_steps):  # collect trajectory
            global_step += cfg.num_envs
            buf.obs[t] = next_obs
            buf.prev_done[t] = next_done

            with torch.no_grad():
                action, log_prob, _ , value = agent.act(next_obs)

            buf.actions[t] = action
            buf.logprobs[t] = log_prob
            buf.values[t] = value

            obs_np, rewards, term, trunc, infos = envs.step(action.cpu().numpy())
            tracker.update(infos)

            dones_np = np.logical_or(term, trunc).astype(np.float32)
            buf.rewards[t] = torch.as_tensor(normalize_reward(rewards, dones_np), device=device, dtype=torch.float32)
            buf.terminated[t] = torch.as_tensor(term, device=device, dtype=torch.float32)
            buf.truncated[t] = torch.as_tensor(trunc, device=device, dtype=torch.float32)

            next_obs = torch.as_tensor(normalize_obs(obs_np), device=device, dtype=torch.float32)
            next_done = torch.as_tensor(dones_np, device=device, dtype=torch.float32)

        with torch.no_grad():
            next_value = agent.value(next_obs)
        buf.compute_gae(next_value, cfg.ppo.gamma, cfg.ppo.gae_lambda)

        metrics = ppo_update(agent, optimizer, buf.flatten(), cfg.ppo)

        for k, v in metrics.items():
            logger.log(k, v)
        tracker.log_to(logger)
        logger.log("charts/SPS", cfg.batch_size / (time.time() - t0))
        logger.log("charts/lr", optimizer.param_groups[0]["lr"])
        logger.log("charts/iteration", iteration)
        logger.dump(step=global_step)


        if iteration % cfg.save_every == 0:
            save(run_dir / "checkpoints" / "latest.pt",
                 agent=agent, optimizer=optimizer, cfg=cfg, iteration=iteration, global_step=global_step, obs_rms=obs_norm.rms if obs_norm else None)

        if tracker.returns:
            mean_return = float(np.mean(tracker.returns))
            if mean_return > best_return:
                best_return = mean_return
                save(run_dir / "checkpoints" / "best.pt",
                     agent=agent, optimizer=optimizer, cfg=cfg, iteration=iteration, global_step=global_step,
                     obs_rms=obs_norm.rms if obs_norm else None,
                     extra={"return": mean_return})

    logger.close()
    envs.close()
    print(f'Training done. Best mean return: {best_return:.2f}. Total time: {time.time() - t0:.2f}s')


if __name__ == "__main__":
    train(tyro.cli(Config))
