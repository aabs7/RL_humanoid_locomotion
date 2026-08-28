import gymnasium as gym
from rl.logger import EpisodeTracker

envs = gym.wrappers.vector.RecordEpisodeStatistics(
    gym.make_vec("LunarLander-v3", num_envs=4, vectorization_mode="sync"))
tracker = EpisodeTracker(window=100)
envs.reset(seed=0)
for _ in range(3000):
    *_, info = envs.step(envs.action_space.sample())
    tracker.update(info)

print(tracker.total_episodes, len(tracker.returns), sum(tracker.returns) / len(tracker.returns))
assert all(x != 0.0 for x in tracker.returns), "padding zeros leaked in"
