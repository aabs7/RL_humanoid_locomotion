import gymnasium as gym
import math, torch
from rl.policies import make_actor_critic

env = gym.make("HalfCheetah-v4", render_mode="human")
# env = gym.make("LunarLander-v3", render_mode="human")

print("Observation space shape:", env.observation_space.shape)
print("Action space shape:", env.action_space.shape)

agent = make_actor_critic(env.observation_space, env.action_space)

B = 4  # batch size
obs = torch.randn(B, env.observation_space.shape[0])  # (B, obs_dim)
action, logprob, entropy, value = agent.act(obs)

assert action.shape == (B, env.action_space.shape[0])
assert logprob.shape == (B,)
assert entropy.shape == (B,)
assert value.shape == (B,)

assert agent.actor.log_std.requires_grad   # log_std should be a learnable
print("ok")
