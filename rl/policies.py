import torch
import torch.nn as nn
import numpy as np

import gymnasium as gym
from torch.distributions import Categorical, Normal


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)  # initialize weights orthogonally
    nn.init.constant_(layer.bias, bias_const)  # initialize bias to a constant
    return layer

def mlp(in_dim: int, out_dim: int, hidden: list[int] = (64, 64), out_std: float = 1.0) -> nn.Sequential:
    layers = []
    prev_dim = in_dim
    for h in hidden:
        layers += [layer_init(nn.Linear(prev_dim, h)), nn.Tanh()]
        prev_dim = h
    layers += [layer_init(nn.Linear(prev_dim, out_dim), std=out_std)]
    return nn.Sequential(*layers)

# For discrete action spaces
class CategoricalHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden=(64, 64)):
        super().__init__()
        self.net = mlp(in_dim, out_dim, hidden, out_std=0.01)

    def dist(self, obs: torch.Tensor) -> Categorical:
        logits = self.net(obs)
        return Categorical(logits=logits)

    @staticmethod
    def log_prob(d: Categorical, action: torch.Tensor) -> torch.Tensor:
        return d.log_prob(action)

    @staticmethod
    def entropy(d: Categorical) -> torch.Tensor:
        return d.entropy()

    @staticmethod
    def mode(d: Categorical) -> torch.Tensor:
        return d.probs.argmax(dim=-1)


class ActorCritic(nn.Module):
    def __init__(self, actor: nn.Module, critic: nn.Module):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def act(self, obs: torch.Tensor):
        d = self.actor.dist(obs)
        action = d.sample()
        return action, self.actor.log_prob(d, action), self.actor.entropy(d), self.value(obs)

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor):
        d = self.actor.dist(obs)
        return self.actor.log_prob(d, action), self.actor.entropy(d), self.value(obs)

    def act_deterministic(self, obs: torch.Tensor):
        d = self.actor.dist(obs)
        return self.actor.mode(d)


def make_actor_critic(obs_space, act_space, hidden=(64, 64)) -> ActorCritic:
    obs_dim = obs_space.shape[0]

    if isinstance(act_space, gym.spaces.Discrete):
        actor = CategoricalHead(obs_dim, int(act_space.n), hidden)
    elif isinstance(act_space, gym.spaces.Box):
        raise NotImplementedError("Continuous action spaces are not implemented yet."
                                  "Create GaussianHead for continuous action spaces.")
    else:
        raise TypeError(f"Unsupported action space type: {type(act_space)}")

    critic = mlp(obs_dim, 1, hidden, out_std=1.0)

    return ActorCritic(actor, critic)
