'''Fixed horizon rollout storage'''

import gymnasium as gym
import torch


class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_space, act_space, device: str = "cpu"):
        self.num_steps, self.num_envs, self.device = num_steps, num_envs, device
        T, N = num_steps, num_envs
        act_dtype = torch.int64 if isinstance(act_space, gym.spaces.Discrete) else torch.float32
        zeros = lambda *shape, dtype=torch.float32: torch.zeros(*shape, dtype=dtype, device=device)

        self.obs = zeros(T, N, *obs_space.shape)
        self.actions = zeros(T, N, *act_space.shape, dtype=act_dtype)
        self.logprobs = zeros(T, N)
        self.values = zeros(T, N)
        self.rewards = zeros(T, N)
        self.terminated = zeros(T, N)  # e.g., robot fell
        self.truncated = zeros(T, N)   # e.g., time limit
        self.prev_done = zeros(T, N)
        self.advantages = zeros(T, N)
        self.returns = zeros(T, N)

    def compute_gae(self, next_value: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        lastgaelam = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.num_steps)):
            # Determine the value of state s_{t+1}
            nextvalue = next_value if t == self.num_steps - 1 else self.values[t + 1]
            nonterminated = 1.0 - self.terminated[t]
            nonend = 1.0 - torch.clamp(self.terminated[t] + self.truncated[t], max=1.0)
            delta = self.rewards[t] + gamma * nextvalue * nonterminated - self.values[t]
            lastgaelam = delta + gamma * gae_lambda * nonend * lastgaelam
            self.advantages[t] = lastgaelam
        self.returns = self.advantages + self.values


    def rewards_to_go(self, gamma: float) -> None:
        last_return = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.num_steps)):
            nonend = 1.0 - torch.clamp(self.terminated[t] + self.truncated[t], max=1.0)
            last_return = self.rewards[t] + gamma * last_return * nonend
            self.returns[t] = last_return
        self.advantages[:] = self.returns - self.values

    def flatten(self) -> dict[str, torch.Tensor]:
        '''(T, N, ...) -> (V, ...), where V = T * N - len(not_valid)'''
        valid = (1.0 - self.prev_done).bool().reshape(-1) # (T * N, )
        out = {}
        for k in ('obs', 'actions', 'logprobs', 'values', 'advantages', 'returns'):
            v = getattr(self, k)
            out[k] = v.reshape(-1, *v.shape[2:])[valid]
        return out
