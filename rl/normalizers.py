'''
Normalizers for observation and rewards.
The observation space might go from -inf to +inf, which causes problems for NN. e.g., see observation space of HalfCheetah.
'''

import numpy as np

class RunningMeanStd:
    def __init__(self, shape: tuple = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total = self.count + batch_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total

        new_mean = self.mean + delta * batch_count / total
        new_var = m2 / total

        self.mean, self.var, self.count = new_mean, new_var, total


class ObsNormalizer:
    '''(x - mean) / std, clipped'''
    def __init__(self, shape: tuple, clip: float = 10.0, epsilon: float=1e-8):
        self.rms = RunningMeanStd(shape)
        self.clip, self.epsilon = clip, epsilon

    def __call__(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        if update:
            self.rms.update(obs)
        norm = (obs - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon)
        return np.clip(norm, -self.clip, self.clip)

class RewardNormalizer:
    '''Divide rewards by the std of the Discounted returns.
    No mean subtraction, because shifting rewards changes which policy is optimal.'''

    def __init__(self, num_envs: int, gamma: float, clip: float = 10.0, epsilon: float = 1e-8):
        self.rms = RunningMeanStd(())
        self.returns = np.zeros(num_envs, dtype=np.float64)
        self.gamma, self.clip, self.epsilon = gamma, clip, epsilon

    def __call__(self, rewards: np.ndarray, dones: np.ndarray, update: bool = True) -> np.ndarray:
        if update:
            self.returns = self.returns * self.gamma * (1.0 - dones) + rewards
            self.rms.update(self.returns)
        norm = rewards / np.sqrt(self.rms.var + self.epsilon)
        return np.clip(norm, -self.clip, self.clip)
