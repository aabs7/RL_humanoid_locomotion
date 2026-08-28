import pytest
import gymnasium as gym
import torch
from rl.buffers import RolloutBuffer

T, N = 8, 3
OBS = gym.spaces.Box(-1, 1, (4,))
ACT = gym.spaces.Discrete(2)

def make_buffer(seed: int, *, truncations: bool = False) -> RolloutBuffer:
    g = torch.Generator().manual_seed(seed)
    buf = RolloutBuffer(T, N, OBS, ACT)
    buf.rewards = torch.randn(T, N, generator=g)
    buf.values = torch.randn(T, N, generator=g)
    buf.terminated = (torch.rand(T, N, generator=g) < 0.2).float()
    if truncations:
        buf.truncated = (torch.rand(T, N, generator=g) < 0.2).float()
        buf.truncated *= 1.0 - buf.terminated  # can't be both terminated and truncated
    return buf

@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.99, 1.0])
def test_gae_lambda_one_equals_rewards_to_go_when_no_truncations(seed: int, gamma: float):
    buf1 = make_buffer(seed, truncations=False)
    buf2 = make_buffer(seed, truncations=False)
    buf1.compute_gae(next_value=torch.zeros(N), gamma=gamma, gae_lambda=1.0)
    buf2.rewards_to_go(gamma=gamma)
    torch.testing.assert_close(buf1.advantages, buf2.advantages)
    torch.testing.assert_close(buf1.returns, buf2.returns)


def test_gae_and_rtg_differ_at_truncations_by_exactly_bootstrap():
    buf = RolloutBuffer(3, 1, OBS, ACT)
    buf.rewards = torch.tensor([[1.0], [2.0], [3.0]])
    buf.values = torch.tensor([[0.0], [0.0], [7.0]])  # V(s_2) = 30.0
    buf.truncated[1] = 1.0  # say truncated at t = 1

    buf.compute_gae(next_value=torch.zeros(1), gamma=1.0, gae_lambda=1.0)
    assert buf.advantages.flatten().tolist() == [10.0, 9.0, -4.0]

    buf.rewards_to_go(gamma=1.0)
    assert buf.advantages.flatten().tolist() == [3.0, 2.0, -4.0]
    # GAE Tells S2 is still good state. The advantage value differs by exactly V(s_2) = 7.0
