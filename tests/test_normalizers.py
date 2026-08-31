import numpy as np
from rl.normalizers import RunningMeanStd, ObsNormalizer, RewardNormalizer

def get_data(size, seed=0):
    rng = np.random.default_rng(seed=seed)
    data = rng.standard_normal(size=size)
    return data

def test_running_mean_std():
    data = get_data(size=(10000, 17)) * np.array([1.0] * 8 + [20.0] * 9) + 3
    print(data.shape)

    rms = RunningMeanStd(shape=(17,))
    for i in range(0, len(data), 137):
        rms.update(data[i:i+137])

    # running mean & var should match the true mean & var
    assert np.allclose(rms.mean, data.mean(0), atol=1e-8)
    assert np.allclose(rms.var, data.var(0), atol=1e-8)

def test_obs_normalizer():
    obs_data = get_data(size=(10000, 17)) * np.array([1.0] * 8 + [20.0] * 9) + 3

    norm = ObsNormalizer(shape=(17, ))
    out = np.concatenate([norm(obs_data[i:i+137], update=True) for i in range(0, len(obs_data), 137)], axis=0)

    # Normalized observation output is ~N(0, 1)
    assert abs(out[2000:].mean()) < 0.05 and abs (out[2000:].std() - 1.0) < 0.05
    assert abs(norm.rms.count - len(obs_data)) < 1e-3

    # update=False should not change the count
    before = norm.rms.count
    norm(obs_data[:100], update=False)
    assert norm.rms.count == before

    # clipping test
    assert np.abs(norm(np.full((1, 17), 1e6))).max() <= 10.0


def test_reward_normalizer():
    rewards = get_data(size=5000) * 50
    done = np.zeros(1)
    rn = RewardNormalizer(num_envs=1, gamma=0.99)
    scaled = [rn(np.array([r]), done)[0] for r in rewards]

    # reward normalizer brings large scale reward near unit variance
    assert 0.05 < np.std(scaled[1000:]) < 2.0, np.std(scaled[1000:])
