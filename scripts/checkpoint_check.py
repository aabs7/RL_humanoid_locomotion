import random
from pathlib import Path
import gymnasium as gym, numpy as np, torch

from rl.checkpoint import save, load, restore, config_of
from rl.config import Config
from rl.policies import make_actor_critic


run = Path("runs/_ck")
env = gym.make("LunarLander-v3")
config = Config()

torch.manual_seed(config.seed)
agent = make_actor_critic(env.observation_space, env.action_space)
opt = torch.optim.Adam(agent.parameters(), lr=config.ppo.lr)


random_obs = torch.randn(env.observation_space.shape[0], 8)
agent.value(random_obs).pow(2).mean().backward()
opt.step()

assert opt.state_dict()["state"], "adam state empty"

path = save(run / "checkpoints" / "latest.pt", agent=agent, optimizer=opt, cfg=config, iteration=7, global_step=7 * 2048)
before = [random.random(), float(np.random.rand()), torch.rand(1).item()]

ck = load(path)
it, gs = restore(ck, agent, opt)
after = [random.random(), float(np.random.rand()), torch.rand(1).item()]

assert (it, gs) == (7, 7 * 2048)
assert config_of(ck)["env_id"] == "LunarLander-v3"
assert before == after, "RNG state not restored correctly"
assert not list((run/"checkpoints").glob("*.tmp")), "temp file not cleaned"
