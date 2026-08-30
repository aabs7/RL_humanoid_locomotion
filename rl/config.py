from dataclasses import dataclass, field

@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coeff: float = 0.5  # scale critic loss
    ent_coeff: float = 0.0 # scale entropy loss
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    num_minibatches: int = 32
    target_kl: float | None = 0.02
    norm_adv: bool = True
    anneal_lr: bool = True

@dataclass
class Config:
    env_id: str = "LunarLander-v3"
    num_envs: int = 1
    num_steps: int = 2048
    total_steps: int = 1_000_000
    seed: int = 42
    tag: str = ""
    obs_norm: bool = True  # not implemented yet
    async_envs: bool = False
    save_every: int = 10
    eval_every: int = 10
    capture_video: bool = False
    ppo: PPOConfig = field(default_factory=PPOConfig)

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps
    @property
    def num_iterations(self):
        return max(1, self.total_steps // self.batch_size)
