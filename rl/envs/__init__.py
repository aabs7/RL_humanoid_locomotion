
from pathlib import Path
import gymnasium as gym

def _single(env_id: str, idx: int, capture_video: bool, video_dir, **kwargs):
    '''returns a thunk that creates a single environment'''
    def thunk():
        record = capture_video and idx == 0
        env = gym.make(env_id, render_mode="rgb_array" if record else None, **kwargs)
        if record:
            env = gym.wrappers.RecordVideo(
                env, str(video_dir), name_prefix="train",
                episode_trigger=lambda ep: ep % 50 == 0
            )
        return env
    return thunk

def make_envs(env_id: str,
              num_envs: int = 1,
              seed: int = 0,
              capture_video: bool = False,
              video_dir = None,
              async_envs: bool = False,
              **kwargs) -> gym.vector.VectorEnv:
    if capture_video and video_dir is None:
        raise ValueError("video_dir must be specified if capture_video is True")

    thunks = [_single(env_id, i, capture_video, video_dir, **kwargs) for i in range(num_envs)]
    cls = gym.vector.AsyncVectorEnv if async_envs else gym.vector.SyncVectorEnv
    envs = cls(thunks, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)

    envs = gym.wrappers.RecordEpisodeStatistics(envs)  # records episode statistics in info dict
    envs.action_space.seed(seed)
    return envs

def make_eval_env(env_id: str, seed: int=0, render_mode: str | None = None, **kwargs) -> gym.vector.VectorEnv:
    env = gym.vector.SyncVectorEnv(
        [lambda: gym.make(env_id, render_mode=render_mode, **kwargs)],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
    )
    envs = gym.wrappers.RecordEpisodeStatistics(env)
    envs.action_space.seed(seed + 10_000)
    return envs
