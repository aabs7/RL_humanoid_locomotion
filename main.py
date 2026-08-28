import gymnasium as gym


def main():
    env = gym.wrappers.vector.RecordEpisodeStatistics(gym.make_vec("LunarLander-v3", num_envs=1, render_mode="human"))
    env.reset()

    for i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("\nStep:", i)
        print(f"  obs: {obs} \n"
              f"  reward: {reward} \n"
              f"  terminated: {terminated} \n"
              f"  truncated: {truncated} \n"
              f"  info: {info}")
        if all(terminated) or all(truncated):
            env.reset()

if __name__ == "__main__":
    main()
