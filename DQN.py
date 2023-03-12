import gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
# create Atari Breakout environment
env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=16, seed=0)

# create the DQN agent
model = DQN('CnnPolicy', env, learning_rate=1e-4, buffer_size=100000, batch_size=32, learning_starts=10000,
            exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.01, train_freq=4,
            target_update_interval=10000, gamma=0.99, verbose=1,device='cuda')

# train the agent
model.learn(total_timesteps=1000000)

# save the model
model.save("dqn_CNN_breakout")

# evaluate the trained agent
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f}")