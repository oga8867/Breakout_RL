from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch

env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=16, seed=0)
model = DQN.load('dqn_CNN_breakout2', env, learning_rate=0.00025, buffer_size=100000, batch_size=32, learning_starts=10000,
            exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.02, train_freq=4,
            target_update_interval=1000, gamma=0.99, verbose=1,device='cuda')
#2 Mean reward: 18.13 +/- 8.02
#3 Mean reward: 20.39 +/- 5.88
model.learn(total_timesteps=1000000)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

model.save("dqn_CNN_breakout3")