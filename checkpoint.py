# from stable_baselines3.common.env_util import make_atari_env
# from stable_baselines3.common.vec_env import VecFrameStack
# from stable_baselines3 import A2C, DQN, PPO
# from stable_baselines3.common.evaluation import evaluate_policy
#
# env = make_atari_env('Breakout-v0', n_envs=8, seed=0)
# # Frame-stacking with 4 frames
# env = VecFrameStack(env, n_stack=8)
#
# model = DQN.load("dqn_breakout6" , env,exploration_fraction=0.05, verbose=1, learning_rate=0.01, device='cuda')
# model.learn(total_timesteps=500000)
#
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
# print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
#
# model.save("dqn_breakout7")



from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch

env = make_atari_env('Breakout-v0', n_envs=8, seed=0)
# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=8)

model = A2C.load("a2c_breakout6" , env,device='cuda')#,exploration_fraction=0.05, verbose=1, learning_rate=0.01,
model.learn(total_timesteps=20000000)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

model.save("a2c_breakout7")