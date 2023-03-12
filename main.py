from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch

env = make_atari_env('Breakout-v0', n_envs=8, seed=0)
# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=8)

model = A2C('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=500000)

# model = DQN('MlpPolicy', env,exploration_fraction=0.05, verbose=1, learning_rate=0.0001, device='cuda')#,batch_size=256) #buffer_size=100000,exploration_fraction=0.2, verbose=1, learning_rate=0.001)
# model.learn(total_timesteps=100000)

# model = PPO('MlpPolicy', env,exploration_fraction=0.05, verbose=1, learning_rate=0.0001, device='cuda' ,n_steps = 8) #buffer_size=100000,exploration_fraction=0.2, verbose=1, learning_rate=0.001)
# model.learn(total_timesteps=1000000)

obs = env.reset()
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=30)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

model.save("a2c_breakout")

# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()