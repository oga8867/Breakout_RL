from stable_baselines3.common.env_util import make_atari_env
# from stable_baselines3.common.vec_env import VecFrameStack
# from stable_baselines3 import DQN
#
# # create the Breakout environment
# env = make_atari_env('Breakout-v0', n_envs=1, seed=0)
# # stack 8 consecutive frames
# env = VecFrameStack(env, n_stack=8)
#
# # load the trained model
# model = DQN.load("a2c_breakout")
#
# # evaluate the model's performance over 10 episodes
# for i in range(100000):
#     obs = env.reset()
#     done = False
#     total_reward = 0
#     while not done:
#         # render the game screen
#         env.render()
#         # get the model's action
#         action, _ = model.predict(obs, deterministic=True)
#         # execute the action on the environment
#         obs, reward, done, info = env.step(action)
#         # update the total reward
#         total_reward += reward
#     print(f"Episode {i + 1} reward: {total_reward}")
#
# # close the environment
# env.close()



import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack

# Load the saved model
model = A2C.load("a2c_breakout7")

# Create the environment
env = make_atari_env('Breakout-v0', n_envs=1, seed=0)

# Preprocess the observation space
env = VecFrameStack(env, n_stack=8)

# Play the game
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done.any():
        obs = env.reset() # Reset the environment when the episode is over
env.close()



# import gym
# from stable_baselines3 import A2C
#
# # Load the trained model
# model = A2C.load("a2c_breakout")
#
# # Create the environment
# env = gym.make('Breakout-v0')
# # Frame-stacking with 4 frames
# env = VecFrameStack(env, n_stack=8)
#
# # Play one episode
# obs = env.reset()
# done = False
# while not done:
#     action, _states = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     env.render()
#
# # Close the environment
# env.close()