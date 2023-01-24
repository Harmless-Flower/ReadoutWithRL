from stable_baselines3 import A2C
import os
import time
from sim import PulseEnv
import gym

models_dir = f"models/{int(time.time())}_A2C"
logdir = f"logs/{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

env = PulseEnv()
env.reset()

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=logdir, device="cuda")
# Can also use CnnPolicy or MultiInputPolicy

TIMESTEPS = 10000
for i in range(1, 1000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"A2C")
    model.save(f"{models_dir}/{TIMESTEPS*i}")