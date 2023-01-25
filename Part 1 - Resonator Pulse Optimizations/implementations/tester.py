from sim import PulseEnv
from stable_baselines3 import PPO, A2C
from sb3_contrib import TRPO, RecurrentPPO
import numpy as np

env = PulseEnv()

models_dir = "models"
model_path = f"{models_dir}/1674586000_A2C/60000.zip"
model = A2C.load(model_path, env=env)
obs = env.reset()
action_, _ = model.predict(obs)

env.grapher(action_)
