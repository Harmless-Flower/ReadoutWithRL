from stable_baselines3.common.env_checker import check_env
from sim import PulseEnv

env = PulseEnv()
check_env(env)