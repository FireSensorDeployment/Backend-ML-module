import numpy as np
from gymnasium import ObservationWrapper, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.env_checker import check_env
from environment import FireSensorEnv

class AddChannelDim(ObservationWrapper): # (H, W) -> (1, H, W)
    def __init__(self, env):
        super().__init__(env)
        h, w = env.observation_space.shape
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1, h, w), dtype=np.float32) # 变成 float32 格式，且增加通道维度

    def observation(self, obs):
        return obs[np.newaxis, ...]  # -> (1, H, W)


def make_env_cnn(): # 这个函数用于创建一个适合CNN输入的环境
    def _thunk():
        env = AddChannelDim(FireSensorEnv())  # 包装环境以增加通道维度
        check_env(env, warn=True)  # 检查环境是否符合Gymnasium标准
        return Monitor(env)  # 监控环境以记录统计信息
    return _thunk

venv = DummyVecEnv([make_env_cnn()])  # 创建向量化环境
venv = VecMonitor(venv)  # 监控向量化环境

model = PPO("CnnPolicy", venv, verbose=1)  # 使用PPO算法和CNN策略
model.learn(total_timesteps=200_000)  # 训练模型
