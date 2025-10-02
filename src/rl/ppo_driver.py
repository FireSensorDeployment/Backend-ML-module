import numpy as np
from gymnasium import ObservationWrapper, spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.env_checker import check_env

from environment import FireSensorEnv
from environment import EnvConfig 

# (H, W) -> (1, H, W)
class AddChannelDim(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        h, w = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1, h, w), dtype=np.float32  # 变成 float32 格式，且增加通道维度
        )
    def observation(self, obs):
        if obs.dtype != np.float32:
            obs = obs.astype(np.float32, copy=False) # 强制转换为 float32
        return obs[np.newaxis, ...]  # (1,H,W) <- (H,W)

def make_env_cnn():
    def _thunk():
        env = FireSensorEnv(config=EnvConfig(
                dataset_paths=["mixed_pattern_training_set_50.npz"]
            ))
        env = AddChannelDim(env)
        # 只在开发期检查一次即可；要静音告警可用 warn=False
        check_env(env, warn=True)
        return env
    return _thunk

# 只用 VecMonitor（不要再用 Monitor 以免重复）
venv = DummyVecEnv([make_env_cnn()]) # 创建向量化环境
venv = VecMonitor(venv) # 监控向量化环境

# model 是 PPO 的实例
model = PPO(
    "CnnPolicy",   # 用 CNN 策略网络（适合图像观测）
    venv,         # 传入环境（向量化后的 FireSensorEnv）
    n_steps=2048,   # 每次收集多少步数据再更新一次网络
    batch_size=64,  # 更新时的小批量大小
    learning_rate=3e-4,  # 学习率（越大更新越快，越容易不稳定）
    gamma=0.99,  # 折扣因子，未来奖励的衰减率
    gae_lambda=0.95,  # GAE 的 λ，用来平衡 bias-variance
    clip_range=0.2,  # PPO 的剪切参数
    ent_coef=0.0,  # 熵正则化系数（控制探索度）
    verbose=1,  # 日志等级（1 打印训练过程）
    tensorboard_log="./tb_logs",             # 日志保存到这个目录，可以用 tensorboard 可视化
    policy_kwargs=dict(normalize_images=False)  # 给策略的额外参数
)

model.learn(total_timesteps=200_000) # 训练 20 万步
model.save("ppo_firesensor_cnn") # 强行停止，不会被保存哦
# 输出的结果：https://stable-baselines3.readthedocs.io/en/master/common/logger.html?utm_source=chatgpt.com