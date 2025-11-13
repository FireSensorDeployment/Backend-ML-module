from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.env_checker import check_env
from environment import FireSensorEnv, EnvConfig
from training_visualization import TBVisualizationCallback

def make_env_cnn():
    def _thunk():
        env = FireSensorEnv(config=EnvConfig(
            dataset_paths=["mixed_pattern_training_set_50.npz"]
        ))
        check_env(env, warn=True)
        return env
    return _thunk

def make_eval_env():
    # 直接返回一个 FireSensorEnv 实例（不是 _thunk）
    return FireSensorEnv(config=EnvConfig(
        dataset_paths=["mixed_pattern_training_set_50.npz"],
        allow_sample_with_replacement=False
    ))

# === Create training environment ===
venv = DummyVecEnv([make_env_cnn()])
venv = VecMonitor(venv)

# === 定义模型 ===
model = PPO(
    "CnnPolicy",
    venv,
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    verbose=1,
    tensorboard_log="./tb_logs",
    policy_kwargs=dict(normalize_images=False),
)

# === 加入可视化回调 ===
viz_cb = TBVisualizationCallback(
    make_env_fn=make_eval_env,
    viz_freq=10_000,
    max_steps=50,
    tag="rollout_viz"
)

# === Start training ===
model.learn(total_timesteps=200_000, callback=viz_cb, tb_log_name="PPO_1")
model.save("ppo_firesensor_cnn")
