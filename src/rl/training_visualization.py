# ==== TensorBoard 可视化回调：把 (2,50,50) + 已放置点 画成一张图 ====
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
import matplotlib.pyplot as plt
import numpy as np

def _render_obs_with_points(obs_2ch: np.ndarray, placed: set[tuple[int, int]]):
    """
    obs_2ch: (2,H,W) = [fire_risk, buildings]
    placed:  {(r,c), ...}
    """
    fire = np.asarray(obs_2ch[0], dtype=np.float32)          # (H,W)
    bld  = np.asarray(obs_2ch[1], dtype=np.float32)          # (H,W)

    fig, ax = plt.subplots(figsize=(5, 5))
    # 背景：火险热力图（0~1）
    ax.imshow(fire, cmap="hot", vmin=0, vmax=1, origin="upper")

    # 覆盖：建筑（白色半透明，bld>0.1 的格子）
    mask = bld > 0.1
    # 用 NaN 只画 True 区域
    overlay = np.where(mask, 1.0, np.nan)
    ax.imshow(overlay, cmap="gray", alpha=0.6, vmin=0, vmax=1, origin="upper")

    # 叠加：已放置点（高对比 + 上层显示）
    if placed and len(placed) > 0:
        # placed 通常是 {(row, col), ...} => y=row, x=col
        ys, xs = zip(*sorted(placed))

        # 1) 白色外圈（做“光晕”更显眼）
        ax.scatter(xs, ys, s=120, facecolors="none",
                edgecolors="lime", linewidths=2.5,
                zorder=6, clip_on=False)

        # 2) 亮色实心点（不被底图吃掉）
        ax.scatter(xs, ys, s=60, marker='o',
                c="cyan", alpha=0.95,
                edgecolors="white", linewidths=0.8,
                zorder=7, clip_on=False)
    else:
        # 没放点时给个提示（可删）
        ax.text(1, 1, "no sensors placed",
                color="white", fontsize=9,
                ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.2",
                        facecolor="black", alpha=0.6),
                zorder=8)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Fire (hot) + Buildings (white) + Sensors (○)", fontsize=10)
    fig.tight_layout()
    return fig

class TBVisualizationCallback(BaseCallback):
    """
    每隔 viz_freq 步，创建一个独立 eval env，用当前策略跑 1 个 episode，
    把最后的观测 + 已放置点画成图，写入 TensorBoard Images。
    make_env_fn: () -> gym.Env（返回单个 FireSensorEnv，非 VecEnv）
    """
    def __init__(self, make_env_fn, viz_freq: int = 10_000, max_steps: int = 50, tag: str = "viz", verbose: int = 0):
        super().__init__(verbose)
        self.make_env_fn = make_env_fn
        self.viz_freq = viz_freq
        self.max_steps = max_steps
        self.tag = tag

    def _on_step(self) -> bool:
        if self.num_timesteps % self.viz_freq != 0:
            return True

        env = self.make_env_fn()
        try:
            obs, info = env.reset()
            placed = set()
            for _ in range(self.max_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                # 从环境内部拿已放置点（你环境里 _placed 是 set[(r,c)])
                placed = getattr(env.unwrapped, "_placed", set())
                if done or truncated:
                    break

            fig = _render_obs_with_points(obs, placed)
            # 记录到 TensorBoard 的 Images 面板
            self.logger.record(f"images/{self.tag}", Figure(fig, close=True),
                               exclude=("stdout", "log", "json", "csv"))
        finally:
            env.close()
        return True
