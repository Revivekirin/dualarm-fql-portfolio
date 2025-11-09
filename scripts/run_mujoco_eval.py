# scripts/run_mujoco_eval.py
import os, argparse, time
import numpy as np
import jax
import gymnasium as gym
from utils.flax_utils import restore_agent
from envs.env_utils import make_env_and_datasets

# (선택) 동영상 저장용
try:
    import imageio.v2 as imageio
except:
    imageio = None

def _obs_to_batched(ob):
    # FQLAgent.sample_actions는 (B, ...) 배치를 기대 -> B=1로 맞춤
    return jax.tree_util.tree_map(lambda x: np.asarray(x)[None, ...], ob)

def run_episode(agent, env, render=False, record_path=None, max_steps=1000, img_hw=(240, 320)):
    key = jax.random.PRNGKey(0)
    frames = []
    obs, info = env.reset()
    ep_ret, ep_len = 0.0, 0

    # pixels-top 리사이즈가 필요한 환경만 적용 (state-only면 무시됨)
    def _resize_top(ob):
        try:
            import cv2
            if isinstance(ob, dict) and "pixels" in ob and "top" in ob["pixels"]:
                img = ob["pixels"]["top"]
                H, W = img_hw
                if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[:2] != (H, W):
                    ob = {**ob}
                    ob["pixels"] = {**ob["pixels"]}
                    ob["pixels"]["top"] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        except Exception:
            pass
        return ob

    obs = _resize_top(obs)

    while True:
        batched = _obs_to_batched(obs)
        key, sub = jax.random.split(key)
        action = np.asarray(agent.sample_actions(batched, seed=sub))[0]

        # env_utils.SafeActionWrapper가 [-1,1]→스무딩/스케일링/마이크로스텝을 처리
        obs, r, terminated, truncated, info = env.step(action)
        obs = _resize_top(obs)

        ep_ret += float(r); ep_len += 1

        if render:
            frame = env.render()  # render_mode="rgb_array"로 생성했을 때 numpy(H,W,3)
            if isinstance(frame, np.ndarray) and imageio is not None:
                frames.append(frame)

        if terminated or truncated or ep_len >= max_steps:
            break

    # 동영상 저장
    if record_path and imageio is not None and len(frames):
        os.makedirs(os.path.dirname(record_path), exist_ok=True)
        fps = 30
        imageio.mimsave(record_path, frames, fps=fps)
        print(f"[video] saved: {record_path}")

    return dict(return_=ep_ret, length=ep_len)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True, help="save_dir (학습 때 만들어진 exp/... 디렉토리)")
    p.add_argument("--step", type=int, default=None, help="저장 스텝(없으면 최신)")
    p.add_argument("--env_name", default="gym-aloha")
    p.add_argument("--aloha_task", default="sim_insertion", choices=["sim_insertion","sim_transfer"])
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--render", action="store_true")
    p.add_argument("--video_out", default="videos/eval_episode_{:03d}.mp4")
    p.add_argument("--frame_stack", type=int, default=None)
    args = p.parse_args()

    # MuJoCo 렌더 백엔드: 화면 없는 서버면 EGL, 로컬 창 띄우려면 glfw
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    # env/datasets 구성 (dataset은 안 써도 agent create 시 예시 배치용으로 유용)
    env, eval_env, train_ds, val_ds = make_env_and_datasets(
        args.env_name, frame_stack=args.frame_stack, aloha_task=args.aloha_task
    )

    # 예시 배치 1개
    example = (train_ds.sample(1) if hasattr(train_ds, "sample") else train_ds)
    ex_obs, ex_act = example["observations"], example["actions"]

    # FQLAgent 클래스를 학습 때와 동일하게 import
    from agents import agents  # registry
    from agents.fql import get_config
    cfg = get_config()
    agent_cls = agents[cfg["agent_name"]]

    # 빈 에이전트 생성 후 체크포인트 로드
    agent = agent_cls.create(
        seed=0,
        ex_observations=ex_obs,
        ex_actions=ex_act,
        config=cfg,
    )
    agent = restore_agent(agent, args.ckpt_dir, args.step)
    print("[restore] loaded params from:", args.ckpt_dir, "step:", args.step)

    # 평가 루프 (eval_env 사용 권장)
    rets = []
    for i in range(args.episodes):
        video_path = args.video_out.format(i) if args.render else None
        info = run_episode(agent, eval_env, render=args.render, record_path=video_path)
        print(f"episode {i}: return={info['return_']:.3f}, length={info['length']}")
        rets.append(info["return_"])

    print(f"[summary] mean_return={np.mean(rets):.3f}  episodes={len(rets)}")

if __name__ == "__main__":
    main()
