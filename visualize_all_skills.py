"""
Particle 환경의 실제 렌더링 영상을 GIF로 생성
Agent들이 2D 공간에서 움직이는 모습을 시각화
"""
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# DUSDi imports
from env_helper import get_single_gym_env
from utils import make_agent, set_seed_everywhere
import utils
from omegaconf import OmegaConf

# ======================
# 설정
# ======================
DOMAIN = "particle"
NUM_STEPS = 100  # 프레임 수
SNAPSHOT_TS = 3000000
SEED = 2

# Particle 환경 설정
NUM_AGENTS = 10
SKILL_DIM = 5

# 테스트할 skill 개수 제한 (50개 전체는 시간 오래 걸림)
MAX_SKILLS_TO_TEST = 50  # 모든 agent × 모든 skill (10 × 5 = 50개)

# 모델 경로
MODEL_DIR = Path("/home/sky/문서/Github/DUSDi/models/states/particle/seed:2 particle dusdi_diayn test/2")
ACTOR_PATH = MODEL_DIR / f"actor_{SNAPSHOT_TS}.pt"

# 출력 디렉토리
OUT_DIR = Path("skill_videos_rendered")
OUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("Particle Skill Video Generator (Rendered)")
print("=" * 80)
print(f"Total possible: {NUM_AGENTS * SKILL_DIM} skills")
print(f"Testing: {MAX_SKILLS_TO_TEST} skills")
print(f"Steps per video: {NUM_STEPS}")
print(f"Output: {OUT_DIR}/")
print("=" * 80)

# ======================
# Config 로드
# ======================
from hydra import compose, initialize_config_dir

config_dir = os.path.abspath(".")

with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name="pretrain", overrides=[
        f"domain={DOMAIN}",
        f"seed={SEED}",
        "obs_type=states",
        "action_repeat=1",
        "env.particle.N=10",
        "env.particle.simplify_action_space=True",
        "env.particle.use_img=False",  # RGB 렌더링 사용
    ])

set_seed_everywhere(SEED)

# ======================
# 환경 및 Agent 생성
# ======================
print("\n[Setup] Creating environment with rendering...")
env = get_single_gym_env(cfg, rank=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dm_env import specs
obs_spec = specs.Array(shape=env.observation_space.shape, 
                       dtype=env.observation_space.dtype, 
                       name='observation')
action_spec = specs.Array(shape=env.action_space.shape,
                         dtype=env.action_space.dtype,
                         name='action')

agent = make_agent(
    obs_type='states',
    obs_spec=obs_spec,
    action_spec=action_spec,
    num_expl_steps=0,
    parent_cfg=cfg,
    cfg=cfg.agent
)

if ACTOR_PATH.exists():
    with ACTOR_PATH.open('rb') as f:
        actor_state = torch.load(f, map_location=device)
    agent.actor.load_state_dict(actor_state)
    print(f"✓ Actor loaded")
else:
    raise FileNotFoundError(f"Actor not found: {ACTOR_PATH}")

# ======================
# 스킬 리스트 생성
# ======================
skill_list = []

# 각 agent별로 각 skill 테스트
count = 0
for agent_idx in range(NUM_AGENTS):
    for skill_idx in range(SKILL_DIM):
        if count >= MAX_SKILLS_TO_TEST:
            break
        skill_vec = [0] * 10
        skill_vec[agent_idx] = skill_idx
        skill_name = f"agent{agent_idx}_skill{skill_idx}"
        skill_list.append((skill_vec, skill_name))
        count += 1
    if count >= MAX_SKILLS_TO_TEST:
        break

print(f"\nGenerating {len(skill_list)} video GIFs...")

# ======================
# 각 스킬 실행 및 렌더링
# ======================
results = []

for idx, (skill_vec, skill_name) in enumerate(tqdm(skill_list, desc="Rendering videos")):
    # 스킬 설정
    skill_vector = np.array(skill_vec, dtype=np.int64)
    meta = agent.get_meta_from_skill(skill_vector, num_envs=1)
    
    # 환경 리셋
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    # 프레임 수집
    frames = []
    rewards = []
    
    # 첫 프레임 렌더링
    try:
        frame = env.render(mode='rgb_array')
        if frame is not None:
            frames.append(frame)
    except:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
    
    # 스킬 실행
    for step in range(NUM_STEPS):
        with torch.no_grad(), utils.eval_mode(agent):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            action = agent.act(obs_tensor, meta, step, eval_mode=True)
        
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().flatten()
        else:
            action = action.flatten()
        
        step_result = env.step(action)
        
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
        
        rewards.append(reward)
        
        # 프레임 렌더링
        try:
            frame = env.render(mode='rgb_array')
            if frame is not None:
                frames.append(frame)
        except:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        if done:
            break
    
    total_reward = sum(rewards)
    
    # 결과 저장
    results.append({
        'name': skill_name,
        'skill_vector': skill_vec,
        'reward': total_reward,
        'frames': len(frames)
    })
    
    # ======================
    # GIF 저장
    # ======================
    if frames:
        # PIL Image로 변환
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        # GIF 저장
        gif_path = OUT_DIR / f"{skill_name}.gif"
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=50,  # 50ms per frame = 20 fps
            loop=0  # 무한 반복
        )
        
        # 보상 정보를 텍스트 파일로 저장
        info_path = OUT_DIR / f"{skill_name}_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Skill: {skill_name}\n")
            f.write(f"Vector: {skill_vec}\n")
            f.write(f"Total Reward: {total_reward:.3f}\n")
            f.write(f"Frames: {len(frames)}\n")

env.close()

# ======================
# 결과 요약
# ======================
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"Total GIFs created: {len(results)}")
print(f"Saved to: {OUT_DIR}/")

# 보상 순 정렬
results_sorted = sorted(results, key=lambda x: x['reward'], reverse=True)

print("\nTop 10 skills by reward:")
for i, r in enumerate(results_sorted[:min(10, len(results_sorted))]):
    print(f"  {i+1:2d}. {r['name']:20s} | Reward: {r['reward']:7.3f} | Frames: {r['frames']:3d}")

print("\nBottom 10 skills by reward:")
for i, r in enumerate(results_sorted[-min(10, len(results_sorted)):]):
    print(f"  {i+1:2d}. {r['name']:20s} | Reward: {r['reward']:7.3f} | Frames: {r['frames']:3d}")

# CSV 저장
csv_path = OUT_DIR / "skill_results.csv"
with open(csv_path, 'w') as f:
    f.write("skill_name,skill_vector,reward,frames\n")
    for r in results:
        skill_str = "_".join(map(str, r['skill_vector']))
        f.write(f"{r['name']},{skill_str},{r['reward']:.3f},{r['frames']}\n")

print(f"\n✓ Results saved to: {csv_path}")
print("=" * 80)
print(f"\n🎬 All rendered GIF animations saved to: {OUT_DIR}/")
print(f"   View example: eog {OUT_DIR}/agent0_skill1.gif")
print("=" * 80)
print("\n💡 To generate all 50 skills, change MAX_SKILLS_TO_TEST=50 in the script")
print("=" * 80)
