"""
스크립트 1: Agent 0의 실제 Action 분석
데이터 출처: 환경을 직접 실행하여 action 수집 (새로 생성)
목적: Skill별로 실제 다른 action을 내는지 확인
"""
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from env_helper import get_single_gym_env
from utils import make_agent, set_seed_everywhere
import utils
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

# ======================
# 설정
# ======================
DOMAIN = "particle"
NUM_STEPS = 100
SNAPSHOT_TS = 3000000
SEED = 2
NUM_TRIALS = 5  # 각 skill을 5번씩 실행

TARGET_AGENT = 0
NUM_SKILLS = 5

MODEL_DIR = Path("/home/sky/문서/Github/DUSDi/models/states/particle/seed:2 particle dusdi_diayn test/2")
ACTOR_PATH = MODEL_DIR / f"actor_{SNAPSHOT_TS}.pt"

print("=" * 80)
print("스크립트 1: Agent 0 Action 분석")
print("데이터: 환경 직접 실행 (새로 생성)")
print("=" * 80)

# Config 로드
config_dir = os.path.abspath(".")
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name="pretrain", overrides=[
        f"domain={DOMAIN}",
        f"seed={SEED}",
        "obs_type=states",
        "action_repeat=1",
        "env.particle.N=10",
        "env.particle.simplify_action_space=True",
    ])

set_seed_everywhere(SEED)

# 환경 및 Agent 생성
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
# 각 Skill별 Action 수집
# ======================
print(f"\n[수집] {NUM_SKILLS} skills × {NUM_TRIALS} trials × {NUM_STEPS} steps")

skill_actions = {skill_idx: [] for skill_idx in range(NUM_SKILLS)}

for skill_idx in tqdm(range(NUM_SKILLS), desc="Skills"):
    for trial in range(NUM_TRIALS):
        skill_vec = [0] * 10
        skill_vec[TARGET_AGENT] = skill_idx
        skill_vector = np.array(skill_vec, dtype=np.int64)
        
        meta = agent.get_meta_from_skill(skill_vector, num_envs=1)
        
        # 같은 seed로 시작
        np.random.seed(SEED + trial)
        torch.manual_seed(SEED + trial)
        
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        trial_actions = []
        
        for step in range(NUM_STEPS):
            with torch.no_grad(), utils.eval_mode(agent):
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                action = agent.act(obs_tensor, meta, step, eval_mode=True)
            
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy().flatten()
            else:
                action = action.flatten()
            
            # Agent 0의 action만 추출 (첫 2D: x, y velocity)
            agent0_action = action[:2]
            trial_actions.append(agent0_action.copy())
            
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
            else:
                obs, reward, done, info = step_result
        
        skill_actions[skill_idx].append(np.array(trial_actions))

env.close()

# ======================
# 분석
# ======================
print("\n[분석] Action 통계")

# Skill별 평균 action
skill_means = {}
skill_stds = {}

for skill_idx in range(NUM_SKILLS):
    all_actions = np.vstack(skill_actions[skill_idx])  # (trials*steps, 2)
    skill_means[skill_idx] = np.mean(all_actions, axis=0)
    skill_stds[skill_idx] = np.std(all_actions, axis=0)
    
    print(f"\nSkill {skill_idx}:")
    print(f"  평균 action: x={skill_means[skill_idx][0]:.3f}, y={skill_means[skill_idx][1]:.3f}")
    print(f"  표준편차:   x={skill_stds[skill_idx][0]:.3f}, y={skill_stds[skill_idx][1]:.3f}")

# Skill 간 차이 계산
print("\n[Skill 간 Action 차이]")
for i in range(NUM_SKILLS):
    for j in range(i+1, NUM_SKILLS):
        diff = np.linalg.norm(skill_means[i] - skill_means[j])
        print(f"  Skill {i} vs {j}: {diff:.4f}")

# Within-skill variance (같은 skill 반복 시 차이)
print("\n[Within-Skill Variance] (같은 skill 재현성)")
for skill_idx in range(NUM_SKILLS):
    actions_list = skill_actions[skill_idx]  # List of (100, 2) arrays
    
    # Trial 간 차이
    trial_diffs = []
    for i in range(len(actions_list)):
        for j in range(i+1, len(actions_list)):
            diff = np.mean(np.linalg.norm(actions_list[i] - actions_list[j], axis=1))
            trial_diffs.append(diff)
    
    avg_within_diff = np.mean(trial_diffs)
    print(f"  Skill {skill_idx}: {avg_within_diff:.4f} (낮을수록 재현성 좋음)")

# ======================
# 시각화
# ======================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Action space plot (평균 action의 2D 분포)
ax = axes[0, 0]
for skill_idx in range(NUM_SKILLS):
    ax.scatter(skill_means[skill_idx][0], skill_means[skill_idx][1], 
               s=200, label=f'Skill {skill_idx}', alpha=0.7)
    ax.arrow(0, 0, skill_means[skill_idx][0]*0.5, skill_means[skill_idx][1]*0.5,
             head_width=0.02, alpha=0.5)
ax.set_xlabel('Action X')
ax.set_ylabel('Action Y')
ax.set_title('Average Action per Skill')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)

# 2-6. 각 Skill의 action trajectory
for skill_idx in range(NUM_SKILLS):
    ax = axes.flatten()[skill_idx + 1]
    
    # 모든 trial의 action trajectory
    for trial_idx in range(NUM_TRIALS):
        actions = skill_actions[skill_idx][trial_idx]
        ax.plot(actions[:, 0], actions[:, 1], alpha=0.5)
    
    ax.set_xlabel('Action X')
    ax.set_ylabel('Action Y')
    ax.set_title(f'Skill {skill_idx} Actions (all trials)')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig('analysis_1_action_distribution.png', dpi=150)
print(f"\n✓ 시각화 저장: analysis_1_action_distribution.png")

# ======================
# 결론
# ======================
print("\n" + "=" * 80)
print("결론")
print("=" * 80)

# Between-skill difference
between_diffs = []
for i in range(NUM_SKILLS):
    for j in range(i+1, NUM_SKILLS):
        diff = np.linalg.norm(skill_means[i] - skill_means[j])
        between_diffs.append(diff)

avg_between = np.mean(between_diffs)
print(f"평균 skill 간 차이: {avg_between:.4f}")

# Within-skill variance
within_vars = []
for skill_idx in range(NUM_SKILLS):
    actions_list = skill_actions[skill_idx]
    trial_diffs = []
    for i in range(len(actions_list)):
        for j in range(i+1, len(actions_list)):
            diff = np.mean(np.linalg.norm(actions_list[i] - actions_list[j], axis=1))
            trial_diffs.append(diff)
    within_vars.append(np.mean(trial_diffs))

avg_within = np.mean(within_vars)
print(f"평균 재현성 오차: {avg_within:.4f}")

ratio = avg_between / (avg_within + 1e-6)
print(f"\n비율 (Skill 간 / 재현성): {ratio:.2f}x")

if ratio > 5:
    print("✓✓ Skills가 명확히 구별되는 action을 생성")
elif ratio > 2:
    print("✓ Skills가 어느 정도 구별됨")
else:
    print("⚠️  Skills 간 action 차이가 재현성 오차와 비슷함")
    print("   → Skills이 의미있게 다르지 않을 수 있음")

print("=" * 80)
