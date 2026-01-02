"""
DUSDi에서 특정 스킬을 실행하고 시각화하는 스크립트
실제 DUSDi 코드베이스 구조에 맞춰 작성됨
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
import matplotlib.animation as animation

# DUSDi imports
from env_helper import get_single_gym_env
from utils import make_agent, set_seed_everywhere
import utils
from omegaconf import OmegaConf

# ======================
# 설정
# ======================
DOMAIN = "particle"
SKILL_VECTOR = [0, 3, 0, 0, 0]  # 스킬 벡터 (채널별 인덱스)
                                 # 예: 채널1=3, 나머지=0
NUM_STEPS = 100
SNAPSHOT_TS = 3000000
SEED = 2

# 모델 경로 (절대 경로)
MODEL_DIR = Path("/home/sky/문서/Github/DUSDi/models/states/particle/seed:2 particle dusdi_diayn test/2")
ACTOR_PATH = MODEL_DIR / f"actor_{SNAPSHOT_TS}.pt"

# 출력 디렉토리
OUT_DIR = Path("skill_viz")
OUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print(f"Testing Skill: {SKILL_VECTOR}")
print(f"Actor path: {ACTOR_PATH}")
print("=" * 80)

# ======================
# Config 로드 (Hydra 없이)
# ======================
# pretrain.yaml 로드
cfg = OmegaConf.load("pretrain.yaml")
cfg.domain = DOMAIN
cfg.seed = SEED
cfg.obs_type = "states"
cfg.action_repeat = 1

# Agent config 로드
agent_cfg = OmegaConf.load("agent/dusdi_diayn.yaml")
cfg.agent = agent_cfg

# 환경 설정 로드
env_cfg = OmegaConf.load("env/env_config.yaml")
cfg.env = env_cfg

# Particle 환경 설정
cfg.env.particle.N = 10
cfg.env.particle.simplify_action_space = True

set_seed_everywhere(SEED)

# ======================
# 환경 생성
# ======================
print("\n[1/4] Creating environment...")
env = get_single_gym_env(cfg, rank=0)
print(f"✓ Environment created: {type(env)}")
print(f"  Observation space: {env.observation_space}")
print(f"  Action space: {env.action_space}")

# ======================
# Agent 생성
# ======================
print("\n[2/4] Creating agent...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dm_env wrapper의 spec 가져오기
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
    parent_cfg=cfg,  # parent_cfg (5th param)
    cfg=cfg.agent    # cfg (6th param) - agent config
)
print(f"✓ Agent created: {type(agent).__name__}")
print(f"  Skill channels: {agent.diayn_skill_channel if hasattr(agent, 'diayn_skill_channel') else 'N/A'}")

# ======================
# Actor 로드
# ======================
print(f"\n[3/4] Loading actor from {ACTOR_PATH}...")
if ACTOR_PATH.exists():
    with ACTOR_PATH.open('rb') as f:
        actor_state = torch.load(f, map_location=device)
    agent.actor.load_state_dict(actor_state)
    print("✓ Actor loaded successfully")
else:
    raise FileNotFoundError(f"Actor not found: {ACTOR_PATH}")

agent.eval()

# ======================
# 스킬 실행 및 데이터 수집
# ======================
print(f"\n[4/4] Running skill for {NUM_STEPS} steps...")

skill_vector = np.array(SKILL_VECTOR, dtype=np.int64)
meta = agent.get_meta_from_skill(skill_vector, num_envs=1)

# 환경 리셋
obs = env.reset()
if isinstance(obs, tuple):  # Gymnasium API
    obs = obs[0]

# 데이터 수집
observations = [obs]
actions_taken = []
rewards = []

for step in range(NUM_STEPS):
    # Action 선택
    with torch.no_grad(), utils.eval_mode(agent):
        # dm_env 형식으로 변환
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        action = agent.act(obs_tensor, meta, step, eval_mode=True)
    
    action = action.cpu().numpy().flatten()
    actions_taken.append(action.copy())
    
    # Step
    step_result = env.step(action)
    
    # Gymnasium vs Gym API 처리
    if len(step_result) == 5:  # Gymnasium
        obs, reward, terminated, truncated, info = step_result
        done = terminated or truncated
    else:  # Gym
        obs, reward, done, info = step_result
    
    observations.append(obs)
    rewards.append(reward)
    
    if step % 20 == 0:
        print(f"  Step {step:3d}: Reward = {sum(rewards):.3f}")
    
    if done:
        print(f"  Episode ended at step {step}")
        break

print(f"\n✓ Rollout complete!")
print(f"  Total steps: {len(observations) - 1}")
print(f"  Total reward: {sum(rewards):.3f}")

# ======================
# 시각화 1: Trajectory Plot
# ======================
print("\n[Visualization 1] Creating trajectory plot...")
observations = np.array(observations)

# Particle 환경의 observation 구조:
# [agent1_dist, agent2_dist, ..., agent10_dist, agent1_x, agent1_y, ...]
# 처음 10개는 각 agent와 landmark 사이의 거리

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 거리 변화
if observations.shape[1] >= 10:
    axes[0].plot(observations[:, :10])  # 첫 10개 distance
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Distance to Landmarks')
    axes[0].set_title(f'Skill {SKILL_VECTOR}: Agent-Landmark Distances')
    axes[0].legend([f'Agent {i}' for i in range(10)], loc='best', fontsize=8)
    axes[0].grid(True)

# 보상 누적
axes[1].plot(np.cumsum(rewards))
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Cumulative Reward')
axes[1].set_title(f'Cumulative Reward (Total: {sum(rewards):.2f})')
axes[1].grid(True)

plt.tight_layout()
plot_path = OUT_DIR / f"skill_{'_'.join(map(str, SKILL_VECTOR))}_trajectory.png"
plt.savefig(plot_path, dpi=150)
print(f"✓ Trajectory plot saved: {plot_path}")

# ======================
# 시각화 2: Action 분석
# ======================
print("\n[Visualization 2] Creating action analysis...")
actions_taken = np.array(actions_taken)

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
for i in range(min(actions_taken.shape[1], 10)):  # 최대 10개 dimension
    ax.plot(actions_taken[:, i], label=f'Action {i}', alpha=0.7)
ax.set_xlabel('Step')
ax.set_ylabel('Action Value')
ax.set_title(f'Skill {SKILL_VECTOR}: Action Sequence')
ax.legend(loc='best', fontsize=8)
ax.grid(True)

action_path = OUT_DIR / f"skill_{'_'.join(map(str, SKILL_VECTOR))}_actions.png"
plt.savefig(action_path, dpi=150)
print(f"✓ Action plot saved: {action_path}")

# ======================
# 결과 요약
# ======================
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"Skill Vector: {SKILL_VECTOR}")
print(f"Steps Executed: {len(observations) - 1}")
print(f"Total Reward: {sum(rewards):.3f}")
print(f"Average Reward per Step: {np.mean(rewards):.3f}")
print(f"\nVisualization files saved to: {OUT_DIR}/")
print(f"  - {plot_path.name}")
print(f"  - {action_path.name}")
print("=" * 80)

env.close()
