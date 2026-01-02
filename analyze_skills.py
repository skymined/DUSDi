"""
스킬별 Agent 위치 변화(trajectory)를 분석하는 스크립트
Observation 데이터에서 agent 위치를 추출하여 비교
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

# DUSDi imports
from env_helper import get_single_gym_env
from utils import make_agent, set_seed_everywhere
import utils
from omegaconf import OmegaConf

# ======================
# 설정
# ======================
DOMAIN = "particle"
NUM_STEPS = 100
SNAPSHOT_TS = 3000000
SEED = 2

# 분석할 agent와 skill
ANALYZE_AGENT = 4  # Agent 0의 trajectory 분석
NUM_SKILLS = 5     # 0~4까지 5개 skill

MODEL_DIR = Path("/home/sky/문서/Github/DUSDi/models/states/particle/seed:2 particle dusdi_diayn test/2")
ACTOR_PATH = MODEL_DIR / f"actor_{SNAPSHOT_TS}.pt"

print("=" * 80)
print(f"Trajectory Analysis for Agent {ANALYZE_AGENT}")
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
    ])

set_seed_everywhere(SEED)

# ======================
# 환경 및 Agent 생성
# ======================
print("\n[Setup] Creating environment and agent...")
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
# 각 스킬 실행 및 trajectory 수집
# ======================
trajectories = {}

print(f"\nCollecting trajectories for Agent {ANALYZE_AGENT}...")

for skill_idx in tqdm(range(NUM_SKILLS), desc="Skills"):
    # Skill vector: 선택된 agent만 해당 skill, 나머지는 0
    skill_vec = [0] * 10
    skill_vec[ANALYZE_AGENT] = skill_idx
    skill_vector = np.array(skill_vec, dtype=np.int64)
    
    meta = agent.get_meta_from_skill(skill_vector, num_envs=1)
    
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    # Agent 위치 저장
    # Observation 구조: [dist(10), pos_vel(40), extra(20)]
    # Agent i의 위치: obs[10 + i*4 : 10 + i*4 + 2] (x, y)
    
    positions = []
    
    for step in range(NUM_STEPS):
        # 현재 agent 위치 추출
        agent_pos_start = 10 + ANALYZE_AGENT * 4
        agent_x = obs[agent_pos_start]
        agent_y = obs[agent_pos_start + 1]
        positions.append([agent_x, agent_y])
        
        # Action 선택 및 실행
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
        
        if done:
            break
    
    trajectories[skill_idx] = np.array(positions)

env.close()

# ======================
# Trajectory 분석 및 시각화
# ======================
print("\n[Analysis] Computing trajectory statistics...")

# 1. Trajectory 거리 계산 (총 이동 거리)
total_distances = {}
for skill_idx, traj in trajectories.items():
    distances = np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1))
    total_distances[skill_idx] = np.sum(distances)

# 2. 시작점 대비 최종 위치
final_displacements = {}
for skill_idx, traj in trajectories.items():
    displacement = np.linalg.norm(traj[-1] - traj[0])
    final_displacements[skill_idx] = displacement

# 3. Trajectory간 차이 (diversity)
trajectory_diversity = []
for i in range(NUM_SKILLS):
    for j in range(i+1, NUM_SKILLS):
        # 두 trajectory의 평균 거리
        traj1 = trajectories[i]
        traj2 = trajectories[j]
        min_len = min(len(traj1), len(traj2))
        diff = np.linalg.norm(traj1[:min_len] - traj2[:min_len], axis=1)
        trajectory_diversity.append(np.mean(diff))

avg_diversity = np.mean(trajectory_diversity)

# ======================
# 시각화
# ======================
fig = plt.figure(figsize=(16, 10))

# 1. Trajectory Plot (2D 궤적)
ax1 = plt.subplot(2, 3, 1)
colors = ['red', 'blue', 'green', 'orange', 'purple']
for skill_idx, traj in trajectories.items():
    ax1.plot(traj[:, 0], traj[:, 1], 
             color=colors[skill_idx], 
             label=f'Skill {skill_idx}',
             linewidth=2, alpha=0.7)
    ax1.scatter(traj[0, 0], traj[0, 1], color=colors[skill_idx], s=100, marker='o', edgecolors='black')
    ax1.scatter(traj[-1, 0], traj[-1, 1], color=colors[skill_idx], s=100, marker='s', edgecolors='black')
ax1.set_xlabel('X Position')
ax1.set_ylabel('Y Position')
ax1.set_title(f'Agent {ANALYZE_AGENT} Trajectories (○=start, □=end)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# 2. X Position over time
ax2 = plt.subplot(2, 3, 2)
for skill_idx, traj in trajectories.items():
    ax2.plot(traj[:, 0], color=colors[skill_idx], label=f'Skill {skill_idx}', linewidth=2)
ax2.set_xlabel('Time Step')
ax2.set_ylabel('X Position')
ax2.set_title('X Position Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Y Position over time
ax3 = plt.subplot(2, 3, 3)
for skill_idx, traj in trajectories.items():
    ax3.plot(traj[:, 1], color=colors[skill_idx], label=f'Skill {skill_idx}', linewidth=2)
ax3.set_xlabel('Time Step')
ax3.set_ylabel('Y Position')
ax3.set_title('Y Position Over Time')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Total distance traveled
ax4 = plt.subplot(2, 3, 4)
skills = list(total_distances.keys())
distances = [total_distances[s] for s in skills]
ax4.bar(skills, distances, color=colors[:len(skills)])
ax4.set_xlabel('Skill ID')
ax4.set_ylabel('Total Distance Traveled')
ax4.set_title('Movement Distance per Skill')
ax4.grid(True, alpha=0.3)

# 5. Final displacement from start
ax5 = plt.subplot(2, 3, 5)
displacements = [final_displacements[s] for s in skills]
ax5.bar(skills, displacements, color=colors[:len(skills)])
ax5.set_xlabel('Skill ID')
ax5.set_ylabel('Final Displacement')
ax5.set_title('Distance from Starting Point')
ax5.grid(True, alpha=0.3)

# 6. Summary statistics
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary_text = f"""
Agent {ANALYZE_AGENT} Trajectory Analysis

Average Diversity: {avg_diversity:.4f}
(mean distance between trajectories)

Total Distance Traveled:
"""
for skill_idx in skills:
    summary_text += f"\n  Skill {skill_idx}: {total_distances[skill_idx]:.4f}"

summary_text += f"\n\nFinal Displacement:"
for skill_idx in skills:
    summary_text += f"\n  Skill {skill_idx}: {final_displacements[skill_idx]:.4f}"

ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
         verticalalignment='center')

plt.tight_layout()
plt.savefig(f'trajectory_analysis_agent{ANALYZE_AGENT}.png', dpi=150)
print(f"\n✓ Analysis saved to: trajectory_analysis_agent{ANALYZE_AGENT}.png")

# ======================
# 결론
# ======================
print("\n" + "=" * 80)
print("Trajectory Analysis Results")
print("=" * 80)
print(f"Average trajectory diversity: {avg_diversity:.4f}")

if avg_diversity < 0.01:
    print("\n❌ VERY LOW DIVERSITY - Skills produce nearly identical trajectories")
    print("   → Skills are NOT well differentiated")
elif avg_diversity < 0.1:
    print("\n⚠️  LOW DIVERSITY - Skills produce similar trajectories")
    print("   → Some differentiation but not strong")
elif avg_diversity < 0.5:
    print("\n✓ MODERATE DIVERSITY - Skills produce different trajectories")
    print("   → Skills are reasonably differentiated")
else:
    print("\n✓✓ HIGH DIVERSITY - Skills produce very different trajectories")
    print("   → Skills are well differentiated!")

print("\n💡 Check the plot to see actual trajectory shapes!")
print("=" * 80)
