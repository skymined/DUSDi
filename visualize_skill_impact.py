"""
Agent 0의 모든 스킬 변화 시 다른 agent들의 영향 분석
히트맵과 trajectory 변화량 시각화
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
from matplotlib.patches import Rectangle
from tqdm import tqdm

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

TARGET_AGENT = 0  # 이 agent의 skill을 변경
NUM_SKILLS = 5    # 0~4

MODEL_DIR = Path("/home/sky/문서/Github/DUSDi/models/states/particle/seed:2 particle dusdi_diayn test/2")
ACTOR_PATH = MODEL_DIR / f"actor_{SNAPSHOT_TS}.pt"

print("=" * 80)
print(f"Analyzing Agent {TARGET_AGENT}'s Impact on Other Agents")
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
# 각 스킬별 모든 agent trajectory 수집
# ======================
def run_episode(skill_value):
    """한 에피소드 실행하고 모든 agent의 trajectory 반환"""
    skill_vec = [0] * 10
    skill_vec[TARGET_AGENT] = skill_value
    skill_vector = np.array(skill_vec, dtype=np.int64)
    
    meta = agent.get_meta_from_skill(skill_vector, num_envs=1)
    
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    all_positions = [[] for _ in range(10)]
    
    for step in range(NUM_STEPS):
        for agent_idx in range(10):
            pos_start = 10 + agent_idx * 4
            x, y = obs[pos_start], obs[pos_start + 1]
            all_positions[agent_idx].append([x, y])
        
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
        else:
            obs, reward, done, info = step_result
    
    return [np.array(pos) for pos in all_positions]

print(f"\n[Collecting] Running {NUM_SKILLS} skills...")
all_trajectories = {}
for skill_idx in tqdm(range(NUM_SKILLS), desc="Skills"):
    all_trajectories[skill_idx] = run_episode(skill_idx)

env.close()

# ======================
# 분석: 스킬 간 각 agent의 변화량
# ======================
print("\n[Analysis] Computing trajectory differences...")

# Baseline: Skill 0
baseline_trajs = all_trajectories[0]

# 각 skill에서 각 agent의 변화량 계산
# change_matrix[skill][agent] = 변화량
change_matrix = np.zeros((NUM_SKILLS, 10))

for skill_idx in range(NUM_SKILLS):
    for agent_idx in range(10):
        baseline_traj = baseline_trajs[agent_idx]
        current_traj = all_trajectories[skill_idx][agent_idx]
        
        # 두 trajectory 간 평균 거리
        diff = np.linalg.norm(baseline_traj - current_traj, axis=1)
        change_matrix[skill_idx, agent_idx] = np.mean(diff)

# ======================
# 시각화
# ======================
fig = plt.figure(figsize=(20, 12))

# 1. Heatmap: Skill × Agent 변화량
ax1 = plt.subplot(2, 3, 1)
im = ax1.imshow(change_matrix, cmap='YlOrRd', aspect='auto')
ax1.set_xticks(range(10))
ax1.set_yticks(range(NUM_SKILLS))
ax1.set_xticklabels([f'A{i}' for i in range(10)])
ax1.set_yticklabels([f'Skill {i}' for i in range(NUM_SKILLS)])
ax1.set_xlabel('Agent ID')
ax1.set_ylabel(f'Agent {TARGET_AGENT} Skill')
ax1.set_title(f'Agent {TARGET_AGENT} Skill Impact on All Agents')

# Annotate values
for i in range(NUM_SKILLS):
    for j in range(10):
        text = ax1.text(j, i, f'{change_matrix[i, j]:.3f}',
                       ha="center", va="center", color="black", fontsize=8)

# Colorbar
cbar = plt.colorbar(im, ax=ax1)
cbar.set_label('Trajectory Change')

# Target agent 강조
rect = Rectangle((TARGET_AGENT-0.5, -0.5), 1, NUM_SKILLS, 
                 fill=False, edgecolor='blue', linewidth=3)
ax1.add_patch(rect)

# 2. 각 skill별 평균 변화량
ax2 = plt.subplot(2, 3, 2)
target_changes = change_matrix[:, TARGET_AGENT]  # Target agent 변화
other_changes = np.mean([change_matrix[:, i] for i in range(10) if i != TARGET_AGENT], axis=0)

x = np.arange(NUM_SKILLS)
width = 0.35
ax2.bar(x - width/2, target_changes, width, label=f'Agent {TARGET_AGENT} (target)', color='red', alpha=0.7)
ax2.bar(x + width/2, other_changes, width, label='Other agents (avg)', color='gray', alpha=0.7)
ax2.set_xlabel(f'Agent {TARGET_AGENT} Skill')
ax2.set_ylabel('Average Trajectory Change')
ax2.set_title('Target vs Others Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels([f'S{i}' for i in range(NUM_SKILLS)])
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Agent별 총 영향 (모든 skill 평균)
ax3 = plt.subplot(2, 3, 3)
avg_per_agent = np.mean(change_matrix, axis=0)
colors = ['red' if i == TARGET_AGENT else 'lightblue' for i in range(10)]
ax3.bar(range(10), avg_per_agent, color=colors)
ax3.set_xlabel('Agent ID')
ax3.set_ylabel('Average Change Across All Skills')
ax3.set_title('Which Agents are Affected Most?')
ax3.grid(True, alpha=0.3)

# 4. 모든 agent trajectories (Skill 0 vs Skill 2 비교)
ax4 = plt.subplot(2, 3, 4)
compare_skill = 2
for agent_idx in range(10):
    traj_0 = all_trajectories[0][agent_idx]
    traj_c = all_trajectories[compare_skill][agent_idx]
    
    if agent_idx == TARGET_AGENT:
        ax4.plot(traj_0[:, 0], traj_0[:, 1], 'r-', linewidth=3, alpha=0.5)
        ax4.plot(traj_c[:, 0], traj_c[:, 1], 'b-', linewidth=3, alpha=0.5,
                label=f'Agent {TARGET_AGENT}')
    else:
        ax4.plot(traj_0[:, 0], traj_0[:, 1], 'lightgray', linewidth=1, alpha=0.3)
        ax4.plot(traj_c[:, 0], traj_c[:, 1], 'gray', linewidth=1, alpha=0.3)

ax4.set_title(f'All Trajectories: Skill 0 (red) vs Skill {compare_skill} (blue)')
ax4.set_xlabel('X Position')
ax4.set_ylabel('Y Position')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.axis('equal')

# 5. Independence ratio per skill
ax5 = plt.subplot(2, 3, 5)
independence_ratios = []
for skill_idx in range(NUM_SKILLS):
    target_change = change_matrix[skill_idx, TARGET_AGENT]
    other_avg = np.mean([change_matrix[skill_idx, i] for i in range(10) if i != TARGET_AGENT])
    ratio = target_change / (other_avg + 1e-6)
    independence_ratios.append(ratio)

colors_ratio = ['green' if r > 2.0 else 'orange' if r > 1.5 else 'red' for r in independence_ratios]
ax5.bar(range(NUM_SKILLS), independence_ratios, color=colors_ratio)
ax5.axhline(y=2.0, color='green', linestyle='--', label='Good (2x)', alpha=0.5)
ax5.axhline(y=1.5, color='orange', linestyle='--', label='Moderate (1.5x)', alpha=0.5)
ax5.set_xlabel(f'Agent {TARGET_AGENT} Skill')
ax5.set_ylabel('Independence Ratio')
ax5.set_title('Target Change / Others Change Ratio')
ax5.set_xticks(range(NUM_SKILLS))
ax5.set_xticklabels([f'S{i}' for i in range(NUM_SKILLS)])
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Summary statistics
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

avg_target_change = np.mean(target_changes[1:])  # Skill 0 제외
avg_other_change = np.mean(other_changes[1:])
overall_ratio = avg_target_change / (avg_other_change + 1e-6)

summary = f"""
Agent {TARGET_AGENT} Skill Impact Summary

Total Skills Analyzed: {NUM_SKILLS}
Target Agent: {TARGET_AGENT}

Average Changes (Skills 1-4):
  Agent {TARGET_AGENT}: {avg_target_change:.4f}
  Other agents: {avg_other_change:.4f}
  Ratio: {overall_ratio:.2f}x

Independence per Skill:
"""

for i, ratio in enumerate(independence_ratios):
    symbol = "✓" if ratio > 2.0 else "~" if ratio > 1.5 else "✗"
    summary += f"\n  Skill {i}: {ratio:.2f}x {symbol}"

summary += "\n\nInterpretation:"
if overall_ratio > 2.5:
    summary += "\n✓✓ Excellent independence"
elif overall_ratio > 1.8:
    summary += "\n✓ Good independence"
elif overall_ratio > 1.3:
    summary += "\n~ Moderate independence"
else:
    summary += "\n✗ Poor independence"

ax6.text(0.1, 0.5, summary, fontsize=10, family='monospace',
         verticalalignment='center')

plt.tight_layout()
plt.savefig(f'skill_impact_analysis_agent{TARGET_AGENT}.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Analysis saved to: skill_impact_analysis_agent{TARGET_AGENT}.png")

# ======================
# 상세 데이터 출력
# ======================
print("\n" + "=" * 80)
print("Detailed Change Matrix")
print("=" * 80)
print("Rows = Skills, Columns = Agents")
print("Values = Average trajectory change from Skill 0")
print()
print("       ", end="")
for i in range(10):
    print(f"  A{i}  ", end="")
print()
print("-" * 80)
for skill_idx in range(NUM_SKILLS):
    print(f"Skill {skill_idx}", end="")
    for agent_idx in range(10):
        value = change_matrix[skill_idx, agent_idx]
        if agent_idx == TARGET_AGENT:
            print(f" *{value:.3f}*", end="")
        else:
            print(f"  {value:.3f} ", end="")
    print()
print("\n* = target agent")
print("=" * 80)

print("\n✓ Open the image to see visual analysis!")
print(f"   eog skill_impact_analysis_agent{TARGET_AGENT}.png")
print("=" * 80)
