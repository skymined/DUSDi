"""
한 스킬을 변경할 때 모든 10개 agent의 움직임을 비교
특정 agent만 움직이고 나머지는 고정되는지 확인
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

# ======================
# 설정
# ======================
DOMAIN = "particle"
NUM_STEPS = 100
SNAPSHOT_TS = 3000000
SEED = 2

# 비교할 설정
TEST_AGENT = 0      # 이 agent의 skill만 변경
SKILL_A = 0         # Baseline skill
SKILL_B = 2         # 비교할 skill

MODEL_DIR = Path("/home/sky/문서/Github/DUSDi/models/states/particle/seed:2 particle dusdi_diayn test/2")
ACTOR_PATH = MODEL_DIR / f"actor_{SNAPSHOT_TS}.pt"

print("=" * 80)
print(f"Comparing: Agent {TEST_AGENT} with Skill {SKILL_A} vs Skill {SKILL_B}")
print("Checking if other agents remain same")
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
# 두 가지 경우 실행
# ======================
def run_episode(skill_value):
    """한 에피소드 실행하고 모든 agent의 trajectory 반환"""
    skill_vec = [0] * 10
    skill_vec[TEST_AGENT] = skill_value
    skill_vector = np.array(skill_vec, dtype=np.int64)
    
    meta = agent.get_meta_from_skill(skill_vector, num_envs=1)
    
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    # 모든 agent의 위치 저장 (10 agents × steps × 2(x,y))
    all_positions = [[] for _ in range(10)]
    
    for step in range(NUM_STEPS):
        # 각 agent의 위치 추출
        for agent_idx in range(10):
            pos_start = 10 + agent_idx * 4
            x, y = obs[pos_start], obs[pos_start + 1]
            all_positions[agent_idx].append([x, y])
        
        # Action
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
    
    return [np.array(pos) for pos in all_positions]

print(f"\n[1/2] Running with Agent {TEST_AGENT} = Skill {SKILL_A}...")
trajectories_A = run_episode(SKILL_A)

print(f"[2/2] Running with Agent {TEST_AGENT} = Skill {SKILL_B}...")
trajectories_B = run_episode(SKILL_B)

env.close()

# ======================
# 분석: 각 agent의 변화량
# ======================
print("\n[Analysis] Computing trajectory differences...")

differences = []
for agent_idx in range(10):
    traj_A = trajectories_A[agent_idx]
    traj_B = trajectories_B[agent_idx]
    
    # 두 trajectory 사이의 평균 거리
    min_len = min(len(traj_A), len(traj_B))
    diff = np.linalg.norm(traj_A[:min_len] - traj_B[:min_len], axis=1)
    avg_diff = np.mean(diff)
    
    differences.append(avg_diff)
    
    if agent_idx == TEST_AGENT:
        print(f"  Agent {agent_idx} (CHANGED): avg diff = {avg_diff:.4f}")
    else:
        print(f"  Agent {agent_idx} (should be same): avg diff = {avg_diff:.4f}")

# ======================
# 시각화
# ======================
fig = plt.figure(figsize=(18, 12))

# 1. 모든 agent의 trajectory (Skill A)
ax1 = plt.subplot(2, 3, 1)
for agent_idx in range(10):
    traj = trajectories_A[agent_idx]
    color = 'red' if agent_idx == TEST_AGENT else 'gray'
    linewidth = 3 if agent_idx == TEST_AGENT else 1
    alpha = 1.0 if agent_idx == TEST_AGENT else 0.3
    ax1.plot(traj[:, 0], traj[:, 1], 
             color=color, linewidth=linewidth, alpha=alpha,
             label=f'Agent {agent_idx}' if agent_idx == TEST_AGENT else None)
ax1.set_title(f'All Agents (Agent {TEST_AGENT} = Skill {SKILL_A})')
ax1.set_xlabel('X Position')
ax1.set_ylabel('Y Position')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# 2. 모든 agent의 trajectory (Skill B)
ax2 = plt.subplot(2, 3, 2)
for agent_idx in range(10):
    traj = trajectories_B[agent_idx]
    color = 'blue' if agent_idx == TEST_AGENT else 'gray'
    linewidth = 3 if agent_idx == TEST_AGENT else 1
    alpha = 1.0 if agent_idx == TEST_AGENT else 0.3
    ax2.plot(traj[:, 0], traj[:, 1], 
             color=color, linewidth=linewidth, alpha=alpha,
             label=f'Agent {agent_idx}' if agent_idx == TEST_AGENT else None)
ax2.set_title(f'All Agents (Agent {TEST_AGENT} = Skill {SKILL_B})')
ax2.set_xlabel('X Position')
ax2.set_ylabel('Y Position')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

# 3. Agent 0의 비교 (빨강 vs 파랑)
ax3 = plt.subplot(2, 3, 3)
ax3.plot(trajectories_A[TEST_AGENT][:, 0], trajectories_A[TEST_AGENT][:, 1], 
         'r-', linewidth=3, label=f'Skill {SKILL_A}', alpha=0.7)
ax3.plot(trajectories_B[TEST_AGENT][:, 0], trajectories_B[TEST_AGENT][:, 1], 
         'b-', linewidth=3, label=f'Skill {SKILL_B}', alpha=0.7)
ax3.set_title(f'Agent {TEST_AGENT} Comparison')
ax3.set_xlabel('X Position')
ax3.set_ylabel('Y Position')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axis('equal')

# 4. 각 agent의 변화량 (bar plot)
ax4 = plt.subplot(2, 3, 4)
colors = ['red' if i == TEST_AGENT else 'lightgray' for i in range(10)]
ax4.bar(range(10), differences, color=colors)
ax4.set_xlabel('Agent ID')
ax4.set_ylabel('Average Trajectory Difference')
ax4.set_title('Trajectory Change per Agent')
ax4.axhline(y=0.01, color='green', linestyle='--', label='Threshold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. 다른 agent 중 하나 비교 (Agent 5)
ax5 = plt.subplot(2, 3, 5)
other_agent = 5
ax5.plot(trajectories_A[other_agent][:, 0], trajectories_A[other_agent][:, 1], 
         'r-', linewidth=2, label=f'When Agent{TEST_AGENT}=Skill{SKILL_A}', alpha=0.7)
ax5.plot(trajectories_B[other_agent][:, 0], trajectories_B[other_agent][:, 1], 
         'b-', linewidth=2, label=f'When Agent{TEST_AGENT}=Skill{SKILL_B}', alpha=0.7)
ax5.set_title(f'Agent {other_agent} (Should be Same)')
ax5.set_xlabel('X Position')
ax5.set_ylabel('Y Position')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.axis('equal')

# 6. Summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

changed_agent_diff = differences[TEST_AGENT]
other_agents_diff = [differences[i] for i in range(10) if i != TEST_AGENT]
avg_other_diff = np.mean(other_agents_diff)

summary = f"""
Skill Independence Test

Changed: Agent {TEST_AGENT}
  Skill {SKILL_A} → Skill {SKILL_B}

Trajectory Difference:
  Agent {TEST_AGENT}: {changed_agent_diff:.4f}
  Other agents avg: {avg_other_diff:.4f}

Ratio: {changed_agent_diff / (avg_other_diff + 1e-6):.2f}x

"""

if changed_agent_diff > 10 * avg_other_diff:
    summary += "✓✓ EXCELLENT Independence\n   Only target agent changed!"
elif changed_agent_diff > 5 * avg_other_diff:
    summary += "✓ GOOD Independence\n   Target agent changed more"
elif changed_agent_diff > 2 * avg_other_diff:
    summary += "~ MODERATE Independence\n   Some coupling exists"
else:
    summary += "✗ POOR Independence\n   All agents affected"

ax6.text(0.1, 0.5, summary, fontsize=11, family='monospace',
         verticalalignment='center')

plt.tight_layout()
plt.savefig(f'independence_test_agent{TEST_AGENT}.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Analysis saved to: independence_test_agent{TEST_AGENT}.png")

# ======================
# 결론
# ======================
print("\n" + "=" * 80)
print("Skill Independence Results")
print("=" * 80)
print(f"Agent {TEST_AGENT} trajectory change: {changed_agent_diff:.4f}")
print(f"Other agents average change: {avg_other_diff:.4f}")
print(f"Ratio: {changed_agent_diff / (avg_other_diff + 1e-6):.2f}x")

if changed_agent_diff > 10 * avg_other_diff:
    print("\n✓✓ Skills are INDEPENDENT - Only the target agent changes!")
elif changed_agent_diff > 5 * avg_other_diff:
    print("\n✓ Skills show GOOD independence")
else:
    print("\n~ Skills show MODERATE independence - some coupling between agents")
print("=" * 80)
