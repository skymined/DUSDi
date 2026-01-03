"""
스크립트 3: Skill Independence 재검증 (Stochasticity 고려)
데이터 출처: 같은 seed의 여러 실행 (새로 생성)
목적: 환경의 랜덤성을 제외하고 실제 skill 효과만 측정
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
NUM_REPEATS = 5  # 각 skill을 같은 seed로 5번 반복

TARGET_AGENT = 0
COMPARE_SKILLS = [0, 2]  # Skill 0 vs Skill 2 비교

MODEL_DIR = Path("/home/sky/문서/Github/DUSDi/models/states/particle/seed:2 particle dusdi_diayn test/2")
ACTOR_PATH = MODEL_DIR / f"actor_{SNAPSHOT_TS}.pt"

print("=" * 80)
print("스크립트 3: Skill Independence (Stochasticity 보정)")
print("데이터: 같은 seed 반복 실행 (환경 랜덤성 측정)")
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
# 데이터 수집
# ======================
def run_episode_with_seed(skill_idx, episode_seed):
    """같은 seed로 episode 실행"""
    skill_vec = [0] * 10
    skill_vec[TARGET_AGENT] = skill_idx
    skill_vector = np.array(skill_vec, dtype=np.int64)
    
    meta = agent.get_meta_from_skill(skill_vector, num_envs=1)
    
    # Seed 고정
    np.random.seed(episode_seed)
    torch.manual_seed(episode_seed)
    
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

print(f"\n[수집] Skill {COMPARE_SKILLS[0]} vs {COMPARE_SKILLS[1]} × {NUM_REPEATS} repeats")

skill_data = {skill: [] for skill in COMPARE_SKILLS}

for skill in tqdm(COMPARE_SKILLS, desc="Skills"):
    for repeat in range(NUM_REPEATS):
        trajectories = run_episode_with_seed(skill, SEED + repeat)
        skill_data[skill].append(trajectories)

env.close()

# ======================
# 분석
# ======================
print("\n[분석]")

# 1. Within-skill variance (같은 skill, 같은 agent 반복)
print("\n1. Within-Skill Variance (재현성 오류)")
for skill in COMPARE_SKILLS:
    print(f"  Skill {skill}:")
    for agent_idx in range(10):
        # 같은 skill, 같은 agent의 여러 trajectories
        trajs = [skill_data[skill][r][agent_idx] for r in range(NUM_REPEATS)]
        
        # 모든 쌍의 차이 계산
        diffs = []
        for i in range(len(trajs)):
            for j in range(i+1, len(trajs)):
                diff = np.mean(np.linalg.norm(trajs[i] - trajs[j], axis=1))
                diffs.append(diff)
        
        avg_diff = np.mean(diffs) if diffs else 0
        if agent_idx == TARGET_AGENT:
            print(f"    Agent {agent_idx} (target): {avg_diff:.4f}")
        elif agent_idx < 3:  # 처음 3개만 표시
            print(f"    Agent {agent_idx}: {avg_diff:.4f}")

# 2. Between-skill difference (skill 바뀔 때 차이)
print("\n2. Between-Skill Difference (실제 skill 효과)")
skill0_data = skill_data[COMPARE_SKILLS[0]]
skill1_data = skill_data[COMPARE_SKILLS[1]]

for agent_idx in range(10):
    # 같은 seed의 trajectory 비교
    diffs = []
    for repeat in range(NUM_REPEATS):
        traj0 = skill0_data[repeat][agent_idx]
        traj1 = skill1_data[repeat][agent_idx]
        diff = np.mean(np.linalg.norm(traj0 - traj1, axis=1))
        diffs.append(diff)
    
    avg_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    
    if agent_idx == TARGET_AGENT:
        print(f"  Agent {agent_idx} (target): {avg_diff:.4f} ± {std_diff:.4f}")
    elif agent_idx < 3:
        print(f"  Agent {agent_idx}: {avg_diff:.4f} ± {std_diff:.4f}")

# ======================
# 시각화
# ======================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Agent 0-5 trajectories
for i, agent_idx in enumerate([0, 1, 2, 3, 4, 5]):
    ax = axes[i // 3, i % 3]
    
    # Skill 0
    for repeat in range(NUM_REPEATS):
        traj = skill_data[COMPARE_SKILLS[0]][repeat][agent_idx]
        ax.plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.3, linewidth=1)
    
    # Skill 1
    for repeat in range(NUM_REPEATS):
        traj = skill_data[COMPARE_SKILLS[1]][repeat][agent_idx]
        ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.3, linewidth=1)
    
    ax.set_title(f'Agent {agent_idx}' + (' (target)' if agent_idx == TARGET_AGENT else ''))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    ax.legend([f'Skill {COMPARE_SKILLS[0]}', f'Skill {COMPARE_SKILLS[1]}'])

plt.tight_layout()
plt.savefig('analysis_3_skill_independence.png', dpi=150)
print(f"\n✓ 시각화 저장: analysis_3_skill_independence.png")

# ======================
# 결론
# ======================
print("\n" + "=" * 80)
print("결론")
print("=" * 80)

# Target agent의 within vs between
target_within = []
for skill in COMPARE_SKILLS:
    trajs = [skill_data[skill][r][TARGET_AGENT] for r in range(NUM_REPEATS)]
    for i in range(len(trajs)):
        for j in range(i+1, len(trajs)):
            diff = np.mean(np.linalg.norm(trajs[i] - trajs[j], axis=1))
            target_within.append(diff)

target_between = []
for repeat in range(NUM_REPEATS):
    traj0 = skill_data[COMPARE_SKILLS[0]][repeat][TARGET_AGENT]
    traj1 = skill_data[COMPARE_SKILLS[1]][repeat][TARGET_AGENT]
    diff = np.mean(np.linalg.norm(traj0 - traj1, axis=1))
    target_between.append(diff)

avg_within = np.mean(target_within)
avg_between = np.mean(target_between)

print(f"Target Agent {TARGET_AGENT}:")
print(f"  재현성 오차 (within): {avg_within:.4f}")
print(f"  Skill 간 차이 (between): {avg_between:.4f}")
print(f"  비율: {avg_between/(avg_within+1e-6):.2f}x")

if avg_between > 3 * avg_within:
    print("\n✓✓ Skill 효과가 재현성 오차보다 훨씬 큼")
    print("   Skills가 실제로 다른 행동을 만듦")
elif avg_between > avg_within:
    print("\n✓ Skill 효과가 재현성 오차보다 큼")
else:
    print("\n⚠️  Skill 효과가 재현성 오차와 비슷하거나 작음")
    print("   Skills이 의미있게 다르지 않을 수 있음")

print("=" * 80)
