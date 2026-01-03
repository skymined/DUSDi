"""
스크립트 2: Discriminator 직접 테스트
데이터 출처: 저장된 discriminator 모델 + 새로 생성한 trajectories
목적: Discriminator가 실제로 skill을 잘 맞추는지 직접 확인
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
NUM_TEST_EPISODES = 10  # 각 skill당 10 episodes

TARGET_AGENT = 0
NUM_SKILLS = 5

MODEL_DIR = Path("/home/sky/문서/Github/DUSDi/models/states/particle/seed:2 particle dusdi_diayn test/2")
ACTOR_PATH = MODEL_DIR / f"actor_{SNAPSHOT_TS}.pt"
DISCRIMINATOR_PATH = MODEL_DIR / f"discriminator_{SNAPSHOT_TS}.pt"

print("=" * 80)
print("스크립트 2: Discriminator 직접 테스트")
print("데이터: Discriminator 모델 + 새 trajectories")
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

# Actor 로드
if ACTOR_PATH.exists():
    with ACTOR_PATH.open('rb') as f:
        actor_state = torch.load(f, map_location=device)
    agent.actor.load_state_dict(actor_state)
    print(f"✓ Actor loaded")
else:
    raise FileNotFoundError(f"Actor not found: {ACTOR_PATH}")

# Discriminator 로드
if DISCRIMINATOR_PATH.exists():
    with DISCRIMINATOR_PATH.open('rb') as f:
        discriminator_state = torch.load(f, map_location=device)
    agent.diayn.load_state_dict(discriminator_state)
    print(f"✓ Discriminator loaded")
else:
    raise FileNotFoundError(f"Discriminator not found: {DISCRIMINATOR_PATH}")

# ======================
# 각 Skill별 trajectories 생성 및 discriminator 예측
# ======================
print(f"\n[테스트] {NUM_SKILLS} skills × {NUM_TEST_EPISODES} episodes")

confusion_matrix = np.zeros((NUM_SKILLS, NUM_SKILLS))
all_predictions = []
all_true_labels = []

for true_skill in tqdm(range(NUM_SKILLS), desc="True Skills"):
    for episode in range(NUM_TEST_EPISODES):
        # Skill vector 생성
        skill_vec = [0] * 10
        skill_vec[TARGET_AGENT] = true_skill
        skill_vector = np.array(skill_vec, dtype=np.int64)
        
        meta = agent.get_meta_from_skill(skill_vector, num_envs=1)
        
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        states = []
        
        for step in range(NUM_STEPS):
            states.append(obs.copy())
            
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
        
        # Discriminator로 예측
        states_tensor = torch.from_numpy(np.array(states)).float().to(device)
        
        with torch.no_grad():
            # DIAYN discriminator는 channel별로 예측
            # partition_utils에서 state partition 사용
            from agent.partition_utils import get_domain_stats
            diayn_dim, state_partition_points = get_domain_stats(DOMAIN, cfg.env)
            
            # Agent 0에 해당하는 state 부분 추출
            # Particle: first N positions
            agent_states = states_tensor[:, state_partition_points[TARGET_AGENT]:state_partition_points[TARGET_AGENT+1]]
            
            # Discriminator 예측
            logits = agent.diayn.discriminators[TARGET_AGENT](agent_states)
            
            # 평균 logits (모든 timesteps)
            avg_logits = logits.mean(dim=0)
            predicted_skill = avg_logits.argmax().item()
        
        # Confusion matrix 업데이트
        confusion_matrix[true_skill, predicted_skill] += 1
        all_predictions.append(predicted_skill)
        all_true_labels.append(true_skill)

env.close()

# 정규화
confusion_matrix = confusion_matrix / NUM_TEST_EPISODES

# ======================
# 분석
# ======================
print("\n[결과] Confusion Matrix:")
print("Rows = True Skill, Columns = Predicted Skill\n")
print("     ", end="")
for i in range(NUM_SKILLS):
    print(f"  S{i}  ", end="")
print()
print("-" * 40)
for i in range(NUM_SKILLS):
    print(f"S{i} |", end="")
    for j in range(NUM_SKILLS):
        print(f" {confusion_matrix[i,j]:.2f} ", end="")
    print()

# Accuracy per skill
print("\n[Per-Skill Accuracy]")
for i in range(NUM_SKILLS):
    acc = confusion_matrix[i, i]
    print(f"  Skill {i}: {acc:.2f} ({int(acc*NUM_TEST_EPISODES)}/{NUM_TEST_EPISODES})")

# Overall accuracy
overall_acc = np.trace(confusion_matrix) / NUM_SKILLS
print(f"\nOverall Accuracy: {overall_acc:.3f}")

# ======================
# 시각화
# ======================
fig, ax = plt.subplots(figsize=(8, 7))

im = ax.imshow(confusion_matrix, cmap='YlOrRd', vmin=0, vmax=1)

# 텍스트 추가
for i in range(NUM_SKILLS):
    for j in range(NUM_SKILLS):
        text = ax.text(j, i, f'{confusion_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black" if confusion_matrix[i, j] < 0.5 else "white",
                       fontsize=12)

ax.set_xticks(range(NUM_SKILLS))
ax.set_yticks(range(NUM_SKILLS))
ax.set_xticklabels([f'S{i}' for i in range(NUM_SKILLS)])
ax.set_yticklabels([f'S{i}' for i in range(NUM_SKILLS)])
ax.set_xlabel('Predicted Skill', fontsize=12)
ax.set_ylabel('True Skill', fontsize=12)
ax.set_title(f'Discriminator Confusion Matrix (Agent {TARGET_AGENT})', fontsize=14)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Proportion', fontsize=11)

plt.tight_layout()
plt.savefig('analysis_2_discriminator_test.png', dpi=150)
print(f"\n✓ 시각화 저장: analysis_2_discriminator_test.png")

# ======================
# 결론
# ======================
print("\n" + "=" * 80)
print("결론")
print("=" * 80)

if overall_acc > 0.9:
    print(f"✓✓ Discriminator가 매우 정확 ({overall_acc:.1%})")
    print("   Skills를 명확히 구별할 수 있음")
elif overall_acc > 0.7:
    print(f"✓ Discriminator가 괜찮음 ({overall_acc:.1%})")
    print("   어느 정도 구별 가능")
else:
    print(f"⚠️  Discriminator 정확도 낮음 ({overall_acc:.1%})")
    print("   Skills가 잘 구별되지 않음")

# 가장 헷갈리는 skill 쌍
max_confusion = 0
confused_pair = None
for i in range(NUM_SKILLS):
    for j in range(i+1, NUM_SKILLS):
        confusion = confusion_matrix[i, j] + confusion_matrix[j, i]
        if confusion > max_confusion:
            max_confusion = confusion
            confused_pair = (i, j)

if confused_pair and max_confusion > 0.2:
    print(f"\n가장 헷갈리는 skills: {confused_pair[0]} ↔ {confused_pair[1]}")
    print(f"  혼동률: {max_confusion:.2f}")

print("=" * 80)
