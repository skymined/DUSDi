"""
Particle 환경 구조 이해 및 시각화 스크립트
"""
import numpy as np

# Particle 환경 정보 (partition_utils.py 기반)
N_AGENTS = 10
SKILL_DIM = 5

print("=" * 80)
print("Particle 환경 구조")
print("=" * 80)

# State 구조
print("\n📊 State (Observation) 구조:")
print(f"  총 Dimension: 70")
print(f"  구성:")
print(f"    - Agent-Landmark 거리: {N_AGENTS} × 1 = {N_AGENTS} dim")
print(f"    - Agent 위치+속도: {N_AGENTS} × 4 = {N_AGENTS * 4} dim")
print(f"    - 추가 정보: {N_AGENTS} × 2 = {N_AGENTS * 2} dim")
print(f"  Partition: [1]*{N_AGENTS} + [4]*{N_AGENTS} + [2]*{N_AGENTS}")

# Action 구조
print("\n🎮 Action 구조 (simplified):")
print(f"  총 Dimension: 20")
print(f"  구성: 각 agent가 2D action (x, y)")
print(f"  Partition: [2]*{N_AGENTS} = [2,2,2,2,2,2,2,2,2,2]")

# Skill 구조
print("\n🎯 Skill 구조:")
print(f"  Channels: {N_AGENTS} (각 agent별)")
print(f"  Skill_dim per channel: {SKILL_DIM}")
print(f"  총 Skill 조합: {SKILL_DIM}^{N_AGENTS} = {SKILL_DIM**N_AGENTS:,}개")

# 예시 스킬
print("\n💡 Skill Vector 예시:")
examples = [
    ([0,0,0,0,0,0,0,0,0,0], "모든 agent가 skill 0"),
    ([1,0,0,0,0,0,0,0,0,0], "Agent 0만 skill 1, 나머지 0"),
    ([0,3,0,0,0,0,0,0,0,0], "Agent 1만 skill 3, 나머지 0"),
    ([1,1,1,1,1,1,1,1,1,1], "모든 agent가 skill 1"),
    ([2,3,1,0,4,2,1,3,0,2], "각 agent가 다른 skill"),
]

for skill, desc in examples:
    print(f"  {skill} → {desc}")

print("\n" + "=" * 80)
print("iGibson 환경 구조")
print("=" * 80)

# iGibson 정보
IGIBSON_CHANNELS = 3
IGIBSON_SKILL_DIM = 4
IGIBSON_PARTITION = [0, 3, 7, 10]

print("\n📊 State (Observation) Discriminator 입력:")
print(f"  총 Dimension for Discriminator: 10 (처음 10 dim만 사용)")
print(f"  Channel partition: {IGIBSON_PARTITION}")
print(f"  의미:")
print(f"    - Channel 0 (dim 0-2): Base 관련 3 dim")
print(f"    - Channel 1 (dim 3-6): Arm 관련 4 dim")
print(f"    - Channel 2 (dim 7-9): Gripper/View 3 dim")

print("\n🎯 Skill 구조:")
print(f"  Channels: {IGIBSON_CHANNELS} (Base, Arm, Gripper)")
print(f"  Skill_dim per channel: {IGIBSON_SKILL_DIM}")
print(f"  총 Skill 조합: {IGIBSON_SKILL_DIM}^{IGIBSON_CHANNELS} = {IGIBSON_SKILL_DIM**IGIBSON_CHANNELS}개")

print("\n💡 Skill Vector 예시:")
igibson_examples = [
    ([0,0,0], "Base=0, Arm=0, Gripper=0"),
    ([1,0,0], "Base만 skill 1"),
    ([0,2,0], "Arm만 skill 2"),
    ([0,0,3], "Gripper만 skill 3"),
    ([1,2,3], "Base=1, Arm=2, Gripper=3 조합"),
]

for skill, desc in igibson_examples:
    print(f"  {skill} → {desc}")

print("\n" + "=" * 80)
print("Key Insight")
print("=" * 80)
print("✓ Particle: 10 agents → 10 channels (agent-wise partition)")
print("✓ iGibson: 1 agent → 3 channels (body-part partition)")
print("✓ Both: Multi-channel skill learning with compositional skills")
print("=" * 80)
