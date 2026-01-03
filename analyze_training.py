"""
DUSDi Particle 학습 로그 (train.csv) 분석
Intrinsic reward, skill classification accuracy, critic Q 값 추이 분석
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# CSV 파일 로드
csv_path = "exp_local/2025.12.31/173349_seed:2 particle dusdi_diayn test/train.csv"
df = pd.read_csv(csv_path)

print("=" * 80)
print("DUSDi Particle 학습 분석")
print("=" * 80)

# 기본 정보
print(f"\n전체 데이터 포인트: {len(df)}")
print(f"학습 step 범위: {df['step'].min():,.0f} ~ {df['step'].max():,.0f}")
print(f"총 episode: {df['episode'].max():.0f}")

# 컬럼 확인
print(f"\n사용 가능한 metrics: {list(df.columns)}")

# 주요 지표 분석
print("\n" + "=" * 80)
print("주요 지표 분석")
print("=" * 80)

# Intrinsic reward 관련 (diayn_reward)
if 'diayn_reward' in df.columns:
    print("\n[Intrinsic Reward (DIAYN)]")
    print(f"  초기 (first 10): {df['diayn_reward'].head(10).mean():.4f}")
    print(f"  중간 (half): {df['diayn_reward'].iloc[len(df)//2:len(df)//2+10].mean():.4f}")
    print(f"  최종 (last 10): {df['diayn_reward'].tail(10).mean():.4f}")
    print(f"  추세: {'증가' if df['diayn_reward'].tail(10).mean() > df['diayn_reward'].head(10).mean() else '감소'}")

# Skill classification accuracy
diayn_acc_cols = [col for col in df.columns if 'diayn_acc' in col]
if diayn_acc_cols:
    print(f"\n[Skill Classification Accuracy] ({len(diayn_acc_cols)} channels)")
    for col in diayn_acc_cols:
        print(f"  {col}:")
        print(f"    초기: {df[col].head(10).mean():.4f}")
        print(f"    최종: {df[col].tail(10).mean():.4f}")
        print(f"    추세: {'증가 ✓' if df[col].tail(10).mean() > df[col].head(10).mean() else '감소'}")

# Critic Q values
if 'critic' in df.columns:
    print(f"\n[Critic Q 값]")
    print(f"  초기 (first 10): {df['critic'].head(10).mean():.4f}")
    print(f"  중간 (half): {df['critic'].iloc[len(df)//2:len(df)//2+10].mean():.4f}")
    print(f"  최종 (last 10): {df['critic'].tail(10).mean():.4f}")
    
    # Q 값 증가율 계산
    initial_q = df['critic'].head(100).mean()
    final_q = df['critic'].tail(100).mean()
    increase_rate = (final_q - initial_q) / abs(initial_q) * 100
    print(f"  증가율: {increase_rate:.1f}%")
    
    if abs(increase_rate) > 500:
        print(f"  ⚠️  경고: Q 값이 매우 빠르게 증가 ({increase_rate:.1f}%)")
    elif abs(increase_rate) > 200:
        print(f"  ⚠️  주의: Q 값 증가 속도 점검 필요")
    else:
        print(f"  ✓ Q 값 증가 안정적")

# Actor loss
if 'actor' in df.columns:
    print(f"\n[Actor Loss]")
    print(f"  초기: {df['actor'].head(10).mean():.4f}")
    print(f"  최종: {df['actor'].tail(10).mean():.4f}")

# ======================
# 시각화
# ======================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. DIAYN reward
if 'diayn_reward' in df.columns:
    axes[0, 0].plot(df['step'], df['diayn_reward'], alpha=0.7)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('DIAYN Reward')
    axes[0, 0].set_title('Intrinsic Reward (DIAYN)')
    axes[0, 0].grid(True, alpha=0.3)

# 2. Skill Classification Accuracy
if diayn_acc_cols:
    for col in diayn_acc_cols:
        axes[0, 1].plot(df['step'], df[col], alpha=0.6, label=col.replace('diayn_acc_', 'Ch'))
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Skill Classification Accuracy')
    axes[0, 1].legend(loc='best', ncol=2, fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

# 3. Critic Q values
if 'critic' in df.columns:
    axes[0, 2].plot(df['step'], df['critic'], alpha=0.7, color='red')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('Q Value')
    axes[0, 2].set_title('Critic Q Values')
    axes[0, 2].grid(True, alpha=0.3)

# 4. Actor loss
if 'actor' in df.columns:
    axes[1, 0].plot(df['step'], df['actor'], alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Actor Loss')
    axes[1, 0].grid(True, alpha=0.3)

# 5. Episode length
if 'episode_length' in df.columns:
    axes[1, 1].plot(df['step'], df['episode_length'], alpha=0.7)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Length')
    axes[1, 1].set_title('Episode Length')
    axes[1, 1].grid(True, alpha=0.3)

# 6. FPS
if 'fps' in df.columns:
    axes[1, 2].plot(df['step'], df['fps'], alpha=0.7, color='purple')
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('FPS')
    axes[1, 2].set_title('Training Speed (FPS)')
    axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('particle_training_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ 시각화 저장: particle_training_analysis.png")

# ======================
# 결론
# ======================
print("\n" + "=" * 80)
print("학습 상태 종합 평가")
print("=" * 80)

conclusions = []

# DIAYN reward 체크
if 'diayn_reward' in df.columns:
    if df['diayn_reward'].tail(10).mean() > df['diayn_reward'].head(10).mean():
        conclusions.append("✓ Intrinsic reward 증가 - Skill discovery 정상 작동")
    else:
        conclusions.append("⚠️  Intrinsic reward 감소 - 점검 필요")

# Accuracy 체크
if diayn_acc_cols:
    avg_acc = np.mean([df[col].tail(10).mean() for col in diayn_acc_cols])
    if avg_acc > 0.85:
        conclusions.append(f"✓ Skill classification accuracy 우수 ({avg_acc:.2f})")
    elif avg_acc > 0.7:
        conclusions.append(f"~ Skill classification accuracy 보통 ({avg_acc:.2f})")
    else:
        conclusions.append(f"⚠️  Skill classification accuracy 낮음 ({avg_acc:.2f})")

# Critic Q 체크
if 'critic' in df.columns:
    initial_q = df['critic'].head(100).mean()
    final_q = df['critic'].tail(100).mean()
    increase_rate = (final_q - initial_q) / abs(initial_q) * 100
    
    if abs(increase_rate) > 500:
        conclusions.append(f"⚠️  Critic Q 값 급증 ({increase_rate:.0f}%) - Reward scale 조정 필요")
    elif abs(increase_rate) > 200:
        conclusions.append(f"⚠️  Critic Q 값 빠르게 증가 ({increase_rate:.0f}%) - 모니터링 필요")
    else:
        conclusions.append(f"✓ Critic Q 값 안정적 증가 ({increase_rate:.0f}%)")

print("\n".join(conclusions))

print("\n" + "=" * 80)
print("💡 권장 사항")
print("=" * 80)
if 'critic' in df.columns and abs(increase_rate) > 200:
    print("1. Critic learning rate 조정 고려")
    print("2. Reward normalization 적용 검토")
    print("3. Target network update 주기 조정")
else:
    print("현재 학습 안정적으로 진행 중")
    print("Downstream task로 진행 가능")

print("=" * 80)
