# Skill 영상 생성 완료! 🎬

## ✅ 성공적으로 생성됨

**생성된 파일:**
- 📁 `skill_videos/` 디렉토리에 **15개 영상** 저장됨
- 각 영상은 PNG 형식 (3개 subplot: 거리, 보상, action)
- `skill_results.csv`에 모든 결과 요약

## 🎯 생성된 Skill 영상 목록

```
agent0_skill0_viz.png - Agent 0가 skill 0 사용
agent0_skill1_viz.png - Agent 0가 skill 1 사용
agent0_skill2_viz.png - Agent 0가 skill 2 사용
agent1_skill0_viz.png - Agent 1이 skill 0 사용 (baseline과 동일)
agent1_skill1_viz.png - Agent 1이 skill 1 사용
agent1_skill2_viz.png - Agent 1이 skill 2 사용
... (총 15개)
```

## 🔧 Skill 조정 방법

`visualize_all_skills.py` 파일 상단의 설정을 변경하세요:

### 방법 1: 테스트 모드 변경

```python
# Line 36-37
TEST_MODE = "single"  # ← 이 부분을 변경!

# 3가지 옵션:
# "single" - 한 번에 하나의 agent만 변경 (현재 설정)
# "all"    - 모든 agent가 같은 skill 사용
# "custom" - 사용자가 직접 정의한 skill 리스트
```

**"all" 모드 예시:**
```python
TEST_MODE = "all"
# 결과: 모든 agent가 skill 0, 모든 agent가 skill 1, ... (총 5개)
```

**"custom" 모드 예시:**
```python
TEST_MODE = "custom"

# Line 113-119 수정:
custom_skills = [
    ([0,0,0,0,0,0,0,0,0,0], "baseline"),
    ([1,0,0,0,0,0,0,0,0,0], "only_agent0_moves"),
    ([0,1,0,0,0,0,0,0,0,0], "only_agent1_moves"),
    ([1,1,0,0,0,0,0,0,0,0], "agent0_and_1_move"),
    ([2,3,1,0,4,0,0,0,0,0], "complex_combo"),
    ([4,4,4,4,4,4,4,4,4,4], "all_max_skill"),
]
```

### 방법 2: Skill 개수 조정

```python
# Line 125-129 (single 모드일 때)
for agent_idx in range(min(NUM_AGENTS, 5)):  # ← 5를 10으로 변경하면 모든 agent
    for skill_idx in range(min(SKILL_DIM, 3)):  # ← 3을 5로 변경하면 모든 skill
```

**모든 조합 테스트 (주의: 50개!):**
```python
for agent_idx in range(NUM_AGENTS):  # 10개 agent
    for skill_idx in range(SKILL_DIM):  # 5개 skill
        # 총 10 × 5 = 50개 영상 생성!
```

### 방법 3: 특정 Skill Vector만 테스트

```python
# custom 모드로 변경 후:
custom_skills = [
    ([0,3,0,0,0,0,0,0,0,0], "z1_channel1_skill3"),  # ← 원하는 skill!
    ([1,2,3,0,0,0,0,0,0,0], "collaboration_123"),
    ([4,0,0,0,0,0,0,0,0,0], "agent0_max_skill"),
]
```

## 🚀 실행 명령어

```bash
cd /home/sky/문서/Github/DUSDi

# 설정을 변경한 후 실행:
/home/sky/miniconda3/envs/dusdi/bin/python visualize_all_skills.py
```

**출력:**
- `skill_videos/*.png` - 각 skill의 시각화
- `skill_videos/skill_results.csv` - 결과 요약

## 📊 결과 분석

### CSV 파일 보기:
```bash
cat skill_videos/skill_results.csv
# 또는
column -t -s, skill_videos/skill_results.csv | less
```

### 영상 파일 열기:
```bash
# 특정 영상 보기
eog skill_videos/agent1_skill1_viz.png

# 모든 영상 슬라이드쇼
eog skill_videos/*.png
```

## 💡 추천 설정

### 1. 빠른 탐색 (각 agent별 1개 skill):
```python
TEST_MODE = "single"
# Line 125
for agent_idx in range(NUM_AGENTS):  # 모든 agent
    for skill_idx in range(1):  # skill 0만
        # → 10개 영상
```

### 2. 각 skill 비교 (모든 agent 동시):
```python
TEST_MODE = "all"
# → 5개 영상 (각 skill별)
```

### 3. 흥미로운 조합만:
```python
TEST_MODE = "custom"
custom_skills = [
    ([0,0,0,0,0,0,0,0,0,0], "baseline"),
    ([1,1,1,1,1,1,1,1,1,1], "all_skill1"),
    ([0,1,2,3,4,0,1,2,3,4], "pattern_01234"),
    ([4,4,4,4,4,4,4,4,4,4], "all_max"),
]
```

## ⚙️ 고급 설정

### Step 수 변경:
```python
NUM_STEPS = 100  # ← 50, 200 등으로 변경
```

### 다른 체크포인트 사용:
```python
SNAPSHOT_TS = 3000000  # ← 2000000, 1000000 등
```

### Seed 변경:
```python
SEED = 2  # ← 다른 seed로 변경하여 다른 실행 생성
```

---

이제 원하는 대로 skill을 조정하고 영상을 생성할 수 있습니다!
