"""
특정 스킬 하나만 실행하고 비디오로 저장하는 스크립트
사용 예시:
    python visualize_single_skill.py \
        agent=dusdi_diayn \
        domain=particle \
        snapshot_ts=3000000 \
        skill_vector="[0,1,0,0,0]" \
        num_episodes=3 \
        max_steps=100
"""
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import hydra
import torch
import numpy as np
from pathlib import Path
from env_helper import make_envs
from utils import make_agent
import utils

@hydra.main(config_path='.', config_name='pretrain')
def visualize_skill(cfg):
    skill_id = cfg.skill_id
    meta = agent.get_meta(skill_id=skill_id, num_envs=1)
    
    num_episodes = cfg.get('num_episodes', 1)
    max_steps = cfg.get('max_steps', 50)
    
    print("=" * 80)
    print(f"Visualizing Skill: {skill_vector}")
    print(f"Episodes: {num_episodes}, Max steps per episode: {max_steps}")
    print("=" * 80)
    
    utils.set_seed_everywhere(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 환경 생성
    train_env, eval_env, use_gym = make_envs(cfg)
    
    # Agent 생성
    agent = make_agent(
        cfg.obs_type,
        train_env.observation_spec(),
        train_env.action_spec(),
        cfg.num_seed_frames // cfg.action_repeat,
        cfg, cfg.agent
    )
    
    # 가중치 로드
    work_dir = Path.cwd()
    snapshot_dir = work_dir / Path(cfg.snapshot_dir)
    actor_path = snapshot_dir / f'actor_{cfg.snapshot_ts}.pt'
    
    if actor_path.exists():
        print(f"Loading actor from: {actor_path}")
        with actor_path.open('rb') as f:
            actor_state = torch.load(f, map_location=device)
        agent.actor.load_state_dict(actor_state)
        print("✓ Actor loaded\n")
    else:
        raise FileNotFoundError(f"Actor not found: {actor_path}")
    
    # 비디오 저장 디렉토리
    video_dir = work_dir / 'skill_videos'
    video_dir.mkdir(exist_ok=True)
    
    # 에피소드 실행
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("-" * 40)
        
        # 스킬 메타 정보 생성
        meta = agent.get_meta_from_skill(skill_vector, num_envs=1)
        
        # 환경 리셋
        time_step = eval_env.reset()
        
        # 프레임 수집
        frames = []
        episode_reward = 0
        step = 0
        
        while not time_step.last() and step < max_steps:
            # 렌더링
            frame = eval_env.render(mode='rgb_array')
            if frame is not None:
                frames.append(frame)
            
            # 행동 선택
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(
                    time_step.observation,
                    meta,
                    step,
                    eval_mode=True
                )
            
            action = action.flatten()
            time_step = eval_env.step(action)
            
            episode_reward += time_step.reward.mean()
            step += 1
            
            if step % 10 == 0:
                print(f"  Step {step}/{max_steps}, Reward: {episode_reward:.3f}")
        
        print(f"  Final Reward: {episode_reward:.3f}, Steps: {step}")
        
        # 비디오 저장
        if frames:
            skill_str_clean = "_".join(str(s) for s in skill_vector)
            video_path = video_dir / f'skill_{skill_str_clean}_ep{episode}_reward{episode_reward:.2f}.mp4'
            
            # OpenCV로 저장
            import cv2
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))
            
            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            
            print(f"  📹 Video saved: {video_path}")
    
    print("\n" + "=" * 80)
    print(f"All videos saved to: {video_dir}")
    print("=" * 80)

if __name__ == '__main__':
    visualize_skill()
