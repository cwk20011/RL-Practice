import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 로봇 팔을 모델링한 클래스
# RobotArmModel 클래스는 간단한 로봇 팔 모델을 나타냅니다. move 메서드는 주어진 행동을 사용하여 로봇 팔의 각도를 업데이트합니다.
class RobotArmModel:
    def __init__(self):
        self.angle = 0.0

    def move(self, action):
        # 행동을 적용하여 각도를 업데이트
        self.angle += action
        # 각도를 0에서 360 사이로 제한
        self.angle = self.angle % 360
        return self.angle

# DDPG 에이전트 클래스
# DDPGAgent 클래스는 DDPG 알고리즘을 구현한 에이전트입니다. __init__ 메서드에서는 Actor 신경망을 정의합니다. 여기서는 단순한 2층 신경망이며, 출력은 -1과 1 사이로 스케일 조정된 값입니다. select_action 메서드는 주어진 상태에서 액션을 선택합니다.
class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # -1과 1 사이의 값으로 스케일 조정
        )
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=1e-3)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action = self.actor_net(state).numpy().flatten()
        return action

    def train(self, state, action, reward, next_state, done):
        # DDPG에서는 크리틱 네트워크를 사용하여 업데이트하지만, 여기서는 단순화를 위해 생략함
        pass

# 환경 초기화
robot_arm = RobotArmModel()
# 로봇 팔 모델과 DDPG 에이전트를 초기화합니다.
state_dim = 1  # 로봇 팔 각도
action_dim = 1  # 각도의 변화량

agent = DDPGAgent(state_dim, action_dim)

# 학습 루프
# 학습 루프에서는 에피소드를 반복하고, 각 에피소드는 10 스텝으로 가정합니다. 각 스텝에서는 현재 상태에서 액션을 선택하고, 로봇 팔을 이동시킵니다. 보상은 180도에 가까워지도록 설정되어 있습니다. 현재는 크리틱 네트워크를 사용하여 학습을 진행하지 않고 있습니다.
for episode in range(100):
    state = np.array([robot_arm.angle])  # 초기 상태는 로봇 팔의 현재 각도
    episode_reward = 0

    for _ in range(10):  # 각 에피소드는 10 스텝으로 가정
        action = agent.select_action(state)
        next_state = np.array([robot_arm.move(action)])
        reward = -(next_state - np.array([180.0]))**2  # 목표는 180도에 가까워지도록 함
        episode_reward += reward

        # 강화 학습 에이전트에 샘플 추가 및 학습 생략
        agent.train(state, action, reward, next_state, False)

        state = next_state

    print(f"Episode: {episode+1}, Reward: {episode_reward.mean()}")

# 최종 로봇 팔 각도 시각화
print(f"Final Robot Arm Angle: {robot_arm.angle}")
