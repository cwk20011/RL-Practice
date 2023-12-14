#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 로봇 팔을 모델링한 클래스
class RobotArmModel:
    def __init__(self):
        # 초기 각도를 랜덤으로 설정 (0도부터 360도 사이)
        self.angle = np.random.uniform(0, 360)

    def move(self, action):
        # 주어진 행동을 사용하여 로봇 팔의 각도 업데이트
        self.angle += action
        self.angle = self.angle % 360
        return self.angle

# Actor 네트워크 클래스
class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        return x

# Critic 네트워크 클래스
class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNet, self).__init__()
        self.fc_state = nn.Linear(state_dim, 128)
        self.fc_action = nn.Linear(action_dim, 128)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, state, action):
        x_state = F.relu(self.fc_state(state))
        x_action = F.relu(self.fc_action(action))
        x = torch.cat((x_state, x_action), dim=1)
        x = F.relu(self.fc2(x))
        return x

# Replay 메모리 클래스
class Memory:
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.data_pointer = 0

    def update(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.data_pointer] = transition
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return []
        return np.random.choice(self.memory, batch_size, replace=False)

# Transition 클래스 정의
class Transition:
    def __init__(self, state=None, action=None, reward=None, next_state=None, done=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

# DDPG 에이전트 클래스
class DDPGAgent:
    def __init__(self, state_dim, action_dim, noise_std=0.1):
        # Actor, Critic 네트워크 및 타겟 네트워크 초기화
        self.actor_net = ActorNet(state_dim, action_dim)
        self.critic_net = CriticNet(state_dim, action_dim)
        self.target_actor_net = ActorNet(state_dim, action_dim)
        self.target_critic_net = CriticNet(state_dim, action_dim)

        # Actor, Critic의 옵티마이저 초기화
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=1e-4)

        # Replay 메모리 초기화
        self.memory = Memory(10000)  # 메모리 크기

        # 타겟 네트워크 업데이트 주기
        self.target_update_freq = 300  # 업데이트 주기
        self.target_update_counter = 0

        # 노이즈 설정
        self.noise_std = noise_std

    def select_action(self, state):
        # 주어진 상태에서 액션 선택
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action = self.actor_net(state).numpy().flatten()
        # 액션에 노이즈 추가
        action += np.random.normal(0, self.noise_std, action.shape)
        return action

    def train(self, state, action, reward, next_state, done):
        transitions = self.memory.sample(32)
        if not transitions:
            return  # 메모리에서 샘플이 부족하면 학습을 진행하지 않음

        batch = Transition(*zip(*transitions))

        # Critic 네트워크 업데이트 시 보상 스케일링
        scale_factor = 10.0
        reward /= scale_factor  # 보상 스케일링

        # 최종 각도에 도달했을 때 추가적인 보상
        done_reward = 100.0  # 보상 설정
        done_reward *= np.abs(self.robot_arm.angle - 180.0) < 3.0  # 종료 조건

        # 최종 각도에 따른 리워드 및 종료 조건 설정
        reward = -np.abs(self.robot_arm.angle - 180.0) + done_reward

        # 타겟 Q 값 계산
        with torch.no_grad():
            target_actions = self.target_actor_net(batch.next_state)
            target_q_values = self.target_critic_net(batch.next_state, target_actions).squeeze()
            target_q_values = target_q_values * (1 - batch.done)
            target_q_values += batch.reward

        # Critic 네트워크 업데이트
        critic_loss = F.mse_loss(self.critic_net(batch.state, batch.action).squeeze(), target_q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor 네트워크 업데이트
        actor_loss = -self.critic_net(batch.state, self.actor_net(batch.state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 타겟 네트워크 주기적 업데이트
        if self.target_update_counter % self.target_update_freq == 0:
            self.target_critic_net.load_state_dict(self.critic_net.state_dict())
            self.target_actor_net.load_state_dict(self.actor_net.state_dict())

        self.target_update_counter += 1

# Main 클래스
class Main:
    def __init__(self):
        # 환경 초기화
        self.robot_arm = RobotArmModel()
        self.state_dim = 1  # 로봇 팔 각도 디멘젼
        self.action_dim = 1  # 각도의 변화량 디멘젼

        # DDPG 에이전트 초기화
        self.agent = DDPGAgent(self.state_dim, self.action_dim, noise_std = 0.1)

        # 학습 루프
        self.episode_rewards = []

    def run(self):
        for episode in range(2500):
            # 초기 각도를 랜덤으로 설정
            self.robot_arm = RobotArmModel()
            state = np.array([self.robot_arm.angle])
            episode_reward = 0


            for _ in range(500):
                action = self.agent.select_action(state)
                next_state = np.array([self.robot_arm.move(action)])

                # 최종 각도에 따른 리워드 및 종료 조건 설정
                done_reward = 50.0  # 보상 설정
                done_reward *= np.abs(self.robot_arm.angle - 180.0) < 5.0  # 종료 조건
                reward = -np.abs(self.robot_arm.angle - 180.0) + done_reward

                episode_reward += reward

                self.agent.train(state, action, reward, next_state, False)

                state = next_state

                if done_reward > 0:
                    break

            if (episode + 1) % 100 == 0:
                self.episode_rewards.append(episode_reward.mean())
                print(f"Episode: {episode + 1}, Reward: {episode_reward.mean()}, Robot Arm Angle: {self.robot_arm.angle}")

        # 최종 로봇 팔 각도 및 리워드 시각화
        print(f"Final Robot Arm Angle: {self.robot_arm.angle}")

        # 보상 그래프 플로팅
        plt.plot(np.arange(100, 2501, 100), self.episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Training Progress')
        plt.show()


if __name__ == "__main__":
    main_instance = Main()
    main_instance.run()

