import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

# 정책 네트워크 정의
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(24, 400)
        self.fc2 = nn.Linear(400, 300)  # BipedalWalker의 액션 차원은 
        self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(300, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        action_mean = torch.tanh(self.fc4(x))
        return action_mean

# 학습 함수
def train(episodes, gamma=0.99):
    rewards = []
    for episode in range(episodes):
        state = env.reset()[0]

        log_probs = []
        episode_rewards = []
        done = False
        truncated = False

        while not done and not truncated:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
			
			# 분포 정의
            action_mean = policy_net(state_tensor)
            scale = torch.tensor([1 - (episode / episodes)])
            scale = torch.tensor([0.01])
            action_dist = torch.distributions.Normal(action_mean.to("cpu"), scale.to("cpu"))
            
			# Action 샘플링
			action = action_dist.sample()
            action = action.clamp(-1, 1)  # BipedalWalker의 액션 범위는 [-1, 1]
            
            next_state, reward, done, truncated, _ = env.step(action.squeeze().numpy())
            log_prob = action_dist.log_prob(action).sum(dim=1)
            log_probs.append(log_prob)
            episode_rewards.append(reward)
			
			# 총 Reward 계산
            if done or truncated:
                discounted_rewards = []
                R = 0
                for r in reversed(episode_rewards):
                    R = r + gamma * R
                    discounted_rewards.insert(0, R)
                discounted_rewards = torch.tensor(discounted_rewards)
                discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-6)

                policy_loss = []
                for log_prob, R in zip(log_probs, discounted_rewards):
                    policy_loss.append(-log_prob * R)
                policy_loss = torch.cat(policy_loss).sum()

                optimizer.zero_grad()
                policy_loss.backward()
                optimizer.step()

                rewards.append(sum(episode_rewards))
                print(f"Episode {episode + 1}/{episodes}, Total Reward: {sum(episode_rewards)}")
                break

    return rewards
	
# 평가 함수
def evaluate(model, num_episodes=5):
    for _ in range(num_episodes):
        state = env.reset()[0]
        done = False
        truncated = False
        total_reward = 0

        while not done and not truncated:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
				# 분포 정의
                action_mean = model(state)
                scale = torch.tensor([0.01])
                action_dist = torch.distributions.Normal(action_mean.to("cpu"), scale.to("cpu"))
				
				# Action 샘플링
                action = action_dist.sample()
                action = action.clamp(-1, 1)  # BipedalWalker의 액션 범위는 [-1, 1] 

            state, reward, done, truncated, _ = env.step(action.squeeze().numpy())
            total_reward += reward

            if done or truncated:
                print("Total reward:", total_reward)
                break

# 학습 실행
print("Training...")
gamma = 0.8
lr = 0.01

# 환경 초기화
env = gym.make('BipedalWalker-v3', render_mode='rgb_array')
policy_net = PolicyNetwork()

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    policy_net = nn.DataParallel(policy_net)
    policy_net.to(device)

optimizer = optim.Adam(policy_net.parameters(), lr=lr)


episode_rewards = train(1000, gamma=gamma)  # 1000 에피소드 동안 학습

# 결과 시각화
plt.plot(np.arange(len(episode_rewards)), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
gamma = str(gamma).replace('.', '')
lr = str(lr).replace('.', '')
plt.savefig("graph/train_graph_{}_{}.png".format(gamma, lr))

# 평가 모드로 전환
policy_net.eval()

# 평가 실행
print("Evaluating...")
evaluate(policy_net, num_episodes=5)

# 환경 종료
env.close()
