#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 학습 기록 저장을 위한 namedtuple 정의
TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_'])

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(3, 100)
        self.a_head = nn.Linear(100, 5)
        self.v_head = nn.Linear(100, 1)

    def forward(self, x):
        x = F.tanh(self.fc(x))
        a = self.a_head(x) - self.a_head(x).mean(1, keepdim=True)
        v = self.v_head(x)
        action_scores = a + v
        return action_scores

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = np.empty(capacity, dtype=object)
        self.capacity = capacity
        self.data_pointer = 0
        self.isfull = False

    def update(self, transition):
        self.memory[self.data_pointer] = transition
        self.data_pointer += 1
        if self.data_pointer == self.capacity:
            self.data_pointer = 0
            self.isfull = True

    def sample(self, batch_size):
        return np.random.choice(self.memory, batch_size)

class DQNAgent():
    action_list = [(i * 4 - 2,) for i in range(5)]
    max_grad_norm = 0.5

    def __init__(self):
        self.training_step = 0
        self.epsilon = 1
        self.eval_net, self.target_net = QNetwork().float(), QNetwork().float()
        self.memory = ReplayMemory(2000)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=1e-3)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        if np.random.random() < self.epsilon:
            action_index = np.random.randint(5) # 이산화한 액션수 (num_actions)
        else:
            probs = self.eval_net(state)
            action_index = probs.max(1)[1].item()
        return self.action_list[action_index], action_index

    def save_param(self):
        torch.save(self.eval_net.state_dict(), './dqn_net_params.pkl')

    def store_transition(self, transition):
        self.memory.update(transition)

    def update(self):
        self.training_step += 1

        transitions = self.memory.sample(32)
        s = torch.tensor([t.s for t in transitions], dtype=torch.float)
        a = torch.tensor([t.a for t in transitions], dtype=torch.long).view(-1, 1)
        r = torch.tensor([t.r for t in transitions], dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in transitions], dtype=torch.float)

        with torch.no_grad():
            a_ = self.eval_net(s_).max(1, keepdim=True)[1]
            q_target = r + 0.9 * self.target_net(s_).gather(1, a_) # Gamma = 0.9
        q_eval = self.eval_net(s).gather(1, a)

        self.optimizer.zero_grad()
        loss = F.smooth_l1_loss(q_eval, q_target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.training_step % 200 == 0: # 200에 한번씩 업데이트
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.epsilon = max(self.epsilon * 0.999, 0.01)

        return q_eval.mean().item()

def main():
    env = gym.make('Pendulum-v1')
    agent = DQNAgent()

    training_records = []
    running_reward, running_q = -1000, 0
#    best_reward = -float('inf')

    for i_ep in range(1000):
        score = 0
        state, _ = env.reset()

        for t in range(200):
            action, action_index = agent.select_action(state)
            state_, reward, terminated, truncated, _ = env.step(action)
            score += reward
            agent.store_transition(Transition(state, action_index, (reward + 8) / 8, state_))
            state = state_
            if agent.memory.isfull:
                q = agent.update()
                running_q = 0.99 * running_q + 0.01 * q

        running_reward = running_reward * 0.9 + score * 0.1
        training_records.append(TrainingRecord(i_ep, running_reward))

        if i_ep % 20 == 0:
            print('Ep {}\tAverage score: {:.2f}\tAverage Q: {:.2f}'.format(
                i_ep, running_reward, running_q))

#         if running_reward > best_reward:
#             best_reward = running_reward
#             agent.save_param()
#             with open('./dqn_training_records.pkl', 'wb') as f:
#                 pickle.dump(training_records, f, protocol=pickle.HIGHEST_PROTOCOL)


            
#         if running_reward > -10:
#             print("Solved! Running reward is now {}!".format(running_reward))
#             agent.save_params()
#             with open('./dqn_training_records.pkl', 'wb') as f:
#                 pickle.dump(training_records, f)
#             break

    env.close()

    plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
    plt.title('DQN')
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.savefig("./dqn.png")
    plt.show()


# 'gamma', type=float, default=0.9
# 'num_actions', type=int, default=5


