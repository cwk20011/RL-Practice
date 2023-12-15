import torch
import torch.nn as nn

# 정책 네트워크 정의
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(24, 400)
        self.fc2 = nn.Linear(400, 300)  
        self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(300, 4)  # BipedalWalker의 액션은 연속형 4개

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        action_mean = torch.tanh(self.fc4(x))
        return action_mean