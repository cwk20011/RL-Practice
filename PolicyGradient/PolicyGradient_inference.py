from network import PolicyNetwork
import torch
from torch import nn
import gym

# load trained-model
def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = PolicyNetwork()
    model_state = torch.load(path)
    policy_net.load_state_dict(model_state)


    # setting gpus
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs")
        policy_net = nn.DataParallel(policy_net)
    print(policy_net.parameters())
    policy_net.to(device)
    print(policy_net.parameters())
    return policy_net

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
                action_mean = model(state)
                scale = torch.tensor([0.01])
                action_dist = torch.distributions.Normal(action_mean.to("cpu"), scale.to("cpu"))
                action = action_dist.sample()
                action = action.clamp(-1, 1)  # BipedalWalker의 액션 범위는 [-1, 1] 

            state, reward, done, truncated, _ = env.step(action.squeeze().numpy())
            total_reward += reward

            if done or truncated:
                print("Total reward:", total_reward)
                break

# 환경 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('BipedalWalker-v3', render_mode='rgb_array')

# 불러오기
policy_net = load_model('models/final.pth')
policy_net.eval()

# 추론
print("Evaluating...")
evaluate(policy_net, num_episodes=5)