import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
import numpy as np
import random
import math

class Net(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=128):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )
    
    def forward(self, x):
        x = self.net(x)
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# hyper parameters
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 2000  # e-greedy threshold decay
GAMMA = 0.99
LR = 1e-2
HIDDEN_SIZE = 128
MINI_BATCH_SIZE = 32
EPISODES = 20000
target_update_freq = 100

env = gym.make("CartPole-v1")
test_env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_Q_net = Net(obs_dim, action_dim, HIDDEN_SIZE)
target_Q_net = Net(obs_dim, action_dim, HIDDEN_SIZE)
target_Q_net.load_state_dict(policy_Q_net.state_dict())
target_Q_net.eval()
optimizer = optim.Adam(policy_Q_net.parameters(), lr=LR)

memory = ReplayMemory(MINI_BATCH_SIZE*100)

def select_action(state, cur_episode, deterministic=False):
    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * cur_episode/ EPS_DECAY)
    if random.random() > eps or deterministic:
        action = torch.argmax(policy_Q_net(state)).item()
    else:
        action = random.randrange(2)
    return torch.LongTensor([[action]])


def learn(cur_episode, target_update_freq=100):
    if len(memory) < MINI_BATCH_SIZE:
        return
    
    mini_batch = memory.sample(MINI_BATCH_SIZE)
    mb_state, mb_action, mb_reward, mb_next_state, mb_mask = zip(*mini_batch)
    
    mb_state = torch.cat(mb_state)
    mb_action = torch.cat(mb_action)
    mb_reward = torch.cat(mb_reward)
    mb_next_state = torch.cat(mb_next_state)
    mb_mask = torch.cat(mb_mask)

    # Q
    cur_q_value = policy_Q_net(mb_state).gather(1, mb_action)

    # Q'
    target_q_value = target_Q_net(mb_next_state).detach().max(1)[0]
    y = mb_reward + GAMMA * mb_mask * target_q_value

    loss = (y.reshape_as(cur_q_value) - cur_q_value).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if cur_episode % target_update_freq == 0:
        target_Q_net.load_state_dict(policy_Q_net.state_dict())

def evaluate():
    state = test_env.reset()
    done = False
    policy_Q_net.eval()
    # Log
    eval_episode_reward = 0
    while not done:
        action = select_action(torch.FloatTensor(state), 0, deterministic=True)
        next_state, reward, done, info = test_env.step(action.item())
        state = next_state
        eval_episode_reward += reward
    print(f"eval_episode_reward = {eval_episode_reward}")
        
if __name__ == '__main__':
    for episode in range(EPISODES):
        state = env.reset()
        done = False

        # Log
        cur_episode_reward = 0
        while not done:
            # env.render()
            action = select_action(torch.FloatTensor(state), episode)

            next_state, reward, done, _ = env.step(action.item())
            cur_episode_reward += reward
            
            memory.push((torch.FloatTensor([state]), 
                        action, 
                        torch.FloatTensor([reward]), 
                        torch.FloatTensor([next_state]), 
                        torch.FloatTensor([int(1-done)])
            ))
            
            state = next_state
        
            learn(episode)

        print(f"episode:{episode} \t reward = {cur_episode_reward}")

        if episode % 100 == 0:
            evaluate()