import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.optim import Adam

import gym
import numpy as np

Device = torch.device("cuda:0")

class ActorCriticNet(nn.Module):
    def __init__(self, observation_space, action_space,
                hidden_sizes=[32,32], activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]
        action_dim = action_space.n
        self.base_net = nn.Sequential(
                            nn.Linear(obs_dim, hidden_sizes[0]),
                            # nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        )
        self.pi = nn.Linear(hidden_sizes[1], action_dim)
        self.vf = nn.Linear(hidden_sizes[1],1)
        self.to(Device)
    
    def forward(self, obs):
        obs = torch.Tensor(obs).to(Device)
        x = F.relu(self.base_net(obs))
        action_logits = F.softmax(self.pi(x), dim=-1)
        value = self.vf(x)
        return action_logits, value

class Agent(object):
    def __init__(self, model=None, lr=1e-2, gamma=0.99):
        self.gamma = gamma
        self.AC = model
        self.optimizer = Adam(AC.parameters(), lr=lr)
        self.logp_as = []
        self.values = []
        self.rewards = []

    def choose_action(self, obs):
        action_logits, value = self.AC(obs)
        distribution = Categorical(action_logits)
        action = distribution.sample()
        self.logp_as.append(distribution.log_prob(action))
        self.values.append(value)
        return action.item()
    
    def learn(self):

        R = 0
        policy_losses = []
        value_losses = []
        returns = []

        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(Device)
        returns = (returns - returns.mean()) / (returns.std() + 0.00001)

        for logp_a, value, R in zip(self.logp_as, self.values, returns):
            advantage = R - value.item()
            # calculate actor (policy) loss 
            policy_losses.append(-logp_a * advantage)
            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(Device)))

        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        self.rewards = []
        self.values = []
        self.logp_as = []
        

# Build env
env = gym.make('CartPole-v1')
state = env.reset()

# Learning setting
lr = 3e-2
EPISODES=30000
GAMMA = 0.99
hidden_sizes = [128,128]
show_every = 100

AC = ActorCriticNet(env.observation_space, env.action_space, hidden_sizes)
agent = Agent(AC, lr=lr, gamma=GAMMA)

for episode in range(EPISODES):
    # For every episode init
    done = False
    obs = env.reset()
    I = 1
    T = 0

    # Logs
    episode_reward = 0
    running_reward = 0
    if episode % show_every == 0:
        is_render = True
    else:
        is_render = False

    while not done:
        # Render
        if is_render:
            env.render("human")
            
        # Predict action and value
        action = agent.choose_action(obs)

        # Step the env
        next_obs, reward, done, _ = env.step(action)

        # Update obs
        obs = next_obs
        agent.rewards.append(reward)
        T += 1

        # Logs
        episode_reward += reward
    
    # Learn once
    agent.learn()

    # Update cumulative reward
    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
    
    print(f"episode_{episode} \t ep_reward = {episode_reward} \t ep_len = {T}")
    if running_reward > env.spec.reward_threshold:
        print("Solved! Running reward is now {} and "
                "the last episode runs to {} time steps!".format(running_reward, T))
        break
