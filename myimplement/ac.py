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
        x = self.base_net(obs)
        action_logits = F.softmax(self.pi(x), dim=-1)
        value = self.vf(x)
        return action_logits, value

class Agent(object):
    def __init__(self, model=None, lr=1e-2, gamma=0.99):
        self.gamma = gamma
        self.AC = model
        self.optimizer = Adam(AC.parameters(), lr=lr)
    
    def choose_action(self, obs):
        action_logits, _ = self.AC.forward(obs)
        distribution = Categorical(action_logits)
        action = distribution.sample()
        self.log_probs = distribution.log_prob(action)
        return action.item()
    
    def learn(self, obs, reward, next_obs, done, I):
        self.optimizer.zero_grad()

        _, value = self.AC.forward(obs)
        _, next_value = self.AC.forward(next_obs)

        # reward = torch.Tensor(reward).to(Device)
        U = reward + self.gamma*(next_value if not done else 0)
        pi_loss = -I*(U-value)*self.log_probs
        value_loss = (U-value)**2

        # delta = reward + self.gamma*(next_value if not done else 0) - value
        # pi_loss = -self.log_probs * delta
        # value_loss = delta**2

        loss = pi_loss + value_loss
        loss.backward()
        self.optimizer.step()
        

# Build env
env = gym.make('CartPole-v1')
state = env.reset()

# Learning setting
lr = 3e-2
EPISODES=30000
GAMMA = 0.99
hidden_sizes = [128,128]
show_every = 1000

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

    # Trajectories
    episode_reward = []
    episode_action = []
    episode_obs = []
    
        
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
        T += 1

        # Logs
        episode_reward += reward
    
    # Learn once
    agent.learn(obs, reward, next_obs, done, I)

    # Update cumulative reward
    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
    
    print(f"episode_{episode} \t ep_reward = {episode_reward} \t ep_len = {T}")
    if running_reward > env.spec.reward_threshold:
        print("Solved! Running reward is now {} and "
                "the last episode runs to {} time steps!".format(running_reward, T))
        break
    
    


