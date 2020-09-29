import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.optim import Adam

import gym
import numpy as np

Device = torch.device("cuda:0")

def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x

class Agent(nn.Module):
    def __init__(self, 
        num_inputs, num_outputs, hidden_size,
        gamma=0.99, lam=0.95):
        super(Agent, self).__init__()
        self.gamma = gamma
        self.lam = lam
        self.eps = 0.2
        self.CRITIC_DISCOUNT = 0.5

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )

        # Log shape is [t, dim]
        self.obs = []
        self.actions = []
        self.logp_as = []
        self.values = []
        self.rewards = []
        self.masks = []
        # next_obs is the last obs
        self.next_obs = None

    def forward(self, obs):
        logits = self.actor(obs)
        dist = Categorical(F.softmax(logits, dim=-1))

        value = self.critic(obs)

        return dist, value
        
    def choose_action(self, obs):
        dist, value = self.forward(obs)

        action = dist.sample()
        self.obs.append(obs)
        self.logp_as.append(dist.log_prob(action))
        self.values.append(value)
        self.actions.append(action)
        return action.item()
    
    def compute_gae(self, values, rewards, masks):
        gae = 0
        returns = []
        for t in range(len(rewards))[::-1]:
            delta = rewards[t] + self.gamma*values[t+1]*masks[t] - values[t]
            gae = delta + self.lam*gae*masks[t]
            returns.insert(0, gae + values[t])
        returns = torch.tensor(returns).to(Device)
        returns = normalize(returns)
        return returns
        
    def learn(self):
        logp_as = self.logp_as.detach()
        actions = self.actions.detach()
        # Learn at end of one episode
        distributions, _ = self.forward(self.obs)
        new_logp_as = distributions.log_prob(self.actions)
        ratio = (new_logp_as - self.logp_as).exp()

        _, next_value = self.forward(self.next_obs)
        self.values = self.values.squeeze(-1)
        returns = self.compute_gae(
            torch.cat((self.values,next_value)), 
            self.rewards, 
            self.masks)
        advantages = returns - self.values
        advantages = normalize(advantages)
        advantages = advantages.detach()

        p1 = ratio * advantages
        p2 = torch.clamp(ratio, 1-self.eps, 1+self.eps)*advantages
        pi_loss = -torch.min(p1,p2).mean()
        value_loss = (returns - self.values).pow(2).mean()
        loss = value_loss + pi_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.obs = []
        self.actions = []
        self.logp_as = []
        self.values = []
        self.rewards = []
        self.masks = []
        self.next_obs = None
        
# Build env
env = gym.make('CartPole-v1')
state = env.reset()

# Learning setting
lr = 3e-2
EPISODES=30000
GAMMA = 0.99
hidden_sizes = 128
show_every = 100
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = Agent(obs_dim, action_dim, hidden_sizes).to(Device)
optimizer = Adam(agent.parameters(), lr=lr)

for episode in range(EPISODES):
    # For every episode init
    done = False
    obs = env.reset()
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
        
        # Add extra dim for cat
        obs = torch.FloatTensor(obs).unsqueeze(0).to(Device)

        # Predict action and value
        action = agent.choose_action(obs)

        # Step the env
        next_obs, reward, done, _ = env.step(action)

        # Update obs
        obs = next_obs
        T += 1

        # Logs
        agent.rewards.append(reward)
        agent.masks.append(done)
        episode_reward += reward
    
    agent.next_obs = torch.FloatTensor(next_obs).to(Device)
    agent.obs = torch.cat(agent.obs)
    agent.actions = torch.cat(agent.actions)
    agent.logp_as = torch.cat(agent.logp_as)
    agent.values = torch.cat(agent.values)

    # Learn once
    agent.learn()

    # Update cumulative reward
    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
    
    print(f"episode_{episode} \t ep_reward = {episode_reward} \t ep_len = {T}")
    if running_reward > env.spec.reward_threshold:
        print("Solved! Running reward is now {} and "
                "the last episode runs to {} time steps!".format(running_reward, T))
        break
