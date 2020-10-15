import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box

Device = torch.device("cuda:0")

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

# class Agent(object):
#     def __init__(self, obs_dim, gamma=0.99, action_dim=None, output
#                 hidden_sizes=[32]):
#         self.gamma = gamma
#         self.reward_history = []
#         self.logp_a_history = []
#         self.policy = mlp(sizes=
#                             [obs_dim]+hidden_sizes+[action_dim]
#                         ).to(Device)
#     def get_action(self, obs):
#         obs = torch.from_numpy(obs).float().to(Device)
#         distributions = self.policy(obs)
#         action = distributions.sample()
#         logp_a = distributions.log_prob(action)
#         self.logp_a_history.append(logp_a)
#         return action.item()
        
        


env = gym.make('CartPole-v1')

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

lr = 1e-2
EPISODES=30000
GAMMA = 0.95
hidden_size = [32]
show_every = 100

logits_net = mlp(sizes=[obs_dim]+hidden_size+[action_dim]).to(Device)
optimizer = Adam(logits_net.parameters(), lr=lr)

def get_policy(obs):
    obs = torch.from_numpy(obs).float().to(Device)
    logits = logits_net(obs)
    return Categorical(logits=logits)

def get_action(obs):
    return get_policy(obs).sample().item()

for episode in range(EPISODES):
    done = False
    rewards = []
    logp_as = []

    obs = env.reset()
    
    if episode % show_every == 0:
        is_render = True
    else:
        is_render = False
    
    # Play one episode
    while not done:
        if is_render:
            env.render("human")
        action = get_action(obs)
        next_obs, reward, done, info = env.step(action)
        rewards.append(reward)
        logp_as.append(get_policy(obs).log_prob(torch.as_tensor(action, dtype=torch.int32).to(Device)))
        obs = next_obs
        
    # Caculate G
    optimizer.zero_grad()
    T = len(rewards)
    G = np.zeros((T, ))
    for t in range(T):
        G_sum = 0
        discount = 1
        for k in range(t, T):
            G_sum += rewards[k] * discount
            discount *= GAMMA
        G[t] = G_sum
    # Norm
    print(f"episode = {episode} \t G = {G[0]} \t ep_len = {T}")

    mean = np.mean(G)
    std = np.std(G)
    G = (G-mean)/std
    
    G = torch.tensor(G, dtype=torch.float).to(Device)
    
    # 每一条轨迹都可以得到一个梯度，这个梯度是把trajectory过程中的每一步的loga都累加起来
    loss = 0
    for g, logp_a in zip(G, logp_as):
        loss += -g * logp_a
    loss.backward()
    optimizer.step()
    