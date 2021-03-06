import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.optim import Adam

import gym
import numpy as np

Device = torch.device("cuda:0")

class Agent(nn.Module):
    def __init__(self, 
        num_inputs, num_outputs, hidden_size,
        gamma=0.99, lam=0.95, dropout=0.5,):
        super(Agent, self).__init__()
        self.gamma = gamma
        self.lam = lam
        self.eps = 0.2
        self.CRITIC_DISCOUNT = 0.5

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )

        # Log shape is [t, dim]
        self.states = []
        self.actions = []
        self.logp_as = []
        self.values = []

        # Log as list
        self.rewards = []
        self.masks = []
        
        # next_obs is the last obs
        self.next_obs = None

    def forward(self, obs):
        logits = self.actor(obs)
        dist = Categorical(F.softmax(logits, dim=-1))

        value = self.critic(obs)

        return dist, value

def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x

def compute_gae(next_value:int, values, rewards, masks, gamma=0.99, lam=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for t in range(len(rewards))[::-1]:
        delta = rewards[t] + gamma*values[t+1]*masks[t] - values[t]
        gae = delta + gamma*lam*gae*masks[t]
        returns.insert(0, gae + values[t])
    return returns
      
def ppo_iter(states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    # generates random mini-batches until we have covered the full batch
    for _ in range(batch_size // MINI_BATCH_SIZE):
        rand_ids = np.random.randint(0, batch_size, MINI_BATCH_SIZE)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], \
            returns[rand_ids, :], advantage[rand_ids, :]

def PPO_update(states, actions, log_probs, returns, advantages, eps=0.2):
        # run EPOCHS time on current memeory
    for state, action, old_log_prob, return_, advantage in zip(
        states, actions, log_probs, returns, advantages):
            # chosen actions
            new_distribution, new_value = agent(state)
            new_logp_prob = new_distribution.log_prob(action)
            # old log probability of each action
            ratio = (new_logp_prob - old_log_prob).exp()

            p1 = ratio * advantage
            p2 = torch.clamp(ratio, 1-eps, 1+eps) * advantage
            pi_loss = -torch.min(p1, p2).mean()
            value_loss = (return_ - new_value).pow(2).mean()
            loss = value_loss + pi_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def evaluate():
    agent.eval()

    done = False
    state = test_env.reset()

    evaluate_rewards = 0
    while not done:
        # test_env.render()
        state = torch.Tensor(state).to(Device)
        with torch.no_grad():
            dist, _ = agent(state)
            action = torch.argmax(dist.logits)
        next_state, reward, done, _ = test_env.step(action.item())
        state = next_state

        evaluate_rewards += reward
        
    return evaluate_rewards

# Build env
env = gym.make('CartPole-v1')
test_env = gym.make("CartPole-v1")

# Learning setting
lr = 1e-2
EPISODES = 300
GAMMA = 0.99
hidden_sizes = 128
EPOCHS = 10
PPO_STEPS = 256 # trajectory steps
MINI_BATCH_SIZE = 32 # caculate loss on one mini batch
show_every = 20
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = Agent(obs_dim, action_dim, hidden_sizes).to(Device)
optimizer = Adam(agent.parameters(), lr=lr)

for episode in range(EPISODES):
    agent.train()
    # For every episode init
    done = False
    state = env.reset()

    # Init memeory
    masks = []
    rewards = []
    states = []
    actions = []
    values = []
    log_probs = []

    episode_reward = 0
    running_reward = 0
    if episode % show_every == 0:
        is_render = True
    else:
        is_render = False

    while not done:
        # Render
        # if is_render:
        #     env.render("human")
        
        # Add extra dim for cat
        state = torch.FloatTensor(state).unsqueeze(0).to(Device)

        # Predict action and value
        dist, value = agent(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Step the env
        next_state, reward, done, _ = env.step(action.item())

        # Logs
        log_probs.append(log_prob.unsqueeze(1).to(Device))
        values.append(value)
        rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(Device))
        masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(Device))
        states.append(state)
        actions.append(action.unsqueeze(1).to(Device))

        episode_reward += reward
    
        # Update obs
        state = next_state

    _, next_value = agent(torch.FloatTensor(next_state).to(Device))

    returns = compute_gae(next_value, values, rewards, masks)
    returns = torch.cat(returns).detach()
    # returns = normalize(returns)
    
    states = torch.cat(states)
    actions = torch.cat(actions).detach()
    log_probs = torch.cat(log_probs).detach()
    values = torch.cat(values).detach()
    
    advantages = returns - values
    advantages = normalize(advantages)

    # Learn once
    PPO_update(states, actions, log_probs, returns, advantages)

    # Test once
    print(f"episode_{episode} \t ep_reward = {evaluate()}")
    