import numpy as np
import itertools
import gym
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
state = env.reset()

# obs = [x, v], position and velocity
print(f"obs lower bound = {env.observation_space.low}")
print(f"obs lower bound = {env.observation_space.high}")
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
print(discrete_os_win_size)

# Init Q-table, the initialization doesn't matter
# Q-table.shape = [20,20,3], means that we have 20 possible x, 20 possible velocity, and 3 possible action
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
print(q_table.shape)

def get_discrete_obs(obs):
    discrete_obs =(obs - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_obs.astype(np.int))
    
# Q-Learning setting
ALPHA = 0.1
DISCOUNT = 0.95
EPISODES = 30000
RENDER_EVERY = 1000
epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# logs
ep_rewards = []

for episode in range(EPISODES):
    # for every episode init
    x, v = get_discrete_obs(env.reset())
    done = False
    episode_reward = 0

    while not done:
        # find action
        if np.random.random() > epsilon:
            # Get action from Q table
            a = np.argmax(q_table[x][v])
        else:
            # Get random action
            a = np.random.randint(0, env.action_space.n)
        
        # step the env and update observation
        new_state, reward, done, info = env.step(a)
        new_x, new_v = get_discrete_obs(new_state)
        episode_reward += reward
        
        if episode > 10000 and episode % RENDER_EVERY == 0:
            env.render("human")

        if not done:
            # Update function: Q[s,a] = Q[s,a] + alpha * (R + gamma * max(Q[s',a]) - Q[s,a])
            current_q = q_table[x][v][a]
            q_table[x][v][a] = current_q + ALPHA*(reward+DISCOUNT*np.max(q_table[new_x][new_v]) - current_q)
        elif new_state[0] >= env.goal_position:
            q_table[x][v][a] = 0
        
        x, v = new_x, new_v
    ep_rewards.append(episode_reward)
    # print(ep_rewards)
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    if episode > 10000 and episode % RENDER_EVERY == 0:
        print(f"episode_{episode} = {max(ep_rewards)}")
        plt.plot(ep_rewards)
        plt.show()
        plt.pause(0.001)