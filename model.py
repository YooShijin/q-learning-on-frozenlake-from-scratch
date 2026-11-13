"""
Q-Learning on FrozenLake from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_q_table
import numpy as np
def init_q_table(num_states, num_actions):
    """Return a zero-initialized Q-table of shape (num_states, num_actions)."""
    # table = np.array(num_states, num_actions)
    # table = [ [float(0)]*num_actions for _ in range(num_states)]
    # table = np.array(table)
    # return table
    return np.zeros((num_states,num_actions))

# Step 2 - max_q_value
import numpy as np

def max_q_value(q_table, state):
    """Return the maximum Q value across all actions for the given state."""
    idx = np.argmax(q_table[state])
    return q_table[state][idx]

# Step 3 - greedy_action
import numpy as np

def greedy_action(q_table, state):
    """Return the action index with the highest Q value at the given state."""
    # TODO: return argmax over the action axis for this state's Q values
    return int(np.argmax(q_table[state]))

# Step 4 - sample_random_action
import gymnasium as gym 
def sample_random_action(action_space):
    # TODO: draw a uniformly random action from the given Gymnasium action space
    return int(action_space.sample())

# Step 5 - should_explore
def should_explore(epsilon, rng):
    """Return True with probability epsilon using the provided numpy Generator."""
    # TODO: draw a uniform sample from rng and compare it to epsilon
    if (rng.random() < epsilon):
        return True
    return False

# Step 6 - epsilon_greedy_action
import numpy as np

def epsilon_greedy_action(q_table, state, epsilon, action_space, rng):
    """Return an epsilon-greedy action for the given state."""
    # TODO: with prob epsilon explore via action_space, else pick a max-Q action (random among ties)
    if(should_explore(epsilon, rng)):
        action = sample_random_action(action_space)
    else:
        val = max(q_table[state])
        arr = []
        for act_idx, act_val in enumerate(q_table[state]):
            if act_val == val:
                arr.append(act_idx)

        action = rng.choice(arr)
    return int(action)

# Step 7 - decay_epsilon
def decay_epsilon(epsilon, decay_rate, min_epsilon):
    # TODO: return max(min_epsilon, epsilon * decay_rate)
    return max(epsilon*decay_rate, min_epsilon)

# Step 8 - td_target
def td_target(reward, gamma, q_table, next_state, done):
    # TODO: compute r + gamma * max_a Q(next_state, a), zeroing the bootstrap when done.
    if done:
        return reward
    return float(reward + gamma*max_q_value(q_table,next_state))

# Step 9 - td_error
def td_error(target, q_table, state, action):
    # TODO: return the TD error: target minus current Q(state, action)
    return float(target - q_table[state][action])

# Step 10 - q_learning_update
def q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma):
    # TODO: apply Q(s,a) += alpha * (target - Q(s,a)) in place and return the new Q value
    target = td_target(reward, gamma, q_table, next_state,done)
    error = td_error(target, q_table,state,action)
    q_table[state][action]= q_table[state][action] + alpha*error
    return float(q_table[state][action])

# Step 11 - interaction_step
def interaction_step(env, q_table, state, epsilon, alpha, gamma, rng):
    # TODO: select epsilon-greedy action, step env, apply Q-learning update, return (next_state, reward, done)
    action = epsilon_greedy_action(q_table, state, epsilon, env.action_space, rng)
    obs, reward, terminated, truncated, info = env.step(action)
    if( terminated or truncated):
        done= True
    else:
        done = False
    q_learning_update(q_table, state, action,reward, obs,done, alpha, gamma)
    return obs, float(reward), done

# Step 12 - run_training_episode
def run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps=200):
    # TODO: reset env, then repeatedly call interaction_step until done or max_steps, returning total reward.
    total_step = 0
    total_reward = 0.0
    done = False
    env.reset(seed=0); env.action_space.seed(0)
    state = 0
    while total_step < max_steps and not done:
        next_state, reward, done = interaction_step(env, q_table, state, epsilon, alpha,gamma,rng)
        state = next_state
        total_step +=1
        total_reward += reward

    return total_reward

# Step 13 - train_q_learning
import numpy as np
def train_q_learning(env, num_episodes, alpha=0.8, gamma=0.95, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.99, seed=0, max_steps=200):
    # TODO: train a Q-learning agent for num_episodes; return (q_table, returns)
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = init_q_table(num_states, num_actions)
    epsilon = epsilon_start
    env.reset(seed=seed)
    env.action_space.seed(seed)
    rng = np.random.default_rng(seed)
    episode_returns = []
    for i in range(num_episodes):
        ep_reward = run_training_episode(env, q_table, epsilon,alpha, gamma, rng, max_steps)
        episode_returns.append(ep_reward)
        epsilon = decay_epsilon(epsilon, epsilon_decay, epsilon_min)
    return q_table, episode_returns

# Step 14 - extract_greedy_policy
def extract_greedy_policy(q_table):
    # TODO: return a 1D int64 array mapping each state to its best (argmax) action.
    # best_actions = []
    # for i in range(len(q_table)):
    #     best_actions.append(np.argmax(np.array(q_table[i])))
    return np.argmax(q_table,1)

# Step 15 - run_greedy_episode
def run_greedy_episode(env, policy, seed=None, max_steps=200):
    """Run one greedy episode and return True if the goal was reached."""
    # TODO: reset env, follow policy[state] each step, return bool(success)
    state = 0
    env.reset(seed= seed)
    env.action_space.seed(seed=seed)
    reward = -0.1
    for i in range(max_steps):
        next_state, reward, terminated, truncated,_ = env.step(policy[state])
        state = next_state

        if(terminated or truncated):
            return reward > 0

    return reward > 0

# Step 16 - evaluate_success_rate (not yet solved)
# TODO: implement

