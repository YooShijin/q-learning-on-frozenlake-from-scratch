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

# Step 12 - run_training_episode (not yet solved)
# TODO: implement

# Step 13 - train_q_learning (not yet solved)
# TODO: implement

# Step 14 - extract_greedy_policy (not yet solved)
# TODO: implement

# Step 15 - run_greedy_episode (not yet solved)
# TODO: implement

# Step 16 - evaluate_success_rate (not yet solved)
# TODO: implement

