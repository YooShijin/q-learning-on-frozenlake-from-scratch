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

# Step 6 - epsilon_greedy_action (not yet solved)
# TODO: implement

# Step 7 - decay_epsilon (not yet solved)
# TODO: implement

# Step 8 - td_target (not yet solved)
# TODO: implement

# Step 9 - td_error (not yet solved)
# TODO: implement

# Step 10 - q_learning_update (not yet solved)
# TODO: implement

# Step 11 - interaction_step (not yet solved)
# TODO: implement

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

