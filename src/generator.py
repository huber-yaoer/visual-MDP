import numpy as np

def make_large_env(reward_value=1, lava_value=-1):
    dim = 15
    reward_pos = (-1,-1)

    rewards = np.zeros((dim, dim))
    rewards[2:-2,2:8] = lava_value
    rewards[2:-2,9:14] = lava_value

    rewards[reward_pos] = 1

    terminal = np.zeros((dim,dim))
    terminal[reward_pos] = 1
    return rewards, terminal

def make_small_env(reward_value=1, lava_value=-1):
    dim = 10
    reward_pos = (5,6)

    rewards = np.zeros((dim, dim))
    rewards[2:-2,3:5] = lava_value
    rewards[2,3:7] = lava_value
    rewards[-2,3:7] = lava_value
    rewards[:2,6] = lava_value

    rewards[reward_pos] = reward_value

    terminal = np.zeros((dim,dim))
    terminal[reward_pos] = 1
    return rewards, terminal

