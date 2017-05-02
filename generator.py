import numpy as np

dim = 15

world = np.zeros((dim, dim))
world[-2:,:] = 1
world[:2,:] = 1
world[:,7:9] = 1
world[:,:2] = 1
world[:,13:] = 1

world -= 1
world[-1,-1] = 1

terminal = np.zeros((dim,dim))
terminal[-1,-1] = 1

default_env = {'rewards': world, 'terminal': terminal}

def make_env(reward_location, reward_value, lava_value):
	dim = 10

	world = np.ones((dim, dim)) * lava_value
	world[-2:,:] = 0
	world[:2,:] = 0
	world[:,:1] = 0

	world[0,-1] = reward_value

	terminal = np.zeros((dim,dim))
	terminal[0,-1] = 1
	return world, terminal

