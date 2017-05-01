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

env = {'rewards': world, 'terminal': terminal}

