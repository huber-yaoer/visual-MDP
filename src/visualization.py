# -*- coding: utf-8 -*-
import numpy as np, scipy.misc
from matplotlib import pyplot as plt

'''
returns a heatmap of values with arrows pointing to the best
neighboring state at each position
'''
def visualize_values(mdp, values, policy, filename=None, title=None, vmin=None, vmax=None):
    states = mdp.states
    plt.clf()
    m = max(states, key=lambda x: x[0])[0] + 1
    n = max(states, key=lambda x: x[1])[1] + 1
    data = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            state = (i,j)
            if type(values) == dict:
                data[i][j] = values[state]
            else:
                data[i][j] = values[i][j]
            action = policy[state]
            ## if using all_reachable actions, pick the best one
            if type(action) == tuple:
                action = action[0]
            if action != None:
                x, y, w, h = arrow(i, j, action)
                plt.arrow(x,y,w,h,head_length=0.4,head_width=0.4,fc='k',ec='k')
                
    heatmap = plt.pcolor(data, cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.axis('off')

    if title:
        plt.title(title, fontsize=20)
    if filename:
        plt.savefig(filename + '.png')

    fig = plt.gcf()
    fig.set_size_inches(10,8)

    return fig

'''
helper function to return parameters of arrow
at position (i,j) for specified action
action is index into ['up', 'down', 'left', 'right']
'''
def arrow(i, j, action):
    ## up, down, left, right
    ## x, y, w, h
    arrows = {0: (.5,.95,0,-.4), 1: (.5,.05,0,.4), 2: (.95,.5,-.4,0), 3: (.05,.5,.4,0)}
    arrow = arrows[action]
    return j+arrow[0], i+arrow[1], arrow[2], arrow[3]


def read_img(path, cell_dim):
    img = scipy.misc.imread(path)
    img = scipy.misc.imresize(img, (cell_dim, cell_dim))
    return img

'''
visualizes map with sprites
rewards and terminal are same as arguments to MDP
each sprite is scaled to cell_dim
'''
def visualize_map(rewards, terminal, cell_dim=100):
    grass = read_img('sprites/grass.png', cell_dim)
    lava = read_img('sprites/lava.png', cell_dim)
    candy = read_img('sprites/candy.png', cell_dim)

    grass_mask = rewards >= 0
    candy_mask = terminal > 0

    channels = 3
    M, N = rewards.shape
    grid = np.zeros((M*cell_dim, N*cell_dim, channels)).astype('uint8')
    for i in range(M):
        for j in range(N):
            if grass_mask[i][j]:
                sprite = grass
            else:
                sprite = lava

            x_start = i*cell_dim
            x_end = (i+1)*cell_dim
            y_start = j*cell_dim
            y_end = (j+1)*cell_dim
            
            grid[x_start:x_end, y_start:y_end] = sprite[:,:,:channels]
    
            if candy_mask[i][j]:
                candy_pixels = np.tile((candy[:,:,-1]>0)[:,:,np.newaxis], (1, 1, 3))
                grid[x_start:x_end, y_start:y_end][candy_pixels] = candy[:,:,:3][candy_pixels]
                
    return grid 


