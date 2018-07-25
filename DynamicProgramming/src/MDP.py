# -*- coding: utf-8 -*-
import numpy as np

class MDP:

    '''
    rewards and terminal are np arrays of the same shape
    rewards gives value of each state
    terminal is binary denoting whether a given state is terminal
    确定地图（环境）情况
    '''
    def __init__(self, rewards, terminal):
        self.reward_map = rewards
        self.terminal_map = terminal
        self.shape = self.reward_map.shape

        self.M, self.N = self.shape
        self.states = [(i,j) for i in range(self.M) for j in range(self.N)]
        self.children = self.get_children( self.M, self.N )#孩子为考虑边界后的4邻域，即得到可走的下一个状态或位置的集合

        self.actions = [(-1,0),(1,0),(0,-1),(0,1)] #上下左右
        self.states = [(i,j) for i in range(self.shape[0]) for j in range(self.shape[1])]

    '''
    returns list of valid action indices
    '''
    def getActions(self):
        return [i for i in range(len(self.actions))]

    '''
    returns list of all possible states 
    expressed as tuples
    '''
    def getStates(self):
        return self.states

    '''
    position: (i,j) tuple
    action_ind: int 

    returns new location resulting from taking
    actions[action_ind] from position
    '''
    def transition(self, position, action_ind):
        action = self.actions[action_ind]
        candidate = tuple(map(sum, zip(position, action)))
        
        ## if new location is valid, 
        ## update the position
        if self.valid(candidate):
            position = candidate
        
        return position

    '''
    returns True iff position is reachable on map
    (cannot be outside range or have negative coordinates)
    '''
    def valid(self, position):
        x, y = position[0], position[1]
        if x >= 0 and x < self.shape[0] and y >= 0 and y < self.shape[1]:
            return True
        else:
            return False

    def reward(self, position):
        rew = self.reward_map[position]
        return rew

    '''
    position is (i,j) tuple
    returns True if terminal_map[i][j]==1
    '''
    def terminal(self, position):
        term = self.terminal_map[position]
        return term

    '''
    returns dict: pos --> children,
    where pos is a tuple denoting state (i,j) 
    and children is a list of states neighboring pos
    '''
    def get_children(self, M, N):
        #孩子为考虑边界后的4邻域，即得到可走的下一个状态或位置的集合
        children = {}
        for i in range(M):
            for j in range(N):
                pos = (i,j)
                children[pos] = []
                for di in range( max(i-1, 0), min(i+1, M-1)+1 ):
                    for dj in range( max(j-1, 0), min(j+1, N-1)+1 ):
                        child = (di, dj)
                        if pos != child and (i == di or j == dj):
                            children[pos].append( child )
        return children


    '''
    values is M x N map of predicted values
    '''
    def get_policy(self, values):
        #根据收益地图确定下一个状态，选择收益最大的一个
        policy = {}
        for state in self.states:
            reachable = self.children[state]
            selected = sorted(reachable, key = lambda x: values[x], reverse=True)
            policy[state] = selected
        return policy



