# Complexity project implementation of Oslo model
# Botond Branyicskai-Nagy
# January 2022

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
import time
from tqdm import tqdm


class Oslo:
    def __init__(self, L, p=0.5, i_after_ss=10**3, thresholds=[1, 2], doAnimation=False, repeatN=5):
        self.L = L
        self.p = p
        self.i_after_ss = i_after_ss
        self.steady_state = 2*L**2
        self.iterations = self.steady_state + i_after_ss
        self.config = np.zeros(L, dtype=int)
        self.thresholds = np.array(thresholds)
        self.z_th = self.randomThresholds()
        self.avalanches = np.zeros(self.iterations)
        self.heights = np.zeros(self.iterations)
        self.doAnimation = doAnimation
        self.t_c = 0
        self.t_c_reached = False
        self.repeatN = repeatN
        self.t_c_list = np.zeros(self.repeatN)
        self.dir = os.path.dirname(os.path.realpath(__file__)) + '/data/' + str(self.L) + '/' #data directory named after L

    def randomThresholds(self):
        # create list of random threshold slopes from threshold_slopes with probability p
        z_th = np.random.choice(self.thresholds, size=self.L, p=[self.p, 1-self.p])

        return z_th
    
    def initialize(self):
        # initialize with empty configuration
        self.config = np.zeros(self.L, dtype=int)

        # create list of random initial threshold slopes
        self.z_th = self.randomThresholds()

        if self.doAnimation:
            # initialize animation
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(0, self.L)
            self.ax.set_ylim(0, self.L*2+2)
            self.ax.set_xlabel('Cell')
            self.ax.set_ylabel('Height')
            self.ax.set_title('Oslo model')
            self.ax.grid()
            self.ax.set_xticks(np.arange(0, self.L+1, 1))
            self.ax.set_yticks(np.arange(0, self.L*2+2, 1))
            self.edges = np.arange(self.L+1)
            self.ax.stairs(self.config, self.edges, color='black')
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)

    def getSlopes(self, i_max):
        # calculate local slopes in configuration
        slopes = self.config[:i_max+1] - np.concatenate((self.config[1:], [0]))[:i_max+1]
        return slopes

    def drive(self):
        # add one grain to the first cell
        self.config[0] += 1

    def relax(self, i):
        avalanche_size = 0
        i_max = 1 # index of last cell to be checked for relaxation

        # relax the configuration while there are cells with slope greater than local threshold slope
        while np.any(self.getSlopes(i_max) > self.z_th[:i_max+1]):
            slopes = self.getSlopes(i_max)
            to_relax = np.where(slopes > self.z_th[:i_max+1])[0] # to_relax is array of cells that need to be relaxed, ie. cells with slope greater than local threshold slope
            self.config[to_relax] -= 1
            self.z_th[to_relax] = self.randomThresholds()[to_relax] # reset threshold slope of relaxed cells

            # if cell is not last cell, add grain to cell to the right, otherwise don't add grain
            
            to_add = to_relax + 1

            if to_add[-1] < self.L:
                self.config[to_add] += 1
            
            else:
                if self.t_c_reached == False:
                    self.t_c_reached = True
                    #print('First grain left the system after {} iterations'.format(i))
                    self.t_c = i #cross-over time
                
                self.config[to_add[:-1]] += 1 #add grain to to_add cells, except last

            # count avalanche size
            avalanche_size += len(to_relax)

            # increase i_max unless it is already at the end of the array
            i_max = to_relax[-1] + 1

        self.avalanches[i] = avalanche_size
        self.heights[i] = self.config[0] #height of first cell

    def animate(self, i, s=0):
        # update animation
        self.ax.clear()
        self.ax.set_xlim(0, self.L)
        self.ax.set_ylim(0, self.L*2+2)
        self.ax.set_xlabel('Cell')
        self.ax.set_ylabel('Height')
        self.ax.set_title('Oslo model')
        #self.ax.grid()
        #self.ax.set_xticks(np.arange(0, self.L+1, 1))
        #self.ax.set_yticks(np.arange(0, self.L*2+2, 1))
        self.ax.stairs(self.config, self.edges, color='black')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.0001)

    def run(self, n):
        #initialize configuration
        self.initialize()

        # run model
        for i in tqdm(range(self.iterations), desc='Running model {}/{} with L = {}'.format(n+1, self.repeatN, self.L)): 
            self.drive()

            # if self.doAnimation:
            #    self.animate(i)

            self.relax(i)

            if self.doAnimation:
                self.animate(i)

        #get steady state values
        self.avalanches_ss = self.avalanches[self.t_c:]
        self.heights_ss = self.heights[self.t_c:]
        
        if self.doAnimation:
            self.anim = animation.FuncAnimation(self.fig, self.animate, frames=self.iterations, interval=1, repeat=False)
            plt.show()
            plt.close()

    def save(self, run_id, dir='data/'):
        # save data
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        #save avalanche sizes (after steady state), heights for each iteration as csv
        np.savetxt(dir + 'avalanches_'+str(run_id)+'.csv', self.avalanches_ss, delimiter=',')
        np.savetxt(dir + 'heights_'+str(run_id)+'.csv', self.heights, delimiter=',')

        if run_id == self.repeatN-1:
            #after last run save list of t_c values as csv
            np.savetxt(dir + 't_c.csv', self.t_c_list, delimiter=',')

    def repeat(self):
        #repeat model run N times
        for n in range(self.repeatN):
            self.run(n)
            self.t_c_list[n] = self.t_c
            self.save(n, dir=self.dir)
            self.t_c = 0
            self.t_c_reached = False


if __name__ == '__main__':
    # run model for system sizes L
    start, stop = 2, 2 #eg for [8, 16, ... 256] use 3, 8
    repeats = 10 #number of times to repeat model for each system size
    i_after_ss = 10**5 #number of iterations after definite steady state
    L = np.logspace(start, stop, stop-start+1, base=2, dtype=int)
    print('Running model for system sizes: {}, each repeated {} times'.format(L, repeats))
    for l in L:
        repeatN = repeats
        model = Oslo(L=l, i_after_ss=i_after_ss, repeatN=repeatN, doAnimation=False)
        model.repeat()

    



