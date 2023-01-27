#analysis of the Oslo model results

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import scipy
# load data
Ls = [4, 8, 16, 32, 64, 128, 256]

Ls = Ls[::-1] #invert for plotting
#every second color from viridis
colors = plt.cm.viridis(np.linspace(0, 1, len(Ls)*2))[::2]
plt.rc('axes', prop_cycle=(plt.cycler('color', colors))) #change default color cycle

datafolder = 'data/'

#read in all avalanches and heights from all L and plot them
def plotAllAvalanches():
    # barplot avalanche sizes vs time
    plt.figure(figsize=(10, 5))
    for L in Ls:
        folder = datafolder+str(L)+'/'
        avalanches = np.loadtxt(folder + 'avalanches.csv', delimiter=',')
        plt.bar(np.arange(len(avalanches)), avalanches, label='L=' + str(L))
    plt.xlabel('Time')
    plt.ylabel('Avalanche size')
    plt.title('Avalanche sizes over time for different L')
    plt.legend()
    plt.savefig(datafolder+'avalanches.png', dpi=300)
    plt.show()

def plotAllHeights():
    # plot heights vs time
    plt.figure(figsize=(10, 7))
    for L in Ls:
        folder = datafolder+str(L)+'/'
        heights = np.loadtxt(folder + 'heights.csv', delimiter=',')
        #log scale
        plt.plot(np.arange(len(heights)), heights, label='L=' + str(L), alpha=0.8, linewidth=0.8)
        

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Time')
    plt.ylabel('Height of pile')
    plt.title('Height of pile over time for different L')
    plt.legend()
    plt.savefig(datafolder+'heights.png', dpi=300)
    plt.show()


plotAllHeights()