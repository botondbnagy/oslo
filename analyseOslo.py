#analysis of the Oslo model results

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import scipy

datafolder = os.path.dirname(os.path.realpath(__file__)) + '/data/run0/'
# load data
Ls = [4, 8, 16, 32, 64, 128, 256]

Ls = Ls[::-1] #invert for plotting
#every second color from viridis
colors = plt.cm.viridis(np.linspace(0, 1, len(Ls)*2))[::2]
plt.rc('axes', prop_cycle=(plt.cycler('color', colors))) #change default color cycle


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

def plot_t_cs():
    # plot t_c vs L
    plt.figure(figsize=(10, 7))
    avg_t_cs = np.zeros(len(Ls))
    errs = np.zeros(len(Ls))
    for i, L in enumerate(Ls):
        folder = datafolder+str(L)+'/'
        t_cs = np.loadtxt(folder + 't_c.csv', delimiter=',')
        avg_t_cs[i] = np.mean(t_cs)
        errs[i] = np.std(t_cs)

    #fit quadratic to points
    def func(x, a):
        x = np.array(x)
        return a * x**2

    popt, pcov = scipy.optimize.curve_fit(func, Ls, avg_t_cs, sigma=errs)

    Xs = np.linspace(0, np.max(Ls), 100)
    plt.plot(Xs, func((Xs), *popt), color='gray', label='quadratic fit')
    print('fit parameters: ', popt)

    plt.errorbar(Ls, avg_t_cs, yerr=errs, fmt='o', color='k')
    
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xticks(np.arange(0, np.max(Ls)+1, 32))
    plt.xlabel('System size L')
    plt.ylabel('Average crossover time')
    plt.title('Crossover time and system size')
    plt.legend()
    plt.grid()    
    plt.savefig(datafolder+'t_c.png', dpi=300)
    plt.show()

