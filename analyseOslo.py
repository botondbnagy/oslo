#analysis of the Oslo model results

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import scipy
import pandas as pd
from tqdm import tqdm

datafolder = os.path.dirname(os.path.realpath(__file__)) + '/data/'
# load data
Ls = [4, 8, 16, 32, 64, 128, 256]
i_after_ss = 10**5
repeatNs = np.zeros(len(Ls), dtype=int) + 10 #number of repeats for each L


class osloAnalysis:

    def __init__(self, Ls=Ls, i_after_ss=i_after_ss, repeatNs=repeatNs, datafolder=datafolder, which='all'):
        self.Ls = np.array(Ls[::-1]) #invert for plotting
        self.steady_states = 2*self.Ls**2
        self.i_after_ss = i_after_ss
        self.datafolder = datafolder
        self.which = which
        self.repeatNs = np.array(repeatNs) #number of repeats for each L
        self.colors = plt.cm.viridis(np.linspace(0, 1, len(self.Ls)*2))[::2] #get a color for each L from viridis
        plt.rc('axes', prop_cycle=(plt.cycler('color', self.colors))) #change default color cycle
    
    def readData(self, L):
        '''read in all avalanches and heights for a given L'''
        
        #read in avalanches_X.csv and heights_X.csv from each run
        repeats = self.repeatNs[self.Ls==L][0]
        steady_state = 2*L**2
        iterations = steady_state + i_after_ss
        all_avalanches = np.zeros((repeats, i_after_ss))
        all_heights = np.zeros((repeats, iterations))
        
        for run_id in range(repeats):
            folder = datafolder+str(L)+'/'
            #all_avalanches[run_id] = np.loadtxt(folder + 'avalanches_'+str(run_id)+'.csv', delimiter=',')[:i_after_ss]
            #all_heights[run_id] = np.loadtxt(folder + 'heights_'+str(run_id)+'.csv', delimiter=',')
            cur_aval = pd.read_csv(folder + 'avalanches_'+str(run_id)+'.csv', header=None, usecols=[0], dtype=int).values[:i_after_ss]
            cur_height = pd.read_csv(folder + 'heights_'+str(run_id)+'.csv', header=None, usecols=[0], dtype=int).values
            #convert to numpy array
            all_avalanches[run_id] = np.array(cur_aval).flatten()
            all_heights[run_id] = np.array(cur_height).flatten()
        
        t_c_list = pd.read_csv(folder + 't_c.csv', header=None, usecols=[0], dtype=int).values
        t_c_list = np.array(t_c_list).flatten()

        return all_avalanches, all_heights, t_c_list
    
    def plotAllAvalanches(self, show=False):
        '''barplot avalanche sizes vs time'''
        print('plotting avalanches')
        plt.figure(figsize=(10, 5))
        for L in self.Ls:
            avalanches = self.readData(L)[0]
            first_aval = avalanches[0]
            print('L = ', L, 'mean = ', np.mean(first_aval), '+/-', np.std(first_aval))
            first_aval = first_aval[:1000]
            plt.bar(np.arange(len(first_aval)), first_aval, label='L='+str(L)) #plot first run only
        plt.xlabel('Time')
        plt.ylabel('Avalanche size')
        #plt.title('Avalanche sizes over time for different L')
        plt.legend()
        plt.savefig(self.datafolder+'avalanches.png', dpi=300)
        if show:
            plt.show()

    def plotMeanHeights(self, runs='all', show=False, scale='linear'):
        # plot heights vs time
        plt.figure(figsize=(10, 7))
        for l, L in enumerate(self.Ls):
            heights = self.readData(L)[1]
            if runs == 'one':
                heights = heights[0]
                mean_heights = heights

            else:
                mean_heights = np.mean(heights, axis=0) #average over repeats

            avg_height = np.mean(mean_heights[self.steady_states[l]:])
            err_height = np.std(mean_heights[self.steady_states[l]:])

            print('L = ', L, 'mean = ', avg_height, '+/-', err_height)
            plt.plot(np.arange(len(mean_heights)), mean_heights, label='L='+str(L), alpha=0.8, linewidth=0.8)
            
        if scale == 'loglog':
            plt.yscale('log')
            plt.xscale('log')
        plt.xlabel('Time')
        plt.ylabel('Height of pile')
        #plt.title('Height of pile (average) over time for different L')
        plt.legend()
        plt.savefig(self.datafolder+'meanheights_'+runs+scale+'.png', dpi=300)
        if show:
            plt.show()
    
    def quad(self, x, a):
            x = np.array(x)
            return a * x**2
    
    def scal(self, L, a_0, a_1, w_1):
        return a_0 * L * (1 - a_1 * L ** (-w_1))
    
    def log(self, x, a, b):
        return a * np.log(x) + b
    
    def plot_t_cs(self, fit=True, show=False):
        # plot t_c vs L
        plt.figure(figsize=(10, 7))
        avg_t_cs = np.zeros(len(self.Ls))
        errs = np.zeros(len(self.Ls))
        for i, L in enumerate(self.Ls):
            folder = datafolder+str(L)+'/'
            t_cs = np.loadtxt(folder + 't_c.csv', delimiter=',')
            avg_t_cs[i] = np.mean(t_cs)
            errs[i] = np.std(t_cs)
            print('L = ', L, 'mean = ', avg_t_cs[i], '+/-', errs[i])

        if fit:
            #fit quadratic to points
            popt, pcov = scipy.optimize.curve_fit(self.quad, self.Ls, avg_t_cs, sigma=errs)
            Xs = np.linspace(0, np.max(self.Ls), 100)
            plt.plot(Xs, self.quad((Xs), *popt), color='gray', label='quadratic fit')
            print('fit parameters: ', popt)

        plt.errorbar(self.Ls, avg_t_cs, yerr=errs, fmt='.', color='k', capsize=3)
        
        #plt.xscale('log')
        #plt.yscale('log')
        plt.xticks(np.arange(0, np.max(self.Ls)+1, 32))
        plt.xlabel('System size L')
        plt.ylabel('Average crossover time')
        plt.title('Crossover time and system size')
        plt.legend()
        plt.grid()    
        plt.savefig(self.datafolder+'t_c.png', dpi=300)
        if show:
            plt.show()

    def collapseHeights(self, show=False, scale='linear'):
        # plot heights vs time
        plt.figure(figsize=(10, 7))
        for l, L in enumerate(self.Ls):
            heights = self.readData(L)[1]
            mean_heights = np.mean(heights, axis=0)
            mod_heights = mean_heights / L
            Xs = np.arange(len(mod_heights))
            mod_Xs = Xs / L**2
            plt.plot(mod_Xs, mod_heights, label='L='+str(L), alpha=0.8, linewidth=0.8)

        if scale == 'loglog':
            plt.yscale('log')
            plt.xscale('log')
        plt.xlabel('Time / $L^2$')
        plt.ylabel('Height / L')
        plt.xlim(0, 5)
        #plt.title('Height of pile (average) over time for different L')
        plt.legend()
        plt.savefig(self.datafolder+'meanheights_collapsed'+scale+'.png', dpi=300)
        if show:
            plt.show()

    def scalingCorrections(self, show=False, fit=True):
        fig, ax = plt.subplots(1, 3, figsize=(15, 7))
        avg_heights = np.zeros(len(self.Ls))
        err_heights = np.zeros(len(self.Ls))

        for l, L in enumerate(self.Ls):
            avalanches, heights, t_cs = self.readData(L)
            t_c = t_cs[0]
            heights_ss = heights[0][t_c:]
            
            avg_heights[l] = np.mean(heights_ss)
            err_heights[l] = np.std(heights_ss)
            manual_std= np.sqrt(np.sum(heights_ss**2) / len(heights_ss) - avg_heights[l]**2)

            print('L = {:.2f}, mean = {:.2f} +/- {:.2f} (manual std = {:.2f})'.format(L, avg_heights[l], err_heights[l], manual_std))

            height_prob = np.bincount(heights_ss.astype(int))
            height_prob = height_prob / np.sum(height_prob)

            ax[0].plot(np.arange(len(height_prob)), height_prob, label='L='+str(L), alpha=0.8, linewidth=0.8)

        if fit:
            #divide by L
            scaled_heights = avg_heights #/ self.Ls
            scaled_err_heights = err_heights #/ self.Ls

            #fit to scaling function
            popt, pcov = scipy.optimize.curve_fit(self.scal, self.Ls, scaled_heights, sigma=scaled_err_heights)
            Xs = np.linspace(0, np.max(self.Ls), 100)
            ax[1].plot(Xs, self.scal((Xs), *popt), color='gray', label='scaling fit')
            print('fit parameters: a_0={:.2f} +/- {:.2f}, a_1={:.2f} +/- {:.2f}, w_1={:.2f} +/- {:.2f}'.format(popt[0], np.sqrt(pcov[0,0]), popt[1], np.sqrt(pcov[1,1]), popt[2], np.sqrt(pcov[2,2])))

            #fit log to std against L
            popt, pcov = scipy.optimize.curve_fit(self.log, self.Ls, scaled_err_heights)
            Xs = np.linspace(0, np.max(self.Ls), 100)
            ax[2].plot(Xs, self.log((Xs), *popt), color='gray', label='log fit')
            print('fit parameters: a={:.2f} +/- {:.2f}, b={:.2f} +/- {:.2f}'.format(popt[0], np.sqrt(pcov[0,0]), popt[1], np.sqrt(pcov[1,1])))
        

        #plot std against L
        ax[2].scatter(self.Ls, scaled_err_heights, color='k', s=10)
        ax[2].set_xlabel('System size L')
        ax[2].set_ylabel('Standard deviation')
        #ax[2].set_title('Standard deviation of height of pile over time for different L')
        ax[2].grid()


        ax[0].set_xlabel('Height')
        ax[0].set_ylabel('Probability')
        #ax[0].set_title('Probability distribution of heights')
        ax[0].legend()

        ax[1].errorbar(self.Ls, avg_heights, yerr=err_heights, fmt='.', color='k', capsize=3)
        ax[1].set_xlabel('System size L')
        ax[1].set_ylabel('Average height')
        #ax[1].set_title('Average height of pile over time for different L')
        ax[1].grid()
        plt.savefig(self.datafolder+'scalingcorrections.png', dpi=300)


        if show:
            plt.show()

        

            




analyser = osloAnalysis()
#analyser.plotAllAvalanches(show=False)
#analyser.plotMeanHeights(runs='one', show=True, scale='loglog')
#analyser.plot_t_cs(fit=True, show=True)
#analyser.collapseHeights(show=True, scale='linear')
analyser.scalingCorrections(show=True)