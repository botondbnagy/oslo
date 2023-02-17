#analysis of the Oslo model results

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import scipy
import pandas as pd
from tqdm import tqdm
from logbin import logbin
from scipy.optimize import curve_fit

datafolder = os.path.dirname(os.path.realpath(__file__)) + '/data/' + '1mil/'
# load data
Ls = [4, 8, 16, 32, 64, 128, 256]
i_after_ss = 10**6 #number of iterations after steady state
repeatNs = np.zeros(len(Ls), dtype=int) + 1 #number of repeats for each L
show=False


class osloAnalysis:

    def __init__(self, Ls=Ls, i_after_ss=i_after_ss, repeatNs=repeatNs, datafolder=datafolder, which='all'):
        self.Ls = np.array(Ls[::-1]) #invert for plotting
        self.steady_states = 2*self.Ls**2
        self.i_after_ss = i_after_ss
        self.datafolder = datafolder
        self.which = which
        self.repeatNs = np.array(repeatNs) #number of repeats for each L
        #color scale from viridis
        self.colors = plt.cm.viridis(np.linspace(0, 1, len(self.Ls)))
        plt.rc('axes', prop_cycle=(plt.cycler('color', self.colors))) #change default color cycle
    
    def readData(self, L):
        '''read in all avalanches and heights for a given L'''
        
        #read in avalanches_X.csv and heights_X.csv from each run
        repeats = self.repeatNs[self.Ls==L][0]
        steady_state = L*(L+1)
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
    
    def plotAllAvalanches(self, show=False, cutoff=1000):
        '''barplot avalanche sizes vs time'''
        print('plotting avalanches')
        plt.figure(figsize=(10, 5))
        for L in self.Ls:
            avalanches = self.readData(L)[0]
            first_aval = avalanches[0]
            print('L = ', L, 'mean = ', np.mean(first_aval), '+/-', np.std(first_aval))
            first_aval = first_aval[:cutoff]
            plt.bar(np.arange(len(first_aval)), first_aval, label='L='+str(L)) #plot first run only
        plt.xlabel('Time')
        plt.ylabel('Avalanche size')
        #plt.title('Avalanche sizes over time for different L')
        plt.legend()
        plt.savefig(self.datafolder+'avalanches.png', dpi=300)
        if show:
            plt.show()

    def plotMeanHeights(self, runs='all', show=False, scale='linear'):
        print('plotting mean heights')
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
        print('plotting t_c vs L')
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
        print('plotting heights data collapse')
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
        print('plotting scaling corrections')
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

    def heightProb(self, show=False):
        print('plotting height probability')
        plt.figure(figsize=(10, 7))
        for l, L in enumerate(self.Ls):
            avalanches, heights, t_cs = self.readData(L)
            t_c = t_cs[0]
            
            heights_ss = heights[0][t_c:]
            std = np.std(heights_ss)
            heights_ss = ( heights_ss / L )
            #scale by variance
            heights_ss = (( heights_ss - np.mean(heights_ss) ) / std + 2)*100

           
            #plot a histogram
            #plt.hist(heights_ss, bins=100, density=True, label='L='+str(L), alpha=1)
            
            #plot data as pdf
            height_prob = np.bincount(heights_ss.astype(int))
            height_prob = height_prob / np.sum(height_prob)

            plt.plot(np.arange(len(height_prob)), height_prob, label='L='+str(L), alpha=0.8, linewidth=0.8)

            
            

            #plot a gaussian
            #x = np.linspace(0, 2, 100)
            #plt.plot(x, scipy.stats.norm.pdf(x, mean, std), label='L='+str(L), alpha=0.8, linewidth=0.8)

            plt.xticks(np.arange(0, 2.1, 0.1))
            #plt.xlim(-0.1, 0.1)

        #plot standard deviation against L
        

        plt.xlabel('Height / L')
        plt.ylabel('Probability')

        plt.legend()
        plt.savefig(self.datafolder+'heightprob.png', dpi=300)
        if show:
            plt.show()
        
    def avalancheProb(self, scale=1, show=False, step=1, skip=None, collapse=False):
        print('plotting avalanche probability')
        plt.figure(figsize=(10, 7))
        D, tau = 2.25, 1.55
        for l, L in enumerate(self.Ls[:None:step]):
            print('L =', L)
            
            avalanches, heights, t_cs = self.readData(L)
            #take all runs from their respective t_c
            avalanches_ss = avalanches[:,t_cs[0]:]
            avalanches_ss = avalanches_ss.flatten().astype(int)
            
            bincentres, bincounts = logbin(avalanches_ss, scale=scale)
            if collapse:
                bincounts = bincounts * (bincentres**tau)
                bincentres = bincentres / (L**D)

            
            plt.plot(bincentres, bincounts, label='L='+str(L), alpha=0.9)

        binning = 'log'
        if scale == 1:
            binning = 'linear'

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('$s/L^D$')
        plt.ylabel('$s^{ \\tau_s}P(s;L)$')
        #plt.xlim(0, 1000)
        #plt.xticks(bincentres)
        plt.legend()
        fname = 'avalancheprob_'+binning+'.png'
        if collapse:
            fname = 'avalancheprob_'+binning+'_collapse.png'
        plt.savefig(self.datafolder+fname, dpi=300)
        if show:
            plt.show()

    def avalanche_moment_k(self, ks, show=False):
        '''measure the kth moment of the avalanche distribution'''  
        print('plotting avalanche moments')
        moments = np.zeros((len(self.Ls), len(ks)))
        for l, L in enumerate(self.Ls):
            avalanches, heights, t_cs = self.readData(L)
            t_c = t_cs[0]

            #take first run from its t_c
            avalanches_ss = avalanches[0][t_c:]
            #flatten and convert to int
            avalanches_ss = avalanches_ss.astype(int)
            #take the kth moment
            for k in ks:
                moment = np.mean(avalanches_ss**k)
                moments[l, k-1] = moment
                print('L =', L, 'k =', k, 'moment =', moment)
        
        #plot the moments in one plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        log_moments = np.log(moments)
        log_Ls = np.log(self.Ls)
        
        for k in ks:
            #fit exp to the data
            #k = 1 popt = [0.99976387 1.00006119]
            #k = 2 popt = [0.28517019 3.21035064]
            #k = 3 popt = [0.11549297 5.45150957]
            #k = 4 popt = [0.09807589 7.56493738]
            #fit linear to log-log data
            popt, pcov = curve_fit(lambda x, a, b: a*x + b, log_Ls, log_moments[:,k-1])
            print('k={:d}, a={:.2e} +/- {:.2e}, b={:.2e} +/- {:.2e}'.format(k, popt[0], np.sqrt(pcov[0,0]), popt[1], np.sqrt(pcov[1,1])))
            ax.plot(log_Ls, log_moments[:,k-1], '.', label='k='+str(k), alpha=0.9, color=self.colors[k-1])
            ax.plot(log_Ls, popt[0]*log_Ls + popt[1], alpha=0.9, linestyle='--', color=self.colors[k-1])
        ax.set_xlabel('log(L)')
        ax.set_ylabel('$log(\\langle s^k \\rangle)$')
        
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.datafolder+'avalanchemoment.png', dpi=300)
        if show:
            plt.show()

analyser = osloAnalysis()
analyser.plotAllAvalanches(show=show)
analyser.plotMeanHeights(runs='one', show=show, scale='loglog')
analyser.plot_t_cs(fit=True, show=show)
analyser.collapseHeights(show=show, scale='linear')
analyser.scalingCorrections(show=show)
analyser.heightProb(show=show)
analyser.avalancheProb(show=show, scale=1.25, step=1, skip=None, collapse=True)
analyser.avalanche_moment_k(ks=[1,2,3,4], show=show)