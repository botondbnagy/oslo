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

datafolder = os.path.dirname(os.path.realpath(__file__)) + '/data/' + '100ks/'
# load data
Ls = [4, 8, 16, 32, 64, 128, 256]
i_after_ss = 10**5 #number of iterations after steady state
repeatNs = np.zeros(len(Ls), dtype=int) + 1 #number of repeats for each L
show=True


class osloAnalysis:

    def __init__(self, Ls=Ls, i_after_ss=i_after_ss, repeatNs=repeatNs, datafolder=datafolder, which='all'):
        self.Ls = np.array(Ls[::-1]) #invert for plotting
        #self.steady_states = 2*self.Ls**2
        self.steady_states = self.Ls*(self.Ls+1)
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
        steady_state = self.steady_states[self.Ls==L][0]
        iterations = steady_state + i_after_ss
        all_avalanches = np.zeros((repeats, i_after_ss))
        all_heights = np.zeros((repeats, iterations))
        all_slopes = np.zeros((repeats, iterations, L))

        for run_id in range(repeats):
            folder = datafolder+str(L)+'/'
            #all_avalanches[run_id] = np.loadtxt(folder + 'avalanches_'+str(run_id)+'.csv', delimiter=',')[:i_after_ss]
            #all_heights[run_id] = np.loadtxt(folder + 'heights_'+str(run_id)+'.csv', delimiter=',')
            cur_aval = pd.read_csv(folder + 'avalanches_'+str(run_id)+'.csv', header=None, usecols=[0], dtype=int).values[:i_after_ss]
            cur_height = pd.read_csv(folder + 'heights_'+str(run_id)+'.csv', header=None, usecols=[0], dtype=int).values
            cur_slope = pd.read_csv(folder + 'slopes_'+str(run_id)+'.csv', header=None, dtype=int).values
            #convert to numpy array
            all_avalanches[run_id] = np.array(cur_aval).flatten().astype(int)
            all_heights[run_id] = np.array(cur_height).flatten().astype(int)
            all_slopes[run_id] = np.array(cur_slope).astype(int)
        
        t_c_list = pd.read_csv(folder + 't_c.csv', header=None, usecols=[0], dtype=int).values
        t_c_list = np.array(t_c_list).flatten()

        return all_avalanches, all_heights, all_slopes, t_c_list
    
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

            mean_heights = mean_heights.astype(int)[::10]
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
            return a * x * (x + 1)
    
    def scal(self, L, a_0, a_1, w_1):
        return a_0 * L * (1 - a_1 * L ** (-w_1))
    
    def scal_z(self, L, a_0, a_1, w_1):
        return a_0 * (1 - a_1 * L ** (-w_1))
    
    def scal_z_std(self, L, a, b):
        return a * L ** (b-1)
        

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
            print('fit: {} * L * (L + 1)'.format(popt[0]), 'with error', np.sqrt(pcov[0][0]))

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
        fig, ax = plt.subplots(1, 2, figsize=(12, 7))
        fig_hp, ax_hp = plt.subplots(1, 1, figsize=(10, 7))
        avg_heights = np.zeros(len(self.Ls))
        err_heights = np.zeros(len(self.Ls))

        for l, L in enumerate(self.Ls):
            avalanches, heights, slopes, t_cs = self.readData(L)
            t_c = t_cs[0]
            heights_ss = heights[0][t_c:]
            
            avg_heights[l] = np.mean(heights_ss)
            err_heights[l] = np.std(heights_ss)
            manual_std= np.sqrt(np.sum(heights_ss**2) / len(heights_ss) - avg_heights[l]**2)

            print('L = {:.2f}, mean = {:.2f} +/- {:.2f} (manual std = {:.2f})'.format(L, avg_heights[l], err_heights[l], manual_std))

            height_prob = np.bincount(heights_ss.astype(int))
            height_prob = height_prob / np.sum(height_prob)
            
            ax_hp.plot(np.arange(len(height_prob)), height_prob, label='L='+str(L), alpha=0.8, linewidth=0.8)
        
        ax_hp.set_xlabel('Height')
        #ax_hp.set_title('Height distribution for different L')
        ax_hp.legend()
        fig_hp.savefig(self.datafolder+'height_prob.png', dpi=300)


        if fit:
            #divide by L
            scaled_heights = avg_heights #/ self.Ls
            scaled_err_heights = err_heights #/ self.Ls

            #fit to scaling function
            popt, pcov = scipy.optimize.curve_fit(self.scal, self.Ls, scaled_heights, sigma=scaled_err_heights)
            Xs = np.linspace(0, np.max(self.Ls), 100)
            ax[0].plot(Xs, self.scal((Xs), *popt), color='gray', label='scaling fit')
            print('fit parameters: a_0={:.2f} +/- {:.2f}, a_1={:.2f} +/- {:.2f}, w_1={:.2f} +/- {:.2f}'.format(popt[0], np.sqrt(pcov[0,0]), popt[1], np.sqrt(pcov[1,1]), popt[2], np.sqrt(pcov[2,2])))
        
            #fit a*L**b to std dev of height with L
            popt, pcov = scipy.optimize.curve_fit(lambda x, a, b: a*x**b, self.Ls, scaled_err_heights)
            Xs = np.linspace(0, np.max(self.Ls), 100)
            ax[1].plot(Xs, popt[0]*Xs**popt[1], color='gray', label='scaling fit')
            print('fit parameters: a={:.2f} +/- {:.2f}, b={:.2f} +/- {:.2f}'.format(popt[0], np.sqrt(pcov[0,0]), popt[1], np.sqrt(pcov[1,1])))

        #plot std against L
        ax[1].scatter(self.Ls, scaled_err_heights, color='k', s=10)
        ax[1].set_xlabel('System size L')
        ax[1].set_ylabel('Standard deviation')
        #ax[1].set_title('Standard deviation of height of pile over time for different L')
        ax[1].grid()

        ax[0].errorbar(self.Ls, avg_heights, yerr=err_heights, fmt='.', color='k', capsize=3)
        ax[0].set_xlabel('System size L')
        ax[0].set_ylabel('Average height')
        #ax[0].set_title('Average height of pile over time for different L')
        ax[0].grid()
        fig.savefig(self.datafolder+'scalingcorrections.png', dpi=300)

        if show:
            plt.show()

    def heightProb(self, show=False):
        print('plotting height probability')
        plt.figure(figsize=(10, 7))
        avg_heights = np.zeros(len(self.Ls))
        err_heights = np.zeros(len(self.Ls))
        #theoretical gaussian distribution
        x = np.linspace(-5, 5, 100)
        mean, std = 0, 1
        gaussian = scipy.stats.norm.pdf(x, mean, std)

        for l, L in enumerate(self.Ls):
            avalanches, heights, slopes, t_cs = self.readData(L)
            t_c = t_cs[0]
            heights_ss = heights[0][t_c:]
            
            avg_heights[l] = np.mean(heights_ss)
            err_heights[l] = np.std(heights_ss)
            manual_std= np.sqrt(np.sum(heights_ss**2) / len(heights_ss) - avg_heights[l]**2)

            print('L = {:.2f}, mean = {:.2f} +/- {:.2f} (manual std = {:.2f})'.format(L, avg_heights[l], err_heights[l], manual_std))
            
            #rescale to same mean and std
            heights_sc = (heights_ss - avg_heights[l]) / err_heights[l]
            
            n_bins = np.arange(np.amin(heights_ss)-0.5, np.amax(heights_ss)+1.5, 1)
            
            n_bins = (n_bins-avg_heights[l]) / err_heights[l]
            

            #plot pdf of heights
            counts, bins = np.histogram(heights_sc, bins=n_bins, density=True)
            height_prob = counts
            scaled_h = bins[:-1] + 0.5*np.diff(bins)
            #check if distribution is gaussian
            #print('KS test: ', scipy.stats.kstest(height_prob, 'norm', args=(0, 1)))
            #chi2 test for normal distribution
            # test_gaussian = scipy.stats.norm.pdf(scaled_h, 0, 1)
            # test_gaussian = test_gaussian / np.sum(test_gaussian) * np.sum(height_prob)
            # print('chi2 test: ', scipy.stats.chisquare(height_prob, test_gaussian, ddof=2))
            plt.plot(scaled_h, height_prob, label='L='+str(L), alpha=0.8, linewidth=0.8)

            
        
        #plot the theoretical gaussian
        plt.plot(x, gaussian, label='Gaussian', color='k', linewidth=0.8)
        
        plt.xlim(-5, 5)
        plt.xlabel('Rescaled height')
        plt.ylabel('Probability')
        #plt.title('Height distribution for different L')
        plt.legend()
        plt.savefig(self.datafolder+'height_prob_collapse.png', dpi=300)
        if show:
            plt.show()

        
    
    def avgSlope(self, show=False):
        #plot time-averaged slope h(t)/L for each L
        print('plotting average slope')
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        avg_slopes = np.zeros(len(self.Ls))
        err_slopes = np.zeros(len(self.Ls))
        for l, L in enumerate(self.Ls):
            avalanches, heights, slopes, t_cs = self.readData(L)
            t_c = t_cs[0]
            heights_ss = heights[0][t_c:]
            slopes_ss = slopes[0][t_c:]

            avg_slopes[l] = np.mean(slopes_ss.mean(axis=1))
            err_slopes[l] = np.std(slopes_ss.mean(axis=1))
            

            print('L = {:.2f}, mean = {:.2f} +/- {:.2f}'.format(L, avg_slopes[l], err_slopes[l]))


        ax[0].errorbar(self.Ls, avg_slopes, fmt='.', color='k', capsize=3)
        ax[1].plot(self.Ls, err_slopes, '.', color='k', linewidth=0.8)

        #prediction from average height
        a_0, a_1, w_1 = 1.74, 0.21, 0.57 #from avg height fit
        x = np.linspace(np.amin(self.Ls), np.amax(self.Ls), 100)
        fit_avg = self.scal_z(x, a_0, a_1, w_1)
        ax[0].plot(x, fit_avg, color='k', linewidth=0.8)

        #prediction from standard deviation of height
        a, b = 0.58, 0.24 #from std height fit
        fit_std = self.scal_z_std(x, 0.58, 0.24)
        ax[1].plot(x, fit_std, color='k', linewidth=0.8)
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        
        ax[0].set_xlabel('System size L')
        ax[0].set_ylabel('Average slope')
        #ax[0].set_title('Average slope of pile over time for different L')
        ax[0].grid()

        ax[1].set_xlabel('System size L')
        ax[1].set_ylabel('Standard deviation of slope')
        #ax[1].set_title('Standard deviation of slope of pile over time for different L')
        ax[1].grid()

        fig.savefig(self.datafolder+'avg_slope.png', dpi=300)

        if show:
            plt.show()


    def avalancheProb(self, scale=1, show=False, step=1, skip=None, collapse=False):
        print('plotting avalanche probability')
        plt.figure(figsize=(10, 7))
        D, tau = 2.25, 1.55
        for l, L in enumerate(self.Ls[:None:step]):
            print('L =', L)
            
            avalanches, heights, slopes, t_cs = self.readData(L)
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
            avalanches, heights, slopes, t_cs = self.readData(L)
            t_c = t_cs[0]

            #take first run from its t_c
            avalanches_ss = avalanches[0][t_c:]
            #flatten and convert to int
            avalanches_ss = avalanches_ss.astype(np.int64),

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
#analyser.plotAllAvalanches(show=show)
# analyser.plotMeanHeights(runs='all', show=show, scale='loglog')
# analyser.plot_t_cs(fit=True, show=show)
# analyser.collapseHeights(show=show, scale='linear')
#analyser.scalingCorrections(show=show)
# analyser.heightProb(show=show)
# analyser.avalancheProb(show=show, scale=1.25, step=1, skip=None, collapse=True)
#analyser.avalanche_moment_k(ks=[1,2,3,4], show=show)
analyser.avgSlope(show=show)