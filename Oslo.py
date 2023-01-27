# Complexity project implementation of Oslo model
# Botond Branyicskai-Nagy
# January 2022

import random
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time



p = 0.5
Ls = np.zeros(5, dtype=int) + 16 #[4, 8, 16, 32, 64, 128, 256]
i_after_ss = 10**3 # number of iterations after steady state
thresholds = np.array([1, 2])



def randomThresholds(p, L):
    # create list of random threshold slopes from threshold_slopes with probability p
    z_th = np.random.choice(thresholds, size=L, p=[p, 1-p])

    return z_th

def initialize(p, L, thresholds):
    # initialize with empty configuration
    config = np.zeros(L, dtype=int)

    # create list of random initial threshold slopes
    z_th = randomThresholds(p, L)

    return config, z_th


def getSlopes(config, i_max):
    # calculate local slopes in configuration
    slopes = config[:i_max+1] - np.concatenate((config[1:], [0]))[:i_max+1]
    return slopes

def drive(config, z_th):
    # add one grain to the first cell
    config[0] += 1

    return config

def relax(config, z_th, L):
    avalanche_size = 0
    i_max = 1 # index of last cell to be checked for relaxation

    # relax the configuration while there are cells with slope greater than local threshold slope
    while np.any(getSlopes(config, i_max) > z_th[:i_max+1]):

        slopes = getSlopes(config, i_max)
        to_relax = np.where(slopes > z_th[:i_max+1])[0] # to_relax is array of cells that need to be relaxed, ie. cells with slope greater than local threshold slope
        config[to_relax] -= 1
        z_th[to_relax] = randomThresholds(p, L)[to_relax] # reset threshold slope of relaxed cells

        # if cell is not last cell, add grain to cell to the right, otherwise don't add grain
        addto = to_relax[to_relax < L-1] + 1
        config[addto] += 1


        # count avalanche size
        avalanche_size += len(to_relax)

        # increase i_max unless it is already at the end of the array
        i_max = to_relax[-1] + 1



    return config, z_th, avalanche_size

def animate(config, z_th, i, s=0):
    # animation of the model with squares representing the cells
    # if called for first time, initialize animation
    
    if not animate.initialized:
        # initialize figure and axes with drive button
        # button_pressed is 1 if button is clicked
        # horizontal axis shows z_th for each cell in centre of cell
        # vertical axis shows height of each cell
        animate.fig, animate.ax = plt.subplots()
        animate.axdrive = animate.fig.add_axes([0.05, 0.8, 0.1, 0.1])
        animate.axdrive.button_pressed = 0
        animate.axdrive.button = plt.Button(animate.axdrive, 'Drive')
        animate.axdrive.button.on_clicked(lambda event: setattr(animate.axdrive, 'button_pressed', 1))
        # add quit button that terminates the program
        animate.axquit = animate.fig.add_axes([0.05, 0.7, 0.1, 0.1])
        animate.axquit.button = plt.Button(animate.axquit, 'Quit')
        animate.axquit.button.on_clicked(lambda event: plt.close(animate.fig))
        animate.fig.canvas.mpl_connect('close_event', lambda event: quit())

        # add iteration counter
        animate.axiter = animate.fig.add_axes([0.05, 0.6, 0.1, 0.1])
        animate.axiter.text(0.5, 0.5, 'Iteration: 0', horizontalalignment='center', verticalalignment='center')
        animate.axiter.axis('off')

        # add avalanche size counter
        animate.axsize = animate.fig.add_axes([0.05, 0.5, 0.1, 0.1])
        animate.axsize.text(0.5, 0.5, 'Last avalanche size: 0', horizontalalignment='center', verticalalignment='center')
        animate.axsize.axis('off')

        # set axes labels and ticks
        animate.ax.set_xlabel('z_th')
        animate.ax.set_ylabel('height')
        animate.ax.set_xticks(np.arange(L))
        # xticks are in centre of each cell
        uplim = 2*L
        animate.ax.set_xticklabels(np.arange(L) + 0.5)
        animate.ax.set_xticklabels(z_th)
        animate.ax.set_yticks(np.arange(uplim))
        animate.ax.set_ylim(0, uplim)
        animate.ax.set_xlim(0, L)
        animate.ax.set_aspect('equal')
        animate.ax.grid(True)
        
        # initialize bars
        animate.rects = animate.ax.bar(np.arange(L), config, width=1, align='edge', color='black')
        animate.initialized = True
        plt.pause(0.0001)

        return

    # update the heights of the bars and the x axis labels
    for rect, h in zip(animate.rects, config):
        rect.set_height(h)
    
    animate.ax.set_xticklabels(z_th)
    animate.axiter.clear()
    animate.axiter.text(0.5, 0.5, 'Iteration: ' + str(i+1), horizontalalignment='center', verticalalignment='center')
    animate.axiter.axis('off')
    animate.axsize.clear()
    animate.axsize.text(0.5, 0.5, 'Last avalanche size: ' + str(s), horizontalalignment='center', verticalalignment='center')
    animate.axsize.axis('off')


    # draw the animation
    animate.fig.canvas.draw()
    animate.fig.canvas.flush_events()
    plt.pause(0.000001)

    return

def checkRecurrence(config, z_th):
    # check if the model has reached a recurrence state
    return False

def run(p, L, iterations, thresholds, manual=True, animation=True, saving=True):

    print('Running system size: ' + str(L))
    # initialize configuration, threshold slopes and animation
    config, z_th = initialize(p, L, thresholds)
    t_c_switch = False

    #observables
    avalanches, heights = np.zeros(iterations), np.zeros(iterations)
    
    animate.initialized = False
    if animation:
        animate(config, z_th, 0)
        print('Animation running. Click "Drive" to drive the model.')

    # drive the model for a number of iterations
    for t in range(iterations):
        #if manual=true wait until 'drive' button is clicked, drive and relax the model

        if animation:
            if manual:
                while not animate.axdrive.button_pressed:
                    plt.pause(0.1)

                animate.axdrive.button_pressed = 0

            config = drive(config, z_th)
            animate(config, z_th, t)
            config, z_th, s = relax(config, z_th, L) # s is avalanche size
            animate(config, z_th, t, s)

        else:
            config = drive(config, z_th)
            config, z_th, s = relax(config, z_th, L) # s is avalanche size

        #check if a grain has left the system
        if np.sum(config) == t and not t_c_switch:
            t_c = t-1
            t_c_switch = True
            print('Grain left system at iteration ' + str(t_c))
            
        avalanches[t] = s
        heights[t] = config[0] #height of pile at drive t is height of first cell after relaxation

    #print('Average avalanche size after steady state: ' + str(np.mean(avalanches[steady_state:])))
    #print('Average height of pile after steady state: ' + str(np.mean(heights[steady_state:])))

    if saving:
        to_save = [avalanches, heights]
        #save avalanche sizes and heights as csv files in a folder named after data/L
        foldername = 'data/' + str(L)
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        for file in to_save:
            np.savetxt(foldername + '/'+str(file)+'.csv', file, delimiter=',')
        
        #add t_c to the end of t_cs.csv
        with open(foldername + '/t_cs.csv', 'a') as f:
            f.write(str(t_c) + '\n')

        print('Saved in data/' + str(L))
    if t_c_switch == False:
        print(config)
        print(np.sum(config))
    return config, avalanches, heights


#final_config, avalanches, heights = run(p, L, iterations, thresholds, manual=False, animation=False, saving=True)

#run the model for L=4, 8, 16, 32, 64, 128, 256...
for L in Ls:
    #start timer
    start_time = time.time()
    steady_state = 2*L**2
    iterations = steady_state + i_after_ss
    #create a file to save t_c in
    foldername = 'data/' + str(L) + '/'
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    with open(foldername + 't_cs.csv', 'w') as f:
        f.write('')
    

    final_config, avalanches, heights = run(p, L, iterations, thresholds, manual=False, animation=False, saving=False)

    #print iterations per second
    #print('Iterations per second: ' + str(iterations/(time.time() - start_time)))



