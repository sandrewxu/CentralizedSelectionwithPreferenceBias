from datetime import datetime
import os
import numpy as np
from scipy.stats import truncnorm
from scipy.stats import pareto
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams['figure.figsize'] = [6, 3]
# import pandas as pd

from kendall_tau_helpers import *
from algorithms import *
from fairness import *
from preferences import *


def simulation_corrpref(n, p, k_inst, dist, beta, type, iter, saveImg=False, showImg=True):
    '''
        n: number of agents
        p: number of institutions
        k_inst: capacity of institutions
        phi: parameter for mallows distribution
        dist: type of distribution -- gaussian or pareto (alpha = 3)
        beta: value of beta
        type: ptop1, ptop5, u
        iter: number of iterations for each gamma
    '''

    # generate distributions of utilities based on n, dist, and beta

    cut = n // 2 # cutoff for normal and disadvantaged
    
    # PTOP1, plot

    ## run gale shapley 50 times, each time, compare num of first n/2 who got first preference.

    gs_means = []
    gs_stds = []
    group_means = []
    group_stds = []
    inst_means = []
    inst_stds = []

    alphas = np.linspace(0, 1, 21)

    for alpha in alphas:
        print(f"phi: {alpha}")
        if alpha == 1:
            alpha = 0.99999999

        gs_res = []
        group_res = []
        inst_res = []

        for i in range(iter):
            #print(f"gamma: {gamma}, trial: {i + 1}")

            # generate utilities based on distribution and beta
            utils_norm, utils_beta = [], []

            if dist == 'gaussian':
                # rv = truncnorm(a = 0)
                utils_norm = truncnorm.rvs(a = 0, b = np.inf, size = cut)
                utils_beta = truncnorm.rvs(a = 0, b = np.inf, size = n - cut)

            elif dist == 'pareto':
                utils_norm = pareto.rvs(b = 3, size = cut)
                utils_beta = pareto.rvs(b = 3, size = n - cut)
            else:
                print("Please enter gaussian or pareto")
                return 
            
            utils_beta = [util * beta for util in utils_beta]
            
            utils_total = [0]*n
            utils_total[:cut] = utils_norm[::]
            utils_total[cut:] = utils_beta[::]

            # Generate preferences

            # preferences = generate_corr_preferences(n, p, alpha, dist)
            preferences = generate_mallows_preferences(num_agents=n, num_institutions=p, gamma=0, phi=alpha)

            # S, groups a and b
            S_set = range(n)
            group_a = S_set[:cut]
            group_b = S_set[cut:]

            #print("----- agents -----")
            #print(S_set)
            #print(len(S_set))

            #print("----- inst -----")
            #print(k_inst)
            #print(len(k_inst))

            #print("----- pref -----")
            #print(preferences)
            #print(len(preferences))

            #print("----- util -----")
            #print(utils_total)
            #print(len(utils_total))

            ### GALE-SHAPLEY RESULTS
            gs_temp = gale_shapley(S = list(range(1, n + 1)), k = k_inst[::], sigma = preferences[::], est_util= utils_total[::])
            
            #print("gs done")
            group_temp = group_wise(k = k_inst[::], sigma = preferences[::], est_util= utils_total[::], A = group_a[::], B = group_b[::])
            #print(group_temp)
            #print(cut)
            #print(k_inst)
            #print(preferences)
            #print(utils_total)
            #print(group_a)
            #print(group_b)
            #print("group-wise done")

            inst_temp = a_inst_wise(k = k_inst[::], sigma = preferences[::], est_util= utils_total[::], A = group_a[::], B = group_b[::])

            if type == 'ptop1':
                gs_res.append(ptop1(gs_temp, preferences, cut))
                group_res.append(ptop1(group_temp, preferences, cut))

                #print("norm res: ")
                #print(group_temp[:cut])
                #frequency_table = pd.Series(group_temp[:cut]).value_counts()
                #print(frequency_table)

                #print("bias res:")
                #print(group_temp[cut:])
                #frequency_table = pd.Series(group_temp[cut:]).value_counts()
                #print(frequency_table)

                #print("overall res")
                #sorted_res = [group_temp for _, group_temp in sorted(zip(utils_total, group_temp), reverse = True)]
                #print(sorted_res)

                #print("choices")
                #print(preferences[0])

                inst_res.append(ptop1(inst_temp, preferences, cut))
            elif type == 'ptop5':
                gs_res.append(ptop5(gs_temp, preferences, cut))
                group_res.append(ptop5(group_temp, preferences, cut))
                inst_res.append(ptop5(inst_temp, preferences, cut))
            elif type == 'u':
                # print(f"gs_res: {gs_temp}")
                gs_res.append(u(gs_temp, utils_total[::], cut = cut, beta = beta))
                # print(f"group_res: {group_temp}")
                group_res.append(u(group_temp, utils_total[::], cut = cut, beta = beta))
                # print(f"inst_res: {inst_temp}")
                inst_res.append(u(inst_temp, utils_total[::], cut = cut, beta = beta))
            else:
                print("Please enter 'ptop1', 'ptop5', or 'u'")
                return

            #print("inst-wise done")
        
        gs_means.append(sum(gs_res)/len(gs_res))
        gs_stds.append(np.std(gs_res)/np.sqrt(len(gs_res)))
        group_means.append(sum(group_res)/len(group_res))
        group_stds.append(np.std(group_res)/np.sqrt(len(group_res)))
        inst_means.append(sum(inst_res)/len(inst_res))
        inst_stds.append(np.std(inst_res)/np.sqrt(len(inst_res)))

    # Sample data
    
    
    # Create a line graph with error bars for each set of data
    #print(gs_means)
    #print(len(gs_means))
    #print(gs_stds)
    #print(len(gs_stds))
    #print(group_means)
    #print(group_stds)
    #print(inst_means)
    #print(inst_stds)

    gs_means = np.array(gs_means)
    gs_stds = np.array(gs_stds)
    group_means = np.array(group_means)
    group_stds = np.array(group_stds)
    inst_means = np.array(inst_means)
    inst_stds = np.array(inst_stds)

    plt.figure().clear() 
    plt.figure().set_size_inches(4, 3, forward=True)
    # plt.figure().set_figheight(2)
    # plt.figure().set_figwidth(4)
    
    # plt.margins(0.1)

    plt.errorbar(alphas, gs_means, yerr=gs_stds, fmt='-', label='Unconstrained', capsize=3)
    plt.errorbar(alphas, group_means, yerr=group_stds, fmt='-', label='Group-wise constraints', capsize=3)
    plt.errorbar(alphas, inst_means, yerr=inst_stds, fmt='--', label='Unit-wise constraints', capsize=3)

    # Add labels and title
    plt.xlabel(r'$\phi$', fontsize=12)
    if type == 'u':
        plt.ylabel("Fraction of optimal latent utility", fontsize=12)
    else:
        plt.ylabel('Preference-based fairness')
    # plt.title(rf'{dist} utilities $\beta$={beta} $n$={n} $p$={p} $k$={k_inst[0]} iter={iter}', fontsize=10)
    # plt.legend()

    # Set y-axis range
    #plt.xlim(0, 10)
    if type == 'u':
        plt.ylim(0.9,1.005)
        plt.yticks(np.arange(0.9, 1.005, 0.025))
    else:
        plt.ylim(0, 1.05)
        plt.yticks(np.arange(0, 1.05, 0.25))

    plt.tight_layout()

    # Show the plot
    if showImg:
        plt.show()

    if saveImg:
        # Get the current date
        current_date = datetime.now().strftime('%m-%d-%Y')
        subdirectory = os.path.join('plots', current_date)
        os.makedirs(subdirectory, exist_ok=True)

        filename = os.path.join(subdirectory, f'corrpref_{type}_{dist}_beta{beta}_iter{iter}.pdf')
        plt.savefig(filename, format="pdf")

    # run the three things many times and do the mean and all that shit

    # depends on ptop1, ptop5, and u