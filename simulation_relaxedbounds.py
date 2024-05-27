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


def simulation_relaxedbounds(n, p, k_inst, gamma, phi, dist, beta, type, iter, saveImg=False, showImg=True):
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

    # CUTOFF
    cut = n // 2

    # X-AXIS - bound
    relaxed_bound = np.linspace(0, 1, 11)

    # FIND VALUES
    gs_means, group_means, inst_means, group_control_means, inst_control_means = [], [], [], [], []
    gs_sems, group_sems, inst_sems, group_control_sems, inst_control_sems  = [], [], [], [], []

    for bound in relaxed_bound:
        print(f"bound: {bound}%")
        # iteration results
        gs_res, group_res, inst_res, group_control_res, inst_control_res = [], [], [], [], []

        for i in range(iter):
            # generate utilities
            utils_norm, utils_beta = [], []

            if dist == 'gaussian':
                # rv = truncnorm(a = 0)
                utils_norm = truncnorm.rvs(a = 0, b = np.inf, size = cut)
                utils_beta = truncnorm.rvs(a = 0, b = np.inf, size = n - cut)

            elif dist == 'pareto':
                alpha = 3
                utils_norm = pareto.rvs(b = alpha, size = cut)
                utils_beta = pareto.rvs(b = alpha, size = n - cut)
            else:
                print("Please enter gaussian or pareto")
                return 

            utils_beta = [util * beta for util in utils_beta]
            utils_total = [0]*n
            utils_total[:cut] = utils_norm[::]
            utils_total[cut:] = utils_beta[::]

            # generate preferences
            preferences = generate_mallows_preferences(n, p, gamma, phi = phi)

            # S, groups a and b
            S_set = range(n)
            group_a = S_set[:cut]
            group_b = S_set[cut:]

            # RUN WITH REGULAR AFTER TO COMPARE

            # Unconstrained results
            gs_temp = gale_shapley(S = list(range(1, n + 1)), k = k_inst[::], sigma = preferences[::], est_util= utils_total[::])

            # Group_wise
            group_control_temp = group_wise(k = k_inst[::], sigma = preferences[::], est_util= utils_total[::], A = group_a[::], B = group_b[::])

            # Group-wise with relaxed bound
            group_temp = group_wise_bounded(k = k_inst[::], sigma = preferences[::], est_util= utils_total[::], A = group_a[::], B = group_b[::], bound = bound/2)

            # Instituion-wise
            inst_control_temp = a_inst_wise(k = k_inst[::], sigma = preferences[::], est_util= utils_total[::], A = group_a[::], B = group_b[::])

            # Institution-wise with relaxed bound
            inst_temp = inst_wise_bounded(k = k_inst[::], sigma = preferences[::], est_util= utils_total[::], A = group_a[::], B = group_b[::], bound = bound/2)


            # print(f"gs_temp: {gs_temp}")
            # print(f"group_temp: {group_temp}")
            # print(f"group_control_temp: {group_control_temp}")
            # print(inst_temp)
            # print(inst_control_temp)

            if type == 'ptop1':
                gs_res.append(ptop1(gs_temp, preferences, cut))
                group_res.append(ptop1(group_temp, preferences, cut))
                inst_res.append(ptop1(inst_temp, preferences, cut))
                inst_control_res.append(ptop1(inst_control_temp, preferences, cut))
                group_control_res.append(ptop1(group_control_temp, preferences, cut))

            elif type == 'ptop5':
                gs_res.append(ptop5(gs_temp, preferences, cut))
                group_res.append(ptop5(group_temp, preferences, cut))
                inst_res.append(ptop5(inst_temp, preferences, cut))
                inst_control_res.append(ptop5(inst_control_temp, preferences, cut))
                group_control_res.append(ptop5(group_control_temp, preferences, cut))
            elif type == 'u':
                # print(f"gs_res: {gs_temp}")
                gs_res.append(u(gs_temp, utils_total[::], cut = cut, beta = beta))
                # print(f"group_res: {group_temp}")
                group_res.append(u(group_temp, utils_total[::], cut = cut, beta = beta))
                # print(f"inst_res: {inst_temp}")
                inst_res.append(u(inst_temp, utils_total[::], cut = cut, beta = beta))
                inst_control_res.append(u(inst_control_temp, utils_total[::], cut = cut, beta = beta))
                group_control_res.append(u(group_control_temp, utils_total[::], cut = cut, beta = beta))
            else:
                print("Please enter 'ptop1', 'ptop5', or 'u'")
                return

        # append means and standard errors to overall results
        gs_means.append(sum(gs_res)/len(gs_res))
        gs_sems.append(np.std(gs_res)/np.sqrt(len(gs_res)))
        group_means.append(sum(group_res)/len(group_res))
        group_sems.append(np.std(group_res)/np.sqrt(len(group_res)))
        inst_means.append(sum(inst_res)/len(inst_res))
        inst_sems.append(np.std(inst_res)/np.sqrt(len(inst_res)))
        group_control_means.append(sum(group_control_res)/len(group_control_res))
        group_control_sems.append(np.std(group_control_res)/np.sqrt(len(group_control_res)))
        inst_control_means.append(sum(inst_control_res)/len(inst_control_res))
        inst_control_sems.append(np.std(inst_control_res)/np.sqrt(len(inst_control_res)))



    plt.figure().clear()
    plt.figure().set_size_inches(4, 3, forward=True)

    # 3 experiment bars
    plt.errorbar(relaxed_bound, gs_means, yerr=gs_sems, fmt='--', label='Unconstrained (control)', capsize=3)
    plt.errorbar(relaxed_bound, group_control_means, yerr=group_control_sems, fmt='--', label="Group-wise (control)", capsize=3)
    plt.errorbar(relaxed_bound, inst_control_means, yerr=inst_control_sems, fmt='--', label="Unit-wise (control)", capsize=3)
    plt.errorbar(relaxed_bound, group_means, yerr=group_sems, fmt='-', label='Group-wise constraints', capsize=3)
    plt.errorbar(relaxed_bound, inst_means, yerr=inst_sems, fmt='-', label='Unit-wise constraints', capsize=3)

    # Control is at the right end of x-axis, where that is perfect (also unconstrained)


    # Axes
    plt.xlabel(r"Lower bound ($\alpha$)", fontsize=12)
    if type == 'u':
        plt.ylabel("Fraction of optimal latent utility",fontsize=12)
    else:
        plt.ylabel('Preference-based fairness', fontsize=12)

    # plt.title(rf'{dist} utilities $\beta$={beta} $n$={n} $p$={p} $k$={k_inst[0]} iter={iter} $\phi$={phi}', fontsize=10)
    
    # Axis ticks
    if type == 'u':
        plt.ylim(0.9,1.005)
        plt.yticks(np.arange(0.9, 1.005, 0.025))
    else:
        plt.ylim(0, 1.02)
        plt.yticks(np.arange(0, 1.05, 0.25))

    if type != 'u':
        plt.xlim(0.4, 1)
        plt.xticks(np.arange(0.4, 1.02, 0.1))

    
    plt.tight_layout()

    if saveImg:
        # Get the current date
        current_date = datetime.now().strftime('%m-%d-%Y')
        subdirectory = os.path.join('plots', current_date)
        os.makedirs(subdirectory, exist_ok=True)

        filename = os.path.join(subdirectory, f'{dist}_{type}_{beta}_relaxedbounds.pdf')
        plt.savefig(filename, format="pdf")
    
    if showImg:
        plt.show()