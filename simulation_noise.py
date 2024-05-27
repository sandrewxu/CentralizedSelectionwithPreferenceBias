import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import seaborn as sns

from scipy.stats import truncnorm
from scipy.stats import pareto
from scipy.stats import norm
from datetime import datetime
import os

from kendall_tau_helpers import *
from algorithms import *
from fairness import *
from preferences import *

def util_noise_simulation(n, p, k_inst, gamma, phi, test, dist, beta, type, iter):
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

    totalres = []
    sigma_a = np.linspace(0, 2, 5)
    sigma_b = np.linspace(0, 2, 5)[::-1]

    for b_noise in sigma_b:
        for a_noise in sigma_a:
            print(fr"\sigma_a: {a_noise}, \sigma_b: {b_noise}")

            res, res_group, res_inst = [], [], []

            for i in range(iter):
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
                
                if a_noise > 0:
                    # print(utils_norm)
                    # noise_norm = truncnorm.rvs(a=0, b=np.inf, size = len(utils_norm))
                    noise_norm = norm.rvs(0, 1, len(utils_norm))
                    noise_norm = noise_norm * a_noise
                    utils_norm = np.add(utils_norm, noise_norm)

                if b_noise > 0:
                    # noise_beta = truncnorm.rvs(a=0, b=np.inf, size = len(utils_beta))
                    noise_beta = norm.rvs(0, 1, len(utils_norm))
                    noise_beta = noise_beta * b_noise
                    utils_beta = np.add(utils_beta, noise_beta)

                utils_total = [0]*n
                utils_total[:cut] = utils_norm[::]
                utils_total[cut:] = utils_beta[::]

                preferences = generate_mallows_preferences(n, p, gamma, phi = phi)

                # S, groups a and b
                S_set = range(n)
                group_a = S_set[:cut]
                group_b = S_set[cut:]
                
                temp = []
                
                if test == 'gs':
                    temp = gale_shapley(S = list(range(1, n + 1)), k = k_inst[::], sigma = preferences[::], est_util= utils_total[::])
                elif test == 'group':
                    temp = group_wise(k = k_inst[::], sigma = preferences[::], est_util= utils_total[::], A = group_a[::], B = group_b[::])
                elif test == 'inst':
                    temp = a_inst_wise(k = k_inst[::], sigma = preferences[::], est_util= utils_total[::], A = group_a[::], B = group_b[::])
                elif test == 'diff':
                    temp_group = group_wise(k = k_inst[::], sigma = preferences[::], est_util= utils_total[::], A = group_a[::], B = group_b[::])
                    temp_inst = a_inst_wise(k = k_inst[::], sigma = preferences[::], est_util= utils_total[::], A = group_a[::], B = group_b[::])

                if type == 'ptop1':
                    if temp == []:
                        res_group.append(ptop1(temp_group, preferences, cut))
                        res_inst.append(ptop1(temp_inst, preferences, cut))
                    else:
                        res.append(ptop1(temp, preferences, cut))
                elif type == 'ptop5':
                    if temp == []:
                        res_group.append(ptop5(temp_group, preferences, cut))
                        res_inst.append(ptop5(temp_inst, preferences, cut))
                    else:
                        res.append(ptop5(temp, preferences, cut))
                else:
                    print("Please enter 'ptop1' or 'ptop5'")
                    return
                
            if res == []:
                mean_group = sum(res_group)/len(res_group)
                mean_inst = sum(res_inst) / len(res_inst)
                totalres.append(mean_inst-mean_group)
            else:
                mean = sum(res)/len(res)
                print(mean)
                totalres.append(mean)


    # HEATMAP GENERATION
    # Correcting the approach to reshape the data for the heatmap
    heatmap_data = np.array(totalres).reshape(len(sigma_a), len(sigma_b))

    # print(heatmap_data)


    # Generate the corrected heatmap
    plt.figure().set_size_inches(3, 3, forward=True)
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", xticklabels=sigma_a, yticklabels=sigma_b, cmap='RdBu', vmin=0.0, vmax=1.0, cbar = False)
    plt.xlabel("$\delta_1$", fontsize=12)
    plt.ylabel("$\delta_2$", fontsize=12)
    plt.tight_layout()

    current_date = datetime.now().strftime('%m-%d-%Y')
    subdirectory = os.path.join('plots', current_date)
    os.makedirs(subdirectory, exist_ok=True)

    filename = os.path.join(subdirectory, f'heatmap_{test}_{type}_{dist}_redblue.pdf')
    plt.savefig(filename, format = "pdf")
    # plt.title("Heatmap for $\sigma_a$ and $\sigma_b$ with Red-Blue Color Scale")
    plt.show()

    # depends on ptop1, ptop5, and u
            


def beta_noise_simulation(n, p, k_inst, gamma, phi, dist, type, iter, saveImg=False, showImg=True, std=0.1):
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

    control_gs_means, control_group_means, control_inst_means = [], [], []
    control_gs_stds, control_group_stds, control_inst_stds = [], [], []

    betas = np.linspace(0, 1, 26)
    for beta in betas:
        if beta == 0:
            beta = 0.00000000000001
        gs_res = []
        group_res = []
        inst_res = []

        control_gs_res, control_group_res, control_inst_res = [], [], []

        for i in range(iter):
            #print(f"gamma: {gamma}, trial: {i + 1}")

            # generate utilities based on distribution and beta
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
            
            ## the estimated utilities for disadvantaged group multiplies by beta
            #utils_norm = [abs(util) for util in utils_norm]
            #utils_beta = [abs(util) for util in utils_beta]

            # Generate truncated gaussian with beta as center

            if std != 0:
                lower, upper = 0, 1
                a, b = (lower - beta) / std, (upper - beta) / std

                beta_dist = truncnorm.rvs(a, b, loc=beta, scale=std, size=len(utils_beta))
            else:
                beta_dist = [beta] * len(utils_beta)
            
            control_beta_dist = np.array([beta] * len(utils_beta))

            control_utils_beta = utils_beta[::] * control_beta_dist
            # print(f"original: {sum(utils_beta)}")
            utils_beta = utils_beta * beta_dist
            

            #plt.figure().clear() 
            #plt.hist(utils_norm, alpha=0.5, bins=50, density=True, label='normal')
            #plt.hist(utils_beta, alpha=0.5, bins=50, density=True, label='biased')
            #plt.legend()
            #plt.show()

            utils_total = [0]*n
            utils_total[:cut] = utils_norm[::]
            utils_total[cut:] = utils_beta[::]

            control_utils_total = [0]*n
            control_utils_total[:cut] = utils_norm[::]
            control_utils_total[cut:] = control_utils_beta[::]

            #print("total utils")
            #print(utils_total)

            #print("distributions done")

            # generate preferences for a and b
            preferences = generate_mallows_preferences(n, p, gamma, phi = phi)

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
            control_gs_temp = gale_shapley(S = list(range(1, n + 1)), k = k_inst[::], sigma = preferences[::], est_util= control_utils_total[::])
            
            #print("gs done")
            group_temp = group_wise(k = k_inst[::], sigma = preferences[::], est_util= utils_total[::], A = group_a[::], B = group_b[::])
            control_group_temp = group_wise(k = k_inst[::], sigma = preferences[::], est_util= control_utils_total[::], A = group_a[::], B = group_b[::])
            #print(group_temp)
            #print(cut)
            #print(k_inst)
            #print(preferences)
            #print(utils_total)
            #print(group_a)
            #print(group_b)
            #print("group-wise done")

            inst_temp = a_inst_wise(k = k_inst[::], sigma = preferences[::], est_util= utils_total[::], A = group_a[::], B = group_b[::])
            control_inst_temp = a_inst_wise(k = k_inst[::], sigma = preferences[::], est_util= control_utils_total[::], A = group_a[::], B = group_b[::])

            if type == 'ptop1':
                gs_res.append(ptop1(gs_temp, preferences, cut))
                group_res.append(ptop1(group_temp, preferences, cut))
                inst_res.append(ptop1(inst_temp, preferences, cut))
                control_gs_res.append(ptop1(gs_temp, preferences, cut))
                control_group_res.append(ptop1(control_group_temp, preferences, cut))
                control_inst_res.append(ptop1(control_inst_temp, preferences, cut))
            elif type == 'ptop5':
                gs_res.append(ptop5(gs_temp, preferences, cut))
                group_res.append(ptop5(group_temp, preferences, cut))
                inst_res.append(ptop5(inst_temp, preferences, cut))
                control_gs_res.append(ptop5(control_gs_temp, preferences, cut))
                control_group_res.append(ptop5(control_group_temp, preferences, cut))
                control_inst_res.append(ptop5(control_inst_temp, preferences, cut))
            elif type == 'u':
                # print(f"modified: {sum(utils_beta)}")
                # utils_beta = utils_beta / beta_dist
                # print(f"original: {sum(utils_beta)}")
                # utils_total[cut:] = utils_beta[::]
                gs_res.append(u_dist(gs_temp, utils_total[::], cut = cut, beta_dist = beta_dist))
                group_res.append(u_dist(group_temp, utils_total[::], cut = cut, beta_dist = beta_dist))
                inst_res.append(u_dist(inst_temp, utils_total[::], cut = cut, beta_dist = beta_dist))

                control_gs_res.append(u_dist(control_gs_temp, control_utils_total[::], cut = cut, beta_dist = control_beta_dist))
                control_group_res.append(u_dist(control_group_temp, control_utils_total[::], cut = cut, beta_dist = control_beta_dist))
                control_inst_res.append(u_dist(control_inst_temp, control_utils_total[::], cut = cut, beta_dist = control_beta_dist))
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

        control_gs_means.append(sum(control_gs_res)/len(control_gs_res))
        control_gs_stds.append(np.std(control_gs_res)/np.sqrt(len(control_gs_res)))
        control_group_means.append(sum(control_group_res)/len(control_group_res))
        control_group_stds.append(np.std(control_group_res)/np.sqrt(len(control_group_res)))
        control_inst_means.append(sum(control_inst_res)/len(control_inst_res))
        control_inst_stds.append(np.std(control_inst_res)/np.sqrt(len(control_inst_res)))

    # Sample data
    
    
    # Create a line graph with error bars for each set of data
    #print(gs_means)
    #print(len(gs_means))
    # print(gs_stds)
    #print(len(gs_stds))
    #print(group_means)
    #print(group_stds)
    #print(inst_means)
    #print(inst_stds)
    #print(control_gs_means)
    #print(control_group_means)
    #print(control_inst_means)

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

    plt.errorbar(betas, gs_means, yerr=gs_stds, fmt='-', label='Unconstrained', capsize=3)
    plt.errorbar(betas, group_means, yerr=group_stds, fmt='-', label='Group-wise constraints', capsize=3)
    plt.errorbar(betas, inst_means, yerr=inst_stds, fmt='-', label='Unit-wise constraints', capsize=3)

    plt.errorbar(betas, control_gs_means, yerr=control_gs_stds, fmt='--', label='Unconstrained control', capsize=3)
    plt.errorbar(betas, control_group_means, yerr=control_group_stds, fmt='--', label='Group-wise control', capsize=3)
    plt.errorbar(betas, control_inst_means, yerr=control_inst_stds, fmt='--', label='Unit-wise control', capsize=3)

    # plt.legend()

    # Add labels and title
    plt.xlabel(r'$\beta$', fontsize=12)
    if type == 'u':
        plt.ylabel("Fraction of optimal latent utility", fontsize=12)
    else:
        plt.ylabel('Preference-based fairness', fontsize=12)
    # plt.title(rf'{dist} utilities $n$={n} $p$={p} $k$={k_inst[0]} iter={iter} $\phi$={phi}', fontsize=10)
    # plt.legend()

    # Set y-axis range
    #plt.xlim(0, 10)
    if type == 'u':
        plt.ylim(0.5,1.02)
        plt.yticks(np.arange(0.5, 1.05, 0.1))
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

        filename = os.path.join(subdirectory, f'{type}_{dist}_phi{phi}_iter{iter}_std{std}.pdf')
        plt.savefig(filename, format = "pdf")

    # run the three things many times and do the mean and all that shit

    # depends on ptop1, ptop5, and u
