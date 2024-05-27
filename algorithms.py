import numpy as np

### ALGORITHMS ###
def gale_shapley (S, k, sigma, est_util):
    '''
        Implementation of the Gale-Shapley algorithm: the preferences of institutions (utility) are matched 
        with the preferences of the agents (inputted)

        S: set of agents
        k: set of institution capacities
        sigma: preferences for each agent in S
        est_util: estimated utilities for each agent in S

        return the list of agents and what institution they're mapped to

        
    '''

    # sort agents in decreasing order of estimated utility
    r_unmatched = [S for _, S in sorted(zip(est_util, S), reverse = True)]

    if sum(k) < len(S):
        r_unmatched = r_unmatched[:sum(k)]

    # initialize matching
    m = [-1] * len(S)

    # institutions have strict preferences -- only the top sum(k) utilities will be chosen, go by first - last preference. 

    for agent in r_unmatched:
        prefs = sigma[S.index(agent)]

        matched = False
        choicek = 0
        
        # if space in top choice available, match, else go second choice, etc. until open
        while (matched == False):
            inst = prefs[choicek]
            if k[inst] > 0:
                # match, matched = True
                m[S.index(agent)] = inst
                k[inst] -= 1
                matched = True
            else:
                # choicek + 1, 
                choicek += 1
    
    return m


    """
    # iterate over institutions in order of priority
    for t in range(p):
        to_remove = []

        # Iterate over unmatched agents
        for agent in range(len(r_unmatched)):
            # Get the index of the next preferred institution for the agent
            inst = sigma[S.index(r_unmatched[agent])][t]

            # If the institution has capacity, match the agent to it
            if k[inst] > 0:
                # Add institution to original order
                m[S.index(r_unmatched[agent])] = inst
                k[inst] -= 1
                to_remove.append(r_unmatched[agent])

        # Remove matched agents from the list of unmatched agents
        r_unmatched = [agent for agent in r_unmatched if agent not in to_remove]
    """

def a_inst_wise (k, sigma, est_util, A, B, ratio=0):
    '''
        k: institutional capacities
        sigma: agent preferences
        est_util: estimated utilities for each agent
        A: list of agents in group A
        B: list of agents in group B
        *note: A and B are disjoint
    '''

    n = len(est_util)

    if ratio == 0:
        ratio = len(A) / n

    # Define virtual institutions
    l_a = [int(x * ratio) for x in k]

    # print(f"l_a: {l_a}")
    #print("l_a: " + ", ".join(map(str, l_a)))
    l_b = [int(x * (1-ratio)) for x in k]

    # print(f"l_b: {l_b}")
    #print("l_b: " + ", ".join(map(str, l_b)))

    # assignments
    m_a = gale_shapley(A, l_a, [sigma[i] for i in A], [est_util[i] for i in A])
    # print(f"m_a: {m_a}")
    m_b = gale_shapley(B, l_b, [sigma[i] for i in B], [est_util[i] for i in B])
    # print(f"m_b: {m_b}")

    #print("m_a: " + ", ".join(map(str, m_a)))
    #print("m_b: " + ", ".join(map(str, m_b)))
    #print([sigma[i] for i in B])

    final_assignment = [-1] * n

    for agent in range(n):
        if agent in A:
            final_assignment[agent] = m_a[A.index(agent)]
        else:
            final_assignment[agent] = m_b[B.index(agent)]

    # print(f"final: {final_assignment}")
    return final_assignment


def group_wise (k, sigma, est_util, A, B, ratio=0):
    '''
        k: institutional capacities
        sigma: agent preferences
        est_util: estimated utilities for each agent
        A: list of agents in group A
        B: list of agents in group B
        *note: A and B are disjoint
    '''
    n = len(est_util)
    total_cap = sum(k)

    if ratio==0:
        ratio = len(A) / n

    # proportional to A
    a_cap = int(total_cap * ratio)
    b_cap = int(total_cap * (1-ratio))

    # utils
    a_utils = [est_util[i] for i in A]
    b_utils = [est_util[i] for i in B]

    # sort top in A and B to make s_a and s_b
    # sort agents in decreasing order of estimated utility
    s_a = [A for _, A in sorted(zip(a_utils, A), reverse = True)]
    s_a = s_a[:a_cap]
    s_b = [B for _, B in sorted(zip(b_utils, B), reverse = True)]
    s_b = s_b[:b_cap]

    s_g = s_a + s_b
    #print(s_g)

    group_utils_a = [est_util[i] for i in s_a]
    group_utils_b = [est_util[i] for i in s_b]

    group_utils = group_utils_a + group_utils_b
    #print(group_utils)

    group_prefs_a = [sigma[i] for i in s_a]
    group_prefs_b = [sigma[i] for i in s_b]
    group_prefs = group_prefs_a + group_prefs_b


    groupres = gale_shapley(S=s_g, k=k, sigma = group_prefs, est_util=group_utils)
    results = [-1] * n

    for i in range(len(groupres)):
        results[s_g[i]] = groupres[i]

    return results


# n = 20
# p = 4
# k = [4] * p
# utils = [0.11, 0.21, 0.31, 0.41, 0.6, 5, 4, 3, 2.1, 1.1, # norm
#          2.5, 2, 1.5, 1, 0.5, 0.7, 0.4, 0.3, 0.2, 0.1] # bias
# pref = [[3, 2, 0, 1]] * n
# A = range(n // 2)
# B = range(n // 2, n)

# inst_temp = a_inst_wise(k = k, sigma = pref, est_util=utils, A=A, B=B)

# from fairness import ptop1

# print(ptop1(inst_temp, pref, len(A), n=0.6))

# inst_wise should return
# [-1, -1, 1, 1, 0, ]

# group_wise should return
# [-1, -1, 1, 1, 0, 3, 3, 3, 2, 2,
# 3, 2, 2, 0, 0, 0, 1, 1, -1, -1]



def group_wise_bounded(k, sigma, est_util, A, B, bound):
    '''
        k: institutional capacities
        sigma: agent preferences
        est_util: estimated utilities for each agent
        A: list of agents in group A
        B: list of agents in group B
        *note: A and B are disjoint

        Assumes groups are of equal proportions
    '''
    n = len(est_util)
    total_cap = sum(k)

    if bound > 0.5:
        print('bound must be less than or equal to 0.5')
        return

    a_quota = int(bound * total_cap)
    b_quota = int(bound * total_cap)
    rem_quota = int((1-2*bound)*total_cap)

    # Run Gale-Shapley and keep going UNLESS a quota is broken

    S = list(A) + list(B)

    # sort agents in decreasing order of estimated utility
    r_unmatched = [S for _, S in sorted(zip(est_util, S), reverse = True)]

    # initialize matching
    m = [-1] * len(S)

    for agent in r_unmatched:
        prefs = sigma[S.index(agent)]

        break_flag = False
        matched = False
        choicek = 0
        
        # if space in top choice available, match, else go second choice, etc. until open
        while (matched == False):
            inst = prefs[choicek]
            if agent in A and a_quota > 0 and k[inst] > 0:
                # match, matched = True
                m[S.index(agent)] = inst
                k[inst] -= 1
                a_quota -= 1
                matched = True
            elif agent in B and b_quota > 0 and k[inst] > 0:
                m[S.index(agent)] = inst
                k[inst] -= 1
                b_quota -= 1
                matched = True
            elif rem_quota > 0 and k[inst] > 0:
                m[S.index(agent)] = inst
                k[inst] -= 1
                rem_quota -= 1
                matched = True
            elif choicek == len(prefs) - 1:
                matched = True
                if a_quota + b_quota + rem_quota == 0:
                    break_flag=True           
            else:
                # choicek + 1, 
                choicek += 1
        
        if break_flag:
            break
    
    return m


def inst_wise_bounded (k, sigma, est_util, A, B, bound):
    '''
        k: institutional capacities
        sigma: agent preferences
        est_util: estimated utilities for each agent
        A: list of agents in group A
        B: list of agents in group B
        *note: A and B are disjoint
    '''

    n = len(est_util)
    total_cap = sum(k)

    if bound > 0.5:
        print('bound must be less than or equal to 0.5')
        return
    
    a_quota = [int(x * bound) for x in k]
    b_quota = [int(x * bound) for x in k]
    rem_quota = [int(x * (1 - 2 * bound)) for x in k]

    # Run Gale-Shapley and keep going UNLESS a quota is broken

    S = list(A) + list(B)

    # sort agents in decreasing order of estimated utility
    r_unmatched = [S for _, S in sorted(zip(est_util, S), reverse = True)]

    # initialize matching
    m = [-1] * len(S)

    # institutions have strict preferences -- only the top sum(k) utilities will be chosen, go by first - last preference. 

    for agent in r_unmatched:
        prefs = sigma[S.index(agent)]

        break_flag = False
        matched = False
        choicek = 0
        
        # if space in top choice available, match, else go second choice, etc. until open
        while (matched == False):
            inst = prefs[choicek]
            if agent in A and a_quota[inst] > 0:
                # match, matched = True
                m[S.index(agent)] = inst
                a_quota[inst] -= 1
                matched = True
            elif agent in B and b_quota[inst] > 0:
                m[S.index(agent)] = inst
                b_quota[inst] -= 1
                matched = True
            elif rem_quota[inst] > 0:
                m[S.index(agent)] = inst
                rem_quota[inst] -= 1
                matched = True
            elif choicek == len(prefs) - 1:
                matched = True
                if sum(a_quota) + sum(b_quota) + sum(rem_quota) == 0:
                    break_flag=True           
            else:
                # choicek + 1, 
                choicek += 1
        
        if break_flag:
            break
    
    return m