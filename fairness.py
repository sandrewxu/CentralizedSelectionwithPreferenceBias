import numpy as np

def ptop1(results, preferences, cut, n=0):
    '''
    ptop1 as defined in paper
    results: return the final matchings for each agent
    preferences: return the preference list of each agent
    cut: index in results where normal ends
    n: optional argument for JEE, total proportion of general population
    '''

    first_choices = [pref[0] for pref in preferences]
    first_achieved = [1 if x == y else 0 for x, y in zip(results, first_choices)]
    first_norm = sum(first_achieved[:cut])
    first_beta = sum(first_achieved[cut:])

    if n==0:
        beta_ratio = first_beta / (len(results) - cut)
        norm_ratio = first_norm / cut
    else:
        beta_ratio = first_beta / (1 - n)
        norm_ratio = first_norm / n

    ratio = beta_ratio / norm_ratio
    return ratio if ratio <= 1 else 1 / ratio

def ptop5(results, preferences, cut, n=0):
    '''
    ptop5 as defined in paper
    '''
    
    # scoring_vec = [1 / np.log(i + 2) for i in range(5)]
    # norm = sum(scoring_vec)
    # scoring_vec = [x / norm for x in scoring_vec]

    res_positions = []

    for agent_prefs, matching in zip(preferences, results):
        if matching == -1:
            res_positions.append(-1)
        else:
            res_positions.append(agent_prefs.tolist().index(matching) + 1)

    top5_scores = []

    for i in range(len(results)):
        ind = res_positions[i]
        if ind > 0 and ind <= 3:
            top5_scores.append(1)
        else:
            top5_scores.append(0)
    
    top5_norm = sum(top5_scores[:cut])
    top5_beta = sum(top5_scores[cut:])

    if n==0:
        beta_ratio = top5_beta / (len(results) - cut)
        norm_ratio = top5_norm / cut
    else:
        beta_ratio = top5_beta / (1 - n)
        norm_ratio = top5_norm / n

    ratio = beta_ratio / norm_ratio
    return ratio if ratio <= 1 else 1 / ratio

def u (results, utilities, cut, beta=1):
    '''
    U as defined in paper
    '''
    act_sum = 0
    counter = 0

    
    utilities[cut:] = [x / beta for x in utilities[cut:]]

    for i in range(len(results)):
        if results[i] != -1:
            act_sum += utilities[i]
            counter += 1

    utilities = sorted(utilities, reverse=True)
    opt_sum = sum(utilities[:counter])
    # print(f"sum of optimal utilities: {opt_sum}")
    # print(f"sum of actual utilities: {act_sum}")
    # print(f"optimal utilities: {utilities[:len(results)]}")

    return act_sum / opt_sum


def u_dist (results, utilities, cut, beta_dist):
    '''
    U as defined in paper
    '''
    act_sum = 0
    counter = 0

    
    utilities[cut:] = utilities[cut:]/beta_dist

    for i in range(len(results)):
        if results[i] != -1:
            act_sum += utilities[i]
            counter += 1

    utilities = sorted(utilities, reverse=True)
    opt_sum = sum(utilities[:counter])
    # print(f"sum of optimal utilities: {opt_sum}")
    # print(f"sum of actual utilities: {act_sum}")
    # print(f"optimal utilities: {utilities[:len(results)]}")

    return act_sum / opt_sum