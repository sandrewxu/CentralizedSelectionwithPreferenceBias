from kendall_tau_helpers import *
import top_k_mallows.mallows_kendall as mk
from scipy.stats import truncnorm
from scipy.stats import pareto

### MISCELLANEOUS ###

def generate_mallows_preferences(num_agents, num_institutions, gamma, phi):
    """
    Generate central preferences, A and B, then generate preferences for first n/2 based on A, last n/2 based on B
    """
    preferences = []

    # Generate central rankings
    rhoA = np.random.permutation(num_institutions)
    # rhoA = range(num_institutions)
    rhoB = random_perm_at_dist(rhoA, gamma)

    #print("central rankings generated")

    # Draw preferences from the Mallows distribution specified by central ranking rhoA

    a_pref = mk.sample(m = num_agents // 2, n = num_institutions, phi = phi, s0 = rhoA)
    b_pref = mk.sample(m = num_agents // 2, n = num_institutions, phi = phi, s0 = rhoB)
    # a_pref, b_pref, preferences = [], [], []

    #for _ in range(num_agents // 2):
    #    a_pref.append(draw_from_mallows(rhoA, gamma))
    #    b_pref.append(draw_from_mallows(rhoB, gamma))
    
    preferences = np.concatenate((a_pref, b_pref), axis = 0)

    #print("preferences generated")

    return preferences

def generate_preferences_jee(num_agents, num_institutions, central, phi):
    """
    Generate preferences given a central ranking and phi
    """
    # preferences = []

    #mallows_a = mallows(rhoA, gamma)
    #mallows_b = mallows(rhoB, gamma)

    #preferences.append(mallows_a.rvs(size = num_agents // 2))
    #preferences.append(mallows_b.rvs(size = num_agents // 2))

    preferences = mk.sample(m = num_agents, n = num_institutions, phi = phi, s0 = central)

    #for _ in range(num_agents):
    #    preferences.append(draw_from_mallows(central, phi * num_institutions * (num_institutions - 1) // 2))
    
    #print("preferences generated")

    return preferences

def draw_from_mallows(central_ranking, gamma):
    n = len(central_ranking)
    perturbed_ranking = np.copy(central_ranking)
    max_gamma = n * (n-1) // 2

    # Introduce randomness based on gamma
    for i in range(n):
        if np.random.rand() < gamma / max_gamma:
            j = np.random.randint(0, n)
            perturbed_ranking[i], perturbed_ranking[j] = perturbed_ranking[j], perturbed_ranking[i]

    return perturbed_ranking


# test = generate_preferences_jee(num_agents = 10000, num_institutions = 10, central = range(10), phi = 0.99)

# print(np.mean(test, axis = 0))
# print(test)


def generate_corr_preferences(num_agents, num_institutions, alpha, distribution):
    """
    Generate central preferences, A and B, then generate preferences for first n/2 based on A, last n/2 based on B
    """
    # preferences = []

    # Generate central and sample rankings
    if distribution == 'gaussian':
        central = truncnorm.rvs(a = 0, b = np.inf, size = num_institutions)
        samples = [truncnorm.rvs(a = 0, b = np.inf, size = num_institutions) for _ in range(num_agents)]
    elif distribution == 'pareto':
        central = pareto.rvs(b = 3, size = num_institutions)
        # print(central)
        samples = [pareto.rvs(b = 3, size = num_institutions) for _ in range(num_agents)]
        # print(samples)

    # Input correlation (alpha * central + original)
    samples = [alpha * central + sample for sample in samples]
    samples = np.array(samples)
    # print(samples)

    preferences = np.argsort(-samples)

    # Convert to 0, 1, 2, 3 ...
    
    #print("preferences generated")

    return preferences
