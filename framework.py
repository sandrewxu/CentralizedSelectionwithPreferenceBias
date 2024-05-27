'''
This is an implementation of the framework in the paper 
"Centralized Selection with Preferences in the Presence of Biases."

The parameters of the following class describe the parameters of the 
optimization problem in the paper and empirical section.

The main function is ___
'''

class SelectionFramework:
    def __init__(self, n = 1000, p = 5, k = None, phi = 0.25, dist = "g", beta = 0.75, test = "p1", iter = 50):
        '''
        Initialize the framework:
        args:
            n:      number of agents                integer greater than 0
            p:      number of institutions          integer greater than 0
            k:      institutional capacities        array of length p
            phi:    dispersion parameter            double between 0 and 1
            beta:   disadvantage ratio              double between 0 and 1
            dist:   distribution type               string with value "g", "p"
            test:   test type                       string with value "p1", "p5", or "u"
            iter:   iteration count                 integer greater than 0
        '''

        if k is None:
            k = [100] * p

        self.n = n
        self.p = p
        self.k = k
        self.phi = phi
        self.dist = dist
        self.beta = beta
        self.test = test
        self.iter = iter

        # PLOT SAVING
        self.saveIMG = False
    

    # SETTER FUNCTIONS
    
    def set_n(self, n_):
        self.n = n_
    
    def set_saveIMG(self, saveIMG_):
        self.saveIMG = saveIMG_


    # MAIN FUNCTIONS
    