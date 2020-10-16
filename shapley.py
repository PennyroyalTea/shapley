import random

class Shapley:

    ITER = 1000  # num of iterations before convergence
    dataset = None # dataset to take training points from
    perf_tolerance = 0.1 # some constant need to be researched later
    shapley_vals = []


    def __init__(self, dataset):
        self.dataset = dataset
        self.shapley_vals = [0] * len(dataset)
        pass

    def calculate_shapley_values(self):
        permutation = list(range(len(dataset)))

        V_D = train_on_subset(permutation).eval() # score of fully-trained model
        V_current = train_on_subset([]).eval() # score of untrained model
        V_prev = V_current

        for _ in range(ITER): # convergence criteria
            random.shuffle(permutation)
            for i in range(len(dataset)):
                V_prev = V_current
                if abs(V_D - V_current) > perf_tolerance:
                    V_current = train_on_subset(permutation[:i]).eval() # score on first i datapoints

                shapley_vals[permutation[i]] = ((t-1) * shapley_vals[permutation[i]] + (V_current - V_prev)) / t # update shapley values