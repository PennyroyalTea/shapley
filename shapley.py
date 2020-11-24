import random

def monte_carlo(train_set, val_set, model, ITER=1000, perf_tolerance=0.1):

    shapley_vals = [0] * len(train_set)

    ################################
    permutation = list(range(len(train_set))) # [0, 1, ..., len(train_set)]

    V_D = train_on_subset(permutation).eval()  # score of fully-trained model
    V_current = train_on_subset([]).eval()  # score of untrained model
    V_prev = V_current

    for _ in range(ITER):  # convergence criteria
        random.shuffle(permutation)
        for i in range(len(dataset)):
            V_prev = V_current
            if abs(V_D - V_current) > perf_tolerance:
                V_current = train_on_subset(permutation[:i]).eval()  # score on first i datapoints

            shapley_vals[permutation[i]] = ((t - 1) * shapley_vals[permutation[i]] + (
                        V_current - V_prev)) / t  # update shapley values
    ################################


    return shapley_vals

def gradient_descend():
    # TODO: implement grad descend approach
    pass