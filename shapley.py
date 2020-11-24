import random

from shapley_utils import train_on_subset

def monte_carlo(train_set, val_set, model_class, ITER=1000, perf_tolerance=0.1):

    shapley_vals = [0] * len(train_set)
    permutation = list(range(len(train_set))) # [0, 1, ..., len(train_set)]

    # score of fully-trained model
    V_D = train_on_subset(
        model=model_class().cuda(),
        train_set=train_set,
        val_set=val_set,
        permutation=permutation,
        exp_id='full'
    )

    # score of untrained model
    V_current = train_on_subset(
        model=model_class().cuda(),
        train_set=train_set,
        val_set=val_set,
        permutation=[],
        exp_id='empty'
    )

    V_prev = V_current

    exp_id = 0
    for _ in range(ITER):  # convergence criteria ???
        random.shuffle(permutation)
        for i in range(len(dataset)):
            V_prev = V_current
            if abs(V_D - V_current) > perf_tolerance:
                # score on first i datapoints
                V_current = train_on_subset(
                    model=model_class().cuda(),
                    train_set=train_set,
                    val_set=val_set,
                    permutation=permutation[:i],
                    exp_id=exp_id
                )
                exp_id += 1

            shapley_vals[permutation[i]] = ((t - 1) * shapley_vals[permutation[i]] + (
                        V_current - V_prev)) / t  # update shapley values
    ################################


    return shapley_vals

def gradient_descend():
    # TODO: implement grad descend approach
    pass