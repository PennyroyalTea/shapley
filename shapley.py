import random
import time

import json

from shapley_utils import train_on_subset
from shapley_utils import save_res

def monte_carlo(train_set, val_set, model_class, ITER=100, perf_tolerance=0.005):
    perm_to_score = dict()
    try:
        with open('score.json', 'r') as f:
            perm_to_score = json.load(f)
            print('score.json loaded')
    except FileNotFoundError:
        print('score.json doesnt exist')


    shapley_vals = [0] * len(train_set)
    permutation = list(range(len(train_set))) # [0, 1, ..., len(train_set)]

    print('full exp')

    # score of fully-trained model
    if str(sorted(permutation)) in perm_to_score.keys():
        print(f'hashed res: {perm_to_score[str(sorted(permutation))]}')
        V_D = perm_to_score[str(sorted(permutation))]
    else:
        print('not hashed, calculate')
        V_D = train_on_subset(
            model_class(),
            train_set, val_set,
            permutation,
            exp_id='full',
        )
        print(f'V_D:{V_D}')
        save_res(perm_to_score, sorted(permutation), V_D)



    print('empty exp')
    # score of untrained model
    if str([]) in perm_to_score.keys():
        print(f'hashed res: {perm_to_score[str([])]}')
        V_current = perm_to_score[str([])]
    else:
        print('not hashed, calculate')
        V_current = train_on_subset(
            model_class(),
            train_set, val_set,
            [],
            exp_id='empty'
        )
        print(f'V_cur:{V_current}')
        save_res(perm_to_score, [], V_current)

    exp_id = 0
    print('start iterating')
    for t in range(1, ITER + 1):  # convergence criteria ???
        print(f'iter {t - 1}')
        random.shuffle(permutation)
        for i in range(len(train_set)):
            # print(f'i:{i}')
            V_prev = V_current
            if abs(V_D - V_current) > perf_tolerance:
                print(f'{exp_id} exp')
                # score on first i datapoints
                if str(sorted(permutation[:i])) in perm_to_score.keys():
                    print(f'hashed res: {perm_to_score[str(sorted(permutation[:i]))]}')
                    V_current = perm_to_score[str(sorted(permutation[:i]))]
                else:
                    print('not hashed, calculate')
                    V_current = train_on_subset(
                        model=model_class(),
                        train_set=train_set,
                        val_set=val_set,
                        permutation=permutation[:i],
                        exp_id=exp_id
                    )
                    print(f'V_cur:{V_current}')
                    save_res(perm_to_score, permutation[:i], V_current)

                exp_id += 1

            shapley_vals[permutation[i]] = ((t - 1) * shapley_vals[permutation[i]] + (
                        V_current - V_prev)) / t  # update shapley values
    return shapley_vals

def gradient_descend():
    # TODO: implement grad descend approach
    pass