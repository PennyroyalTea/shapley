import random
import time

from shapley_utils import train_on_subset

from trains import Task

def huy():
    print('111')

def run_train_on_subs(
        controller_task, model, train_set,
        val_set, permutation, exp_id):

    child = controller_task.create_function_task(
        huy
    )
    # child = controller_task.create_function_task(
    #     train_on_subset,
    #     arguments={
    #         'model': model,
    #         'train_set': train_set,
    #         'val_set': val_set,
    #         'permutation': permutation,
    #         'exp_id': exp_id
    #     },
    #     func_name=f'subset_{exp_id}'
    # )

    Task.enqueue(child, queue_name='default')
    child.wait_for_status(status=['completed'])
    child.reload()
    return child.get_last_scalar_metrics()

def monte_carlo(train_set, val_set, model_class, ITER=1000, perf_tolerance=0.1):

    controller_task = Task.init(project_name="astral", task_name="controller")


    shapley_vals = [0] * len(train_set)
    permutation = list(range(len(train_set))) # [0, 1, ..., len(train_set)]

    print('full exp')
    # score of fully-trained model
    V_D = run_train_on_subs(
        controller_task=controller_task,
        model=model_class().cuda(),
        train_set=train_set,
        val_set=val_set,
        permutation=permutation,
        exp_id='full'
    )

    print('empty exp')
    # score of untrained model
    V_current = run_train_on_subset(
        controller_task=controller_task,
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
        for i in range(len(train_set)):
            V_prev = V_current
            if abs(V_D - V_current) > perf_tolerance:
                print(f'{ex_id} exp')
                # score on first i datapoints
                V_current = run_train_on_subs(
                    controller_task=controller_task,
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