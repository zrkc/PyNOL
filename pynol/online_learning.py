from multiprocessing import Pool
from typing import Union
import time
import numpy as np
import copy

from pynol.environment.environment import Environment
from pynol.learner.base import Base
from pynol.learner.models.model import Model


def online_learning(T, env: Environment, learner: Union[Base, Model]):
    """Combine the environment and learner, start the online learning process.

    Args:
        T (int): Number of total round.
        env (Environment): Environment.
        learner (Base, Model): Learner.

    Returns:
        tuple: tuple contains:
            x (numpy.ndarray): Decisions over :math:`T` rounds. \n
            loss (numpy.ndarray): Losses over :math:`T` rounds. \n
            surrogate_loss (numpy.ndarray): Surrogate losses over :math:`T` rounds. \n
            tm (numpy.ndarray): Time cost over :math:`T` rounds.
    """
    if hasattr(learner, 'domain'):
        dimension = learner.domain.dimension
    else:
        dimension = learner.schedule.bases[0].domain.dimension
    x = np.zeros((T, dimension))
    loss, surrogate_loss = np.zeros(T), np.zeros(T)
    tm = np.zeros(T)
    _minF = 0. # min_x sum_{t=1}^T f_t(x)
    for t in range(T):
        start_time = time.time()
        x[t] = learner.get_decision(env[t-1]) # after opt x with optimism
        env.upd_func_with_decision(t, x[t]) # work if non-oblivious
        minF = env.func_sequence.minF if hasattr(env.func_sequence, 'minF') else 0.
        x[t], loss[t], surrogate_loss[t] = learner.opt(env[t])
        loss[t] = loss[t] + _minF - minF # sum loss = regret
        _minF = minF
        tm[t] = time.time() - start_time
        if t % 100 == 0 and hasattr(learner, 'meta'): print(t, learner.meta.prob)
    return x, loss, surrogate_loss, tm


def multiple_online_learning(T, env: Environment, learners: list, processes=4):
    """Combine the environment and multiple learners, start the online learning
    process with multiprocessing to speed up.

    Args:
        T (int): Number of total round.
        env (Environment): Environment.
        learners (list): Learners.

    Returns:
        tuple: tuple contains:
            x (numpy.ndarray): Decisions of all learners over :math:`T` rounds. \n
            loss (numpy.ndarray): Losses of all learners over :math:`T` rounds. \n
            surrogate_loss (numpy.ndarray): Surrogate losses of all learners over :math:`T` rounds. \n
            tm (numpy.ndarray): Time cost of all learners over :math:`T` rounds.
    """
    if hasattr(learners[0][0], 'domain'):
        dimension = learners[0][0].domain.dimension
    else:
        dimension = learners[0][0].schedule.bases[0].domain.dimension
    num_learners, num_repeat = len(learners), len(learners[0])
    x = np.zeros((num_learners, num_repeat, T, dimension))
    loss, surrogate_loss = np.zeros((num_learners, num_repeat, T)), np.zeros((num_learners, num_repeat, T))
    tm = np.zeros_like(loss)
    p = Pool(processes=processes)
    results = []
    # copy env in case of non-oblivious
    envs = [[copy.deepcopy(env) for j in range(num_repeat)]
            for i in range(num_learners)]
    for i in range(num_learners):
        for j in range(num_repeat):
            results.append(
                (i, j, p.apply_async(online_learning,
                                     (T, envs[i][j], learners[i][j]))))
    p.close()
    p.join()
    for i, j, result in results:
        x[i][j], loss[i][j], surrogate_loss[i][j], tm[i][j] = result.get()
    return x, loss, surrogate_loss, tm
