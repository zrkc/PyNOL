import os,sys
sys.path.insert(0, os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from pynol.environment.domain import Ball
from pynol.environment.environment import Environment
from pynol.environment.loss_function import ContaminatedStronglyConvexForLowerBound, ContaminatedOCO_E2
from pynol.learner.base import OGD
from pynol.learner.models.universal.usc import USC
from pynol.online_learning import multiple_online_learning
from pynol.utils.data_generator import LinearRegressionGenerator
from pynol.utils.plot import plot

import numpy as np
import PyPDF2

def get_contaminated_seq_by_lambda(T, g): # K = g(T)
    is_cont = np.zeros(T) # first round is no cont by default
    K = 0
    for t in range(1, T):
        if K + 1 <= g(t):
            is_cont[t] = 1
            K += 1
    return is_cont

if __name__ == "__main__":

    T = 10000 # TODO 1
    cont_type = ['No contamination', 'K=(log T)^2']
    cont_type_id = 1 # TODO 2
    if cont_type_id == 0:
        is_cont = get_contaminated_seq_by_lambda(T, lambda t : 0)
    elif cont_type_id == 1:
        is_cont = get_contaminated_seq_by_lambda(T, lambda t : 2 * ((np.log(t)) ** 2))

    loss_func = ContaminatedStronglyConvexForLowerBound(T=T, is_contaminated=is_cont)
 
    dimension, D, G, sigma = loss_func.dim, loss_func.D, loss_func.G, loss_func.sigma
    domain = Ball(dimension=dimension, radius=D/2)
    
    seeds = range(4)
    ogd_con = [OGD(domain, step_size=[D/(G*np.sqrt(t+1)) for t in range(T)], seed=seed) for seed in seeds]
    ogd_str = [OGD(domain, step_size=[1/(sigma*(t+1)) for t in range(T)], seed=seed) for seed in seeds]
    usc = [USC(domain, T, G, seed=seed) for seed in seeds]
    learners = [ogd_con, ogd_str, usc] #
    labels = ['OGD-Con', 'OGD-Str', 'USC'] #
    
    env = Environment(func_sequence=loss_func, use_surrogate_grad=False)
    
    _, loss, _, _ = multiple_online_learning(T, env, learners)
    loss = np.cumsum(loss, axis=2) # already taken pre-sum
    loss_r, labels_r = np.zeros_like(loss), []
    for i in range(loss.shape[0]):
        if labels[i] == 'OGD-Con':
            cmp_order = lambda t : np.sqrt(t+1) # t+1> 0
            labels_r.append(labels[i]+' / sqrt(T)')
        else:
            cmp_order = lambda t : np.log(t+2) # t+2> 1
            labels_r.append(labels[i]+' / log(T)')
        for j in range(loss_r.shape[1]):
            for t in range(loss_r.shape[2]):
                loss_r[i, j, t] = loss[i, j, t] / cmp_order(t)

    exp_name = f'{cont_type[cont_type_id]}, T{T}, D{D}, G{G}, sigma{sigma}'
    if os.path.exists('./results') is False:
        os.makedirs('./results')
    plot(loss, labels, cum=False, title=f'regret, {exp_name}',
         file_path='./results/tmp.pdf')
    plot(loss_r, labels_r, cum=False, title=f'regret / cmp-order, {exp_name}',
         file_path='./results/tmp-r.pdf')
    merger = PyPDF2.PdfMerger()
    merger.append('./results/tmp.pdf')
    merger.append('./results/tmp-r.pdf')
    merger.write(f'./results/{exp_name}.pdf')
    merger.close()
    os.remove('./results/tmp.pdf')
    os.remove('./results/tmp-r.pdf')