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

if __name__ == "__main__":
    # 为什么 seed=0 跑的那么好？为什么 OGD-Str 跑的那么差？
    T, KlogT, multiple, loss_func_seed = 10000, False, 1000, 0
    # using new loss func, with comparator fixed
    # and each dT with fixed k randomized inside, which is either O(log T) or O(log^2 T)

    loss_func = ContaminatedStronglyConvexForLowerBound(
       T=T, KlogT=KlogT, multiple=multiple, seed=loss_func_seed)
    # loss_func = ContaminatedOCO_E2(seed=loss_func_seed)

    dimension, D, G, sigma = loss_func.dim, loss_func.D, loss_func.G, loss_func.sigma
    domain = Ball(dimension=dimension, radius=D/2)
    
    seeds = range(4)
    ogd_con = [OGD(domain, step_size=[D/(G*np.sqrt(t+1)) for t in range(T)], seed=seed) for seed in seeds]
    ogd_str = [OGD(domain, step_size=[1/(sigma*(t+1)) for t in range(T)], seed=seed) for seed in seeds]
    # ogd_str_1 = [OGD(domain, step_size=[1/(4*sigma*(t+1)) for t in range(T)], seed=seed) for seed in seeds]
    # ogd_str_2 = [OGD(domain, step_size=[1/(16*sigma*(t+1)) for t in range(T)], seed=seed) for seed in seeds]
    # ogd_str_3 = [OGD(domain, step_size=[1/(64*sigma*(t+1)) for t in range(T)], seed=seed) for seed in seeds]
    usc = [USC(domain, T, G, seed=seed) for seed in seeds]

    learners = [usc] #, ogd_con, ogd_str, ogd_str_1, ogd_str_2, ogd_str_3]
    labels = ['USC'] #, 'OGD-Con', 'OGD-Str', 'OGD-Str-1', 'OGD-Str-2', 'OGD-Str-3']
    
    env = Environment(func_sequence=loss_func, use_surrogate_grad=False)
    
    _, loss, _, _ = multiple_online_learning(T, env, learners)
    loss = np.cumsum(loss, axis=2)
    loss_exp = np.exp(loss) # take exponential on regret to see if it's O(log T)
    if os.path.exists('./results') is False:
        os.makedirs('./results')
    assume = ['K', 'KlogT']
    plot(loss, labels, cum=False, 
         title=f'acc-loss, verify Ω(logT+sqrt({assume[KlogT]})), m={multiple}, seed={loss_func_seed}',
         file_path='./results/tmp.pdf')
    plot(loss_exp, labels, cum=False,
         title=f'acc-loss-exp, verify Ω(logT+sqrt({assume[KlogT]})), m={multiple}, seed={loss_func_seed}',
         file_path='./results/tmp-e.pdf')
    merger = PyPDF2.PdfMerger()
    merger.append('./results/tmp.pdf')
    merger.append('./results/tmp-e.pdf')
    merger.write(f'./results/con-verlb-{assume[KlogT]}-m{multiple}-s{loss_func_seed}.pdf')
    merger.close()
    os.remove('./results/tmp.pdf')
    os.remove('./results/tmp-e.pdf')