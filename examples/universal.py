import os,sys
sys.path.insert(0, os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from pynol.environment.domain import Ball
from pynol.environment.environment import Environment
from pynol.environment.loss_function import SquareLoss
from pynol.learner.base import OGD
from pynol.learner.models.universal.usc import USC
from pynol.online_learning import multiple_online_learning
from pynol.utils.data_generator import LinearRegressionGenerator
from pynol.utils.plot import plot

T, dimension, stage, R, Gamma, scale, seed = 10000, 3, 100, 1, 1, 1 / 2, 0
feature, label = LinearRegressionGenerator().generate_data(
    T, dimension, stage, R, Gamma, seed=seed)
D, r = 2 * R, R
G = scale * D * Gamma**2
C = scale * 1 / 2 * (D * Gamma)**2
L_smooth = Gamma**2

seeds = range(5)
domain = Ball(dimension=dimension, radius=R)
min_step_size, max_step_size = D / (G * T**0.5), D / G
ogd = [OGD(domain, step_size=min_step_size, seed=seed) for seed in seeds]
usc = [USC(domain, T, G, seed=seed) for seed in seeds]

learners = [ogd, usc]
labels = ['OGD', 'USC']

if __name__ == "__main__":
    loss_func = SquareLoss(feature=feature, label=label, scale=scale)
    env = Environment(func_sequence=loss_func)
    _, loss, _, _ = multiple_online_learning(T, env, learners)
    if os.path.exists('./results') is False:
        os.makedirs('./results')
    plot(loss, labels, file_path='./results/universal.pdf')