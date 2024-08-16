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

seeds = range(3)
domain = Ball(dimension=dimension, radius=R)
min_step_size, max_step_size = D / (G * T**0.5), D / G

meta_kwargs = {
    'domain'    : domain,
    'T' : T,
    'G' : G,
}
base_kwargs = {
    'domain'    : domain,
    'T' : T,
    'G' : G,
}

usc = [USC(domain, T, G, False, seed=seed) for seed in seeds]