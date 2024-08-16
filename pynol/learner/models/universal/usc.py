from typing import Optional, Union

import numpy as np
from pynol.environment.domain import Domain, Simplex
from pynol.learner.base import OGD
from pynol.learner.meta import Hedge
from pynol.learner.models.model import Model
from pynol.learner.schedule.schedule import Schedule
from pynol.learner.schedule.ssp import SSP, DiscreteSSP

class USC(Model):

    def __init__(self,
                 domain: Domain,
                 T: int,
                 G: float,
                 surrogate: bool = False,
                 prior: Union[list, np.ndarray, None] = None,
                 seed: Union[int, None] = None):
        D = 2 * domain.R

        # get base learners for convex, sigma-str and alpha-exp functions
        bases_con = [
            OGD(step_size=[D/(G*np.sqrt(t+1)) for t in range(T)], \
                domain=domain, prior=prior, seed=seed)
        ]
        sigma_pool = DiscreteSSP.discretize(1/T, 1)
        bases_str = [
            OGD(step_size=[1/(sigma*(t+1)) for t in range(T)], \
                domain=domain, prior=prior, seed=seed)
            for sigma in sigma_pool
        ]
        # TODO: ONS and base_exp
        bases_exp = []
        base_learners = SSP(bases_con + bases_str + bases_exp)

        schedule = Schedule(base_learners)

        # TODO: get meta Adapt-ML-Prod
