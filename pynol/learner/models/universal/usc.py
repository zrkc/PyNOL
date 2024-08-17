from typing import Optional, Union

import numpy as np
from pynol.environment.domain import Domain, Simplex
from pynol.learner.base import OGD
from pynol.learner.meta import Adapt_ML_Prod
from pynol.learner.models.model import Model
from pynol.learner.schedule.schedule import Schedule
from pynol.learner.schedule.ssp import SSP, DiscreteSSP
from pynol.learner.specification.surrogate_meta import InnerSurrogateMeta

class USC(Model):
    """``USC``, a universal strategy for online convex optimization.

    Args:
        domain (Domain): Feasible set for the algorithm.
        T (int): Total number of rounds.
        G (float): Upper bound of gradient.
        prior (str, numpy.ndarray, optional): The initial decisions of all
            base-learners are set as `domain(prior=prior, see=seed)` for the
            algorithm.
        seed (int, optional): The initial decisions of all base-learners are set
            as `domain(prior=prior, see=seed)` for the algorithm.

    References:
        TODO

    """

    def __init__(self,
                 domain: Domain,
                 T: int,
                 G: float,
                 prior: Union[list, np.ndarray, None] = None,
                 seed: Union[int, None] = None):
        D = 2 * domain.R

        # get base learners for convex, sigma-str and alpha-exp functions
        bases_con = [
            OGD(step_size=[D/(G*np.sqrt(t+1)) for t in range(T)], \
                domain=domain, prior=prior, seed=seed)
        ]
        sigma_pool = DiscreteSSP.discretize(1/T, 1) # assume [1/T, 1]
        bases_str = [
            OGD(step_size=[1/(sigma*(t+1)) for t in range(T)], \
                domain=domain, prior=prior, seed=seed)
            for sigma in sigma_pool
        ]
        # TODO: ONS and base_exp
        bases_exp = []
        base_learners = SSP(bases_con + bases_str + bases_exp)
        schedule = Schedule(base_learners)

        # get meta learner Adapt-ML-Prod
        meta = Adapt_ML_Prod(len(base_learners), G, D)

        # get specification
        surrogate_base, surrogate_meta = None, InnerSurrogateMeta()
        
        super().__init__(schedule, meta, surrogate_base=surrogate_base, surrogate_meta=surrogate_meta)
