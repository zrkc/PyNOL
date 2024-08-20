from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

import numpy as np
import math


class LossFunction(ABC):
    """An abstract class for loss function.

    Users can define their loss functions by inheriting from this class and override the method :meth:`~ pynol.environment.loss_function.LossFunction.__getitem__`.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def __getitem__(self, t: int) -> Callable[[np.ndarray], float]:
        pass


class InnerLoss(LossFunction):
    """This class defines the inner loss function.

    Args:
        feature (numpy.ndarray): Features of the environment.
        scale (float): Scale coefficient of the loss function.

    Example:
    ::

        import numpy as np
        func = InnerLoss(feature=np.random.rand(1000, 5))  # 1000 rounds, 5 dimension

    Then, call ``func[t]`` will return the inner loss function :math:`f_t(x) =
    \langle \\varphi_t, x \\rangle`, where :math:`\\varphi_t` is the feature at
    round :math:`t`.
    """

    def __init__(self, feature: np.ndarray = None, scale: float = 1.) -> None:
        self.feature = feature
        self.scale = scale

    def __getitem__(self, t: int):
        return lambda x: self.scale * np.dot(x, self.feature[t])


class SquareLoss(LossFunction):
    """This class defines the logistic loss function.

    Args:
        feature (numpy.ndarray): Features of the environment.
        label (numpy.ndarray): Labels of the environment.
        scale (float): Scale coefficient of the loss function.

    Example:
    ::

        import numpy as np
        feature, label = np.random.rand(1000, 5), np.random.randint(2, size=1000)
        func = SquareLoss(feature, label)  # 1000 rounds, 5 dimension

    Then, call ``func[t]`` will return the square loss function :math:`f_t(x) =
    \\frac{1}{2} (y_t - \langle \\varphi_t, x \\rangle)^2`, where
    :math:`\\varphi_t` and :math:`y_t` are the feature and label at
    round :math:`t`.
    """

    def __init__(self,
                 feature: np.ndarray = None,
                 label: np.ndarray = None,
                 scale: float = 1.) -> None:
        self.feature = feature
        self.label = label
        self.scale = scale

    def __getitem__(self, t: int) -> Callable[[np.ndarray], float]:
        return lambda x: self.scale * 1 / 2 * (
            (np.dot(x, self.feature[t]) - self.label[t])**2)


class LogisticLoss(LossFunction):
    """This class defines the logistic loss function.

    Args:
        Feature (numpy.ndarray): Features of the environment.
        label (numpy.ndarray): Labels of the environment.
        scale (float): Scale coefficient of the loss function.

    Example:
    ::

        import numpy as np
        feature, label = np.random.rand(1000, 5), np.random.randint(2, size=1000)
        func = LogisticLoss(feature, label)  # 1000 rounds, 5 dimension

    Then, call ``func[t]`` will return the loss function :math:`f_t(x) = y \log (\\frac{1}{1+e^{-\\varphi_t^\\top x}})+(1-y) \log (1-\\frac{1}{1+e^{-\\varphi_t^\\top x}})` where
    :math:`\\varphi_t` and :math:`y_t` are the feature and label at
    round :math:`t`.
    """

    def __init__(self,
                 feature: np.ndarray = None,
                 label: np.ndarray = None,
                 scale: float = 1.) -> None:
        self.feature = feature
        self.label = label
        self.scale = scale

    def __getitem__(self, t: int) -> Callable[[np.ndarray], float]:
        return partial(self.func, t=t)

    def func(self, x, t):
        prediction = 1 / (1 + np.e**(-np.dot(x, self.feature[t])))
        loss = prediction * np.log(
            self.label[t]) + (1 - prediction) * np.log(1 - self.y[t])
        return self.scale * loss


class FuncWithSwitch:
    """This class defines the loss function with switching cost.

    Args:
        f (LossFunction): Origin loss function.
        penalty (float): Penalty coefficient of the switching cost.
        norm (non-zero int, numpy.inf): Order of the norm. The default is 2 norm.
        order (int): Order the the switching cost. The default is 2.

    Example:
    ::

        import numpy as np
        feature, label = np.random.rand(1000, 5), np.random.randint(2, size=1000)
        f = SquareLoss(feature, label)
        func = FuncWithSwitch(f, penalty=1, norm=2, order=2)

    Then, call ``func[t]`` will return the square loss function with switching
    cost :math:`f_t(x) = \\frac{1}{2} (y_t - \langle \\varphi_t, x \\rangle)^2 +
    \lVert x - x_{t-1}\\rVert_2^2` where :math:`\\varphi_t` and :math:`y_t` are
    the feature and label at round :math:`t`.
    """

    def __init__(self,
                 f: LossFunction = None,
                 penalty: float = 1.,
                 norm: int = 2,
                 order: int = 2) -> None:
        self.f = f
        self.penalty = penalty
        self.norm = norm
        self.order = order
        self.x_last = None

    def __getitem__(self, t: int) -> Callable[[np.ndarray], float]:
        return partial(self.func, f=self.f[t])

    def func(self, x: np.ndarray, f: Callable[[np.ndarray], float]):
        assert x.ndim == 1 or x.ndim == 2
        if self.x_last is None:
            self.x_last = x
        if x.ndim == 1:
            loss = f(x) + self.penalty * np.linalg.norm(
                x - self.x_last, ord=self.norm)**self.order
        else:
            loss = f(x) + self.penalty * np.linalg.norm(
                x - self.x_last, ord=self.norm, axis=1)**self.order
        self.x_last = x
        return loss


class HuberLoss(LossFunction):
    """This class defines the huber loss function.

    Args:
        Feature (numpy.ndarray): Features of the environment.
        label (numpy.ndarray): Labels of the environment.
        scale (float): Scale coefficient of the loss function.

    Example:
    ::

        import numpy as np
        feature, label = np.random.rand(1000, 5), np.random.randint(2, size=1000)
        func = HuberLoss(feature, label)  # 1000 rounds, 5 dimension
    """
    def __init__(self,
                 feature: np.ndarray = None,
                 label: np.ndarray = None,
                 threshold: float = 1.,
                 scale: float = 1.) -> None:
        self.feature = feature
        self.label = label
        self.threshold = threshold
        self.scale = scale

    def __getitem__(self, t: int) -> Callable[[np.ndarray], float]:
        return partial(self.func, t=t)

    def func(self, x, t):
        prediction = np.dot(self.feature[t], x)
        if abs(prediction - self.label[t]) < self.threshold:
            return self.scale * 1 / 2 * (prediction - self.label[t])**2
        else:
            return self.scale * (self.threshold * abs(prediction - self.label[t]) - self.threshold**2 / 2)
        

class ContaminatedStronglyConvexForLowerBound(LossFunction):
    """This class defines the a simple 
    :math:`(K\\times{\\rm d} T)`-contaminated
    :math:`\\sigma`-strongly-convex case, where 
    the loss functions are not str-con in :math:`K` rounds
    out of each :math:`{\\rm d} T` rounds.

    Args:
        T (int): Number of total round.
        dT (int): The size of the interval length, factor of T.
        K (int): The contamination constant, does not exceed dT.
        dimension (int): Dimension of the feasible set.
        D (float): Bounded 2-Norm of the decisions.
        sigma (float): Strongly convex coefficient.
        seed (int): For generating bias sequence.

    Example:
    ::

        import numpy as np
        func = ContaminatedStronglyConvex()

    Then, call ``func[t]`` will return the function :math:`f_t(x) =
    \\frac{\\sigma}{2}(\\Vert x+b_t \\Vert_2^2 - \\Vert b_t \\Vert_2^2)` 
    at round :math:`t` if it is not contaminated,
    otherwise :math:`f_t(x) = \\langle b_t, x \\rangle`,
    where :math:`b_t` is sampled from the standard normal distribution. \n
    Besides, the upper bound of gradient norm is calculated as ``self.G``.
    """

    def __init__(self,
                 T: int = 10000,
                 KlogT: bool = True, # True then K=mlogT, else K=m*(logT)^2
                 multiple: int = 5, # m=0 in no contamination case 
                 dimension: int = 3,
                 D: float = 2,
                 sigma: float = 1e-3,
                 seed: int = 0) -> None:
        np.random.seed(seed)
        self.dim, self.D, self.sigma = dimension, D, sigma
        # construct loss func
        self.bias = np.random.randn(T, dimension)
        for _ in range(0, T, 100):
            fixed_bias = np.random.randn(dimension) * 0
            for t in range(_, _+100): self.bias[t] += fixed_bias
        self.is_sc, self.KlogT, self.mul = np.ones(T), KlogT, multiple
        for t in range(1, T): self.is_sc[t] = 0 if self.is_contaminated(t+1) else 1
        # calculate comparator loss
        self.Fmin = np.zeros(T) # min_x sum_{t=1}^T f_t(x) = min_x F(x)
        Fa, Fb = sigma / 2, sigma * self.bias[0] # F(x) = Fa x^2 + Fb x
        self.Fmin[0] = self.cal_minimum(Fa, Fb)
        for t in range(1, T):
            Fa += self.is_sc[t] * sigma / 2
            Fb += sigma * self.bias[t]
            self.Fmin[t] = self.cal_minimum(Fa, Fb)
        # calculate G
        G1, G2 = 0., 0.
        for t in range(T):
            bias_norm = np.linalg.norm(self.bias[t])
            if self.is_sc[t]: G1 = max(G1, sigma * (D + bias_norm))
            else: G2 = max(G2, sigma * bias_norm)
        print(f'G1 = {G1}, G2 = {G2}')
        self.G = max(G1, G2)
        
    def __getitem__(self, t: int):
        b, Fm, _Fm = self.bias[t], self.Fmin[t], self.Fmin[t-1] if t else 0
        if self.is_sc[t]:
            return lambda x: \
                self.sigma / 2 * (np.dot(x+b, x+b) - np.dot(b, b)) - Fm + _Fm
        else:
            return lambda x: \
                self.sigma * np.dot(x, b) - Fm + _Fm
        
    def is_contaminated(self, t):
        """
        decide whether contaminated at (t>0)-th round.
        """
        if self.KlogT: 
            return math.floor(np.log(t+1) * self.mul) \
                 > math.floor(np.log(t) * self.mul)
        else:
            return math.floor((np.log(t+1) ** 2) * self.mul) \
                 > math.floor((np.log(t) ** 2) * self.mul) \
    
    def cal_minimum(self, a, b):
        """
        calculate minimum of F(x) = a x^2 + b x, where |x|_2 <= D
        """
        x = - b / (a * 2)
        norm = np.linalg.norm(x)
        if norm> self.D: x *= self.D / norm
        return a * np.dot(x, x) + np.dot(b, x)
    

class ContaminatedOCO_E2(LossFunction):
    """
    Contaminated Online Convex Optimization,
    Tomoya Kamijima and Shinji Ito,
    C.2 Experiment 2: k-Contaminated Strongly Convex.
    """

    def __init__(self,
                 T: int = 10000,
                 K: int = 1000, # satisfy 2K < T
                 seed: int = 0) -> None:
        np.random.seed(seed)
        self.dim, self.D, self.G, self.sigma = 1, 1., 1., 1.
        contaminated_index = np.random.choice(T, K)
        self.is_contaminated = np.zeros(T)
        for idx in contaminated_index:
            self.is_contaminated[idx] = 1
        self.C1 = - (T - K * 2) / (T - K)
        self.C2 = - ((K / (T - K)) ** 2) / 2
        
    def __getitem__(self, t: int):
        if self.is_contaminated[t]:
            return lambda x: x + self.C1
        else:
            return lambda x: ((x - 1)**2) / 2 + self.C2