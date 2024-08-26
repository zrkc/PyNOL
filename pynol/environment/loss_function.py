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

    """

    def __init__(self,
                 T: int = 10000,
                 is_contaminated = None,
                 dimension: int = 3, # dim>=3 to ensure non-empty null-space
                 D: float = 10.,
                 G: float = 1.,
                 sigma: float = 1e-1) -> None:
        self.T, self.is_contaminated = T, is_contaminated
        if is_contaminated is None:
            print('Warning: Set no contamination case by default.')
            self.is_contaminated = np.zeros(T)
        self.oblivious = False # non-oblivious by upd_with_decision
        self.eps = 1e-9
        self.dim, self.D, self.R, self.G, self.sigma = dimension, D, D/2, G, sigma
        # calculate comparator loss sum_{t=1}^T f_t(x) = F(x) = Fa x^2 + Fb x
        self.FA, self.FB = 0., np.zeros(dimension) # FA = sigma_{1:t} / 2
        self.argminF, self.minF = np.zeros(dimension), 0.
        self.t, self.fa, self.fb, self.fc = -1, None, None, None
        # f_t = fa x^2 + fb x + fc, fc = min F_{t-1} - min F_t is additional term
        # t indicates which round is the loss function at, and add 1 when upd
        
    def __getitem__(self, t: int):
        if self.t != t: return None
        return lambda x : \
            self.fa * np.dot(x, x) + np.dot(self.fb, x) + self.fc
    
    def upd_with_decision(self, t: int, x_t: np.ndarray):
        """
        get x_t at the beginning of t-th round, calculate f_t
        """
        if self.t + 1 != t:
            print(f"Err: Loss function at round {self.t} upd by x[{t}].")
            raise(SystemError)
        self.t += 1
        _x, _minF = self.argminF, self.minF
        if self.is_contaminated[self.t]:
            # f_t(x) = v * x, s.t. |v| = G, v perpendicular to x_t and last FB
            # v
            _, _, vh = np.linalg.svd(np.array([x_t, self.FB]))
            v = vh[-1] / np.linalg.norm(vh[-1]) * self.G
            if max(abs(np.dot(v, x_t)), abs(np.dot(v, self.FB))) > self.eps:
                print("Err: v is not perpendicular.")
                raise(SystemError)
            self.FB += v
            # upd
            self.argminF, self.minF = self.cal_minimum(self.FA, self.FB)
            self.fa, self.fb, self.fc = 0., v, 0. # - self.minF + _minF
        else:
            # f_t(x) = v * x + \sigma / 2 * (|x-x_t|^2 - |x_t|^2)
            #        = (v - \sigma x_t) * x + \sigma / 2 * |x|^2
            # u in X, s.t. |u - x_{t-1}^*| = G / sigma_{1:t}
            # v = G * (x_t - u) / |x_t - u|
            self.FA += self.sigma / 2
            # u
            if np.linalg.norm(_x) < self.eps: u = np.zeros(self.dim)
            else: u = _x - _x / np.linalg.norm(_x) * (self.G / self.FA / 2)
            if np.linalg.norm(u)> self.R:
                # print(f'u at {self.t}!')
                u = u / np.linalg.norm(u) * self.R
            # v - \sigma x_t
            if np.linalg.norm(x_t - u) < self.eps:
                v = np.ones(self.dim) / np.sqrt(self.dim) * self.G - self.sigma * x_t
            else: v = self.G * (x_t - u) / np.linalg.norm(x_t - u) - self.sigma * x_t
            self.FB += v
            # upd
            self.argminF, self.minF = self.cal_minimum(self.FA, self.FB)
            self.fa, self.fb, self.fc = self.sigma / 2, v, 0. # - self.minF + _minF
        # print(f'loss func {id(self)} upd into {self.t}')
        # print(f'f[{self.t}] = {self.fa, self.fb}')
        # print(f'at round {t}, receive ', x_t)
        # print(f'  f[{t}](x) : fa = {self.fa}, fb = {self.fb}, fc = {self.fc}')
    
    def cal_minimum(self, a, b):
        """
        calculate minimum of F(x) = a x^2 + b x, where |x|_2 <= R
        """
        x = - b / (a * 2)
        norm = np.linalg.norm(x)
        if norm> self.R:
            # print(f'- x* at {self.t}!')
            x = x / norm * self.R
        return x, a * np.dot(x, x) + np.dot(b, x)
    

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