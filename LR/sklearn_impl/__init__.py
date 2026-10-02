"""CPU Python/NumPy/SciPy 线性回归模型。"""

from .linear import LinearRegression
from .ridge import Ridge
from .lasso import Lasso
from .sgd import SGDRegressor

__all__ = ["LinearRegression", "Ridge", "Lasso", "SGDRegressor"]
