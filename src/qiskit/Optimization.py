
from qiskit.primitives import Estimator
from qiskit_algorithms.gradients import LinCombEstimatorGradient
from qiskit_algorithms.optimizers import GradientDescent, NELDER_MEAD


optimizer_gd = GradientDescent(maxiter=200, learning_rate=1.08)
optimizer_n_m = NELDER_MEAD(maxiter=200)
estimator = Estimator()
grad = LinCombEstimatorGradient(estimator)  # optional estimator gradient
