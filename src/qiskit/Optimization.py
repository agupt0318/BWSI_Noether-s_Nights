import inspect
import logging
import pprint
import unittest
from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from enum import IntEnum
from functools import partial
from typing import Any, Callable, Dict, Optional, Union, Iterator

import numpy as np
from numpy import ndarray
from scipy.optimize import Bounds, minimize

type cost_function_t[Data] = Callable[[ndarray], OptimizerGuess[Data]]
type testFunction = Callable[[ndarray], float]


class OptimizerGuess[Data]:
    """
    A class representing an optimizer guess. It contains the guessed point, the cost function at the point, and some
    data for the guess, and can be partially ordered
    """

    def __init__(self, point: ndarray, cost: float, data: Data):
        self.point = point.copy()
        self.cost = cost
        self.data = data

    # Functions to allow comparison of guesses
    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return self.cost == other.cost and np.array_equal(self.point, other.point)

    # Following functions are implemented in terms of __lt__ and __eq__
    def __gt__(self, other):
        return other < self

    def __le__(self, other):
        return not (self > other)

    def __ge__(self, other):
        return not (self < other)

    def __ne__(self, other):
        return not (self == other)


def rosenbrock(x) -> float:
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


class test_Optimization(unittest.TestCase):
    @staticmethod
    def create_cost_function(func: testFunction) -> cost_function_t[None]:
        return lambda x: OptimizerGuess(x.copy(), func(x), None)

    def test_gradient_descent(self):
        s = 10
        ITERATION_COUNT = 200
        test_function = rosenbrock
        optimizer = GradientDescent(maxiter=ITERATION_COUNT, learning_rate=1.08)
        initial_point = np.random.uniform(-s, s, 2),
        x_opt, fx_opt, nfevs = optimizer.optimize(2, test_Optimization.create_cost_function(test_function),
                                                  initial_point)

        print(f"Found minimum {x_opt} at a value of {fx_opt} using {nfevs} evaluations.")


class AlgorithmResult(ABC):
    """Abstract Base Class for algorithm results."""

    def __str__(self) -> str:
        result = {}
        for name, value in inspect.getmembers(self):
            if (
                    not name.startswith("_")
                    and not inspect.ismethod(value)
                    and not inspect.isfunction(value)
                    and hasattr(self, name)
            ):
                result[name] = value

        return pprint.pformat(result, indent=4)

    def combine(self, result: "AlgorithmResult") -> None:
        """
        Any property from the argument that exists in the receiver is
        updated.
        Args:
            result: Argument result with properties to be set.
        Raises:
            TypeError: Argument is None
        """
        if result is None:
            raise TypeError("Argument result expected.")
        if result == self:
            return

        # find any result public property that exists in the receiver
        for name, value in inspect.getmembers(result):
            if (
                    not name.startswith("_")
                    and not inspect.ismethod(value)
                    and not inspect.isfunction(value)
                    and hasattr(self, name)
            ):
                try:
                    setattr(self, name, value)
                except AttributeError:
                    # some attributes may be read only
                    pass


logger = logging.getLogger(__name__)

POINT = Union[float, ndarray]


class OptimizerResult(AlgorithmResult):
    """The result of an optimization routine."""

    def __init__(self) -> None:
        super().__init__()
        self._x = None  # pylint: disable=invalid-name
        self._fun = None
        self._jac = None
        self._nfev = None
        self._njev = None
        self._nit = None

    @property
    def x(self) -> Optional[POINT]:
        """The final point of the minimization."""
        return self._x

    @x.setter
    def x(self, x: Optional[POINT]) -> None:
        """Set the final point of the minimization."""
        self._x = x

    @property
    def fun(self) -> Optional[float]:
        """The final value of the minimization."""
        return self._fun

    @fun.setter
    def fun(self, fun: Optional[float]) -> None:
        """Set the final value of the minimization."""
        self._fun = fun

    @property
    def jac(self) -> Optional[POINT]:
        """The final gradient of the minimization."""
        return self._jac

    @jac.setter
    def jac(self, jac: Optional[POINT]) -> None:
        """Set the final gradient of the minimization."""
        self._jac = jac

    @property
    def nfev(self) -> Optional[int]:
        """The total number of function evaluations."""
        return self._nfev

    @nfev.setter
    def nfev(self, nfev: Optional[int]) -> None:
        """Set the total number of function evaluations."""
        self._nfev = nfev

    @property
    def njev(self) -> Optional[int]:
        """The total number of gradient evaluations."""
        return self._njev

    @njev.setter
    def njev(self, njev: Optional[int]) -> None:
        """Set the total number of gradient evaluations."""
        self._njev = njev

    @property
    def nit(self) -> Optional[int]:
        """The total number of iterations."""
        return self._nit

    @nit.setter
    def nit(self, nit: Optional[int]) -> None:
        """Set the total number of iterations."""
        self._nit = nit


class Minimizer(Protocol):
    """Callable Protocol for minimizer.

    This interface is based on `SciPy's optimize module
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`__.

     This protocol defines a callable taking the following parameters:

         fun
             The objective function to minimize (for example the energy in the case of the VQE).
         x0
             The initial point for the optimization.
         jac
             The gradient of the objective function.
         bounds
             Parameters bounds for the optimization. Note that these might not be supported
             by all optimizers.

     and which returns a minimization result object (either SciPy's or Qiskit's).
    """

    # pylint: disable=invalid-name
    def __call__(
            self,
            fun: Callable[[ndarray], float],
            x0: ndarray,
            jac: Callable[[ndarray], ndarray] | None,
            bounds: list[tuple[float, float]] | None,
    ) -> OptimizeResult | OptimizerResult:
        """Minimize the objective function.

        This interface is based on `SciPy's optimize module <https://docs.scipy.org/doc
        /scipy/reference/generated/scipy.optimize.minimize.html>`__.

        Args:
            fun: The objective function to minimize (for example the energy in the case of the VQE).
            x0: The initial point for the optimization.
            jac: The gradient of the objective function.
            bounds: Parameters bounds for the optimization. Note that these might not be supported
                by all optimizers.

        Returns:
             The minimization result object (either SciPy's or Qiskit's).
        """
        ...


class OptimizerSupportLevel(IntEnum):
    """Support Level enum for features such as bounds, gradient and initial point"""

    # pylint: disable=invalid-name
    not_supported = 0  # Does not support the corresponding parameter in optimize()
    ignored = 1  # Feature can be passed as non None but will be ignored
    supported = 2  # Feature is supported
    required = 3  # Feature is required and must be given, None is invalid


class Optimizer(ABC):
    """Base class for optimization algorithm."""

    @abstractmethod
    def __init__(self):
        """
        Initialize the optimization algorithm, setting the support
        level for _gradient_support_level, _bound_support_level,
        _initial_point_support_level, and empty options.
        """
        self._gradient_support_level = self.get_support_level()["gradient"]
        self._bounds_support_level = self.get_support_level()["bounds"]
        self._initial_point_support_level = self.get_support_level()["initial_point"]
        self._options = {}
        self._max_evals_grouped = 1

    @abstractmethod
    def get_support_level(self):
        """Return support level dictionary"""
        raise NotImplementedError

    def set_options(self, **kwargs):
        """
        Sets or updates values in the options dictionary.

        The options dictionary may be used internally by a given optimizer to
        pass additional optional values for the underlying optimizer/optimization
        function used. The options dictionary may be initially populated with
        a set of key/values when the given optimizer is constructed.

        Args:
            kwargs (dict): options, given as name=value.
        """
        for name, value in kwargs.items():
            self._options[name] = value
        logger.debug("options: %s", self._options)

    # pylint: disable=invalid-name
    @staticmethod
    def gradient_num_diff(x_center, f, epsilon, max_evals_grouped=1):
        """
        We compute the gradient with the numeric differentiation in the parallel way,
        around the point x_center.

        Args:
            x_center (ndarray): point around which we compute the gradient
            f (func): the function of which the gradient is to be computed.
            epsilon (float): the epsilon used in the numeric differentiation.
            max_evals_grouped (int): max evals grouped
        Returns:
            grad: the gradient computed

        """
        forig = f(*((x_center,)))
        grad = []
        ei = np.zeros((len(x_center),), float)
        todos = []
        for k in range(len(x_center)):
            ei[k] = 1.0
            d = epsilon * ei
            todos.append(x_center + d)
            ei[k] = 0.0

        counter = 0
        chunk = []
        chunks = []
        length = len(todos)
        # split all points to chunks, where each chunk has batch_size points
        for i in range(length):
            x = todos[i]
            chunk.append(x)
            counter += 1
            # the last one does not have to reach batch_size
            if counter == max_evals_grouped or i == length - 1:
                chunks.append(chunk)
                chunk = []
                counter = 0

        for chunk in chunks:  # eval the chunks in order
            parallel_parameters = np.concatenate(chunk)
            todos_results = f(parallel_parameters)  # eval the points in a chunk (order preserved)
            if isinstance(todos_results, float):
                grad.append((todos_results - forig) / epsilon)
            else:
                for todor in todos_results:
                    grad.append((todor - forig) / epsilon)

        return np.array(grad)

    @staticmethod
    def wrap_function(function, args):
        """
        Wrap the function to implicitly inject the args at the call of the function.

        Args:
            function (func): the target function
            args (tuple): the args to be injected
        Returns:
            function_wrapper: wrapper
        """

        def function_wrapper(*wrapper_args):
            return function(*(wrapper_args + args))

        return function_wrapper

    @property
    def setting(self):
        """Return setting"""
        ret = f"Optimizer: {self.__class__.__name__}\n"
        params = ""
        for key, value in self.__dict__.items():
            if key[0] == "_":
                params += f"-- {key[1:]}: {value}\n"
        ret += f"{params}"
        return ret

    @property
    def settings(self) -> Dict[str, Any]:
        """The optimizer settings in a dictionary format.

        The settings can for instance be used for JSON-serialization (if all settings are
        serializable, which e.g. doesn't hold per default for callables), such that the
        optimizer object can be reconstructed as

        .. code-block::

            settings = optimizer.settings
            # JSON serialize and send to another server
            optimizer = OptimizerClass(**settings)

        """
        raise NotImplementedError("The settings method is not implemented per default.")

    @abstractmethod
    def minimize(
            self,
            fun: Callable[[POINT], float],
            x0: POINT,
            jac: Optional[Callable[[POINT], POINT]] = None,
            bounds: Optional[list[tuple[float, float]]] = None,
    ) -> OptimizerResult:
        """Minimize the scalar function.

        Args:
            fun: The scalar function to minimize.
            x0: The initial point for the minimization.
            jac: The gradient of the scalar function ``fun``.
            bounds: Bounds for the variables of ``fun``. This argument might be ignored if the
                optimizer does not support bounds.

        Returns:
            The result of the optimization, containing e.g. the result as attribute ``x``.
        """
        raise NotImplementedError()

    @property
    def gradient_support_level(self):
        """Returns gradient support level"""
        return self._gradient_support_level

    @property
    def is_gradient_ignored(self):
        """Returns is gradient ignored"""
        return self._gradient_support_level == OptimizerSupportLevel.ignored

    @property
    def is_gradient_supported(self):
        """Returns is gradient supported"""
        return self._gradient_support_level != OptimizerSupportLevel.not_supported

    @property
    def is_gradient_required(self):
        """Returns is gradient required"""
        return self._gradient_support_level == OptimizerSupportLevel.required

    @property
    def bounds_support_level(self):
        """Returns bounds support level"""
        return self._bounds_support_level

    @property
    def is_bounds_ignored(self):
        """Returns is bounds ignored"""
        return self._bounds_support_level == OptimizerSupportLevel.ignored

    @property
    def is_bounds_supported(self):
        """Returns is bounds supported"""
        return self._bounds_support_level != OptimizerSupportLevel.not_supported

    @property
    def is_bounds_required(self):
        """Returns is bounds required"""
        return self._bounds_support_level == OptimizerSupportLevel.required

    @property
    def initial_point_support_level(self):
        """Returns initial point support level"""
        return self._initial_point_support_level

    @property
    def is_initial_point_ignored(self):
        """Returns is initial point ignored"""
        return self._initial_point_support_level == OptimizerSupportLevel.ignored

    @property
    def is_initial_point_supported(self):
        """Returns is initial point supported"""
        return self._initial_point_support_level != OptimizerSupportLevel.not_supported

    @property
    def is_initial_point_required(self):
        """Returns is initial point required"""
        return self._initial_point_support_level == OptimizerSupportLevel.required

    def print_options(self):
        """Print algorithm-specific options."""
        for name in sorted(self._options):
            logger.debug("%s = %s", name, str(self._options[name]))

    def set_max_evals_grouped(self, limit):
        """Set max evals grouped"""
        self._max_evals_grouped = limit


CALLBACK = Callable[[int, ndarray, float, float], None]


class GradientDescent(Optimizer):
    r"""The gradient descent minimization routine.

    For a function :math:`f` and an initial point :math:`\vec\theta_0`, the standard (or "vanilla")
    gradient descent method is an iterative scheme to find the minimum :math:`\vec\theta^*` of
    :math:`f` by updating the parameters in the direction of the negative gradient of :math:`f`

    .. math::

        \vec\theta_{n+1} = \vec\theta_{n} - \vec\eta\nabla f(\vec\theta_{n}),

    for a small learning rate :math:`\eta > 0`.

    You can either provide the analytic gradient :math:`\vec\nabla f` as ``gradient_function``
    in the ``optimize`` method, or, if you do not provide it, use a finite difference approximation
    of the gradient. To adapt the size of the perturbation in the finite difference gradients,
    set the ``perturbation`` property in the initializer.

    This optimizer supports a callback function. If provided in the initializer, the optimizer
    will call the callback in each iteration with the following information in this order:
    current number of function values, current parameters, current function value, norm of current
    gradient.

    Examples:

        A minimum example that will use finite difference gradients with a default perturbation
        of 0.01 and a default learning rate of 0.01.

        .. code-block::python

            from qiskit.algorithms.optimizers import GradientDescent

            def f(x):
                return (np.linalg.norm(x) - 1) ** 2

            initial_point = np.array([1, 0.5, -0.2])

            optimizer = GradientDescent(maxiter=100)
            x_opt, fx_opt, nfevs = optimizer.optimize(initial_point.size,
                                                      f,
                                                      initial_point=initial_point)

            print(f"Found minimum {x_opt} at a value of {fx_opt} using {nfevs} evaluations.")

        An example where the learning rate is an iterator and we supply the analytic gradient.
        Note how much faster this convergences (i.e. less ``nfevs``) compared to the previous
        example.

        .. code-block::python

            from qiskit.algorithms.optimizers import GradientDescent

            def learning_rate():
                power = 0.6
                constant_coeff = 0.1

                def powerlaw():
                    n = 0
                    while True:
                        yield constant_coeff * (n ** power)
                        n += 1

                return powerlaw()

            def f(x):
                return (np.linalg.norm(x) - 1) ** 2

            def grad_f(x):
                return 2 * (np.linalg.norm(x) - 1) * x / np.linalg.norm(x)

            initial_point = np.array([1, 0.5, -0.2])

            optimizer = GradientDescent(maxiter=100, learning_rate=learning_rate)
            x_opt, fx_opt, nfevs = optimizer.optimize(initial_point.size,
                                                      f,
                                                      gradient_function=grad_f,
                                                      initial_point=initial_point)

            print(f"Found minimum {x_opt} at a value of {fx_opt} using {nfevs} evaluations.")

    """

    def __init__(
            self,
            maxiter: int = 100,
            learning_rate: Union[float, Callable[[], Iterator]] = 0.01,
            tol: float = 1e-7,
            callback: Optional[CALLBACK] = None,
            perturbation: Optional[float] = None,
    ) -> None:
        r"""
        Args:
            maxiter: The maximum number of iterations.
            learning_rate: A constant or generator yielding learning rates for the parameter
                updates. See the docstring for an example.
            tol: If the norm of the parameter update is smaller than this threshold, the
                optimizer is converged.
            perturbation: If no gradient is passed to ``GradientDescent.optimize`` the gradient is
                approximated with a symmetric finite difference scheme with ``perturbation``
                perturbation in both directions (defaults to 1e-2 if required).
                Ignored if a gradient callable is passed to ``GradientDescent.optimize``.
        """
        super().__init__()

        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.perturbation = perturbation
        self.tol = tol
        self.callback = callback

    @property
    def settings(self) -> Dict[str, Any]:
        # if learning rate or perturbation are custom iterators expand them
        if callable(self.learning_rate):
            iterator = self.learning_rate()
            learning_rate = np.array([next(iterator) for _ in range(self.maxiter)])
        else:
            learning_rate = self.learning_rate

        return {
            "maxiter": self.maxiter,
            "tol": self.tol,
            "learning_rate": learning_rate,
            "perturbation": self.perturbation,
            "callback": self.callback,
        }

    def _minimize(self, loss, grad, initial_point):
        # set learning rate
        if isinstance(self.learning_rate, float):
            eta = constant(self.learning_rate)
        else:
            eta = self.learning_rate()

        if grad is None:
            eps = 0.01 if self.perturbation is None else self.perturbation
            grad = partial(
                Optimizer.gradient_num_diff,
                f=loss,
                epsilon=eps,
                max_evals_grouped=self._max_evals_grouped,
            )

        # prepare some initials
        x = np.asarray(initial_point)
        nfevs = 0

        for _ in range(1, self.maxiter + 1):
            # compute update -- gradient evaluation counts as one function evaluation
            update = grad(x)
            nfevs += 1

            # compute next parameter value
            x_next = x - next(eta) * update

            # send information to callback
            stepsize = np.linalg.norm(update)
            if self.callback is not None:
                self.callback(nfevs, x_next, loss(x_next), stepsize)

            # update parameters
            x = x_next

            # check termination
            if stepsize < self.tol:
                break

        return x, loss(x), nfevs

    def get_support_level(self):
        """Get the support level dictionary."""
        return {
            "gradient": OptimizerSupportLevel.supported,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.required,
        }

    # pylint: disable=unused-argument
    def optimize(
            self,
            num_vars,
            objective_function,
            gradient_function=None,
            variable_bounds=None,
            initial_point=None,
    ):
        return self._minimize(objective_function, gradient_function, initial_point)


def constant(eta=0.01):
    """Yield a constant."""

    while True:
        yield eta


def validate_min(name: str, value: float, minimum: float) -> None:
    """
    Args:
        name: value name.
        value: value to check.
        minimum: minimum value allowed.
    Raises:
        ValueError: invalid value
    """
    if value < minimum:
        raise ValueError(f"{name} must have value >= {minimum}, was {value}")


class SciPyOptimizer(Optimizer):
    """A general Qiskit Optimizer wrapping scipy.optimize.minimize.

    For further detail, please refer to
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    _bounds_support_methods = {"l-bfgs-b", "tnc", "slsqp", "powell", "trust-constr"}
    _gradient_support_methods = {
        "cg",
        "bfgs",
        "newton-cg",
        "l-bfgs-b",
        "tnc",
        "slsqp",
        "dogleg",
        "trust-ncg",
        "trust-krylov",
        "trust-exact",
        "trust-constr",
    }

    def __init__(
            self,
            method: Union[str, Callable],
            options: Optional[Dict[str, Any]] = None,
            max_evals_grouped: int = 1,
            **kwargs,
    ):
        """
        Args:
            method: Type of solver.
            options: A dictionary of solver options.
            kwargs: additional kwargs for scipy.optimize.minimize.
            max_evals_grouped: Max number of default gradient evaluations performed simultaneously.
        """
        # pylint: disable=super-init-not-called
        self._method = method.lower() if isinstance(method, str) else method
        # Set support level
        if self._method in self._bounds_support_methods:
            self._bounds_support_level = OptimizerSupportLevel.supported
        else:
            self._bounds_support_level = OptimizerSupportLevel.ignored
        if self._method in self._gradient_support_methods:
            self._gradient_support_level = OptimizerSupportLevel.supported
        else:
            self._gradient_support_level = OptimizerSupportLevel.ignored
        self._initial_point_support_level = OptimizerSupportLevel.required

        self._options = options if options is not None else {}
        validate_min("max_evals_grouped", max_evals_grouped, 1)
        self._max_evals_grouped = max_evals_grouped
        self._kwargs = kwargs

    def get_support_level(self):
        """Return support level dictionary"""
        return {
            "gradient": self._gradient_support_level,
            "bounds": self._bounds_support_level,
            "initial_point": self._initial_point_support_level,
        }

    @property
    def settings(self) -> Dict[str, Any]:
        settings = {
            "max_evals_grouped": self._max_evals_grouped,
            "options": self._options,
            **self._kwargs,
        }
        # the subclasses don't need the "method" key as the class type specifies the method
        if self.__class__ == SciPyOptimizer:
            settings["method"] = self._method

        return settings

    def optimize(
            self,
            num_vars,
            objective_function,
            gradient_function=None,
            variable_bounds: Optional[Union[Sequence, Bounds]] = None,
            initial_point=None,
    ):
        # Remove ignored parameters to supress the warning of scipy.optimize.minimize
        if self.is_bounds_ignored:
            variable_bounds = None
        if self.is_gradient_ignored:
            gradient_function = None

        if self.is_gradient_supported and gradient_function is None and self._max_evals_grouped > 1:
            if "eps" in self._options:
                epsilon = self._options["eps"]
            else:
                epsilon = (
                    1e-8 if self._method in {"l-bfgs-b", "tnc"} else np.sqrt(np.finfo(float).eps)
                )
            gradient_function = Optimizer.wrap_function(
                Optimizer.gradient_num_diff, (objective_function, epsilon, self._max_evals_grouped)
            )

        # Workaround for L_BFGS_B because it does not accept ndarray.
        # See https://github.com/Qiskit/qiskit-terra/pull/6373.
        if gradient_function is not None and self._method == "l-bfgs-b":
            gradient_function = self._wrap_gradient(gradient_function)

        # Validate the input
        super().optimize(
            num_vars,
            objective_function,
            gradient_function=gradient_function,
            variable_bounds=variable_bounds,
            initial_point=initial_point,
        )

        res = minimize(
            fun=objective_function,
            x0=initial_point,
            method=self._method,
            jac=gradient_function,
            bounds=variable_bounds,
            options=self._options,
            **self._kwargs,
        )
        return res.x, res.fun, res.nfev

    @staticmethod
    def _wrap_gradient(gradient_function):
        def wrapped_gradient(x):
            gradient = gradient_function(x)
            if isinstance(gradient, ndarray):
                return gradient.tolist()
            return gradient

        return wrapped_gradient


class NELDER_MEAD(SciPyOptimizer):  # pylint: disable=invalid-name
    """
    Nelder-Mead optimizer.

    The Nelder-Mead algorithm performs unconstrained optimization; it ignores bounds
    or constraints.  It is used to find the minimum or maximum of an objective function
    in a multidimensional space.  It is based on the Simplex algorithm. Nelder-Mead
    is robust in many applications, especially when the first and second derivatives of the
    objective function are not known.

    However, if the numerical computation of the derivatives can be trusted to be accurate,
    other algorithms using the first and/or second derivatives information might be preferred to
    Nelder-Mead for their better performance in the general case, especially in consideration of
    the fact that the Nelder–Mead technique is a heuristic search method that can converge to
    non-stationary points.

    Uses scipy.optimize.minimize Nelder-Mead.
    For further detail, please refer to
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    _OPTIONS = ["maxiter", "maxfev", "disp", "xatol", "adaptive"]

    # pylint: disable=unused-argument
    def __init__(
            self,
            maxiter: Optional[int] = None,
            maxfev: int = 1000,
            disp: bool = False,
            xatol: float = 0.0001,
            tol: Optional[float] = None,
            adaptive: bool = False,
            options: Optional[dict] = None,
            **kwargs,
    ) -> None:
        """
        Args:
            maxiter: Maximum allowed number of iterations. If both maxiter and maxfev are set,
                minimization will stop at the first reached.
            maxfev: Maximum allowed number of function evaluations. If both maxiter and
                maxfev are set, minimization will stop at the first reached.
            disp: Set to True to print convergence messages.
            xatol: Absolute error in xopt between iterations that is acceptable for convergence.
            tol: Tolerance for termination.
            adaptive: Adapt algorithm parameters to dimensionality of problem.
            options: A dictionary of solver options.
            kwargs: additional kwargs for scipy.optimize.minimize.
        """
        if options is None:
            options = {}
        for k, v in list(locals().items()):
            if k in self._OPTIONS:
                options[k] = v
        super().__init__(method="Nelder-Mead", options=options, tol=tol, **kwargs)
