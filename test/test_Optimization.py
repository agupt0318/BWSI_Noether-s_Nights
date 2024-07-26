import unittest

import matplotlib
from matplotlib import pyplot as plt

from classical.Optimization import *

type testFunction = Callable[[ndarray], float]


def rosenbrock(x: ndarray) -> float:
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def booth(x: ndarray) -> float:
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


def himmelblau(x: ndarray) -> float:
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2



class test_Optimization(unittest.TestCase):
    @staticmethod
    def create_cost_function(func: testFunction) -> cost_function_t[None]:
        return lambda x: OptimizerGuess(x.copy(), func(x), None)

    def test_gradient_descent(self):
        s = 10
        ITERATION_COUNT = 200
        test_function = himmelblau

        optimizer = GradientDescentOptimizer(
            cost_function=test_Optimization.create_cost_function(test_function),
            cost_cutoff=0.1,
            initial_point=np.random.uniform(-s, s, 2),
            learning_rate=0.1,
        )
        # optimizer = NelderMeadOptimizer(
        #     cost_function=test_Optimization.create_cost_function(test_function),
        #     cost_cutoff=0.1,
        #     dimensionality=2,
        #     random_simplex_generator=lambda: [np.random.uniform(-s, s, 2) for _ in range(3)]
        # )

        for _ in range(ITERATION_COUNT):
            optimizer.step()

        plt.figure()
        plt.subplot(111)
        _points = np.array([[
            np.log(test_function(np.array([x, y])))
            for x in np.arange(-10, 10, 0.05)] for y in
                            np.arange(-10, 10, 0.05)])
        plt.imshow(
            (_points - min(_points.flatten())) / (max(_points.flatten()) - min(_points.flatten())),
            cmap='binary_r', extent=(-10, 10, 10, -10)
        )

        history = optimizer.get_history()
        x_coords = [guess.point[0] for guess in history]
        y_coords = [guess.point[1] for guess in history]
        colors = [color(i) for i in np.linspace(0, 1, len(history))]

        for i in range(len(history) - 1):
            plt.plot(x_coords[i:i + 2], y_coords[i:i + 2], color=colors[i], marker='o')

        plt.xlim(-s, s)
        plt.ylim(-s, s)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Gradient Descent')

        plt.show()
