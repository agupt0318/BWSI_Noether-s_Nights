from typing import Callable


class Hamiltonian:
    def __init__(self, calculation_function: Callable[[list[float]], float]):
        """
        :param calculation_function: A function taking the measurement in the Z basis and returning the value of the
                                     Hamiltonian
        """
        self.calculation_function = calculation_function

    def calculate(self, bits: list[bool]):
        values = [-1 if i else 1 for i in bits]
        return self.calculation_function(values)
