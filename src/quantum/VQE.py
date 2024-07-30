import random

from numpy import ndarray
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator as AerEstimator

from classical.Optimization import CostFunctionEvaluation
from classical.util import bitstring_10


class HamiltonianEstimator:
    def __init__(
            self,
            ansatz: QuantumCircuit,
            hamiltonian_operator: SparsePauliOp,
            shots_per_estimate: int = 20,
            seed: int | None = 170,
    ):
        self.ansatz = ansatz

        self.hamiltonian_op = hamiltonian_operator

        self.seed = seed if seed is not None else random.randint(0, 0xffffffff)

        self.estimator = AerEstimator(
            run_options={"seed": seed, "shots": shots_per_estimate},
            transpile_options={"seed_transpiler": seed},
        )

    def evaluate_cost(self, ansatz_parameter_values: ndarray) -> CostFunctionEvaluation[bitstring_10]:
        assert len(ansatz_parameter_values) == self.ansatz.num_parameters

        estimator_result = self.estimator.run(
            self.ansatz,
            self.hamiltonian_op,
            ansatz_parameter_values
        ).result()

        hamiltonian_expectation_value: float = estimator_result.values[0]

        # noinspection PyTypeChecker
        return CostFunctionEvaluation(
            ansatz_parameter_values.copy(),
            hamiltonian_expectation_value,
            tuple([False] * 10)
        )