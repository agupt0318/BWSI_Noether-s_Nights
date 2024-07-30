''' Qiskit VQE algorithm is based off of IBM's function
https://learning.quantum.ibm.com/tutorial/variational-quantum-eigensolver
'''

from quantum.util import Hamiltonian
# Imports
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
# Pre-defined ansatz circuit and operator class for Hamiltonian
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp

# SciPy minimizer routine
from scipy.optimize import minimize

class VQE_Qiskit(QuantumCircuit):
    def __init__(self, num_qubits):
        self.circuit = QuantumCircuit(num_qubits)

    def construct_ansatz(self, hamiltonian):
        ansatz = EfficientSU2(hamiltonian.num_qubits)
        ansatz.decompose().draw("mpl", style="iqp")
        num_params = ansatz.num_parameters

    def cost_func(params, ansatz, hamiltonian, estimator):
        """Return estimate of energy from estimator

        Parameters:
            params (ndarray): Array of ansatz parameters
            ansatz (QuantumCircuit): Parameterized ansatz circuit
            hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
            estimator (EstimatorV2): Estimator primitive instance
            cost_history_dict: Dictionary for storing intermediate results

        Returns:
            float: Energy estimate
        """
        pub = (ansatz, [hamiltonian], [params])
        result = estimator.run(pubs=[pub]).result()
        energy = result[0].data.evs[0]

        cost_history_dict = {
            "prev_vector": None,
            "iters": 0,
            "cost_history": [],
        }
        cost_history_dict["iters"] += 1
        cost_history_dict["prev_vector"] = params
        cost_history_dict["cost_history"].append(energy)
        print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy}]")

        return energy


        #constructing random hamiltonian
        hamiltonian = Hamiltonian
        hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)

        fig, ax = plt.subplots()
        ax.plot(range(cost_history_dict["iters"]), cost_history_dict["cost_history"])
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Cost")
        plt.draw()
        plt.show()


