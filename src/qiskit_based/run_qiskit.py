from qiskit import QuantumRegister
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import GradientDescent, NELDER_MEAD
from qiskit_algorithms.utils import algorithm_globals

from CostFunctionQiskit import get_hamil_pauli_op
from VQE_circuit_for_qiskit import VQE_circuit_qiskit
from classical.S_DES import *
from classical.util import *
from quantum.QuantumSDES import QuantumSDES

counts = []
values = []
import pylab


def store_intermediate_result(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)


if __name__ == "__main__":
    # ansatz according to paper following qiskit tutorial
    key_register = QuantumRegister(10)
    text_register = QuantumRegister(8)
    # wavefunction = QuantumCircuit(key_register, text_register)
    # params = ParameterVector("theta", length=10)
    # it = iter(params)
    # for qubit in range(len(wavefunction.qubits[:])):
    #     wavefunction.h(qubit)
    #     wavefunction.ry(next(it), qubit)
    # for qubit in range(len(wavefunction.qubits[:])-1):
    #     wavefunction.cx(qubit, qubit+1)
    # wavefunction.cx(len(wavefunction.qubits[:])-1, 0)
    #
    # wavefunction.barrier()
    # wavefunction.compose(QuantumSDES(key=key_register, data=text_register), inplace=True)
    # wavefunction.barrier()

    known_plaintext = generate_random_message()
    secret_key = generate_random_key()
    known_ciphertext = encrypt_sdes(known_plaintext, secret_key)

    circuit = VQE_circuit_qiskit(known_plaintext)


    def callback_fn(i, a, f, d):
        print('Called energy estimation function')


    print(
        f'Testing with key={bits_to_string(secret_key)}, message={bits_to_string(known_plaintext)}, ciphertext={bits_to_string(known_ciphertext)}')
    # gradient descent vqe:
    hamiltonian = get_hamil_pauli_op(known_ciphertext=known_ciphertext)
    print(f"Number of qubits: {hamiltonian.num_qubits}")
    ansatz = circuit
    circuit_with_measurements = circuit.copy()
    circuit_with_measurements.measure_all()

    gd = GradientDescent(maxiter=4000, learning_rate=0.05)
    nm = NELDER_MEAD(maxiter=None, maxfev=1000, disp=False, xatol=0.0001)

    seed = 170
    algorithm_globals.random_seed = seed

    noiseless_estimator = AerEstimator(
        run_options={"seed": seed, "shots": 1024},
        transpile_options={"seed_transpiler": seed},
    )
    vqe = VQE(noiseless_estimator, ansatz, optimizer=gd, callback=store_intermediate_result)
    result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)

    measurements = AerSimulator().run(
        circuit_with_measurements.assign_parameters(result.optimal_point),
        shots=100,
        memory=True
    ).result().get_memory()

    # Calculate expected value of hamiltonian
    total = 0
    ciphertexts_found: dict[str, int] = dict()
    for measurement in measurements:
        measured_measurement = bits_to_string(QuantumSDES.get_message_from_measurement(measurement))
        if measured_measurement not in ciphertexts_found:
            ciphertexts_found[measured_measurement] = 0
        ciphertexts_found[measured_measurement] += 1

    print(ciphertexts_found)

    print(f"VQE on Aer qasm simulator (no noise): {result.eigenvalue.real:.5f}")

    pylab.rcParams["figure.figsize"] = (12, 4)
    pylab.plot(counts, values)
    pylab.xlabel("Eval count")
    pylab.ylabel("Energy")
    pylab.title("Convergence with no noise")
    pylab.show()

    # vqe_gd = VQE(estimator=estimator, ansatz=circuit, optimizer=optimizer_gd, gradient=grad, callback=callback_fn)
    # hamiltonian = get_hamil_pauli_op(known_ciphertext=known_ciphertext)
    # result = vqe_gd.compute_minimum_eigenvalue(hamiltonian)
    print("Result for optimum of Gradient Descent VQE:", result)
    # nelder-mead vqe:
    # vqe_nm = VQE(estimator=estimator, ansatz=circuit, optimizer=optimizer_n_m, gradient=grad)
    # result = vqe_gd.compute_minimum_eigenvalue(get_hamil_pauli_op(known_ciphertext=known_ciphertext))
    # print("Result of Nelder-Mead VQE:", result.optimal_value)
