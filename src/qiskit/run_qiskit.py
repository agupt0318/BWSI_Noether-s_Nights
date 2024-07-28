from qiskit.circuit import ParameterVector
from qiskit_algorithms import VQE
from classical.S_DES import *
from classical.util import *
from CostFunctionQiskit import get_hamil_pauli_op
from Optimization import optimizer_gd, optimizer_n_m, estimator, grad
from qiskit import QuantumCircuit

#ansatz according to paper following qiskit tutorial
wavefunction = QuantumCircuit(10)
params = ParameterVector("theta", length=10)
it = iter(params)
for qubit in range(len(wavefunction.qubits[:])):
    wavefunction.h(qubit)
    wavefunction.ry(next(it), qubit)
for qubit in range(len(wavefunction.qubits[:])-1):
    wavefunction.cx(qubit, qubit+1)
wavefunction.cx(len(wavefunction.qubits[:])-1, 0)

known_plaintext = generate_random_message()
secret_key = generate_random_key()
known_ciphertext = encrypt_sdes(known_plaintext, secret_key)
print(f'Testing with key={bits_to_string(secret_key)}, message={bits_to_string(known_plaintext)}, ciphertext={bits_to_string(known_ciphertext)}')
# gradient descent vqe:
vqe_gd = VQE(estimator=estimator, ansatz=wavefunction, optimizer=optimizer_gd, gradient=grad)
result = vqe_gd.compute_minimum_eigenvalue(get_hamil_pauli_op(known_ciphertext=known_ciphertext))
print("Result of Gradient Descent VQE:", result.optimal_value)
# nelder-mead vqe:
vqe_nm = VQE(estimator=estimator, ansatz=wavefunction, optimizer=optimizer_n_m, gradient=grad)
result = vqe_gd.compute_minimum_eigenvalue(get_hamil_pauli_op(known_ciphertext=known_ciphertext))
print("Result of Nelder-Mead VQE:", result.optimal_value)
