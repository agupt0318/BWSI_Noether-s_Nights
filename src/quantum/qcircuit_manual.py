# second half: part 1

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import C4XGate

qreg_key = QuantumRegister(10, 'key')
qreg_data = QuantumRegister(8, 'data')
creg_c = ClassicalRegister(18, 'c')
circuit = QuantumCircuit(qreg_key, qreg_data, creg_c)

circuit.cx(qreg_key[5], qreg_data[5])
circuit.cx(qreg_key[7], qreg_data[0])
circuit.cx(qreg_key[1], qreg_data[1])
circuit.cx(qreg_key[6], qreg_data[2])
circuit.barrier(qreg_key[0], qreg_key[1], qreg_key[2], qreg_key[3], qreg_key[4], qreg_key[5], qreg_key[6], qreg_key[7],
                qreg_key[8], qreg_key[9], qreg_data[0], qreg_data[1], qreg_data[2], qreg_data[3], qreg_data[4],
                qreg_data[5], qreg_data[6], qreg_data[7])
circuit.cx(qreg_data[2], qreg_data[6])
circuit.append(C4XGate(), [qreg_data[5], qreg_data[2], qreg_data[0], qreg_data[1], qreg_data[6]])
circuit.ccx(qreg_data[0], qreg_data[1], qreg_data[6])
circuit.x(qreg_data[0])
circuit.ccx(qreg_data[0], qreg_data[1], qreg_data[6])
