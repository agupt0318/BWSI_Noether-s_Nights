from qiskit import QuantumCircuit, QuantumRegister

def A_ansatz_Y_Cx_model(theta_list):
    register = QuantumRegister(10)
    circuit = QuantumCircuit(register)
    for i in range(10):
        circuit.h(i)
        circuit.ry(theta_list[i],i)
        if i != 9:
            circuit.cx(i,i+1)
    circuit.cx(9,0)
    return circuit

def A_ansatz_Y_Cy_model(theta_list):
    register = QuantumRegister(10)
    circuit = QuantumCircuit(register)
    for i in range(10):
        circuit.h(i)
        circuit.ry(theta_list[i],i)
        if i != 9:
            circuit.cy(i,i+1)
    circuit.cy(9,0)
    return circuit

def A_ansatz_Y_Cz_model(theta_list):
    register = QuantumRegister(10)
    circuit = QuantumCircuit(register)
    for i in range(10):
        circuit.h(i)
        circuit.ry(theta_list[i],i)
        if i != 9:
            circuit.cz(i,i+1)
    circuit.cz(9,0)
    return circuit

def B_ansatz_Y_Cx_model(theta_list):
    register = QuantumRegister(10)
    circuit = QuantumCircuit(register)
    for i in range(10):
        circuit.h(i)
        circuit.ry(theta_list[i],i)
        if i != 9:
            circuit.cx(i,i+1)
    return circuit

def B_ansatz_Y_Cy_model(theta_list):
    register = QuantumRegister(10)
    circuit = QuantumCircuit(register)
    for i in range(10):
        circuit.h(i)
        circuit.ry(theta_list[i],i)
        if i != 9:
            circuit.cy(i,i+1)
    return circuit

def B_ansatz_Y_Cz_model(theta_list):
    register = QuantumRegister(10)
    circuit = QuantumCircuit(register)
    for i in range(10):
        circuit.h(i)
        circuit.ry(theta_list[i],i)
        if i != 9:
            circuit.cz(i,i+1)
    return circuit




