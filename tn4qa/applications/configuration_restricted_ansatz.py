"""
This application considers the construction of a number-preserving ansatz with few variational parameters.
"""

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import Aer
from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

IBMProvider.save_account(token="", overwrite=True)


# Use the Aer simulator
backend = Aer.get_backend("aer_simulator")

# Use the FakeSherbrooke backend
fake_backend = FakeSherbrooke()


# Define registers and circuit
num_counters = 4
num_qubits = 16
c = ClassicalRegister(num_counters, "c")  # Classical register for measurements

counter = QuantumRegister(num_counters, "counter")  # Counter register
q = QuantumRegister(num_qubits, "q")  # Main quantum register
ancilla = QuantumRegister(1, "ancilla")  # Ancilla qubit for phase kickback

# Create the quantum circuit
circuit = QuantumCircuit(q, counter, ancilla, c)

# Apply Hadamard gates to all qubits in the main register
for i in range(num_qubits):
    circuit.h(q[i])

# Initialise the ancilla
circuit.h(ancilla[0])
circuit.z(ancilla[0])

# Controlled operations to prepare counter register
for i in range(num_qubits):
    for j in range(num_counters - 1, -1, -1):  # Apply multi-controlled Toffoli gates
        circuit.mcx(control_qubits=[q[i]] + counter[:j], target_qubit=counter[j])

circuit.barrier()

# Phase kickback for the state |0110⟩
circuit.x(counter[0])
circuit.x(counter[3])
circuit.mcx(
    control_qubits=counter[:num_counters], target_qubit=ancilla[0]
)  # Phase kickback
circuit.x(counter[0])
circuit.x(counter[3])

# Uncompute the ancilla and counter register
circuit.barrier()

# Undo the counter preparation
for i in range(num_qubits - 1, -1, -1):  # Reverse order
    for j in range(num_counters):
        circuit.mcx(control_qubits=[q[i]] + counter[:j], target_qubit=counter[j])

# Reset the ancilla to its initial state
circuit.z(ancilla[0])
circuit.h(ancilla[0])

circuit.barrier()

# Measure the counter register
for i in range(num_counters):
    circuit.measure(counter[i], c[i])

# Draw the circuit
circuit.draw()

compiled_circuit = transpile(circuit, backend=fake_backend)

# Run the circuit
job = backend.run(compiled_circuit, shots=1000)
result = job.result()
counts = job.result().get_counts()

print("RESULT:", counts, "\n")

for outcome, count in counts.items():
    print(f"{outcome} observed {count} times")
