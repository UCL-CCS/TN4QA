from qiskit import QuantumCircuit

from ..mpo import MatrixProductOperator
from ..quantum_algorithms.variational.ansatz_circuits import identity_brickwork_circuit
from .mpo_to_circuit import MPOOptimiser


class CircuitCompression:
    """A class to build shallow quantum circuit approximations"""

    def __init__(self, circuit: QuantumCircuit, max_bond: int | None = None):
        """Constructor

        Args:
            circuit: The quantum circuit to compress
            max_bond: The maximum bond dimension to use in the MPO for circuit
        """
        self.circuit = circuit
        self.max_bond = max_bond
        self.mpo = MatrixProductOperator.from_qiskit_circuit(circuit, max_bond=max_bond)

    def run(
        self,
        num_optimisation_sweeps: int = 10,
        ansatz: QuantumCircuit | None = None,
        layers: int | None = None,
    ) -> QuantumCircuit:
        """Run the circuit compression

        Args:
            num_optimisation_sweeps: The number of optimisation sweeps to run
            ansatz: Optionally provide an ansatz circuit, defaults to a brickwork circuit
            layers: The number of layers for the default brickwork ansatz, defaults to num_qubits

        Returns:
            A quantum circuit that approximates the input circuit
        """
        if layers is None:
            layers = self.circuit.num_qubits
        if ansatz is None:
            ansatz = identity_brickwork_circuit(self.circuit.num_qubits, layers=layers)
        optimiser = MPOOptimiser(ansatz, self.mpo)
        opt_circ = optimiser.run(num_sweeps=num_optimisation_sweeps)
        return opt_circ
