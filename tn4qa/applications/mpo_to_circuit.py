from qiskit import QuantumCircuit

from ..mps import MatrixProductOperator
from ..tn import TensorNetwork


class MPOOptimiser:
    """
    A class for locally optimising a quantum circuit with respect to a reference MPO and the HS distance
    """

    def __init__(self, qc: QuantumCircuit, reference: MatrixProductOperator) -> None:
        """
        Constructor

        Args:
            qc: The quantum circuit that will be optimised
            reference: The reference MPO
        """
        self.num_qubits = qc.num_qubits
        self.tn = TensorNetwork.from_qiskit_circuit(qc)
        self.tn_dag = TensorNetwork.from_qiskit_circuit(qc, dagger=True)
        for t in self.tn.tensors:
            t.labels.append(f"variational_site_{self.tn.tensors.index(t)+1}")
        for t in self.tn_dag.tensors:
            t.labels.append(
                f"variational_site_{len(self.tn_dag.tensors)-self.tn_dag.tensors.index(t)}"
            )
        self.reference = reference

    # def trace_rrdag(self) -> complex:
    #     """
    #     Calculate Tr(RR^dag) where R is the reference MPO
    #     """
    #     r1 = copy.deepcopy(self.reference)
    #     r2 = copy.deepcopy(self.reference)
    #     return hilbert_schmidt_inner_product(r1, r2)

    # def trace_rtdag(self) -> complex:
    #     """
    #     Calculate Tr(RT^dag) where R is the reference MPO and T is the TN
    #     """
    #     r = copy.deepcopy(self.reference)
    #     t = copy.deepcopy(self.tn)
    #     return 0

    # def trace_trdag(self) -> complex:
    #     """
    #     Calculate Tr(TR^dag) where R is the reference MPO and T is the TN
    #     """
    #     return self.trace_rtdag().conjugate()

    # def trace_ttdag(self) -> complex:
    #     """
    #     Calculate Tr(TT^dag) where T is the TN
    #     """
    #     return 2**self.num_qubits

    # def calculate_fidelity(self) -> float:
    #     """
    #     Calculate the squared Frobenius norm between the reference MPO and the TN
    #     """
    #     fid = (
    #         self.trace_rrdag()
    #         - self.trace_rtdag()
    #         - self.trace_trdag()
    #         + self.trace_ttdag()
    #     )
    #     return fid.real  # It will be real anyway

    # def build_trace_rtdag_tn(self) -> TensorNetwork:
    #     return

    # def build_trace_ttdag_tn(self) -> TensorNetwork:
    #     return

    # def get_environment_matrix(self, variational_index: int) -> ndarray:
    #     return

    # def get_environment_vector(self, variational_index: int) -> ndarray:
    #     return

    # def solve_linear_system(self, env_mat: ndarray, env_vec: ndarray) -> ndarray:
    #     return

    # def get_closest_unitary(self, mat: ndarray) -> ndarray:
    #     return

    # def local_update(self, variational_index: int) -> None:
    #     return

    # def run(self, num_sweeps: int) -> TensorNetwork:
    #     return
