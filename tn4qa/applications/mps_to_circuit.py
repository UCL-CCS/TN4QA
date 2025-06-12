from qiskit import QuantumCircuit

from ..mps import MatrixProductState
from ..tn import TensorNetwork


class MPSOptimiser:
    """
    A class for locally optimising a quantum circuit with respect to a reference MPS and the HS distance
    """

    def __init__(self, qc: QuantumCircuit, reference: MatrixProductState) -> None:
        """
        Constructor

        Args:
            qc: The quantum circuit that will be optimised
            reference: The reference MPS
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

    # def ip_rr(self) -> complex:
    #     """
    #     Calculate <R|R> where R is the reference MPS
    #     """
    #     return 1.0 + 0.0j

    # def ip_tr(self) -> complex:
    #     """
    #     Calculate <T|R> where R is the reference MPS and T is the TN
    #     """
    #     r = copy.deepcopy(self.reference)
    #     t = copy.deepcopy(self.tn)
    #     return 0

    # def ip_rt(self) -> complex:
    #     """
    #     Calculate <R|T> where R is the reference MPS and T is the TN
    #     """
    #     return self.ip_tr().conjugate()

    # def ip_tt(self) -> complex:
    #     """
    #     Calculate <T|T> where T is the TN
    #     """
    #     return 1.0 + 0.0j

    # def calculate_fidelity(self) -> float:
    #     """
    #     Calculate the squared Frobenius norm between the reference MPS and the TN
    #     """
    #     fid = self.ip_rr() - self.ip_rt() - self.ip_tr() + self.ip_tt()
    #     return fid.real  # It will be real anyway

    # def build_ip_tr_tn(self) -> TensorNetwork:
    #     return

    # def build_ip_tt_tn(self) -> TensorNetwork:
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
