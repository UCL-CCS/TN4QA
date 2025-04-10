from .variational import VariationalAlgorithm
from qiskit import QuantumCircuit
from qiskit_algorithms.optimizers import Optimizer
from qiskit.primitives import Estimator
import matplotlib.pyplot as plt
from numpy import ndarray
from typing import Union
from pyscf.gto import Mole
from .ansatz import * 
from .optimiser import *
class VQE(VariationalAlgorithm):

    def __init__(self, ansatz : QuantumCircuit) -> "VQE":
        super().__init__(ansatz)
        return 
    
    def __init__(self,
                 n_qubits : int,
                 hamiltonian : any,
                 max_iterations_vqe : int = 1e3,
                 convergence_threshold : float = 1e-6,
                 initial_points : ndarray = None,
                 ansatz : Union[QuantumCircuit, str] = None,
                 scf_obj : Mole = None,
                 hamiltonian_encoding : str = None,
                 optimiser : Union[Optimizer, str] = None) -> None:
        
        self._nqubits = n_qubits
        self._hamiltonian = hamiltonian
        self._scf_obj = scf_obj
        self._hamiltonian_encoding = hamiltonian_encoding
        self._optimisation_index = 0

        self._clear_optimisation_dict()
        self._clear_warm_starting_dict()

        self.set_max_iterations_vqe(max_iterations=max_iterations_vqe)
        self.set_convergence_threshold(convergence_threshold=convergence_threshold)
        self.set_initial_points(initial_points=initial_points)
        self.set_estimator()
        self.set_ansatz(ansatz=ansatz)
        self.set_optimiser(optimiser=optimiser)
        self.set_callback()

        self._driver = self._vqe_driver()


    def set_initial_points(self, initial_points=None):
        self._initial_points = initial_points
        return
    
    def set_max_iterations_vqe(self, max_iterations=1e3):
        self._max_iterations_vqe = max_iterations
        return
    
    def set_convergence_threshold(self, convergence_threshold=1e-6):
        self._convergence_threshold = convergence_threshold
        return

    def set_estimator(self, estimator=None):
        if not estimator:
            self._estimator = Estimator()
        else:
            self._estimator = estimator
        return
    
    def set_ansatz(self, ansatz=None):
        if not ansatz or ansatz == "number_preserving_ansatz":
            qc = number_preserving_ansatz(self._nqubits)
            self._ansatz = qc
        elif ansatz == "hardware_efficient_ansatz":
            qc = hea_ansatz(self._nqubits)
            self._ansatz = qc
        elif ansatz == "uccsd_ansatz":
            assert self._scf_obj is not None and self._hamiltonian_encoding is not None
            qc = uccsd_ansatz(self._scf_obj, encoding=self._hamiltonian_encoding)
            self._ansatz = qc
        elif ansatz == "pauli_two_design_ansatz":
            qc = pauli_two_design_ansatz(self._nqubits)
            self._ansatz = qc
        elif isinstance(ansatz, str):
            print("Ansatz string not recognised. Must be one of 'hea_ansatz', 'number_preserving_ansatz', 'pauli_two_design_ansatz', 'uccsd_ansatz'")
        else:
            assert isinstance(ansatz, QuantumCircuit)
            self._ansatz = ansatz
        return 
    
    def set_optimiser(self, optimiser=None):
        self._optimisation_index = 0
        if not optimiser or optimiser == "QNSPSA":
            self._optimiser = qnspsa_optimiser(self._ansatz, self._max_iterations_vqe, opt_dict=self._optimisation_dict, index=self._optimisation_index)
        elif optimiser == "ADAM":
            self._optimiser = adam_optimiser(self._max_iterations_vqe, opt_dict=self._optimisation_dict, index=self._optimisation_index)
        elif optimiser == "BFGS":
            self._optimiser = bfgs_optimiser(self._max_iterations_vqe, opt_dict=self._optimisation_dict, index=self._optimisation_index)
        elif optimiser == "COBYLA":
            self._optimiser = cobyla_optimiser(self._max_iterations_vqe, self._convergence_threshold, opt_dict=self._optimisation_dict, index=self._optimisation_index)
        elif isinstance(optimiser, str):
            print("Optimiser string not recognised. Must be one of 'ADAM', 'COBYLA', 'QNSPSA', 'BFGS'")
        else:
            assert isinstance(optimiser, Optimizer)
            self._optimiser = optimiser
        return
    
    def set_callback(self, callback=None):
        if not callback:
            def default_callback(i, a, f, _):
                # print(i, a, f)
                self._optimisation_dict["optimisation_number"].append(i)
                self._optimisation_dict["optimisation_parameters"].append(a)
                self._optimisation_dict["optimisation_value"].append(f)
                return
            self._callback = default_callback
        else:
            self._callback = callback
        return
    
    def _clear_optimisation_dict(self):
        self._optimisation_dict = {"optimisation_number" : [], "optimisation_parameters" : [], "optimisation_value" : []}
        return
    
    def _clear_warm_starting_dict(self):
        self._warm_starting_dict = {"optimisation_number" : [], "optimisation_parameters" : [], "optimisation_value" : []}
        return 
    
    def _vqe_driver(self):
        driver = VQE(self._estimator, self._ansatz, self._optimiser, initial_point=self._initial_points)
        return driver
    
    # def run_vqe(self):
    #     if not isinstance(self._hamiltonian, SparsePauliOp):
    #         self._hamiltonian = self._qubitop_to_pauliop()
    #     observable = self._hamiltonian
    #     self._clear_optimisation_dict()
    #     result = self._driver.compute_minimum_eigenvalue(observable)
    #     self._minimum_eigenvalue = result.eigenvalue
    #     return result
    
    def plot_convergence(self):
        plt.plot(self._optimisation_dict["optimisation_number"], self._optimisation_dict["optimisation_value"])
        plt.show()
        return