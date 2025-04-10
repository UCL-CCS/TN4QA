from .qa_base import QuantumAlgorithm
from qiskit import QuantumCircuit

class VariationalAlgorithm(QuantumAlgorithm):

    def __init__(self, ansatz : QuantumCircuit):
        """
        Constructor for the variational algorithms class. 
        """
        super().__init__(ansatz)
        self.parameters = ansatz.parameters
        return

    # def _dmrg_parameter_warm_starting_basic(self, max_bond = 10):

    #     ansatz_circ = self._ansatz
    #     dmrg_solution = self._get_dmrg_solution(max_bond=max_bond)
    #     maxiter = self._max_iterations_warm_starting
    #     params = ansatz_circ.parameters
    #     num_params = len(params)
        
    #     def minimise_func(values):
    #         param_dict = {params[i]:values[i] for i in range(num_params)}
    #         temp_circ = ansatz_circ.assign_parameters(param_dict)
    #         quimb_circ = qiskit_to_quimb_circ(temp_circ)
    #         full_tn = dmrg_solution & quimb_circ.psi
    #         exp_value = np.abs(full_tn.contract_compressed("greedy", output_inds=(), max_bond=max_bond)) ** 2
    #         # print(exp_value)
    #         return 1 - exp_value
        
    #     def optimiser_callback(xk):
    #         val = minimise_func(xk)
    #         self._warm_starting_dict["optimisation_number"].append(len(self._warm_starting_dict["optimisation_number"])+1)
    #         self._warm_starting_dict["optimisation_parameters"].append(xk)
    #         self._warm_starting_dict["optimisation_value"].append(val)
    #         return

    #     res = scipy.optimize.minimize(minimise_func, [random.random() for _ in range(num_params)], method="COBYLA", callback=optimiser_callback, options={"maxiter":maxiter})
    #     self._initial_points = np.array(res.x)
    #     return 
    
    # def _warm_start(self, warm_start, method="tn_parameter_warm_starting_basic"):
    #     if not warm_start:
    #         self._initial_points = None
    #         return
    #     self._clear_warm_starting_dict()
    #     if method == "tn_parameter_warm_starting_basic":
    #         self._initial_points = tn_parameter_warm_starting_basic(self._hamiltonian, self._max_iterations_warm_starting, self._ansatz.parameters, self._ansatz, self._warm_starting_dict)
    #     elif method == "dmrg_parameter_warm_starting_basic":
    #         self._initial_points = dmrg_parameter_warm_starting_basic(self._hamiltonian, self._max_bond, self._max_iterations_warm_starting, self._ansatz, self._warm_starting_dict)
    #     else:
    #         print("Warm starting method not recognised. Must be one of 'tn_parameter_warm_starting_basic', 'dmrg_parameter_warm_starting_basic'")
    #     return 
    