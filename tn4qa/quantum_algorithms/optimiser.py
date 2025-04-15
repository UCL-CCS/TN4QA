from qiskit_algorithms.optimizers import COBYLA, QNSPSA, L_BFGS_B, ADAM
from qiskit.primitives import Sampler
from qiskit import QuantumCircuit
from qiskit_algorithms.optimizers import Optimizer

def get_optimiser_callback(opt_dict : dict[str, any], index : int) -> function:
    """
    Get callback function for optimisation.
    """
    def optimiser_callback(num_function_evals, parameters, function_value, stepsize, flag):
        ind = index
        if num_function_evals and stepsize and flag:
            pass
        ind += 1
        opt_dict["optimisation_number"].append(index)
        opt_dict["optimisation_parameters"].append(parameters)
        opt_dict["optimisation_value"].append(function_value)
    
    return optimiser_callback

def cobyla_optimiser(max_iterations : int, convergence_threshold : float, opt_dict : dict[str, any], index : int=0) -> Optimizer:
    """
    COBYLA optimiser.
    """
    opt_callback = get_optimiser_callback(opt_dict=opt_dict, index=index)
    return COBYLA(maxiter=max_iterations, tol=convergence_threshold, callback=opt_callback)

def qnspsa_optimiser(ansatz : QuantumCircuit, max_iterations : int, opt_dict : dict[str, any], index : int=0) -> Optimizer:
    """
    Quantum natural gradient SPSA optimiser.
    """
    opt_callback = get_optimiser_callback(opt_dict=opt_dict, index=index)
    fidelity = QNSPSA.get_fidelity(ansatz, sampler=Sampler())
    return QNSPSA(fidelity, maxiter=max_iterations, callback=opt_callback)

def bfgs_optimiser(maximum_iterations : int, opt_dict : dict[str, any], index : int=0) -> Optimizer:
    """
    BFGS optimiser. 
    """
    opt_callback = get_optimiser_callback(opt_dict=opt_dict, index=index)
    return L_BFGS_B(maxiter=maximum_iterations, callback=opt_callback)

def adam_optimiser(maximum_iterations : int, opt_dict : dict[str, any], index : int=0) -> Optimizer:
    """
    ADAM optimiser.
    """
    opt_callback = get_optimiser_callback(opt_dict=opt_dict, index=index)
    return ADAM(maxiter=maximum_iterations, callback=opt_callback)
