from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import ADAM, COBYLA, L_BFGS_B, QNSPSA, Optimizer


def get_optimiser_callback(opt_dict: dict[str, any], index: int) -> callable:
    """
    Get callback function for optimisation.

    Args:
        opt_dict: The optimiser callback dictionary
        index: The optimisation iteration numver

    Returns:
        An optimiser callback function
    """

    def optimiser_callback(
        num_function_evals, parameters, function_value, stepsize, flag
    ):
        ind = index
        if num_function_evals and stepsize and flag:
            pass
        ind += 1
        opt_dict["optimisation_number"].append(index)
        opt_dict["optimisation_parameters"].append(parameters)
        opt_dict["optimisation_value"].append(function_value)

    return optimiser_callback


def cobyla_optimiser(
    max_iterations: int,
    convergence_threshold: float,
    opt_dict: dict[str, any],
    index: int = 0,
) -> Optimizer:
    """
    COBYLA optimiser.

    Args:
        max_iterations: The maximum number of optimisation iterations
        convergence_threshold: The convergence threshold
        opt_dict: The dictionary to pass to the optimiser callback setter
        index: The optimisation iteration number

    Returns:
        COBYLA optimiser
    """
    opt_callback = get_optimiser_callback(opt_dict=opt_dict, index=index)
    return COBYLA(
        maxiter=max_iterations, tol=convergence_threshold, callback=opt_callback
    )


def qnspsa_optimiser(
    ansatz: QuantumCircuit,
    max_iterations: int,
    opt_dict: dict[str, any],
    index: int = 0,
) -> Optimizer:
    """
    Quantum natural gradient SPSA optimiser.

    Args:
        ansatz: The ansatz circuit
        max_iterations: The maximum number of iterations
        opt_dict: The dictionary to pass to the optimiser callback setter
        index: The optimisation iteration number

    Returns:
        QNSPSA optimiser
    """
    opt_callback = get_optimiser_callback(opt_dict=opt_dict, index=index)
    fidelity = QNSPSA.get_fidelity(ansatz, sampler=Sampler())
    return QNSPSA(fidelity, maxiter=max_iterations, callback=opt_callback)


def bfgs_optimiser(
    maximum_iterations: int, opt_dict: dict[str, any], index: int = 0
) -> Optimizer:
    """
    BFGS optimiser.

    Args:
        max_iterations: The maximum number of iterations
        opt_dict: The dictionary to pass to the optimiser callback setter
        index: The optimisation iteration number

    Returns:
        BFGS optimiser
    """
    opt_callback = get_optimiser_callback(opt_dict=opt_dict, index=index)
    return L_BFGS_B(maxiter=maximum_iterations, callback=opt_callback)


def adam_optimiser(
    maximum_iterations: int, opt_dict: dict[str, any], index: int = 0
) -> Optimizer:
    """
    ADAM optimiser.

    Args:
        max_iterations: The maximum number of iterations
        opt_dict: The dictionary to pass to the optimiser callback setter
        index: The optimisation iteration number

    Returns:
        ADAM optimiser
    """
    opt_callback = get_optimiser_callback(opt_dict=opt_dict, index=index)
    return ADAM(maxiter=maximum_iterations, callback=opt_callback)
