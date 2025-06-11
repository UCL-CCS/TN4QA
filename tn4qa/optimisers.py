import copy
from typing import Any

import numpy as np
from numpy import ndarray
from numpy.linalg import eig, svd

from .fidelity_metrics import hilbert_schmidt_inner_product
from .mpo import MatrixProductOperator
from .mps import MatrixProductState
from .tensor import Tensor
from .tn import TensorNetwork


class TNOptimiser(TensorNetwork):
    """
    A base class for optimisers.
    """

    def __init__(self, tn: TensorNetwork, reference: Any) -> None:
        """
        Constructor

        Args:
            tn: The tensor network that will be optimised. Should be contractable to an MPS or MPO
            reference: The reference state or operator
        """
        if isinstance(reference, TensorNetwork):
            tensors = tn.tensors + reference.tensors
        else:
            tensors = tn.tensors
        super().__init__(tensors, name="TNOptimiser")
        self.tn = tn
        self.reference = reference
        label_to_tensor_dict = tn.get_label_to_tensor_dict()
        self.variational_tensors = label_to_tensor_dict.get("variational", [])


class MPSOptimiser(TNOptimiser):
    """
    A class for locally optimising tensors in a TN with respect to a reference MPS and the HS distance
    """

    def __init__(self, tn: TensorNetwork, reference: MatrixProductState) -> None:
        """
        Class constructor.

        Args:
            tn: The tensor network that will be optimised. Should be contractable to an MPS
            reference: The reference state or operator
        """
        super().__init__(tn, reference)


class MPOOptimiser(TNOptimiser):
    """
    A class for locally optimising tensors in a TN with respect to a reference MPO and the HS distance
    """

    def __init__(self, tn: TensorNetwork, reference: MatrixProductOperator) -> None:
        """
        Class constructor.

        Args:
            tn: The tensor network that will be optimised. Should be contractable to an MPO
            reference: The reference state or operator
        """
        super().__init__(tn, reference)


class ApproximateDiagonalisation(TNOptimiser):
    """
    A class for approximately diagonalising an MPO
    """

    def __init__(self, mpo: MatrixProductOperator, max_bond: int) -> None:
        """
        Class constructor.

        Args:
            mpo: The MPO that will be appoximately diagonalised
            max_bond: The maximum allowed bond dimension
        """
        reference = MatrixProductOperator.from_increasing_diagonal_matrix(mpo.num_sites)
        self.ansatz = MatrixProductOperator.random_mpo(mpo.num_sites, max_bond)
        self.mpo_to_diag = mpo
        self.max_bond = max_bond
        self.ansatz_dag = copy.deepcopy(self.ansatz)
        self.ansatz_dag.dagger()
        for t in self.ansatz.tensors:
            t.labels.append("variational")
            t.labels.append(f"variational_site_{self.ansatz.tensors.index(t)+1}")
        for t in self.ansatz_dag.tensors:
            t.labels.append(f"variational_site_{self.ansatz_dag.tensors.index(t)+1}")
        reference.set_default_indices(
            internal_prefix="A", input_prefix="T", output_prefix="V"
        )
        self.ansatz_dag.set_default_indices(
            internal_prefix="B", input_prefix="V", output_prefix="W"
        )
        self.mpo_to_diag.set_default_indices(
            internal_prefix="C", input_prefix="W", output_prefix="X"
        )
        self.ansatz.set_default_indices(
            internal_prefix="D", input_prefix="X", output_prefix="T"
        )
        tn = TensorNetwork(
            reference.tensors
            + self.ansatz_dag.tensors
            + self.mpo_to_diag.tensors
            + self.ansatz.tensors,
            name="ApproximateDiagonalisation",
        )
        super().__init__(tn, None)
        self.reference = reference
        self.approximately_diagonalised_mpo = (
            self.construct_approximately_diagonalised_mpo()
        )

    def construct_approximately_diagonalised_mpo(self) -> "MatrixProductOperator":
        """
        Construct the approximately diagonalised MPO
        """
        adag = copy.deepcopy(self.ansatz_dag)
        mpo = copy.deepcopy(self.mpo_to_diag)
        a = copy.deepcopy(self.ansatz)
        approximately_diagonalised_mpo = adag * mpo
        approximately_diagonalised_mpo = approximately_diagonalised_mpo * a
        return approximately_diagonalised_mpo

    def get_local_indices(self, variational_idx: int) -> tuple[list[str], list[str]]:
        """
        For the local site get the expected indices of the environment tensor

        Args:
            variational_idx: The index of the current ansatz site

        Returns:
            output_inds, input_inds for the environment tensor
        """
        ansatz_tensor = self.ansatz.tensors[variational_idx - 1]
        ansatz_dag_tensor = self.ansatz_dag.tensors[variational_idx - 1]
        input_inds = ansatz_tensor.indices
        output_inds = ansatz_dag_tensor.indices
        return output_inds, input_inds

    def form_environment_matrix(self, variational_idx: int) -> ndarray:
        """
        Form the environment matrix for a local variational tensor

        Args:
            variational_idx: The index of the current ansatz site

        Returns:
            A matrix for the environment
        """
        tn_copy = copy.deepcopy(self.tn)
        site_label = f"variational_site_{variational_idx}"
        tn_copy.pop_tensors_by_label([site_label])
        env_tensor = tn_copy.contract_entire_network()
        env_copy = copy.deepcopy(env_tensor)
        output_inds, input_inds = self.get_local_indices(variational_idx)
        env_copy.tensor_to_matrix(input_inds, output_inds)
        env_mat = env_copy.data.todense()
        return env_mat

    def get_maximum_eigenvector(self, mat: ndarray) -> ndarray:
        """
        For a given matrix, get the eigenvector associated to the maximum eigenvalue

        Args:
            mat: The input matrix

        Returns:
            The maximum eigenvector
        """
        evals, evecs = eig(mat)
        max_eval = max(evals)
        max_eval_index = list(evals).index(max_eval)
        max_evec = evecs[:, max_eval_index]
        return max_evec

    def get_closest_isometry(self, data: ndarray, variational_index: int) -> ndarray:
        """
        The global unitary needs to be unitary. We can achieve this by ensuring each local update is an isometry.

        Args:
            data: The optimised data for a local update
            variational_index: Which tensor are we looking at

        Returns:
            The data for the closest isometry
        """
        if variational_index == 1:
            shape = data.shape
            data.transpose(2, 0, 1).reshape(shape[1], shape[0] * shape[2])
            u, _, vh = svd(data, full_matrices=False)
            new_data = u @ vh
            new_data.reshape(shape[1], shape[0], shape[2]).transpose(1, 2, 0)
        elif variational_index == self.ansatz.num_sites:
            shape = data.shape
            data.transpose(0, 2, 1).reshape(shape[0] * shape[1], shape[2])
            u, _, vh = svd(data, full_matrices=False)
            new_data = u @ vh
            new_data.reshape(shape[0], shape[1], shape[2]).transpose(0, 2, 1)
        else:
            shape = data.shape
            data.transpose(0, 3, 1, 2).reshape(shape[0] * shape[1], shape[2] * shape[3])
            u, _, vh = svd(data, full_matrices=False)
            new_data = u @ vh
            new_data.reshape(shape[0], shape[1], shape[2], shape[3]).transpose(
                0, 2, 3, 1
            )
        return new_data

    def local_update(self, variational_index: int) -> None:
        """
        Perform a local optimisation at the given index

        Args:
            variational_index: The index of the current local site
        """
        local_tensor = self.ansatz.tensors[variational_index - 1]
        local_indices = local_tensor.indices
        local_dims = [local_tensor.get_dimension_of_index(idx) for idx in local_indices]
        local_labels = local_tensor.labels

        # local_dag_tensor = self.ansatz_dag.tensors[variational_index-1]
        # local_dag_indices = local_dag_tensor.indices
        # local_dag_labels = local_dag_tensor.labels

        env_mat = self.form_environment_matrix(variational_index)
        max_evec = self.get_maximum_eigenvector(env_mat)
        new_site_data = max_evec.reshape(tuple(local_dims))
        new_site_data = self.get_closest_isometry(new_site_data, variational_index)
        new_tensor = Tensor(new_site_data, indices=local_indices, labels=local_labels)
        # new_dag_tensor = Tensor(new_site_data, indices=local_dag_indices, labels=local_dag_labels)
        # new_dag_tensor.dagger()
        site_label = f"variational_site_{variational_index}"
        self.ansatz.pop_tensors_by_label([site_label])
        self.ansatz.add_tensor(new_tensor, variational_index - 1)
        # self.ansatz_dag.pop_tensors_by_label([site_label])
        # self.ansatz_dag.add_tensor(new_dag_tensor, variational_index-1)
        self.ansatz_dag = copy.deepcopy(self.ansatz)
        self.ansatz_dag.dagger()
        self.ansatz_dag.set_default_indices(
            internal_prefix="B", input_prefix="V", output_prefix="W"
        )
        return

    def check_global_unitarity(self) -> None:
        """
        The output MPO will be a scaled version of a unitary. Here we enforce full unitarity.
        """
        # Check the magnitude
        ansatz = copy.deepcopy(self.ansatz)
        ansatz_copy = copy.deepcopy(self.ansatz)
        ip = hilbert_schmidt_inner_product(ansatz, ansatz_copy).real
        scale_factor = np.sqrt(ip / (2**self.ansatz.num_sites))
        self.ansatz.multiply_by_constant(1 / scale_factor)
        self.ansatz_dag.multiply_by_constant(1 / scale_factor)

        # Check the sign
        if ip < 0:
            self.ansatz.tensors[0].multiply_by_constant(-1)

        return

    def run(self, num_sweeps: int = 10, max_bond: int = 16) -> MatrixProductOperator:
        """
        Optimise the ansatz to approximately diagonalise the given MPO

        Args:
            num_sweeps: The number of sweeps to perform
            max_bond: The maximum allowed bond dimension

        Returns:
            The optimised ansatz
        """
        for _ in range(num_sweeps):
            for idx in range(1, self.ansatz.num_sites + 1):
                self.local_update(idx)
            self.tn = TensorNetwork(
                self.reference.tensors
                + self.ansatz_dag.tensors
                + self.mpo_to_diag.tensors
                + self.ansatz.tensors,
                name="ApproximateDiagonalisation",
            )
            self.approximately_diagonalised_mpo = (
                self.construct_approximately_diagonalised_mpo()
            )
        self.check_global_unitarity()
        self.approximately_diagonalised_mpo = (
            self.construct_approximately_diagonalised_mpo()
        )
        return self.ansatz
