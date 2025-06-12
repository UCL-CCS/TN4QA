import copy

import numpy as np
from numpy import ndarray
from numpy.linalg import eig, svd

from ..fidelity_metrics import hilbert_schmidt_inner_product
from ..mpo import MatrixProductOperator
from ..tensor import Tensor
from ..tn import TensorNetwork


class ApproximateDiagonalisation:
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
        self.tn = TensorNetwork(
            reference.tensors
            + self.ansatz_dag.tensors
            + self.mpo_to_diag.tensors
            + self.ansatz.tensors,
            name="ApproximateDiagonalisation",
        )
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

    def get_two_site_indices(self, variational_idx: int) -> tuple[list[str], list[str]]:
        """
        For the current site get the expected indices of the environment tensor

        Args:
            variational_idx: The index of the current ansatz site

        Returns:
            output_inds, input_inds for the environment tensor
        """
        ansatz_tensor1 = self.ansatz.tensors[variational_idx - 1]
        ansatz_tensor2 = self.ansatz.tensors[variational_idx]
        for idx in ansatz_tensor1.indices:
            if idx in ansatz_tensor2.indices:
                ansatz_shared_index = idx
        ansatz_dag_tensor1 = self.ansatz_dag.tensors[variational_idx - 1]
        ansatz_dag_tensor2 = self.ansatz_dag.tensors[variational_idx]
        for idx in ansatz_dag_tensor1.indices:
            if idx in ansatz_dag_tensor2.indices:
                ansatz_dag_shared_index = idx
        input_inds = ansatz_tensor1.indices + ansatz_tensor2.indices
        input_inds = [x for x in input_inds if x != ansatz_shared_index]
        output_inds = ansatz_dag_tensor1.indices + ansatz_dag_tensor2.indices
        output_inds = [x for x in output_inds if x != ansatz_dag_shared_index]
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

    def form_two_site_environment_matrix(self, variational_idx: int) -> ndarray:
        """
        Form the environment matrix for a local variational tensor

        Args:
            variational_idx: The index of the current ansatz site

        Returns:
            A matrix for the environment
        """
        tn_copy = copy.deepcopy(self.tn)
        site_label1 = f"variational_site_{variational_idx}"
        site_label2 = f"variational_site_{variational_idx+1}"
        tn_copy.pop_tensors_by_label([site_label1])
        tn_copy.pop_tensors_by_label([site_label2])
        env_tensor = tn_copy.contract_entire_network()
        env_copy = copy.deepcopy(env_tensor)
        output_inds, input_inds = self.get_two_site_indices(variational_idx)
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

    def get_closest_isometry(self, data: ndarray) -> ndarray:
        """
        The global unitary needs to be unitary. We can achieve this by ensuring each local update is an isometry.

        Args:
            data: The optimised data for a local update

        Returns:
            The data for the closest isometry
        """
        u, _, vh = svd(data, full_matrices=False)
        new_data = u @ vh
        return new_data

    def get_closest_two_site_isometry(
        self, data: ndarray, variational_index: int
    ) -> ndarray:
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
            mat = data.transpose(1, 4, 2, 0, 3).reshape(
                shape[1] * shape[4], shape[2] * shape[0] * shape[3]
            )
            u, _, vh = svd(mat, full_matrices=False)
            new_mat = u @ vh
            new_data = new_mat.reshape(
                shape[1], shape[4], shape[2], shape[0], shape[3]
            ).transpose(3, 0, 2, 4, 1)
        elif variational_index == self.ansatz.num_sites - 1:
            shape = data.shape
            mat = data.transpose(0, 2, 4, 1, 3).reshape(
                shape[0] * shape[2], shape[4] * shape[1] * shape[3]
            )
            u, _, vh = svd(mat, full_matrices=False)
            new_mat = u @ vh
            new_data = new_mat.reshape(
                shape[0], shape[2], shape[4], shape[1], shape[3]
            ).transpose(0, 3, 1, 4, 2)
        else:
            shape = data.shape
            mat = data.transpose(0, 2, 5, 3, 1, 4).reshape(
                shape[0] * shape[2] * shape[5], shape[1] * shape[4] * shape[3]
            )
            u, _, vh = svd(mat, full_matrices=False)
            new_mat = u @ vh
            new_data = new_mat.reshape(
                shape[0], shape[2], shape[5], shape[1], shape[4], shape[3]
            ).transpose(0, 4, 1, 3, 5, 2)

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

        env_mat = self.form_environment_matrix(variational_index)
        max_evec = self.get_maximum_eigenvector(env_mat)
        new_site_data = max_evec.reshape(tuple(local_dims))
        new_site_data = self.get_closest_isometry(new_site_data)
        new_tensor = Tensor(new_site_data, indices=local_indices, labels=local_labels)
        site_label = f"variational_site_{variational_index}"
        self.ansatz.pop_tensors_by_label([site_label])
        self.ansatz.add_tensor(new_tensor, variational_index - 1)
        self.update_ansatz_dag()
        return

    def two_site_update(self, variational_index: int, max_bond: int) -> None:
        """
        Perform a two-site optimisation at the given index

        Args:
            variational_index: The index of the current local site
            max_bond: The maximum allowed bond dimension
        """
        local_tensor1 = self.ansatz.tensors[variational_index - 1]
        local_indices1 = local_tensor1.indices
        local_labels1 = local_tensor1.labels

        local_tensor2 = self.ansatz.tensors[variational_index]
        local_indices2 = local_tensor2.indices
        local_labels2 = local_tensor2.labels

        for idx in local_indices1:
            if idx in local_indices2:
                shared_idx = idx

        input_dims = [
            local_tensor1.get_dimension_of_index(idx)
            for idx in local_indices1
            if idx != shared_idx
        ]
        output_dims = [
            local_tensor2.get_dimension_of_index(idx)
            for idx in local_indices2
            if idx != shared_idx
        ]
        all_dims = input_dims + output_dims

        env_mat = self.form_two_site_environment_matrix(variational_index)
        max_evec = self.get_maximum_eigenvector(env_mat)
        max_evec = max_evec.reshape(all_dims)

        new_two_site_data = self.get_closest_two_site_isometry(
            max_evec, variational_index
        )

        new_two_site_data = max_evec.reshape(
            (np.prod(output_dims), np.prod(input_dims))
        )
        u, s, vh = svd(new_two_site_data, full_matrices=False)
        max_bond = min(
            [max_bond, new_two_site_data.shape[0], new_two_site_data.shape[1]]
        )
        new_site_data1 = vh[:max_bond, :]
        new_site_data2 = u[:, :max_bond] * s[:max_bond]

        new_dims1 = [max_bond] + input_dims
        new_dims2 = output_dims + [max_bond]
        new_site_data1 = new_site_data1.reshape(tuple(new_dims1))
        new_site_data2 = new_site_data2.reshape(tuple(new_dims2))
        if variational_index != 1:
            new_site_data1 = np.moveaxis(new_site_data1, [1], [0])
        if variational_index != self.ansatz.num_sites - 1:
            new_site_data2 = np.moveaxis(new_site_data2, [3], [0])
        else:
            new_site_data2 = np.moveaxis(new_site_data2, [2], [0])

        new_site_data1 = self.get_closest_isometry(new_site_data1)
        new_site_data2 = self.get_closest_isometry(new_site_data2)

        new_tensor1 = Tensor(
            new_site_data1, indices=local_indices1, labels=local_labels1
        )
        new_tensor2 = Tensor(
            new_site_data2, indices=local_indices2, labels=local_labels2
        )
        site_label1 = f"variational_site_{variational_index}"
        site_label2 = f"variational_site_{variational_index+1}"
        self.ansatz.pop_tensors_by_label([site_label1])
        self.ansatz.add_tensor(new_tensor1, variational_index - 1)
        self.ansatz.pop_tensors_by_label([site_label2])
        self.ansatz.add_tensor(new_tensor2, variational_index)
        self.update_ansatz_dag()
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
        return

    def update_ansatz_dag(self) -> None:
        """
        Update ansatz_dag after changes to ansatz.
        """
        self.ansatz_dag = copy.deepcopy(self.ansatz)
        self.ansatz_dag.dagger()
        self.ansatz_dag.set_default_indices(
            internal_prefix="B", input_prefix="V", output_prefix="W"
        )
        return

    def update_tn(self) -> None:
        """
        Update tn after changes to ansatz.
        """
        self.tn = TensorNetwork(
            self.reference.tensors
            + self.ansatz_dag.tensors
            + self.mpo_to_diag.tensors
            + self.ansatz.tensors,
            name="ApproximateDiagonalisation",
        )
        return

    def run(self, num_sweeps: int = 10) -> MatrixProductOperator:
        """
        Optimise the ansatz to approximately diagonalise the given MPO

        Args:
            num_sweeps: The number of sweeps to perform

        Returns:
            The optimised ansatz
        """
        for _ in range(num_sweeps):
            for idx in range(1, self.ansatz.num_sites + 1):
                self.local_update(idx)
                self.update_tn()
            for idx in list(range(1, self.ansatz.num_sites + 1))[::-1]:
                self.local_update(idx)
                self.update_tn()
        self.check_global_unitarity()
        self.approximately_diagonalised_mpo = (
            self.construct_approximately_diagonalised_mpo()
        )
        return self.ansatz

    def run_two_site(self, num_sweeps: int = 10) -> MatrixProductOperator:
        """
        Optimise the ansatz to approximately diagonalise the given MPO

        Args:
            num_sweeps: The number of sweeps to perform
            max_bond: The maximum allowed bond dimension

        Returns:
            The optimised ansatz
        """
        for _ in range(num_sweeps):
            for idx in range(1, self.ansatz.num_sites):
                self.two_site_update(idx, self.max_bond)
                self.update_tn()
            for idx in list(range(1, self.ansatz.num_sites))[::-1]:
                self.two_site_update(idx, self.max_bond)
                self.update_tn()
        self.check_global_unitarity()
        self.approximately_diagonalised_mpo = (
            self.construct_approximately_diagonalised_mpo()
        )
        return self.ansatz
