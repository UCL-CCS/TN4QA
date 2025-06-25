import copy
from typing import Tuple

import numpy as np
import sparse
from numpy import ndarray
from scipy.sparse.linalg import eigs
from sparse import SparseArray

from .mpo import MatrixProductOperator
from .mps import MatrixProductState
from .tensor import Tensor
from .tn import TensorNetwork


class DMRG:
    def __init__(
        self,
        hamiltonian: dict[str, complex]
        | tuple[ndarray, ndarray, float]
        | MatrixProductOperator,
        max_mps_bond: int,
        method: str = "two-site",
        convergence_threshold: float = 1e-9,
        initial_state: MatrixProductState | None = None,
    ) -> "DMRG":
        """
        Constructor for the DMRG class.

        Args:
            hamiltonian: A dict of the form {pauli_string : weight} or a tuple of (one_e_integrals, two_e_integrals, nuc_energy) or an MPO
            max_mpo_bond: The maximum bond to use for the Hamiltonian MPO construction.
            max_mps_bond: The maximum bond to use for MPS during DMRG.
            method: Which method to use. One of "one-site", and "two-site". Defaults to "one-site".
            convergence_threshold: DMRG terminates once two successive sweeps differ in energy by less than this value.
            initial_state: The starting point for the DMRG calculation. If not provided, random MPS is generated.

        Returns:
            The DMRG object.
        """
        self.hamiltonian = hamiltonian
        self.method = method
        self.convergence_threshold = convergence_threshold
        if isinstance(hamiltonian, dict):
            self.num_sites = len(list(hamiltonian.keys())[0])
            self.hamiltonian_type = "qubit"
        elif isinstance(hamiltonian, MatrixProductOperator):
            self.num_sites = hamiltonian.num_sites
            self.hamiltonian_type = "MPO"
        else:
            self.num_sites = len(hamiltonian[0])
            self.nuc_energy = hamiltonian[2]
            self.hamiltonian_type = "fermionic"
        self.max_mps_bond = max_mps_bond
        self.current_max_mps_bond = 2
        self.mps = self.set_initial_state(initial_state)
        if self.hamiltonian_type == "MPO":
            self.mpo = self.add_trivial_tensors_mpo(hamiltonian)
        else:
            self.mpo = self.set_hamiltonian_mpo()
        self.energy = self.set_initial_energy()
        self.all_energies = [self.energy]
        self.left_block_cache = []
        self.right_block_cache = []
        self.left_block, self.right_block = self.initialise_blocks()

        return

    def set_initial_state(
        self, input: MatrixProductState | None = None
    ) -> MatrixProductState:
        """
        Set the initial state for DMRG.

        Args:
            input: Either a given MPS or a string indicating a construction method, one of "random" or "HF"
        """
        match input:
            case None:
                mps = MatrixProductState.random_quantum_state_mps(
                    self.num_sites, self.current_max_mps_bond
                )
            case _:
                if isinstance(input, MatrixProductState):
                    mps = input
                else:
                    raise ValueError("Given initial_state not supported")

        mps = self.add_trivial_tensors_mps(mps)

        return mps

    def set_hamiltonian_mpo(self) -> MatrixProductOperator:
        """
        Convert the Hamiltonian to an MPO for DMRG.
        """

        match self.hamiltonian_type:
            case "qubit":
                mpo = MatrixProductOperator.from_hamiltonian(self.hamiltonian, np.inf)
            case "fermionic":
                mpo = MatrixProductOperator.from_electron_integral_arrays(
                    self.hamiltonian[0], self.hamiltonian[1]
                )

        mpo = self.add_trivial_tensors_mpo(mpo)

        return mpo

    def set_initial_energy(self) -> float:
        mps = copy.deepcopy(self.mps)
        mpo = copy.deepcopy(self.mpo)
        return mps.compute_expectation_value(mpo)

    def add_trivial_tensors_mps(self, mps: MatrixProductState) -> MatrixProductState:
        """
        Add trivial tensors to MPS.
        """
        mps.tensors[0].data = sparse.reshape(
            mps.tensors[0].data, (1,) + mps.tensors[0].dimensions
        )
        mps.tensors[-1].data = sparse.reshape(
            mps.tensors[-1].data,
            (mps.tensors[-1].dimensions[0], 1, mps.tensors[-1].dimensions[1]),
        )

        trivial_array = sparse.COO.from_numpy(
            np.array([1], dtype=complex).reshape(1, 1)
        )
        all_arrays = (
            [trivial_array]
            + [mps.tensors[i].data for i in range(self.num_sites)]
            + [trivial_array]
        )
        mps = MatrixProductState.from_arrays(all_arrays)
        return mps

    def add_trivial_tensors_mpo(
        self, mpo: MatrixProductOperator
    ) -> MatrixProductOperator:
        """
        Add trivial tensors to MPO.
        """
        mpo.tensors[0].data = sparse.reshape(
            mpo.tensors[0].data, (1,) + mpo.tensors[0].dimensions
        )
        mpo.tensors[-1].data = sparse.reshape(
            mpo.tensors[-1].data,
            (
                mpo.tensors[-1].dimensions[0],
                1,
                mpo.tensors[-1].dimensions[1],
                mpo.tensors[-1].dimensions[2],
            ),
        )

        trivial_array = sparse.COO.from_numpy(
            np.array([1], dtype=complex).reshape(1, 1, 1)
        )
        all_arrays = (
            [trivial_array]
            + [mpo.tensors[i].data for i in range(self.num_sites)]
            + [trivial_array]
        )
        mpo = MatrixProductOperator.from_arrays(all_arrays)
        return mpo

    def remove_trivial_tensors_mps(self, mps: MatrixProductState) -> MatrixProductState:
        """
        Remove trivial tensors from MPS.
        """
        zero_array = sparse.COO.from_numpy(
            np.array([1], dtype=complex).reshape(
                1,
            )
        )
        zero_tensor_top = Tensor(zero_array, ["P1"], ["ZERO"])
        zero_tensor_bottom = Tensor(zero_array, [f"P{self.num_sites+2}"], ["ZERO"])
        self.mps.add_tensor(zero_tensor_top)
        self.mps.add_tensor(zero_tensor_bottom)
        self.mps.contract_index("P1")
        self.mps.contract_index(f"P{self.num_sites+2}")
        self.mps.contract_index("B1")
        self.mps.contract_index(f"B{self.num_sites+1}")
        arrays = []
        for idx in range(2, self.num_sites + 2):
            arrays.append(self.mps.get_tensors_from_index_name(f"P{idx}")[0].data)
        mps = MatrixProductState.from_arrays(arrays)
        # mps.multiply_by_constant(2)

        return mps

    def remove_trivial_tensors_mpo(
        self, mpo: MatrixProductOperator
    ) -> MatrixProductOperator:
        """
        Remove trivial tensors from MPS.
        """
        zero_array = sparse.COO.from_numpy(
            np.array([1], dtype=complex).reshape(
                1,
            )
        )
        zero_tensor_top_right = Tensor(zero_array, ["R1"], ["ZERO"])
        zero_tensor_bottom_right = Tensor(
            zero_array, [f"R{self.num_sites+2}"], ["ZERO"]
        )
        zero_tensor_top_left = Tensor(zero_array, ["L1"], ["ZERO"])
        zero_tensor_bottom_left = Tensor(zero_array, [f"L{self.num_sites+2}"], ["ZERO"])
        self.mpo.add_tensor(zero_tensor_top_right)
        self.mpo.add_tensor(zero_tensor_bottom_right)
        self.mpo.add_tensor(zero_tensor_top_left)
        self.mpo.add_tensor(zero_tensor_bottom_left)
        self.mpo.contract_index("R1")
        self.mpo.contract_index(f"R{self.num_sites+2}")
        self.mpo.contract_index("L1")
        self.mpo.contract_index(f"L{self.num_sites+2}")
        self.mpo.contract_index("B1")
        self.mpo.contract_index(f"B{self.num_sites+1}")
        arrays = []
        for idx in range(2, self.num_sites + 2):
            arrays.append(self.mpo.get_tensors_from_index_name(f"R{idx}")[0].data)
        mpo = MatrixProductOperator.from_arrays(arrays)
        # mpo.multiply_by_constant(4)

        return mpo

    def initialise_left_block(self) -> Tensor:
        """
        Initialise the left block of the DMRG routine.
        """
        dag = copy.deepcopy(self.mps.tensors[0].data.conj())
        ham = copy.deepcopy(self.mpo.tensors[0].data)
        mps = copy.deepcopy(self.mps.tensors[0].data)

        dag_tensor = Tensor(dag, ["Ldag", "TEMP1"], ["DAG"])
        ham_tensor = Tensor(ham, ["Lham", "TEMP1", "TEMP2"], ["HAM"])
        mps_tensor = Tensor(mps, ["Lmps", "TEMP2"], ["MPS"])

        tn = TensorNetwork([dag_tensor, ham_tensor, mps_tensor])
        left_block = tn.contract_entire_network()
        left_block.reorder_indices(["Ldag", "Lham", "Lmps"])
        left_block.labels.append("LEFT_OF_SITE_1")

        return left_block

    def initialise_blocks(self) -> Tuple[Tensor]:
        """
        Initialise the left and right blocks of the DMRG routine.

        Returns:
            A tuple of the initial left block and the initial right block.
        """
        # Set up the rightmost block
        dag = copy.deepcopy(self.mps.tensors[-1].data.conj())
        ham = copy.deepcopy(self.mpo.tensors[-1].data)
        mps = copy.deepcopy(self.mps.tensors[-1].data)
        dag_tensor = Tensor(dag, ["Rdag", "TEMP1"], ["DAG"])
        ham_tensor = Tensor(ham, ["Rham", "TEMP1", "TEMP2"], ["HAM"])
        mps_tensor = Tensor(mps, ["Rmps", "TEMP2"], ["MPS"])
        tn = TensorNetwork([dag_tensor, ham_tensor, mps_tensor])
        right_block = tn.contract_entire_network()
        right_block.reorder_indices(["Rdag", "Rham", "Rmps"])
        right_block.labels.append(f"RIGHT_OF_SITE_{self.num_sites}")
        self.right_block_cache.append(right_block)

        for i in list(range(2, self.num_sites + 1))[::-1]:
            self.mps.move_orthogonality_centre(i, i + 1)
            dag = copy.deepcopy(self.mps.tensors[i].data.conj())
            ham = copy.deepcopy(self.mpo.tensors[i].data)
            mps = copy.deepcopy(self.mps.tensors[i].data)
            temp_right_block = copy.deepcopy(right_block)
            temp_right_block.indices = ["TEMP1", "TEMP3", "TEMP5"]
            dag_tensor = Tensor(dag, ["Rdag", "TEMP1", "TEMP2"], ["DAG"])
            ham_tensor = Tensor(ham, ["Rham", "TEMP3", "TEMP2", "TEMP4"], ["HAM"])
            mps_tensor = Tensor(mps, ["Rmps", "TEMP5", "TEMP4"], ["MPS"])
            tn = TensorNetwork([temp_right_block, dag_tensor, ham_tensor, mps_tensor])
            right_block = tn.contract_entire_network()
            right_block.reorder_indices(["Rdag", "Rham", "Rmps"])
            right_block.labels.append(f"RIGHT_OF_SITE_{i-1}")
            self.right_block_cache.append(right_block)

        left_block = self.initialise_left_block()
        self.right_block_cache.pop()

        if self.method == "two-site":
            right_block = copy.deepcopy(self.right_block_cache[-1])
            self.right_block_cache.pop()

        return left_block, right_block

    def update_blocks_left_sweep(self) -> None:
        """
        Update the blocks after each local optimisation.
        """
        current_site = len(self.left_block_cache) + 1
        self.left_block_cache.append(self.left_block)
        self.mps.move_orthogonality_centre(current_site + 2, current_site + 1)
        left_block = copy.deepcopy(self.left_block)

        mps = copy.deepcopy(self.mps.tensors[current_site].data)
        ham = copy.deepcopy(self.mpo.tensors[current_site].data)
        dag = copy.deepcopy(self.mps.tensors[current_site].data.conj())

        left_block.indices = ["TEMP1", "TEMP2", "TEMP3"]
        mps_tensor = Tensor(mps, ["TEMP3", "Lmps", "TEMP5"], ["MPS"])
        ham_tensor = Tensor(ham, ["TEMP2", "Lham", "TEMP4", "TEMP5"], ["HAM"])
        dag_tensor = Tensor(dag, ["TEMP1", "Ldag", "TEMP4"], ["DAG"])

        tn = TensorNetwork([left_block, mps_tensor, ham_tensor, dag_tensor])
        self.left_block = tn.contract_entire_network()
        self.left_block.reorder_indices(["Ldag", "Lham", "Lmps"])
        self.left_block.labels.append(f"LEFT_OF_SITE_{current_site+1}")

        self.right_block = copy.deepcopy(self.right_block_cache[-1])
        self.right_block_cache.pop()

        return

    def update_blocks_right_sweep(self) -> None:
        """
        Update the blocks after each local optimisation.
        """
        current_site = self.num_sites - len(self.right_block_cache)
        self.right_block_cache.append(self.right_block)
        self.mps.move_orthogonality_centre(current_site, current_site + 1)
        right_block = copy.deepcopy(self.right_block)

        mps = copy.deepcopy(self.mps.tensors[current_site].data)
        ham = copy.deepcopy(self.mpo.tensors[current_site].data)
        dag = copy.deepcopy(self.mps.tensors[current_site].data.conj())

        right_block.indices = ["TEMP1", "TEMP2", "TEMP3"]
        mps_tensor = Tensor(mps, ["Rmps", "TEMP3", "TEMP5"], ["MPS"])
        ham_tensor = Tensor(ham, ["Rham", "TEMP2", "TEMP4", "TEMP5"], ["HAM"])
        dag_tensor = Tensor(dag, ["Rdag", "TEMP1", "TEMP4"], ["DAG"])

        tn = TensorNetwork([right_block, mps_tensor, ham_tensor, dag_tensor])
        self.right_block = tn.contract_entire_network()
        self.right_block.reorder_indices(["Rdag", "Rham", "Rmps"])
        self.right_block.labels.append(f"RIGHT_OF_SITE_{current_site-1}")

        self.left_block = copy.deepcopy(self.left_block_cache[-1])
        self.left_block_cache.pop()

        return

    def combine_neighbouring_sites(self) -> SparseArray:
        """
        For the two_site method, combine neighbouring Hamiltonian sites.
        """
        current_site = len(self.left_block_cache) + 1
        next_site = current_site + 1
        ham1 = copy.deepcopy(self.mpo.tensors[current_site])
        ham2 = copy.deepcopy(self.mpo.tensors[next_site])

        tn = TensorNetwork([ham1, ham2])
        combined = tn.contract_entire_network()
        combined.combine_indices([f"R{current_site+1}", f"R{current_site+2}"], "R")
        combined.combine_indices([f"L{current_site+1}", f"L{current_site+2}"], "L")
        combined.reorder_indices([f"B{current_site}", f"B{current_site+2}", "R", "L"])

        return combined.data

    def construct_effective_matrix(
        self, current_site_mat: SparseArray | None = None
    ) -> SparseArray:
        """
        Construct the effective matrix for a step of DMRG.

        Args:
            current_site_mat (optional): The MPO tensor at the site(s) to be optimised. Defaults to single site tensor.
        """
        left_block = copy.deepcopy(self.left_block)
        right_block = copy.deepcopy(self.right_block)
        current_site = len(self.left_block_cache) + 1
        if current_site_mat is None:
            current_site_mat = copy.deepcopy(self.mpo.tensors[current_site].data)

        ham_tensor = Tensor(current_site_mat, ["Lham", "Rham", "pdag", "p"], ["HAM"])
        tn = TensorNetwork([ham_tensor, right_block, left_block])
        tensor = tn.contract_entire_network()
        tensor.reorder_indices(["Ldag", "pdag", "Rdag", "Lmps", "p", "Rmps"])
        tensor.indices = ["udag", "pdag", "ddag", "u", "p", "d"]
        tensor.tensor_to_matrix(["u", "p", "d"], ["udag", "pdag", "ddag"])

        return tensor.data

    def optimise_local_tensor(self) -> None:
        """
        Optimise the local tensor at the current site.

        Args:
            site: The site index.
        """
        site = len(self.left_block_cache) + 1
        if self.method == "two-site":
            original_dims = (
                self.mps.tensors[site].dimensions[0],
                self.mps.tensors[site + 1].dimensions[1],
                self.mps.tensors[site].dimensions[2],
                self.mps.tensors[site + 1].dimensions[2],
            )
            ham_mat = self.combine_neighbouring_sites()
            effective_matrix = self.construct_effective_matrix(ham_mat)
        else:
            original_dims = self.mps.tensors[site].dimensions + (1,)
            effective_matrix = self.construct_effective_matrix()
        w, v = eigs(effective_matrix, k=1, which="SR")
        eigval = w[0]
        eigvec = sparse.COO.from_numpy(
            v[:, 0]
        )  # This is the new optimal value at site i

        new_data = sparse.reshape(
            eigvec,
            (original_dims[0], original_dims[2] * original_dims[3], original_dims[1]),
        )
        new_data = sparse.moveaxis(new_data, [0, 1, 2], [0, 2, 1])

        if self.method == "two-site":
            new_data = sparse.moveaxis(new_data, [0, 1, 2], [2, 1, 0])
            new_data = sparse.reshape(
                new_data,
                (
                    original_dims[2],
                    original_dims[3],
                    original_dims[1],
                    original_dims[0],
                ),
            )
            new_data = sparse.moveaxis(new_data, [0, 1, 2, 3], [2, 3, 1, 0])
            temp_t = Tensor(new_data, ["u", "d", "p1", "p2"], ["TEMP"])
            temp_tn = TensorNetwork([temp_t])
            temp_tn.svd(
                temp_t,
                ["u", "p1"],
                ["d", "p2"],
                max_bond=self.current_max_mps_bond,
                new_index_name="b",
            )

            t1 = temp_tn.tensors[0]
            t1.reorder_indices(["u", "b", "p1"])
            indices1 = self.mps.tensors[site].indices
            labels1 = self.mps.tensors[site].labels
            t1.indices = indices1
            t1.labels = labels1

            t2 = temp_tn.tensors[1]
            t2.reorder_indices(["b", "d", "p2"])
            indices2 = self.mps.tensors[site + 1].indices
            labels2 = self.mps.tensors[site + 1].labels
            t2.indices = indices2
            t2.labels = labels2

            self.mps.pop_tensors_by_label(labels1)
            self.mps.add_tensor(t1, site)
            self.mps.pop_tensors_by_label(labels2)
            self.mps.add_tensor(t2, site + 1)

        else:
            original_indices = self.mps.tensors[site].indices
            original_labels = self.mps.tensors[site].labels
            self.mps.pop_tensors_by_label(original_labels)
            new_t = Tensor(new_data, original_indices, original_labels)
            self.mps.add_tensor(new_t, site)

        self.energy = eigval.real

        return

    def sweep_left_one_site(self):
        """
        Perform a left sweep.
        """
        sites = list(range(1, self.num_sites + 1))
        for site in sites:
            self.optimise_local_tensor()
            if site != self.num_sites:
                self.update_blocks_left_sweep()
        return

    def sweep_right_one_site(self):
        """
        Perform a right sweep.
        """
        sites = list(range(1, self.num_sites + 1))[::-1]
        for site in sites:
            self.optimise_local_tensor()
            if site != 1:
                self.update_blocks_right_sweep()
        return

    def sweep_left_two_site(self):
        """
        Perform a left sweep.
        """
        sites = list(range(1, self.num_sites))
        for site in sites:
            self.optimise_local_tensor()
            if site != self.num_sites - 1:
                self.update_blocks_left_sweep()
        return

    def sweep_right_two_site(self):
        """
        Perform a right sweep.
        """
        sites = list(range(1, self.num_sites))[::-1]
        for site in sites:
            self.optimise_local_tensor()
            if site != 1:
                self.update_blocks_right_sweep()
        return

    def perform_bond_expansion(self) -> None:
        """
        Expand the MPS bond dimension.
        """
        new_bond_dim = min(2 * self.current_max_mps_bond, self.max_mps_bond)
        diff = new_bond_dim - self.current_max_mps_bond
        self.current_max_mps_bond = new_bond_dim
        self.mps = self.mps.expand_bond_dimension_list(
            diff, list(range(2, self.num_sites + 1))
        )
        self.left_block_cache = []
        self.right_block_cache = []
        self.left_block, self.right_block = self.initialise_blocks()
        return

    def sub_convergence_check(self) -> None:
        """
        Check if the convergence threshold has been met for the current bond dimension.
        """
        if len(self.all_energies) < 2:
            return False
        if np.abs(self.all_energies[-1] - self.all_energies[-2]) < 1e-3:
            return True
        return False

    def convergence_check(self) -> None:
        """
        Check if the convergence threshold has been met for the max bond dimension.
        """
        if len(self.all_energies) < 2:
            return False
        if self.current_max_mps_bond == self.max_mps_bond:
            if (
                np.abs(self.all_energies[-1] - self.all_energies[-2])
                < self.convergence_threshold
            ):
                return True
        return False

    def run(self, maxiter: int) -> Tuple[float, MatrixProductState]:
        """
        Find the groundstate of an MPO with DMRG.

        Args:
            maxiter: The maximum number of DMRG sweeps.

        Returns:
            A tuple of the DMRG energy and the DMRG state.
        """
        if self.method == "one-site":
            for _ in range(maxiter):
                self.sweep_left_one_site()
                self.sweep_right_one_site()
                self.all_energies.append(self.energy)
                if self.convergence_check():
                    break
                elif self.sub_convergence_check():
                    self.perform_bond_expansion()
        elif self.method == "two-site":
            for _ in range(maxiter):
                self.sweep_left_two_site()
                self.sweep_right_two_site()
                self.all_energies.append(self.energy)
                if self.convergence_check():
                    break
                elif self.sub_convergence_check():
                    self.perform_bond_expansion()

        self.mps = self.remove_trivial_tensors_mps(self.mps)
        self.mpo = self.remove_trivial_tensors_mpo(self.mpo)

        if self.hamiltonian_type == "fermionic":
            self.energy = self.energy + self.nuc_energy

        return (self.energy, self.mps)
