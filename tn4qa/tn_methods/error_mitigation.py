import copy

import numpy as np
from numpy import ndarray
from qiskit import QuantumCircuit
from qiskit.providers import BackendV2

from ..mpo import MatrixProductOperator
from ..mps import MatrixProductState
from ..noise_modelling.build import NoiseModelBuilder
from ..noise_modelling.noise_data import extract_noise_data
from ..noise_modelling.noisy_circuit_decomposer import NoisyCircuitDecomposer


class TNQuantumErrorMitigation:
    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: BackendV2,
        max_bond_dimension: int,
        representation: str = "PTM",
    ):
        """Constructor

        Args:
            circuit: The input quantum circuit
            max_bond_dimension: Maximum allowed bond dimension for TNQEM MPO
            calibration_data: The calibration data for the quantum device
        """
        self.circuit = circuit
        self.num_qubits = circuit.num_qubits
        self.backend = backend
        self.max_bond_dimension = max_bond_dimension
        self.representation = representation
        self.noise_data = extract_noise_data(backend)
        self.noise_model = NoiseModelBuilder(
            self.noise_data,
            twoq_gate_errors=True,
            oneq_gate_errors=True,
            decoherence=False,
            readout_errors=False,
        ).build()
        self.noisy_circuit = NoisyCircuitDecomposer(
            backend, self.noise_model
        ).decompose(circuit)
        self.tnqem_mpo = self.build_tnqem_mpo()

    def bitstring_to_vectorised_dm_mps(self, bitstring: str) -> MatrixProductState:
        """Convert a bitstring into an MPS representing the vectorised density matrix

        Args:
            bitstring: Input bitstring

        Returns:
            A MatrixProductState
        """
        if len(bitstring) == 1:
            if bitstring[0] == "0":
                first_array = (
                    np.array([1, 0, 0, 0], dtype=complex).reshape((4,))
                    if self.representation == "Liouville"
                    else np.array([1, 0, 0, 1], dtype=complex).reshape((4,))
                )
            else:
                first_array = (
                    np.array([0, 0, 0, 1], dtype=complex).reshape((4,))
                    if self.representation == "Liouville"
                    else np.array([1, 0, 0, -1], dtype=complex).reshape((4,))
                )
            mps = MatrixProductState.from_arrays([first_array])
            return mps

        arrays = []

        first_bit = bitstring[0]
        if first_bit == "0":
            first_array = (
                np.array([1, 0, 0, 0], dtype=complex).reshape((1, 4))
                if self.representation == "Liouville"
                else np.array([1, 0, 0, 1], dtype=complex).reshape((1, 4))
            )
        else:
            first_array = (
                np.array([0, 0, 0, 1], dtype=complex).reshape((1, 4))
                if self.representation == "Liouville"
                else np.array([1, 0, 0, -1], dtype=complex).reshape((1, 4))
            )
        arrays.append(first_array)

        for bit in bitstring[1:-1]:
            if bit == "0":
                array = (
                    np.array([1, 0, 0, 0], dtype=complex).reshape((1, 1, 4))
                    if self.representation == "Liouville"
                    else np.array([1, 0, 0, 1], dtype=complex).reshape((1, 1, 4))
                )
            else:
                array = (
                    np.array([0, 0, 0, 1], dtype=complex).reshape((1, 1, 4))
                    if self.representation == "Liouville"
                    else np.array([1, 0, 0, -1], dtype=complex).reshape((1, 1, 4))
                )
            arrays.append(array)

        last_bit = bitstring[-1]
        if last_bit == "0":
            last_array = (
                np.array([1, 0, 0, 0], dtype=complex).reshape((1, 4))
                if self.representation == "Liouville"
                else np.array([1, 0, 0, 1], dtype=complex).reshape((1, 4))
            )
        else:
            last_array = (
                np.array([0, 0, 0, 1], dtype=complex).reshape((1, 4))
                if self.representation == "Liouville"
                else np.array([1, 0, 0, -1], dtype=complex).reshape((1, 4))
            )
        arrays.append(last_array)

        mps = MatrixProductState.from_arrays(arrays)
        return mps

    def superoperator_to_submpo(self, superop: ndarray):
        """Convert a superoperator to an MPO"""
        dim = len(superop)
        if dim == 4:
            mpo = MatrixProductOperator.from_arrays([superop])
        elif dim == 16:
            superop = np.reshape(superop, (4, 4, 4, 4))
            superop = np.moveaxis(superop, [0, 1, 2, 3], [0, 2, 1, 3])
            superop = np.reshape(superop, (16, 16))
            U, S, Vh = np.linalg.svd(superop, full_matrices=False)
            if self.max_bond_dimension:
                chi = min(len(S), self.max_bond_dimension)
            else:
                chi = len(S)
            U = U[:, :chi]
            S = S[:chi]
            Vh = Vh[:chi, :]
            first_array = U.reshape(4, 4, chi).transpose(2, 0, 1)
            second_array = (np.diag(S) @ Vh).reshape(chi, 4, 4)
            mpo = MatrixProductOperator.from_arrays([first_array, second_array])
        return mpo

    def apply_one_site_term(
        self, mpo: MatrixProductOperator, data: ndarray, site: int, left: bool = False
    ) -> None:
        """
        Apply a one-qubit gate in place

        Args:
            data: The one-qubit matrix
            site: Where to apply the gate to
            left: If true, applies the gate to the left of the mpo
        """
        if mpo.num_sites == 1:
            contraction = "ij,jk->ik" if left else "ij,ki->kj"
        elif site == 1 or site == mpo.num_sites:
            contraction = "ijk,kl->ijl" if left else "ijk,lj->ilk"
        else:
            contraction = "hijk,kl->hijl" if left else "hijk,lj->hilk"
        mpo.tensors[site - 1].data = np.einsum(
            contraction, mpo.tensors[site - 1].data, data
        )
        return

    def two_site_term_to_full_length_mpo(
        self, data: ndarray, sites: list[int], num_sites: int
    ) -> "MatrixProductOperator":
        site0, site1 = sites[0], sites[1]
        term_mpo = self.superoperator_to_submpo(data)
        gate_mpo_bond = term_mpo.tensors[0].dimensions[0]

        first_array = term_mpo.tensors[0].data
        last_array = term_mpo.tensors[1].data
        q0 = min(site0, site1)
        q1 = max(site0, site1)
        num_intermediate_sites = q1 - q0 - 1
        middle_array = np.array([[np.zeros((4, 4))] * gate_mpo_bond] * gate_mpo_bond)
        for x in range(gate_mpo_bond):
            middle_array[x, x, :, :] = np.eye(4)
        middle_arrays = [middle_array for _ in range(num_intermediate_sites)]
        arrays = [first_array] + middle_arrays + [last_array]
        nonlocal_mpo = MatrixProductOperator.from_arrays(arrays)
        if q0 == 1:
            arrays = [
                nonlocal_mpo.tensors[x].data for x in range(nonlocal_mpo.num_sites)
            ]
            if q1 == num_sites:
                pass
            else:
                shape = arrays[-1].shape
                arrays[-1] = arrays[-1].reshape((shape[0], 1, shape[1], shape[2]))
                post_arrays = [np.eye(4).reshape(1, 1, 4, 4)] * (num_sites - q1 - 1) + [
                    np.eye(4).reshape(1, 4, 4)
                ]
                nonlocal_mpo = MatrixProductOperator.from_arrays(arrays + post_arrays)
        else:
            prior_arrays = [np.eye(4).reshape(1, 4, 4)] + [
                np.eye(4).reshape(1, 1, 4, 4)
            ] * (q0 - 2)
            shape = nonlocal_mpo.tensors[0].data.shape
            first_nonlocal_array = nonlocal_mpo.tensors[0].data.reshape(
                (1, shape[0], shape[1], shape[2])
            )
            remaining_arrays = [
                nonlocal_mpo.tensors[x].data for x in range(1, nonlocal_mpo.num_sites)
            ]
            if q1 == num_sites:
                nonlocal_mpo = MatrixProductOperator.from_arrays(
                    prior_arrays + [first_nonlocal_array] + remaining_arrays
                )
            else:
                shape = remaining_arrays[-1].shape
                remaining_arrays[-1] = remaining_arrays[-1].reshape(
                    (shape[0], 1, shape[1], shape[2])
                )
                post_arrays = [np.eye(4).reshape(1, 1, 4, 4)] * (num_sites - q1 - 1) + [
                    np.eye(4).reshape(1, 4, 4)
                ]
                nonlocal_mpo = MatrixProductOperator.from_arrays(
                    prior_arrays
                    + [first_nonlocal_array]
                    + remaining_arrays
                    + post_arrays
                )
        return nonlocal_mpo

    def build_identity_mpo(self) -> MatrixProductOperator:
        """Build identity MPO with physical dim 4"""
        id_array = np.eye(4)
        first_array = np.reshape(id_array, (1, 4, 4))
        middle_arrays = []
        for _ in range(self.num_qubits - 2):
            middle_array = np.reshape(id_array, (1, 1, 4, 4))
            middle_arrays.append(middle_array)
        last_array = np.reshape(id_array, (1, 4, 4))
        all_arrays = [first_array] + middle_arrays + [last_array]
        mpo = MatrixProductOperator.from_arrays(all_arrays)
        return mpo

    def build_tnqem_mpo(self) -> MatrixProductOperator:
        tnqem_mpo = self.build_identity_mpo()
        for data in self.noisy_circuit.values():
            data = data[0]
            qidxs = data["qubits"]
            qidxs = [q + 1 for q in qidxs]
            if self.representation == "Liouville":
                ideal_gate = data["ideal_liouville"]
                noisy_gate = data["noisy_liouville"]
            else:
                ideal_gate = data["ideal_ptm"]
                noisy_gate = data["noisy_ptm"]
            noisy_inverse = np.linalg.inv(noisy_gate)
            if len(qidxs) == 1:
                self.apply_one_site_term(tnqem_mpo, ideal_gate, qidxs[0])
                self.apply_one_site_term(tnqem_mpo, noisy_inverse, qidxs[0], left=True)
            else:
                full_left_mpo = self.two_site_term_to_full_length_mpo(
                    noisy_inverse, qidxs, tnqem_mpo.num_sites
                )
                full_right_mpo = self.two_site_term_to_full_length_mpo(
                    ideal_gate, qidxs, tnqem_mpo.num_sites
                )
                tnqem_mpo = tnqem_mpo.multiply_and_compress_three(
                    full_left_mpo, full_right_mpo, self.max_bond_dimension
                )
        return tnqem_mpo

    def marginalise_to_qubit(
        self, A: np.ndarray, is_top: bool, is_bottom: bool
    ) -> np.ndarray:
        """
        Collapse the d=4 physical index of a Liouville-basis MPS tensor down to
        d=2 by summing over the diagonal elements of the density matrix.

        index 0 → |0><0|  maps to qubit outcome 0
        index 3 → |1><1|  maps to qubit outcome 1
        indices 1,2 (coherences) are discarded

        Tensor shapes in → out:
        top      (χ_d, 4)       → (χ_d, 2)
        bottom   (χ_u, 4)       → (χ_u, 2)
        interior (χ_u, χ_d, 4) → (χ_u, χ_d, 2)
        """
        if self.representation == "Liouville":
            if is_top or is_bottom:
                return A[..., [0, 3]]
            else:
                return A[..., [0, 3]]
        else:
            rI = A[..., 0]
            rZ = A[..., 3]
            p0 = 0.5 * (rI + rZ)
            p1 = 0.5 * (rI - rZ)
            return np.stack([p0, p1], axis=-1)

    def sample_qubit_bitstrings_from_liouville_mps(
        self,
        mps: "MatrixProductState",
        num_samples: int = 1,
        seed: int | None = None,
    ) -> dict[str, int]:
        """
        Sample qubit bitstrings from an MPS in the Liouville (vectorised DM)
        representation with d=4 physical indices.

        Collapses each site's physical index from d=4 to d=2 by retaining only
        the |0><0| (index 0) and |1><1| (index 3) components, then delegates
        to the existing sample_from_mps.

        Parameters
        ----------
        mps         : MPS with d=4 physical indices in Liouville basis
        num_samples : number of bitstrings to draw
        seed        : RNG seed

        Returns
        -------
        dict mapping qubit bitstring → count
        """
        N = mps.num_sites

        # Canonicalise once here so we can read off dense tensors in canonical form
        canonical_mps = copy.deepcopy(mps)
        canonical_mps.move_orthogonality_centre(1)

        # Build reduced d=2 arrays
        reduced_arrays = []
        for site in range(1, N + 1):
            is_top = site == 1
            is_bottom = site == N
            A = np.asarray(canonical_mps.tensors[site - 1].data.todense())
            reduced_arrays.append(self.marginalise_to_qubit(A, is_top, is_bottom))

        # Reconstruct a new MPS with d=2 physical indices
        qubit_mps = MatrixProductState.from_arrays(reduced_arrays)

        # Delegate entirely to the existing sampler (already handles canonicalisation)
        return qubit_mps.sample_bitstrings(num_samples=num_samples, seed=seed)

    def run_single_shot_tnqem(
        self, input_bitstring: str, num_samples: int = 1
    ) -> list[str]:
        """Perform single shot TNQEM"""
        input_mps = self.bitstring_to_vectorised_dm_mps(input_bitstring)
        output_mps = input_mps.apply_mpo(self.tnqem_mpo)
        output_mps.normalise()
        samples = self.sample_qubit_bitstrings_from_liouville_mps(
            output_mps, num_samples
        )
        return samples

    def run_tnqem(self, counts: dict[str, int], samples_per_shot: int = 1):
        new_counts = {}
        for bitstring, count in counts.items():
            samples = self.run_single_shot_tnqem(bitstring, samples_per_shot * count)
            for s, c in samples.items():
                new_counts[s] = new_counts.get(s, 0) + c

        return new_counts
