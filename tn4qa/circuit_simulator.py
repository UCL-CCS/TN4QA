import copy

import numpy as np
import sparse
from numpy.linalg import svd
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
from scipy.sparse.linalg import svds
from sparse import COO

from tn4qa.mpo import MatrixProductOperator
from tn4qa.mps import MatrixProductState


class CircuitSimulator:
    """
    A class to simulate quantum circuits built using Qiskit
    """

    def __init__(
        self, circuit: QuantumCircuit, input_state: MatrixProductState | None = None
    ) -> None:
        """
        Class constructor.

        Args:
            circuit: The Qiskit QuantumCircuit object
        """
        self.circuit = circuit
        self.num_qubits = circuit.num_qubits
        self.set_input_state(input_state)
        self.current_state = copy.deepcopy(self.input_state)
        self.output_state = None
        self.site_mapping = {
            str(idx): str(idx) for idx in range(1, self.num_qubits + 1)
        }  # {physical_site:logical_site}
        self.mpo = MatrixProductOperator.identity_mpo(self.num_qubits)

    def set_input_state(self, input_state: MatrixProductState | None) -> None:
        """
        Set the input state to the circuit

        Args:
            input_state: The input state, defaults to the all zero state
        """
        if not input_state:
            input_state = MatrixProductState.all_zero_mps(self.num_qubits)

        self.input_state = input_state
        self.input_state.set_default_indices()

    def apply_one_qubit_gate(self, data: COO, site: int) -> None:
        """
        Apply a one-qubit gate in place

        Args:
            data: The one-qubit matrix
            site: Where to apply the gate to
        """
        site_loc = int(self.site_mapping[str(site)])
        tensor = self.current_state.tensors[site_loc - 1]
        if site_loc == 1 or site_loc == self.num_qubits:
            contraction = "ij,kj->ik"
        else:
            contraction = "ijk,lk->ijl"
        self.current_state.tensors[site_loc - 1].data = sparse.einsum(
            contraction, tensor.data, data
        )
        return

    def move_site_to_location(self, site_source: int, site_destination: int) -> None:
        """
        Move a site to a different location

        Args:
            site_source: The starting location
            site_destination: The final location
        """
        if site_source < site_destination:
            for site in range(site_source, site_destination):
                self.current_state.swap_neighbouring_sites(site)
                reverse_mapping = {v: k for k, v in self.site_mapping.items()}
                (
                    self.site_mapping[reverse_mapping[str(site)]],
                    self.site_mapping[reverse_mapping[str(site + 1)]],
                ) = (
                    str(site + 1),
                    str(site),
                )
        else:
            for site in range(site_source, site_destination, -1):
                self.current_state.swap_neighbouring_sites(site - 1)
                reverse_mapping = {v: k for k, v in self.site_mapping.items()}
                (
                    self.site_mapping[reverse_mapping[str(site - 1)]],
                    self.site_mapping[reverse_mapping[str(site)]],
                ) = (
                    str(site),
                    str(site - 1),
                )
        return

    def apply_two_qubit_gate(
        self,
        data: COO,
        sites: list[int],
        max_bond: int | None = None,
        tol: float = 1e-12,
    ) -> None:
        """
        Apply a two qubit gate in place

        Args:
            data: The two-qubit matrix
            sites: The sites to apply it to
            max_bond: The maximum allowed bond dimension
        """
        sites_locs = [int(self.site_mapping[str(site)]) for site in sites]
        site0, site1 = sites_locs[0], sites_locs[1]
        data = sparse.reshape(data, (2, 2, 2, 2))

        if self.num_qubits == 2:
            data = sparse.moveaxis(data, [0, 1, 2, 3], [1, 0, 3, 2])
            if site1 < site0:
                data = sparse.moveaxis(data, [0, 1, 2, 3], [1, 0, 3, 2])
            data = sparse.reshape(data, (4, 4))
            gate = UnitaryGate(data.todense())
            qc = QuantumCircuit(2)
            qc.append(gate, [site0 - 1, site1 - 1])
            gate_mpo = MatrixProductOperator.from_qiskit_gate(qc.data[0])
            self.current_state = self.current_state.apply_mpo(gate_mpo)
            return

        if site1 < site0:
            data = sparse.moveaxis(data, [0, 1, 2, 3], [1, 0, 3, 2])
            if site1 == site0 - 1:
                pass
            else:
                self.move_site_to_location(site1, site0 - 1)
            tensor0 = self.current_state.tensors[site0 - 2]
            tensor1 = self.current_state.tensors[site0 - 1]
            if site0 - 1 == 1:
                contraction = "ij,ikl,mnjl->knm"
                output_shape = (tensor1.dimensions[1], 2, 2)
                mat_shape = (2 * tensor1.dimensions[1], 2)
            elif site0 == self.num_qubits:
                contraction = "hij,ik,lmjk->mhl"
                output_shape = (2, tensor0.dimensions[0], 2)
                mat_shape = (2, tensor0.dimensions[0] * 2)
            else:
                contraction = "hij,ikl,mnjl->knhm"
                output_shape = (tensor1.dimensions[1], 2, tensor0.dimensions[0], 2)
                mat_shape = (tensor1.dimensions[1] * 2, tensor0.dimensions[0] * 2)
        else:
            if site1 == site0 + 1:
                pass
            else:
                self.move_site_to_location(site1, site0 + 1)
            tensor0 = self.current_state.tensors[site0 - 1]
            tensor1 = self.current_state.tensors[site0]
            if site0 == 1:
                contraction = "ij,ikl,mnjl->knm"
                output_shape = (tensor1.dimensions[1], 2, 2)
                mat_shape = (2 * tensor1.dimensions[1], 2)
            elif site0 + 1 == self.num_qubits:
                contraction = "hij,ik,lmjk->mhl"
                output_shape = (2, tensor0.dimensions[0], 2)
                mat_shape = (2, tensor0.dimensions[0] * 2)
            else:
                contraction = "hij,ikl,mnjl->knhm"
                output_shape = (tensor1.dimensions[1], 2, tensor0.dimensions[0], 2)
                mat_shape = (tensor1.dimensions[1] * 2, tensor0.dimensions[0] * 2)

        output_data = sparse.einsum(contraction, tensor0.data, tensor1.data, data)
        output_data = np.reshape(output_data, mat_shape)

        if max_bond:
            bond_dim = min([max_bond, mat_shape[0], mat_shape[1]])
        else:
            bond_dim = min([mat_shape[0], mat_shape[1]])

        if bond_dim >= min([mat_shape[0], mat_shape[1]]) - 1:
            u, s, vh = svd(output_data.todense(), full_matrices=False)
        else:
            u, s, vh = svds(output_data, k=bond_dim)

        s = s[s > 1e-14]
        sq = s**2
        cumulative = np.cumsum(sq[::-1])[::-1]
        keep_dim = len(s)
        for k in range(len(s)):
            if cumulative[k] < tol**2:
                keep_dim = k + 1
                break
        keep_dim = min(keep_dim, bond_dim)

        new_data0 = sparse.COO.from_numpy(vh[:keep_dim, :])
        new_data1 = sparse.COO.from_numpy(u[:, :keep_dim] * s[:keep_dim])

        if site1 < site0:
            if site0 - 1 == 1:
                new_data0 = sparse.reshape(new_data0, (keep_dim,) + (output_shape[2],))
                new_data1 = sparse.reshape(new_data1, output_shape[:2] + (keep_dim,))
                new_data1 = sparse.moveaxis(new_data1, [2], [0])
            elif site0 == self.num_qubits:
                new_data0 = sparse.reshape(new_data0, (keep_dim,) + output_shape[-2:])
                new_data0 = sparse.moveaxis(new_data0, [0], [1])
                new_data1 = sparse.reshape(new_data1, (output_shape[0],) + (keep_dim,))
                new_data1 = sparse.moveaxis(new_data1, [0], [1])
            else:
                new_data0 = sparse.reshape(new_data0, (keep_dim,) + output_shape[-2:])
                new_data0 = sparse.moveaxis(new_data0, [0], [1])
                new_data1 = sparse.reshape(new_data1, output_shape[:2] + (keep_dim,))
                new_data1 = sparse.moveaxis(new_data1, [2], [0])
            self.current_state.tensors[site0 - 2].data = new_data0
            self.current_state.tensors[
                site0 - 2
            ].dimensions = self.current_state.tensors[site0 - 2].data.shape
            self.current_state.tensors[site0 - 1].data = new_data1
            self.current_state.tensors[
                site0 - 1
            ].dimensions = self.current_state.tensors[site0 - 1].data.shape
            self.current_state.bond_dims = [
                t.dimensions[0] for t in self.current_state.tensors[1:]
            ]
            self.current_state.bond_dimension = max(self.current_state.bond_dims)
        else:
            if site0 == 1:
                new_data0 = sparse.reshape(new_data0, (keep_dim,) + (output_shape[2],))
                new_data1 = sparse.reshape(new_data1, output_shape[:2] + (keep_dim,))
                new_data1 = sparse.moveaxis(new_data1, [2], [0])
            elif site0 + 1 == self.num_qubits:
                new_data0 = sparse.reshape(new_data0, (keep_dim,) + output_shape[-2:])
                new_data0 = sparse.moveaxis(new_data0, [0], [1])
                new_data1 = sparse.reshape(new_data1, (output_shape[0],) + (keep_dim,))
                new_data1 = sparse.moveaxis(new_data1, [0], [1])
            else:
                new_data0 = sparse.reshape(new_data0, (keep_dim,) + output_shape[-2:])
                new_data0 = sparse.moveaxis(new_data0, [0], [1])
                new_data1 = sparse.reshape(new_data1, output_shape[:2] + (keep_dim,))
                new_data1 = sparse.moveaxis(new_data1, [2], [0])
            self.current_state.tensors[site0 - 1].data = new_data0
            self.current_state.tensors[
                site0 - 1
            ].dimensions = self.current_state.tensors[site0 - 1].data.shape
            self.current_state.tensors[site0].data = new_data1
            self.current_state.tensors[site0].dimensions = self.current_state.tensors[
                site0
            ].data.shape
            self.current_state.bond_dims = [
                t.dimensions[0] for t in self.current_state.tensors[1:]
            ]
            self.current_state.bond_dimension = max(self.current_state.bond_dims)
        return

    def apply_general_gate(
        self,
        inst: CircuitInstruction,  # type: ignore
        max_bond: int | None = None,
    ) -> None:  # type: ignore
        """
        Apply a gate with no better option

        Args:
            inst: The circuit instruction
        """
        qidxs = [inst.qubits[i]._index + 1 for i in range(inst.operation.num_qubits)]
        qidxs = [int(self.site_mapping[str(qidx)]) for qidx in qidxs]
        mpo = MatrixProductOperator.from_qiskit_gate(inst)
        self.current_state = self.current_state.apply_sub_mpo(mpo, qidxs, max_bond)
        return

    def restore_ordering(self) -> None:
        """
        Restore correct site ordering
        """
        reversed_mapping = {v: k for k, v in self.site_mapping.items()}
        target_site_ordering = [
            int(reversed_mapping[str(site)]) for site in range(1, self.num_qubits + 1)
        ]
        self.current_state.reorder_sites(target_site_ordering)
        self.site_mapping = {
            str(idx): str(idx) for idx in range(1, self.num_qubits + 1)
        }
        return

    def run(
        self, max_bond_dimension: int | None = None, samples: int | None = None
    ) -> MatrixProductState | dict[str, int]:
        """
        Execute the quantum circuit

        Args:
            max_bond_dimension: The maximum allowed bond dimension
            samples: If provided will return this number of bitstring samples from the output state
        """
        for inst in self.circuit.data:
            qidxs = [
                inst.qubits[i]._index + 1 for i in range(inst.operation.num_qubits)
            ]
            data = COO.from_numpy(Operator(inst.operation).reverse_qargs().data)
            if len(qidxs) == 1:
                self.apply_one_qubit_gate(data, qidxs[0])
            elif len(qidxs) == 2:
                self.apply_two_qubit_gate(data, qidxs, max_bond_dimension)
            else:
                self.apply_general_gate(inst, max_bond_dimension)
        self.restore_ordering()
        self.output_state = self.current_state

        if samples:
            sample_dict = self.output_state.sample_bitstrings(samples)
            return sample_dict

        return self.output_state

    def get_operator_mpo(
        self, after_gate: int | None = None, max_bond: int | None = None
    ) -> MatrixProductOperator:
        """
        Build the MPO representing the quantum circuit

        Args:
            after_gate: Builds the MPO representing the circuit up to after the given gate number. Defaults to full circuit
            max_bond: Maximum allowed bond dimension

        Returns:
            An MPO
        """
        if not after_gate:
            qc_data = self.circuit.data
        else:
            qc_data = self.circuit.data[:after_gate]

        qc_after_gate = QuantumCircuit(self.num_qubits)
        for inst in qc_data:
            qc_after_gate.append(inst)
        mpo = MatrixProductOperator.from_qiskit_circuit(qc_after_gate, max_bond)
        return mpo
