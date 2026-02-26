import copy

import numpy as np
import sparse
from numpy.linalg import svd
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
from sparse import COO

from tn4qa.mpo import MatrixProductOperator
from tn4qa.mps import MatrixProductState
from tn4qa.tensor import Tensor
from tn4qa.tn import TensorNetwork


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
        tensor = self.current_state.tensors[site - 1]
        if site == 1 or site == self.num_qubits:
            contraction = "ij,kj->ik"
        else:
            contraction = "ijk,lk->ijl"
        self.current_state.tensors[site - 1].data = sparse.einsum(
            contraction, tensor.data, data
        )
        return

    def from_qiskit_gate(self, inst: CircuitInstruction) -> MatrixProductOperator:  # type: ignore
        """
        Create an MPO from a single Qiskit gate

        Args:
            inst: The Qiskit CircuitInstruction

        Returns:
            An MPO
        """
        qidxs = [inst.qubits[i]._index + 1 for i in range(inst.operation.num_qubits)]
        indices = [f"out{qidxs[i]}" for i in range(inst.operation.num_qubits)] + [
            f"in{qidxs[i]}" for i in range(inst.operation.num_qubits)
        ]
        if len(qidxs) == 1:
            arrays = [Operator(inst.operation).reverse_qargs().data]
        elif len(qidxs) == 2:
            tensor = Tensor.from_qiskit_gate(inst, indices=indices)
            tn = TensorNetwork([tensor])
            tn.svd(
                tn.tensors[0],
                input_indices=[indices[0], indices[2]],
                output_indices=[indices[1], indices[3]],
                new_index_name=f"C{qidxs[0]}",
            )
            tn.tensors[0].reorder_indices(
                [f"C{qidxs[0]}", f"out{qidxs[0]}", f"in{qidxs[0]}"]
            )
            tn.tensors[1].reorder_indices(
                [f"C{qidxs[0]}", f"out{qidxs[1]}", f"in{qidxs[1]}"]
            )
            arrays = [tn.tensors[i].data for i in range(2)]
        else:
            tensor = Tensor.from_qiskit_gate(inst, indices=indices)
            tn = TensorNetwork([tensor])
            for idx in range(len(qidxs) - 1):
                t = tn.tensors[idx]
                input_inds = [indices[idx], indices[len(qidxs) + idx]]
                output_inds = (
                    indices[idx + 1 : len(qidxs)] + indices[len(qidxs) + idx + 1 :]
                )
                if idx != 0:
                    input_inds.insert(0, f"C{idx}")
                tn.svd(
                    t,
                    input_indices=input_inds,
                    output_indices=output_inds,
                    new_index_name=f"C{idx+1}",
                    new_labels=[[f"T{idx+1}"], [f"T{idx+2}"]],
                )
                if idx == 0:
                    new_idx_order1 = [
                        f"C{idx+1}",
                        f"out{qidxs[idx]}",
                        f"in{qidxs[idx]}",
                    ]
                    new_idx_order2 = [f"C{idx+1}"] + output_inds
                else:
                    new_idx_order1 = [
                        f"C{idx}",
                        f"C{idx+1}",
                        f"out{qidxs[idx]}",
                        f"in{qidxs[idx]}",
                    ]
                new_idx_order2 = [f"C{idx+1}"] + output_inds
                tn.tensors[idx].reorder_indices(new_idx_order1)
                tn.tensors[idx + 1].reorder_indices(new_idx_order2)
            arrays = [tn.tensors[i].data for i in range(len(qidxs))]
        mpo = MatrixProductOperator.from_arrays(arrays)
        return mpo

    def apply_local_two_qubit_gate(
        self,
        data: COO,
        sites: list[int],
        max_bond: int | None = None,
        tol: float = 1e-12,
    ) -> None:
        """
        Apply a two qubit gate to neighbouring qubits

        Args:
            data: The two-qubit matrix
            sites: The sites to apply it to
            max_bond: The maximum allowed bond dimension
        """
        site0, site1 = sites[0], sites[1]

        if self.current_state.num_sites == 2:
            data = sparse.reshape(data, (2, 2, 2, 2))
            if site1 < site0:
                data = sparse.moveaxis(data, [0, 1, 2, 3], [1, 0, 3, 2])
            data = sparse.reshape(data, (4, 4))
            gate = UnitaryGate(data.todense())
            qc = QuantumCircuit(2)
            qc.append(gate, [site0 - 1, site1 - 1])
            gate_mpo = self.from_qiskit_gate(qc.data[0])
            self.current_state = self.current_state.apply_sub_mpo(
                gate_mpo, [site0, site1], max_bond=max_bond
            )
            return

        data = sparse.reshape(data, (2, 2, 2, 2))
        if site1 < site0:
            data = sparse.moveaxis(data, [0, 1, 2, 3], [1, 0, 3, 2])
            assert site1 == site0 - 1
            tensor0 = self.current_state.tensors[site0 - 2]
            tensor1 = self.current_state.tensors[site0 - 1]
            if site0 - 1 == 1:
                contraction = "hi,hkl,noil->kon"
                output_shape = (
                    tensor1.dimensions[1],
                    2,
                    2,
                )
                mat_shape = (
                    2 * tensor1.dimensions[1],
                    2,
                )
            elif site0 == self.current_state.num_sites:
                contraction = "hij,il,nojl->ohn"
                output_shape = (
                    2,
                    tensor0.dimensions[0],
                    2,
                )
                mat_shape = (
                    2,
                    tensor0.dimensions[0] * 2,
                )
            else:
                contraction = "hij,ilm,opjm->lpho"
                output_shape = (
                    tensor1.dimensions[1],
                    2,
                    tensor0.dimensions[0],
                    2,
                )
                mat_shape = (
                    tensor1.dimensions[1] * 2,
                    tensor0.dimensions[0] * 2,
                )
        else:
            assert site1 == site0 + 1
            tensor0 = self.current_state.tensors[site0 - 1]
            tensor1 = self.current_state.tensors[site0]
            if site0 == 1:
                contraction = "hi,hkl,noil->kon"
                output_shape = (
                    tensor1.dimensions[1],
                    2,
                    2,
                )
                mat_shape = (
                    2 * tensor1.dimensions[1],
                    2,
                )
            elif site0 + 1 == self.current_state.num_sites:
                contraction = "hij,il,nojl->ohn"
                output_shape = (
                    2,
                    tensor0.dimensions[0],
                    2,
                )
                mat_shape = (
                    2,
                    tensor0.dimensions[0] * 2,
                )
            else:
                contraction = "hij,ilm,opjm->lpho"
                output_shape = (
                    tensor1.dimensions[1],
                    2,
                    tensor0.dimensions[0],
                    2,
                )
                mat_shape = (
                    tensor1.dimensions[1] * 2,
                    tensor0.dimensions[0] * 2,
                )

        output_data = sparse.einsum(contraction, tensor0.data, tensor1.data, data)
        output_data = np.reshape(output_data, mat_shape)

        if max_bond:
            bond_dim = min([max_bond, mat_shape[0], mat_shape[1]])
        else:
            bond_dim = min([mat_shape[0], mat_shape[1]])

        u, s, vh = svd(output_data.todense(), full_matrices=False)
        s = s[s > 1e-16]
        sq = s**2
        cumulative = np.cumsum(sq[::-1])[::-1]
        keep_dim = len(s)
        for k in range(len(s)):
            if cumulative[k] < tol**2:
                keep_dim = k + 1
                break
        keep_dim = min(keep_dim, bond_dim)

        threshold = 1e-14
        data0 = vh[:keep_dim, :]
        data0[np.abs(data0) < threshold] = 0.0
        data1 = u[:, :keep_dim] * s[:keep_dim]
        data1[np.abs(data1) < threshold] = 0.0

        new_data0 = sparse.COO.from_numpy(data0)
        new_data1 = sparse.COO.from_numpy(data1)

        if site1 < site0:
            if site0 - 1 == 1:
                new_data0 = sparse.reshape(new_data0, (keep_dim,) + output_shape[-1:])
                new_data1 = sparse.reshape(new_data1, output_shape[:2] + (keep_dim,))
                new_data1 = sparse.moveaxis(new_data1, [2], [0])
            elif site0 == self.current_state.num_sites:
                new_data0 = sparse.reshape(new_data0, (keep_dim,) + output_shape[-2:])
                new_data0 = sparse.moveaxis(new_data0, [0], [1])
                new_data1 = sparse.reshape(new_data1, output_shape[:1] + (keep_dim,))
                new_data1 = sparse.moveaxis(new_data1, [1], [0])
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
            self.bond_dims = [t.dimensions[0] for t in self.current_state.tensors[1:]]
            self.bond_dimension = max(self.bond_dims)
        else:
            if site0 == 1:
                new_data0 = sparse.reshape(new_data0, (keep_dim,) + output_shape[-1:])
                new_data1 = sparse.reshape(new_data1, output_shape[:2] + (keep_dim,))
                new_data1 = sparse.moveaxis(new_data1, [2], [0])
            elif site0 + 1 == self.current_state.num_sites:
                new_data0 = sparse.reshape(new_data0, (keep_dim,) + output_shape[-2:])
                new_data0 = sparse.moveaxis(new_data0, [0], [1])
                new_data1 = sparse.reshape(new_data1, output_shape[:1] + (keep_dim,))
                new_data1 = sparse.moveaxis(new_data1, [1], [0])
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
            self.bond_dims = [t.dimensions[0] for t in self.current_state.tensors[1:]]
            self.bond_dimension = max(self.bond_dims)
        return self

    def apply_nonlocal_two_qubit_gate(
        self,
        data: COO,
        sites: list[int],
        max_bond: int | None = None,
    ) -> None:
        """
        Apply a two qubit gate on distant qubits

        Args:
            data: The two-qubit matrix
            sites: The sites to apply it to
            max_bond: The maximum allowed bond dimension
        """
        site0, site1 = sites[0], sites[1]
        if site0 == site1 - 1 or site1 == site0 - 1:
            return self.apply_local_two_qubit_gate(data, sites, max_bond)

        data = sparse.reshape(data, (2, 2, 2, 2))
        data = sparse.moveaxis(data, [0, 1, 2, 3], [1, 0, 3, 2])
        if site1 < site0:
            data = sparse.moveaxis(data, [0, 1, 2, 3], [1, 0, 3, 2])
        data = sparse.reshape(data, (4, 4))
        gate = UnitaryGate(data.todense())
        qc = QuantumCircuit(2)
        qc.append(gate, [0, 1])
        gate_mpo = self.from_qiskit_gate(qc.data[0])
        gate_mpo_bond = gate_mpo.tensors[0].dimensions[0]

        first_array = gate_mpo.tensors[0].data
        last_array = gate_mpo.tensors[1].data
        q0 = min(site0, site1)
        q1 = max(site0, site1)
        num_intermediate_sites = q1 - q0 - 1
        middle_array = np.array([[np.zeros((2, 2))] * gate_mpo_bond] * gate_mpo_bond)
        for x in range(gate_mpo_bond):
            middle_array[x, x, :, :] = np.eye(2)
        middle_arrays = [middle_array for _ in range(num_intermediate_sites)]
        arrays = [first_array] + middle_arrays + [last_array]
        nonlocal_mpo = MatrixProductOperator.from_arrays(arrays)
        if q0 == 1:
            arrays = [
                nonlocal_mpo.tensors[x].data for x in range(nonlocal_mpo.num_sites)
            ]
            if q1 == self.current_state.num_sites:
                pass
            else:
                shape = arrays[-1].shape
                arrays[-1] = arrays[-1].reshape((shape[0], 1, shape[1], shape[2]))
                post_arrays = [np.eye(2).reshape(1, 1, 2, 2)] * (
                    self.current_state.num_sites - q1 - 1
                ) + [np.eye(2).reshape(1, 2, 2)]
                nonlocal_mpo = MatrixProductOperator.from_arrays(arrays + post_arrays)
        else:
            prior_arrays = [np.eye(2).reshape(1, 2, 2)] + [
                np.eye(2).reshape(1, 1, 2, 2)
            ] * (q0 - 2)
            shape = nonlocal_mpo.tensors[0].data.shape
            first_nonlocal_array = nonlocal_mpo.tensors[0].data.reshape(
                (1, shape[0], shape[1], shape[2])
            )
            remaining_arrays = [
                nonlocal_mpo.tensors[x].data for x in range(1, nonlocal_mpo.num_sites)
            ]
            if q1 == self.current_state.num_sites:
                nonlocal_mpo = MatrixProductOperator.from_arrays(
                    prior_arrays + [first_nonlocal_array] + remaining_arrays
                )
            else:
                shape = remaining_arrays[-1].shape
                remaining_arrays[-1] = remaining_arrays[-1].reshape(
                    (shape[0], 1, shape[1], shape[2])
                )
                post_arrays = [np.eye(2).reshape(1, 1, 2, 2)] * (
                    self.current_state.num_sites - q1 - 1
                ) + [np.eye(2).reshape(1, 2, 2)]
                nonlocal_mpo = MatrixProductOperator.from_arrays(
                    prior_arrays
                    + [first_nonlocal_array]
                    + remaining_arrays
                    + post_arrays
                )

        self.current_state = self.current_state.apply_mpo(nonlocal_mpo, max_bond)
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
                self.apply_nonlocal_two_qubit_gate(data, qidxs, max_bond_dimension)
                self.current_state.normalise()
        self.output_state = self.current_state
        self.output_state.normalise()

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
