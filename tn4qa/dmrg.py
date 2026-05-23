from __future__ import annotations

import numpy as np
from pyblock2.algebra.io import MPSTools
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from .mps import MatrixProductState


class DMRG:
    def __init__(
        self,
        hamiltonian: dict[str, complex],
        max_mps_bond: int = 16,
        initial_mps: MatrixProductState | None = None,
    ) -> DMRG:
        self.hamiltonian = hamiltonian
        for pauli, weight in hamiltonian.items():
            num_y = pauli.count("Y")
            new_weight = weight.real / (1 - num_y % 4)
            hamiltonian[pauli] = new_weight
        pauli = {
            key: value for key, value in hamiltonian.items() if np.abs(value) > 0.0
        }
        pauli = list(pauli.items())
        self.pauli_terms = pauli
        self.nqubits = len(self.pauli_terms[0][0])
        self.max_bond = max_mps_bond

        self.driver = self.set_driver()
        self.mpo = self.set_mpo()
        self.initial_state = initial_mps

        self.ket = self.set_initial_state(initial_mps)

        self.energy = None
        self.mps = None
        return

    def set_driver(self):
        driver_obj = DMRGDriver(
            symm_type=SymmetryTypes.SGB,
            n_threads=4,
        )
        driver_obj.initialize_system(
            n_sites=self.nqubits,
            pauli_mode=True,
        )
        return driver_obj

    def set_mpo(self):
        mpo = self.driver.get_mpo_any_pauli(self.pauli_terms)
        return mpo

    def set_initial_state(self, initial_mps: MatrixProductState | None):
        if not initial_mps:
            ket = self.driver.get_random_mps(tag="GS", bond_dim=2, nroots=1)
            ket = self.driver.get_random_mps(tag="GS", bond_dim=2, nroots=1)
        else:
            arrays = []
            for t in initial_mps.tensors:
                array = t.to_dense()
                array[...] = array[..., ::-1]
                arrays.append(array)
            initial_mps = MatrixProductState.from_arrays(arrays)

            for bidx in range(1, initial_mps.num_sites):
                if initial_mps.tensors[bidx].data.shape[0] == 1:
                    initial_mps = initial_mps.expand_bond_dimension(1, bidx)
            initial_mps.update_bond_information()
            self.initial_state = initial_mps
            initial_mps.reshape("pud")
            tensor_list = [
                initial_mps.tensors[i].to_dense() for i in range(initial_mps.num_sites)
            ]
            tensor_list[0] = np.expand_dims(tensor_list[0], axis=1)
            tensor_list[-1] = np.expand_dims(tensor_list[-1], axis=2)
            ket = get_mps_from_tensors(
                self.driver,
                tag="GS",
                tensors=tensor_list,
                bond_dim=initial_mps.bond_dimension,
            )
        return ket

    def run(
        self,
        nsweeps: int,
        bond_dims: list[int] | None = None,
        noises: list[int] | None = None,
        thrds: list[int] | None = None,
    ):
        if not bond_dims:
            bond_dims = [
                round(2 + i * (self.max_bond - 2) / (nsweeps - 1))
                for i in range(nsweeps)
            ]
        if not noises:
            noises = (
                [1e-1] * max(1, nsweeps // 4)
                + [1e-2] * max(1, nsweeps // 4)
                + [1e-4] * max(1, nsweeps // 4)
                + [1e-6] * max(1, nsweeps - 3 * (nsweeps // 4) - 1)
                + [0]
            )
        if not thrds:
            thrds = [1e-12] * len(bond_dims)
        self.energy = self.driver.dmrg(
            self.mpo,
            self.ket,
            n_sweeps=nsweeps,
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            noise_type="Perturbative",
            iprint=0,
        )
        self.mps = self.ket_to_tn4qa_mps()
        return self.energy, self.mps

    def ket_to_tn4qa_mps(self):
        ket = self.driver.adjust_mps(self.ket, dot=1)[0]
        pyket = MPSTools.from_block2(ket)

        arrays = []
        for tensor in pyket.tensors:
            array = np.array(tensor.blocks[0].reduced)
            arrays.append(array)

        first, last = arrays[0], arrays[-1]
        if first.ndim == 3:
            arrays[0] = first[:, :, 0]
        if last.ndim == 3:
            arrays[-1] = last[0, :, :]

        first[...] = first[::-1, ...]
        last[...] = last[..., ::-1]
        for a in arrays[1:-1]:
            a[...] = a[:, ::-1, :]

        mps = MatrixProductState.from_arrays(arrays, shape="upd")
        mps.normalise()

        return mps


def get_mps_from_tensors(self, tag, tensors, bond_dim=None, center=0, target=None):
    """
    Initialize an MPS from a list of explicit dense numpy tensors.

    Args:
        tensors : list[np.ndarray]
            List of n_sites tensors, each of shape (left_bond, phys_dim, right_bond).
            Boundary tensors should have a dummy size-1 index on the open boundary side,
            i.e. site 0 has shape (1, phys, right) and site n-1 has shape (left, phys, 1).
    """
    bw = self.bw

    if target is None:
        target = self.target

    max_bd = bond_dim or max(t.shape[2] for t in tensors[:-1])

    mps_info = bw.brs.MPSInfo(self.n_sites, self.vacuum, target, self.ghamil.basis)
    mps_info.tag = tag
    mps_info.set_bond_dimension_full_fci(self.left_vacuum, self.vacuum)
    mps_info.set_bond_dimension(max_bd)
    mps_info.bond_dim = max_bd

    mps = bw.bs.MPS(self.n_sites, center, 1)
    mps.initialize(mps_info)
    # No random_canonicalize here

    for i in range(self.n_sites):
        ts = mps.tensors[i]
        phys, left, right = tensors[i].shape
        data = tensors[i].transpose(1, 0, 2).reshape(phys, left * right).ravel()
        ts[0] = data

    mps.save_mutable()
    mps.save_data()
    mps_info.save_mutable()
    mps_info.save_data(self.scratch + "/%s-mps_info.bin" % tag)

    # No adjust_mps here — let the first DMRG sweep canonicalize it
    return mps
