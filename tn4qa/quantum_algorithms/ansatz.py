from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliTwoDesign, ExcitationPreserving, RXGate, RYGate, RZGate, CXGate, CZGate
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper
from typing import List
from pyscf import scf

def hea_ansatz(n_qubits : int, layers : int, sq_rotations : List[str], mq_gate : str) -> QuantumCircuit:
    """
    Hardware Efficient Ansatz.
    """
    hea = QuantumCircuit(n_qubits)
    num_params = 3 * n_qubits * layers 
    params = [Parameter(str(a)) for a in range(num_params)]


    def get_sq_gate(idx, param):
        sq = sq_rotations[idx]
        if "rx" == sq: return RXGate(param)
        if "ry" == sq: return RYGate(param)
        if "rz" == sq: return RZGate(param)

    if mq_gate == "cx":
        mq_gate = CXGate()
    elif mq_gate == "cz":
        mq_gate = CZGate()

    for _ in range(layers):

        for q_idx in range(n_qubits):
            gate = get_sq_gate(0, params[0])
            hea.append(gate, q_idx)
            params.pop(0)
        for q_idx in range(n_qubits):
            gate = get_sq_gate(1, params[0])
            hea.append(gate, q_idx)
            params.pop(0)
        for q_idx in range(n_qubits):
            gate = get_sq_gate(0, params[0])
            hea.append(gate, q_idx)
            params.pop(0)
        
        for q_idx in range(n_qubits-1):
            hea.append(mq_gate, (q_idx, q_idx+1))
        hea.append(mq_gate, (n_qubits-1, 0))

    return hea

def pauli_two_design_ansatz(n_qubits : int, layers : int) -> QuantumCircuit:
    """
    Pauli two design ansatz.
    """
    return PauliTwoDesign(n_qubits, reps=layers).decompose()

def number_preserving_ansatz(n_qubits : int, layers : int, entanglement : str) -> QuantumCircuit:
    """
    Number preserving ansatz.
    """
    return ExcitationPreserving(n_qubits, reps=layers, entanglement=entanglement).decompose()

def uccsd_ansatz(scf_obj : scf, reps : int, encoding : str) -> QuantumCircuit:
    """
    UCCSD ansatz.
    """
    if encoding == "JW":
        qm = JordanWignerMapper()
    elif encoding == "BK":
        qm = BravyiKitaevMapper()
    else:
        print("Encoding not supported, must be one of 'JW' or 'BK'")
        return 

    nsp = scf_obj.nao
    np = scf_obj.nelec

    return UCCSD(num_spatial_orbitals=nsp, num_particles=np, qubit_mapper=qm, reps=reps).decompose().decompose().decompose()
