import logging
from typing import Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class QubitNoise:
    id: int
    t1_ns: float
    t2_ns: float
    readout_p0: float
    readout_p1: float
    gates_1q: dict[str, float]
    gates_2q: dict[int, float]

@dataclass
class NoiseData:
    n_qubits: int
    coupling_map: list[list[int]]
    qubit_noise: list[QubitNoise]


def get_coupling_map(nqubits: int, data: dict[str, Any]) -> list[list[int]]:
    logger.debug("Getting coupling map.")
    coupling_map = []
    for qi in range(1, nqubits + 1):
        for qj in range(1, nqubits + 1):
            dict_string = f"TC-{qi}-{qj}.cz_gate_fidelity"
            if dict_string in data["metrics"]:
                coupling_map.append([qi - 1, qj - 1])
                coupling_map.append([qj - 1, qi - 1])
    return coupling_map


def _get_readout_fid(qidx: int, data: dict[str, Any]) -> float:
    dict_string = f"QB{qidx+1}.single_shot_readout_fidelity"
    fid = float(data["metrics"][dict_string]["value"])
    return fid


def _get_t1_time(qidx: int, data: dict[str, Any]) -> float:
    dict_string = f"QB{qidx+1}.t1_time"
    t1 = float(data["metrics"][dict_string]["value"])
    return t1


def _get_t2_time(qidx: int, data: dict[str, Any]) -> float:
    dict_string = f"QB{qidx+1}.t2_time"
    t2 = float(data["metrics"][dict_string]["value"])
    return t2


def _get_1q_gate_fid(qidx: int, data: dict[str, Any]) -> float:
    dict_string = f"QB{qidx+1}.fidelity_1qb_gates_averaged"
    fid = float(data["metrics"][dict_string]["value"])
    return fid


def _get_2q_gate_fid(
    control_qidx: int,
    target_qidx: int,
    data: dict[str, Any],
) -> float:
    # cz is symmetric
    dict_string_tc = f"TC-{target_qidx+1}-{control_qidx+1}.cz_gate_fidelity"
    dict_string_ct = f"TC-{control_qidx+1}-{target_qidx+1}.cz_gate_fidelity"
    if data["metrics"].get(dict_string_tc) is not None:
        fid = float(data["metrics"][dict_string_tc]["value"])
    elif data["metrics"].get(dict_string_ct) is not None:
        fid = float(data["metrics"][dict_string_ct]["value"])
    else:
        error_string = f"Could not find {dict_string_tc} or {dict_string_ct} in data."
        logger.error(error_string)
        raise ValueError(error_string)
    return fid


def generate_noise_data(num_qubits: int, data: dict[str, Any]) -> NoiseData:
    logger.debug("Generating noise data for:")
    noise_data = NoiseData()
    noise_data.coupling_map = get_coupling_map(num_qubits, data)
    noise_data.n_qubits = num_qubits
    for qidx in range(num_qubits):
        logger.debug("qubit %d", qidx)
        single_qubit_noise = QubitNoise()
        single_qubit_noise.id = qidx
        single_qubit_noise.t1_ns = _get_t1_time(qidx, data) * 1e9
        single_qubit_noise.t2_ns = _get_t2_time(qidx, data) * 1e9
        readout_fid = _get_readout_fid(qidx, data)
        single_qubit_noise.readout_p0 = readout_fid
        single_qubit_noise.readout_p1 = readout_fid
        single_qubit_noise.gates_1q = {"r": _get_1q_gate_fid(qidx, data)}
        single_qubit_noise.gates_2q = {}
        for q1, q2 in noise_data.coupling_map:
            if q1 == qidx:
                single_qubit_noise.gates_2q[str(q2)] = _get_2q_gate_fid(q1, q2, data)

        noise_data.qubit_noise.append(single_qubit_noise)
    return noise_data
