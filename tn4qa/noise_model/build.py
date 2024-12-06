from collections import defaultdict
from typing import Any
import numpy as np
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    thermal_relaxation_error,
)
from device_characterisation import get_coupling_map, generate_noise_data

def _add_readout_noise(
    num_qubits: int,
    noise_data: dict[str | int, Any],
    target: NoiseModel | dict[str, dict[str, np.ndarray]],
) -> None:
    logger.debug("Adding readout noise for:")
    for qidx in range(num_qubits):
        logger.debug("qubit %d", qidx)
        p0given0 = noise_data[str(qidx)]["measurements"]["p00"]
        p1given1 = noise_data[str(qidx)]["measurements"]["p11"]
        readout_error_mat = np.array(
            [[p0given0, 1 - p0given0], [1 - p1given1, p1given1]]
        )
        if isinstance(target, NoiseModel):
            readout_error = ReadoutError(readout_error_mat)
            target.add_readout_error(readout_error, [qidx])
        else:
            inv_readout_error = np.linalg.inv(readout_error_mat)
            target[str(qidx)]["measurement"] = inv_readout_error


def _add_incoherent_noise(
    num_qubits: int,
    noise_data: dict[str | int, Any],
    target: NoiseModel | dict[str, dict[str, np.ndarray]],
    gate_duration_ns_1q: int = 20,
) -> None:
    logger.debug("Adding incoherent noise for:")
    for qidx in range(num_qubits):
        logger.debug("qubit %d", qidx)
        T1_ns = noise_data[str(qidx)]["T1_ns"]
        T2_ns = noise_data[str(qidx)]["T2_ns"]
        thermal_relaxation = thermal_relaxation_error(
            T1_ns * 10e-9, T2_ns * 10e-9, gate_duration_ns_1q * 10e-9
        )
        if isinstance(target, NoiseModel):
            target.add_quantum_error(thermal_relaxation, ["r", "id"], [qidx])
        else:
            thermal_relaxation_mat = thermal_relaxation.to_quantumchannel().data
            inv_thermal_relaxation = np.linalg.inv(thermal_relaxation_mat)
            target[str(qidx)]["thermal relaxation"] = inv_thermal_relaxation


def _add_coherent_noise(
    num_qubits: int,
    noise_data: dict[str | int, Any],
    target: NoiseModel | dict[str, dict[str, np.ndarray]],
    gate_duration_ns_1q: int = 20,
) -> None:
    logger.debug("Adding coherent noise for:")
    for qidx in range(num_qubits):
        logger.debug("qubit %d", qidx)
        T1_ns = noise_data[str(qidx)]["T1_ns"]
        T2_ns = noise_data[str(qidx)]["T2_ns"]
        p_incoherent_error_1q = (gate_duration_ns_1q / T1_ns) + (
            gate_duration_ns_1q / T2_ns
        )
        p_total_error_1q = noise_data[str(qidx)]["gates_1q"]["r"]
        p_coherent_error_1q = max(p_total_error_1q - p_incoherent_error_1q, 0.0)
        if p_coherent_error_1q <= 0:
            continue
        coherent_error_1q = depolarizing_error(p_coherent_error_1q, 1)
        if isinstance(target, NoiseModel):
            target.add_quantum_error(coherent_error_1q, ["r"], [qidx])
        else:
            coherent_error_1q_mat = coherent_error_1q.to_quantumchannel().data
            inv_coherent_error_1q = np.linalg.inv(coherent_error_1q_mat)
            target[str(qidx)]["coherent error"] = inv_coherent_error_1q


def _add_single_qubit_noise(
    num_qubits: int,
    noise_data: dict[str | int, Any],
    target: NoiseModel | dict[str, dict[str, np.ndarray]],
    gate_duration_ns_1q: int = 20,
) -> None:
    logger.debug("Adding single qubit noise.")
    _add_incoherent_noise(
        num_qubits,
        noise_data,
        target,
        gate_duration_ns_1q,
    )
    _add_coherent_noise(
        num_qubits,
        noise_data,
        target,
        gate_duration_ns_1q,
    )


def _add_two_qubit_noise(
    coupling_map: list[list[int]],
    noise_data: dict[str | int, Any],
    target: NoiseModel | dict[str, np.ndarray],
) -> None:
    logger.debug("Adding two qubit noise.")
    for q1, q2 in coupling_map:
        logger.debug("q%dq%d", q1, q2)
        error_rate = noise_data[str(q1)]["gates_2q"][str(q2)]
        coherent_error_2q = depolarizing_error(error_rate, 2)

        if isinstance(target, NoiseModel):
            target.add_quantum_error(coherent_error_2q, ["cz"], [q1, q2])
        else:
            coherent_error_2q_mat = coherent_error_2q.to_quantumchannel().data
            inv_coherent_error_2q = np.linalg.inv(coherent_error_2q_mat)
            target[f"q{q1}q{q2}"] = inv_coherent_error_2q


def get_noise_model(
    num_qubits: int,
    basis_gates: list[str],
    data: dict[str, Any],
    gate_duration_ns_1q: int = 20,
) -> NoiseModel:
    logger.debug("Building noise model.")
    noise_model = NoiseModel(basis_gates=basis_gates)
    coupling_map = get_coupling_map(num_qubits, data)
    noise_data = generate_noise_data(num_qubits, data)
    # noise contributions
    _add_readout_noise(num_qubits, noise_data, noise_model)
    _add_single_qubit_noise(
        num_qubits,
        noise_data,
        noise_model,
        gate_duration_ns_1q,
    )
    _add_two_qubit_noise(coupling_map, noise_data, noise_model)
    return noise_model


def build_noise_inversion_channels(
    num_qubits: int,
    data: dict[str, Any],
    gate_duration_ns_1q: int = 20,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    logger.debug("Building noise inversion channels.")
    coupling_map = get_coupling_map(num_qubits, data)
    noise_data = generate_noise_data(num_qubits, data)
    # the following line uses nested default dicts, providing an Identity as default
    # if you query unknown keys like dict["a"]["b"], it will return the identity matrix
    noise_inversion_channels_1q: dict[str, dict[str, np.ndarray]] = defaultdict(
        lambda: defaultdict(lambda: np.eye(4))
    )  # avoids ugly if check in _add_coherent_noise
    noise_inversion_channels_2q: dict[str, np.ndarray] = {}
    # error contribution
    _add_readout_noise(num_qubits, noise_data, noise_inversion_channels_1q)
    _add_single_qubit_noise(
        num_qubits,
        noise_data,
        noise_inversion_channels_1q,
        gate_duration_ns_1q,
    )
    _add_two_qubit_noise(
        coupling_map,
        noise_data,
        noise_inversion_channels_2q,
    )

    return noise_inversion_channels_1q, noise_inversion_channels_2q