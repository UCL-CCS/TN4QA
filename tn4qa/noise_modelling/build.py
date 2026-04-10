"""
Qiskit Noise Model Builder
=====================================
Build a Qiskit AER noise model from standard device noise data:
  - Two-qubit gate depolarising errors
  - Single-qubit gate errors
  - Readout (measurement) errors
  - T1/T2 thermal relaxation (decoherence / dephasing)
"""

from __future__ import annotations

from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    thermal_relaxation_error,
)
from qiskit_aer.noise.errors import QuantumError


class NoiseModelBuilder:
    """
    Construct a Qiskit AER ``NoiseModel`` from a structured noise data dict.

    Noise Data Format
    -----------------
    All keys are optional; omit a section to skip that error source.

    .. code-block:: python

        noise_data = {
            # --- Two-qubit gate errors ---
            "cx_errors": {
                (0, 1): 0.01,   # depolarising error rate for CX on qubits (0,1)
                (1, 2): 0.008,
            },
            "cz_errors": {
                (0, 1): 0.009,
            },
            "ecr_errors": {
                (0, 1): 0.012,
            },

            # --- Single-qubit gate errors ---
            "sx_errors": {
                0: 0.001,
                1: 0.0012,
            },
            "x_errors": {
                0: 0.001,
            },
            "rz_errors": {  # typically zero for virtual Z
                0: 0.0,
            },

            # --- Readout errors (per qubit) ---
            # p0given1 = P(measure 0 | state 1)
            # p1given0 = P(measure 1 | state 0)
            "readout_errors": {
                0: {"p0given1": 0.02, "p1given0": 0.01},
                1: {"p0given1": 0.015, "p1given0": 0.012},
            },

            # --- Thermal relaxation (T1/T2 decoherence) ---
            # t1, t2 in microseconds; gate_time in nanoseconds
            "thermal_relaxation": {
                0: {"t1": 150.0, "t2": 80.0},
                1: {"t1": 120.0, "t2": 60.0},
            },
            "gate_times": {  # nanoseconds
                "cx":  300,
                "cz":  300,
                "ecr": 280,
                "sx":  35,
                "x":   35,
                "measure": 4000,
            },
        }

    Parameters
    ----------
    noise_data : dict
        Noise data in the format described above.
    two_qubit_gate : str
        Native two-qubit gate name (``"cx"``, ``"cz"``, ``"ecr"``).
        Determines which error dict from ``noise_data`` is treated as the
        primary two-qubit gate.
    """

    DEFAULT_GATE_TIMES = {
        "cx": 300,  # ns
        "cz": 300,
        "ecr": 280,
        "sx": 35,
        "x": 35,
        "rz": 0,
        "measure": 4000,
        "reset": 500,
    }

    def __init__(
        self,
        noise_data: dict,
        two_qubit_gate: str = "cx",
        oneq_gate_errors: bool = True,
        twoq_gate_errors: bool = True,
        decoherence: bool = True,
        readout_errors: bool = True,
    ) -> None:
        self.noise_data = noise_data
        self.two_qubit_gate = two_qubit_gate.lower()
        self._model: NoiseModel | None = None
        self.oneq = oneq_gate_errors
        self.twoq = twoq_gate_errors
        self.decoherence = decoherence
        self.readout = readout_errors

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> NoiseModel:
        """Construct and return the ``NoiseModel``."""
        model = NoiseModel()
        nd = self.noise_data

        gate_times = {**self.DEFAULT_GATE_TIMES, **nd.get("gate_times", {})}
        thermal_data = nd.get("thermal_relaxation", {})
        if not self.decoherence:
            thermal_data = {}

        # --- Two-qubit gate errors ---
        for gate in ("cx", "cz", "ecr"):
            key = f"{gate}_errors"
            if key in nd:
                for qpair, rate in nd[key].items():
                    if self.twoq:
                        err = self._two_qubit_error(
                            rate=rate,
                            qubits=list(qpair),
                            gate=gate,
                            gate_times=gate_times,
                            thermal_data=thermal_data,
                        )
                        if err is not None:
                            model.add_quantum_error(err, gate, list(qpair))

        # --- Single-qubit gate errors ---
        for gate in ("sx", "x", "y", "z", "rx", "ry", "rz", "h", "u", "u1", "u2", "u3"):
            key = f"{gate}_errors"
            if key in nd:
                for qubit, rate in nd[key].items():
                    if self.oneq:
                        err = self._single_qubit_error(
                            rate=rate,
                            qubit=qubit,
                            gate=gate,
                            gate_times=gate_times,
                            thermal_data=thermal_data,
                        )
                        if err is not None:
                            model.add_quantum_error(err, gate, [qubit])

        # --- Readout errors ---
        if "readout_errors" in nd and self.readout:
            for qubit, rerr in nd["readout_errors"].items():
                p0g1 = rerr.get("p0given1", 0.0)
                p1g0 = rerr.get("p1given0", 0.0)
                # ReadoutError([[P(0|0), P(1|0)], [P(0|1), P(1|1)]])
                ro_err = ReadoutError([[1.0 - p1g0, p1g0], [p0g1, 1.0 - p0g1]])
                model.add_readout_error(ro_err, [qubit])

        self._model = model
        return model

    @property
    def noise_model(self) -> NoiseModel:
        """Return cached model (builds if not yet built)."""
        if self._model is None:
            self.build()
        return self._model

    # ------------------------------------------------------------------
    # Error construction helpers
    # ------------------------------------------------------------------

    def _single_qubit_error(
        self,
        rate: float,
        qubit: int,
        gate: str,
        gate_times: dict,
        thermal_data: dict,
    ) -> QuantumError | None:
        """Depolarising + optional thermal relaxation for a 1Q gate."""
        errors = []

        if rate > 0:
            errors.append(depolarizing_error(rate, 1))

        if qubit in thermal_data:
            t1 = thermal_data[qubit]["t1"] * 1e3  # µs → ns
            t2 = thermal_data[qubit]["t2"] * 1e3
            gt = gate_times.get(gate, 35)
            if gt > 0:
                t2 = min(t2, 2 * t1)  # enforce physical constraint T2 ≤ 2T1
                errors.append(thermal_relaxation_error(t1, t2, gt))

        if not errors:
            return None
        result = errors[0]
        for e in errors[1:]:
            result = result.compose(e)
        return result

    def _two_qubit_error(
        self,
        rate: float,
        qubits: list[int],
        gate: str,
        gate_times: dict,
        thermal_data: dict,
    ) -> QuantumError:
        """Depolarising + optional thermal relaxation for a 2Q gate."""
        errors = []

        if rate > 0:
            errors.append(depolarizing_error(rate, 2))

        gt = gate_times.get(gate, 300)
        for qubit in qubits:
            if qubit in thermal_data and gt > 0:
                t1 = thermal_data[qubit]["t1"] * 1e3
                t2 = thermal_data[qubit]["t2"] * 1e3
                t2 = min(t2, 2 * t1)
                errors.append(
                    thermal_relaxation_error(t1, t2, gt).expand(
                        depolarizing_error(0, 1)  # tensor with identity
                    )
                    if qubit == qubits[0]
                    else thermal_relaxation_error(t1, t2, gt).expand(
                        depolarizing_error(0, 1)
                    )
                )

        if not errors:
            # Fallback: identity-like error with zero rate
            return depolarizing_error(0, 2)

        result = errors[0]
        for e in errors[1:]:
            result = result.compose(e)
        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of all noise sources."""
        nd = self.noise_data
        lines = ["Noise Model Summary", "=" * 40]

        for gate_key in ("cx_errors", "cz_errors", "ecr_errors"):
            if gate_key in nd:
                gate = gate_key.replace("_errors", "").upper()
                lines.append(f"\n{gate} errors:")
                for qpair, rate in nd[gate_key].items():
                    lines.append(f"  qubits {qpair}: ε = {rate:.4g}")

        for gate_key in ("sx_errors", "x_errors", "rz_errors"):
            if gate_key in nd:
                gate = gate_key.replace("_errors", "").upper()
                lines.append(f"\n{gate} errors:")
                for q, rate in nd[gate_key].items():
                    lines.append(f"  qubit {q}: ε = {rate:.4g}")

        if "readout_errors" in nd:
            lines.append("\nReadout errors:")
            for q, e in nd["readout_errors"].items():
                lines.append(
                    f"  qubit {q}: P(0|1)={e['p0given1']:.4g}, "
                    f"P(1|0)={e['p1given0']:.4g}"
                )

        if "thermal_relaxation" in nd:
            lines.append("\nThermal relaxation:")
            for q, v in nd["thermal_relaxation"].items():
                lines.append(f"  qubit {q}: T1={v['t1']:.1f} µs, T2={v['t2']:.1f} µs")

        return "\n".join(lines)
