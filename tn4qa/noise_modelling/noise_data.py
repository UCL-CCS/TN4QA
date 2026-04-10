def extract_noise_data(backend):
    props = backend.properties()
    config = backend.configuration()

    noise_data = {
        "cx_errors": {},
        "cz_errors": {},
        "ecr_errors": {},
        "sx_errors": {},
        "x_errors": {},
        "rz_errors": {},
        "readout_errors": {},
        "thermal_relaxation": {},
        "gate_times": {},
    }

    for gate in props.gates:
        name = gate.gate
        qubits = tuple(gate.qubits)

        error = None
        duration = None

        for param in gate.parameters:
            if param.name == "gate_error":
                error = param.value
            elif param.name == "gate_length":
                duration = param.value  # seconds

        # Store errors
        if name == "cx" and error is not None:
            noise_data["cx_errors"][qubits] = error
        elif name == "cz" and error is not None:
            noise_data["cz_errors"][qubits] = error
        elif name == "ecr" and error is not None:
            noise_data["ecr_errors"][qubits] = error
        elif name == "sx" and error is not None:
            noise_data["sx_errors"][qubits[0]] = error
        elif name == "x" and error is not None:
            noise_data["x_errors"][qubits[0]] = error
        elif name == "rz":
            # virtual gate
            noise_data["rz_errors"][qubits[0]] = 0.0

        # Store representative gate times (convert to ns)
        if duration is not None:
            noise_data["gate_times"].setdefault(name, duration)

    # -------------------------
    # Readout errors
    # -------------------------
    for q in range(config.n_qubits):
        readout = props.readout_error(q)

        # Qiskit gives symmetric readout error by default,
        # but we can reconstruct asymmetric probabilities
        prob_meas0_prep1 = None
        prob_meas1_prep0 = None

        for param in props.qubits[q]:
            if param.name == "prob_meas0_prep1":
                prob_meas0_prep1 = param.value
            elif param.name == "prob_meas1_prep0":
                prob_meas1_prep0 = param.value

        noise_data["readout_errors"][q] = {
            "p0given1": prob_meas0_prep1 if prob_meas0_prep1 is not None else readout,
            "p1given0": prob_meas1_prep0 if prob_meas1_prep0 is not None else readout,
        }

    # -------------------------
    # Thermal relaxation (T1, T2)
    # -------------------------
    for q in range(config.n_qubits):
        t1 = None
        t2 = None

        for param in props.qubits[q]:
            if param.name == "T1":
                t1 = param.value
            elif param.name == "T2":
                t2 = param.value

        # convert to microseconds
        noise_data["thermal_relaxation"][q] = {
            "t1": t1 if t1 else None,
            "t2": t2 if t2 else None,
        }

    return noise_data
