from dataclasses import dataclass

@dataclass
class QubitNoise:
    id: int
    t1_ns: float
    t2_ns: float
    readout_p0: float
    readout_p1: float
    gates_1q: dict[str, float]
    gates_2q: dict[int, float]

    def __init__(
        self, 
        id: int=0, 
        t1_ns: int=0, 
        t2_ns: int=0, 
        readout_p0: int=0, 
        readout_p1: int=0, 
        gates_1q: dict[str, float]={}, 
        gates_2q: dict[int, float]={}
    ):
        """
        Constructor for QubitNoise class.
        
        Args:
            id: Qubit's index.
            t1_ns: Time T1 in nanoseconds.
            t2_ns: Time T2 in nanoseconds.
            readout_p0: Readout probability p0.
            readout_p1: Readout probability p1.
            gates_1q: Dict containing the one-qubit gates' fidelities.
            gates_2q: Dict containing the two-qubit gates' fidelities (for each coupled qubit).
        """
        self.id = id
        self.t1_ns = t1_ns
        self.t2_ns = t2_ns
        self.readout_p0 = readout_p0
        self.readout_p1 = readout_p1
        self.gates_1q = gates_1q
        self.gates_2q = gates_2q


@dataclass
class NoiseData:
    n_qubits: int
    coupling_map: list[list[int]]
    qubit_noise: list[QubitNoise]

    def __init__(
        self, 
        n_qubits: int=0, 
        coupling_map: list[list[int]]=[], 
        qubit_noise: list[QubitNoise]=[]
    ):
        """
        Constructor for NoiseData class.

            Args:
            n_qubits: Number of qubits of the device.
            coupling_map: Map of coupled qubits on the device.
            qubit_noise: List of QubitNoise instances.
        """
        self.n_qubits = n_qubits
        self.coupling_map = coupling_map
        self.qubit_noise = qubit_noise