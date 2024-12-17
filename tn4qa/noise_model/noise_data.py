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
        self.n_qubits = n_qubits
        self.coupling_map = coupling_map
        self.qubit_noise = qubit_noise