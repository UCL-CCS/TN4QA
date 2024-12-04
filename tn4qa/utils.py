from dataclasses import dataclass
from enum import Enum

class PauliTerm(Enum):
    I = "I"
    X = "X"
    Y = "Y"
    Z = "Z"

@dataclass(frozen=True)
class UpdateValues:
    indices: tuple[int, int, int, int]
    weights: tuple[complex, complex]

def _update_array(
    array: list,
    data: list,
    weight: complex,
    p_string_idx: int,
    term: str,
    offset: bool = False,
) -> None:
    match term:
        case PauliTerm.I.value:
            update_values = UpdateValues((0, 0, 1, 1), (1, 1))
        case PauliTerm.X.value:
            update_values = UpdateValues((0, 1, 1, 0), (1, 1))
        case PauliTerm.Y.value:
            update_values = UpdateValues((0, 1, 1, 0), (-1j, 1j))
        case PauliTerm.Z.value:
            update_values = UpdateValues((0, 0, 1, 1), (1, -1))

    for i in [0, 1]:
        array[0].append(p_string_idx)
        if offset:
            array[1].append(p_string_idx)

        array[1 + int(offset)].append(update_values.indices[2 * i])
        array[2 + int(offset)].append(update_values.indices[(2 * i) + 1])
        data.append(update_values.weights[i] * weight)
        