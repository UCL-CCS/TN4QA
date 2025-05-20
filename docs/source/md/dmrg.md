# Tutorial: DMRG

In this tutorial we go through using TN4QA's DMRG solver. We use the following imports

```
from tn4qa.dmrg import DMRG
from tn4qa.utils import ham_dict_from_scf

import numpy as np
```

## Getting the Hamiltonian

The input to the DMRG solver is a Hamiltonian. This can be a second quantised Hamiltonian for a fermionic system represented by a tuple containing the array of one-electron integrals, the array of two-electron integrals and the nuclear energy contribution. Alternatively this can be a qubit Hamiltonian represented by a dictionary of the form {pauli_string : weight}. Calculating these Hamiltonians can be done using [Nbed](https://github.com/UCL-CCS/Nbed). Some pre-made examples are given in the molecules folder within the top-level directory.

## Running DMRG

Given a Hamiltonian, running DMRG is straightforward. First construct the driver by supplying the hamiltonian and the maximum allowed bond dimension:

```
dmrg = DMRG(ham, 8)
```

The default DMRG method is `two-site` however "one-site" and "subspace-expansion" are also available through the "method" keyword argument. To run DMRG simply use the run method and supply the maximum number of sweeps:

```
dmrg.run(10)
```

The resulting energy is stored in `dmrg.energy` and the corresponding groundstate approximation is `dmrg.mps`.
