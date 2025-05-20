# Tutorial: Basic Usage

In this tutorial we go through some example usage of the base classes offered by TN4QA. These are:

* Tensor
* TensorNetwork
* MatrixProductState
* MatrixProductOperator

To get started you'll need these imports:

```
from tn4qa.tensor import Tensor
from tn4qa.tn import TensorNetwork
from tn4qa.mps import MatrixProductState
from tn4qa.mpo import MatrixProductOperator

import numpy as np
from qiskit import QuantumCircuit
from copy import deepcopy
```

## Tensor

Tensor is the base class to store and manipulate tensors within tensor networks. Tensors can be built directly from the constructor by supplying the underlying data, a list of index labels and a list of tensor labels:

```
data = np.array(range(12)).reshape(2,3,2)
t = Tensor(data, ["a", "b", "c"], ["MyFirstTensor"])
```

Alternatively there are several pre-built tensors that can be initialised:

```
t2 = Tensor.rank_3_copy(indices=["a","d","e"])
```

Some of the most useful methods of the tensor class are `combine_indices` and `tensor_to_matrix`. Details for these methods can be found in the class API.

## TensorNetwork

TensorNetwork is the class to store and manipulate tensor networks. They can be initialised directly through the constructor by supplying a list of tensors:

```
tn = TensorNetwork([t,t2])
```

Alternatively they can be constructed directly from a qiskit circuit with the `from_qiskit_circuit` method. Some of the most useful methods in this class are `contract_index`, `contract_entire_network`, `compress_index`, `compress`, and `svd`. Details for these methods can be found in the class API.

## MatrixProductState

MatrixProductState is the class to store and mainpulate matrix product states which are representations of quantum states. They can be initialised directly using the constructor but often it is most useful to use one of the many class methods:

```
qc = QuantumCircuit(4)
qc.h(0)
qc.cx(0,1)
qc.cx(1,2)
qc.cx(2,3)
mps = MatrixProductState.from_qiskit_circuit(qc, 16)
```

A MatrixProductState can be visualised with the `draw` method. The underlying quantum state can be recovered with the `to_dense_array` method.

MatrixProductStates can be added/subtracted, and multiplication by a constant is achieved with the `multiply_by_constant` method.

A MatrixProductState can be normalised with the `normalise` method.

Inner products can be calculated with the `compute_inner_product` method and expectation values (for a given MatrixProductOperator) can be calculated with the `compute_expectation_value` method.

Details for these methods can be found in the class API.

## MatrixProductOperator

MatrixProductOperator is the class to store and manipulate matrix product operators which represent operators. They can be initialised directly using the constructor but often it is most useful to use one of the many class methods:

```
ham_dict = {"XX" : 0.1, "II" : -0.1}
mpo = MatrixProductOperator.from_hamiltonian(ham_dict, 16)
```

A MatrixProductOperators can be visualised with the `draw` method. The underlying operator can be recovered with the `to_dense_array` method.

MatrixProductOperators can be added/subtracted or multiplied together. Multiplication by a constant is achieved with the `multiply_by_constant` method.
