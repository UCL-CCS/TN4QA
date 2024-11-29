from tn4qa.tensor import Tensor 
# Underlying tensor objects can either be NumPy arrays or Sparse arrays
import numpy as np
from numpy import ndarray
import sparse
from sparse import SparseArray

# Qiskit Imports
from qiskit.circuit.library import XGate, HGate, CXGate
from qiskit.quantum_info import Operator

TEST_ARRAY = np.array([[[[0,1,0],[1,0,0]],[[0,1,0],[1,0,0]]],[[[0,1,0],[1,0,0]],[[0,1,0],[1,0,0]]]], dtype=complex)

def test_constructor_sparse():
    sparse_array = sparse.COO.from_numpy(TEST_ARRAY)
    tensor = Tensor(sparse_array, ["INDEX_1", "INDEX_2", "INDEX_3", "INDEX_4"], ["LABEL_1"])

    assert np.allclose(TEST_ARRAY, tensor.data.todense())
    assert tensor.indices[0] == "INDEX_1" and tensor.indices[1] == "INDEX_2" and tensor.indices[2] == "INDEX_3" and tensor.indices[3] == "INDEX_4"
    assert tensor.labels[0] == "LABEL_1"
    assert tensor.rank == 4
    assert tensor.dimensions == (2,2,2,3)

    return 

def test_constructor_numpy():
    x_mat = TEST_ARRAY
    tensor = Tensor(x_mat, ["INDEX_1", "INDEX_2", "INDEX_3", "INDEX_4"], ["LABEL_1"])

    assert np.allclose(TEST_ARRAY, tensor.data.todense())
    assert tensor.indices[0] == "INDEX_1" and tensor.indices[1] == "INDEX_2" and tensor.indices[2] == "INDEX_3" and tensor.indices[3] == "INDEX_4"
    assert tensor.labels[0] == "LABEL_1"
    assert tensor.rank == 4
    assert tensor.dimensions == (2,2,2,3)

    return

def test_from_array_numpy_3D():
    # High-dimensional NumPy array
    numpy_array_3d = np.random.rand(2, 3, 4)
    tensor_3d = Tensor.from_array(numpy_array_3d, indices=["H1", "H2", "H3"], labels=["L1", "L2", "L3"])
    assert isinstance(tensor_3d.data.todense(), ndarray), "Data should be a NumPy ndarray"
    assert tensor_3d.indices == ["H1", "H2", "H3"], "Indices should match the array dimensions"
    assert tensor_3d.labels == ["L1", "L2", "L3"], "Labels should match the given labels"

def test_from_array_numpy_empty():
    # Empty NumPy array
    empty_numpy_array = np.array([])
    tensor_empty_numpy = Tensor.from_array(empty_numpy_array)
    print(tensor_empty_numpy.indices)
    assert tensor_empty_numpy.indices == [], "Indices should be empty for empty array"
    assert tensor_empty_numpy.labels == ["T1"], "Labels should default to ['T1']"

def test_from_array_sparse_3D():
    # High-dimensional Sparse array
    sparse_array_3d = sparse.COO(np.random.randint(0, 2, size=(2, 3, 4)))
    tensor_sparse_3d = Tensor.from_array(sparse_array_3d, indices=["S1", "S2", "S3"], labels=["L1", "L2", "L3"])
    assert isinstance(tensor_sparse_3d.data, SparseArray), "Data should be a SparseArray"
    assert tensor_sparse_3d.indices == ["S1", "S2", "S3"], "Indices should match the array dimensions"
    assert tensor_sparse_3d.labels == ["L1", "L2", "L3"], "Labels should match the given labels"

def test_from_array_sparse_empty():
    # Empty Sparse array
    empty_sparse_array = sparse.COO(np.array([]))
    tensor_empty_sparse = Tensor.from_array(empty_sparse_array)
    assert tensor_empty_sparse.indices == [], "Indices should be empty for empty Sparse array"
    assert tensor_empty_sparse.labels == ["T1"], "Labels should default to ['T1']"

def test_from_array_invalid():
    # Invalid input
    try:
        invalid_input = [1, 2, 3]
        Tensor.from_array(invalid_input)
        assert False, "An error should have been raised for invalid input"
    except AttributeError:
        pass  # Expected for invalid input without ".shape"

def test_from_array_large():
    # Large arrays
    large_array = np.random.rand(1000, 1000)
    tensor_large = Tensor.from_array(large_array)
    assert len(tensor_large.indices) == 2, "Indices should match the number of dimensions"
    assert tensor_large.data.shape == (1000, 1000), "Data shape should match input array"

def test_from_qiskit_gate_xgate():
    # Test single-qubit gate (XGate)
    x_gate = XGate()
    tensor_x = Tensor.from_qiskit_gate(x_gate)
    assert tensor_x.indices == ["O1", "I1"], "Indices for XGate should be ['O1', 'I1']"
    assert tensor_x.labels == ["T1", "x"], "Labels for XGate should be ['T1', 'x']"
    assert tensor_x.data.shape == (2, 2), "Data shape for XGate should be (2, 2)"

def test_from_qiskit_gate_hgate():
    # Test single-qubit gate (HGate)
    h_gate = HGate()
    tensor_h = Tensor.from_qiskit_gate(h_gate, labels=["Label"])
    assert tensor_h.indices == ["O1", "I1"], "Indices for HGate should be ['O1', 'I1']"
    assert tensor_h.labels == ["Label", "h"], "Labels for HGate should be ['Label', 'h']"
    assert tensor_h.data.shape == (2, 2), "Data shape for HGate should be (2, 2)"

def test_from_qiskit_gate_cxgate():
    # Test two-qubit gate (CXGate)
    cx_gate = CXGate()
    tensor_cx = Tensor.from_qiskit_gate(cx_gate)
    assert tensor_cx.indices == ["O1", "O2", "I1", "I2"], "Indices for CXGate should be ['O1', 'O2', 'I1', 'I2']"
    assert tensor_cx.labels == ["T1", "cx"], "Labels for CXGate should be ['T1', 'cx']"
    assert tensor_cx.data.shape == (2, 2, 2, 2), "Data shape for CXGate should be (2, 2, 2, 2)"

def test_from_qiskit_gate_custom():
    # Test custom indices
    h_gate = HGate()
    tensor_custom_indices = Tensor.from_qiskit_gate(h_gate, indices=["I1", "O1"], labels=["Custom"])
    assert tensor_custom_indices.indices == ["I1", "O1"], "Custom indices should be used"
    assert tensor_custom_indices.labels == ["Custom", "h"], "Labels should include custom label and gate name"

def test_rank_3_copy():
  tensor = Tensor.rank_3_copy()
        
  # Expected data
  expected_data = np.array(
      [[[1, 0], [0, 1]], [[0, 0], [0, 1j * np.sqrt(2)]]],
      dtype=complex
  )
  expected_shape = (2, 2, 2)
  expected_rank = 3
  expected_indices = ["B1", "R1", "L1"]
  expected_labels = ["T1", "copy3"]

  # Test properties
  assert (tensor.data.todense() == expected_data).all(),  "Data does not match expected"
  assert tensor.dimensions == expected_shape,  " Invalid shape"
  assert tensor.rank == expected_rank,  " Validate the rank"
  assert tensor.indices == expected_indices,  "Indices do not match expected"
  assert tensor.labels == expected_labels,  "Labels do not match expected"

def test_rank_4_copy():
  tensor = Tensor.rank_4_copy()
        
  # Expected data
  expected_data = np.array(
      [[[[1, 0], [0, 1]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[0, 0], [0, 1]]]],
      dtype=complex
  )
  expected_shape = (2, 2, 2, 2)
  expected_rank = 4
  expected_indices = ["B1", "B2", "R1", "L1"]
  expected_labels = ["T1", "copy4"]

  # Test properties
  assert (tensor.data.todense() == expected_data).all(),  "Data does not match expected"
  assert tensor.dimensions == expected_shape,  " Invalid shape"
  assert tensor.rank == expected_rank,  " Incorrect rank"
  assert tensor.indices == expected_indices,  "Indices do not match expected"
  assert tensor.labels == expected_labels,  "Labels do not match expected"

def test_rank_3_copy_open():
  tensor = Tensor.rank_3_copy_open()
        
  # Expected data
  expected_data = np.array([[[1,0],[0,1]], [[1j*np.sqrt(2),0],[0,0]]], dtype=complex)
  expected_shape = (2, 2, 2)
  expected_rank = 3
  expected_indices = ["B1", "R1", "L1"]
  expected_labels = ["T1", "copy3open"]

  # Test properties
  assert (tensor.data.todense() == expected_data).all(),  "Data does not match expected"
  assert tensor.dimensions == expected_shape,  " Invalid shape"
  assert tensor.rank == expected_rank,  " Incorrect rank"
  assert tensor.indices == expected_indices,  "Indices do not match expected"
  assert tensor.labels == expected_labels,  " Labels do not match expected"

def test_rank_4_copy_open():
  tensor = Tensor.rank_4_copy_open()
        
  # Expected data
  expected_data = np.array([[[[1,0],[0,1]], [[0,0],[0,0]]], [[[0,0],[0,0]], [[1,0],[0,0]]]], dtype=complex)
  expected_shape = (2, 2, 2, 2)
  expected_rank = 4
  expected_indices = ["B1", "B2", "R1", "L1"]
  expected_labels = ["T1", "copy4open"]

  # Test properties
  assert (tensor.data.todense() == expected_data).all(),  "Data does not match expected"
  assert tensor.dimensions == expected_shape,  " Invalid shape"
  assert tensor.rank == expected_rank,  " Incorrect rank"
  assert tensor.indices == expected_indices,  "Indices do not match expected"
  assert tensor.labels == expected_labels,  " Labels do not match expected"

def test_rank_3_qiskit_gate():
    h_gate = HGate()
    tensor = Tensor.rank_3_qiskit_gate(h_gate)

    # Expected data
    h_matrix = Operator(h_gate).data
    id_matrix = np.eye(2, dtype=complex)
    expected_data = np.array(
        [id_matrix, (1j / np.sqrt(2)) * (id_matrix - h_matrix)]
    ).reshape(2, 2, 2)

    # Test properties
    assert np.allclose(tensor.data.todense(), expected_data), "Data does not match expected"
    assert tensor.indices == ["B1", "R1", "L1"], "Indices do not match expected"
    assert tensor.labels == ["T1", "rank3h"], "Labels do not match expected"

def test_rank_4_qiskit_gate():
    h_gate = HGate()
    tensor = Tensor.rank_4_qiskit_gate(h_gate)

    # Expected data
    h_matrix = Operator(h_gate).data
    id_array = np.array([[1,0],[0,1]], dtype=complex).reshape(2,2)
    zero_array = np.array([[0,0],[0,0]], dtype=complex).reshape(2,2)
    expected_data = np.array(
        [[id_array, zero_array], [zero_array, -0.5*(h_matrix-id_array)]]
    ).reshape(2, 2, 2, 2)


    # Test properties
    assert np.allclose(tensor.data.todense(), expected_data), "Data does not match expected"
    assert tensor.indices == ["B1", "B2", "R1", "L1"], "Indices do not match expected"
    assert tensor.labels == ["T1", "rank4h"], "Labels do not match expected"
    assert tensor.dimensions == (2, 2, 2, 2), "Dimensions do not match expected"

def test_reorder_indices():
    tensor = Tensor(TEST_ARRAY, ["I1", "I2", "I3", "I4"], ["Label1"])
    tensor.reorder_indices(["I3", "I4", "I1", "I2"])
    
    # Check indices
    assert tensor.indices == ["I3", "I4", "I1", "I2"], "Indices not reordered correctly"
    
    # Check data shape
    expected_shape = (TEST_ARRAY.shape[2], TEST_ARRAY.shape[3], TEST_ARRAY.shape[0], TEST_ARRAY.shape[1])
    assert tensor.dimensions == expected_shape, "Shape not updated correctly after reordering"
    
    # Check data manipulation
    expected_data = sparse.moveaxis(TEST_ARRAY, [0, 1, 2, 3], [2, 3, 0, 1])
    assert np.allclose(tensor.data.todense(), expected_data), "Data not moved correctly (sparse array)"

def test_new_index_name():
    tensor = Tensor(TEST_ARRAY, ["B1", "B2", "B3"], ["Label1"])
    new_index = tensor.new_index_name("B", 1)

    assert new_index == "B4"
    
def test_get_dimension_of_index():
    tensor = Tensor(TEST_ARRAY, ["I1", "I2", "I3", "I4"], ["Label1"])
    dim = tensor.get_dimension_of_index("I2")

    assert dim == TEST_ARRAY.shape[1]

def test_get_total_dimension_of_indices():
    tensor = Tensor(TEST_ARRAY, ["I1", "I2", "I3", "I4"], ["Label1"])
    total_dim = tensor.get_total_dimension_of_indices(["I1", "I2"])

    assert total_dim == TEST_ARRAY.shape[0] * TEST_ARRAY.shape[1]
    
def test_combine_indices():
    tensor = Tensor(TEST_ARRAY, ["I1", "I2", "I3", "I4"], ["Label1"])
    tensor.combine_indices(["I1", "I2"], "Combined")
    
    # Check indices
    assert "Combined" in tensor.indices, "New combined index not found in indices"
    assert len(tensor.indices) == 3, "Incorrect number of indices after combining"

    # Check shape
    combined_dim = TEST_ARRAY.shape[0] * TEST_ARRAY.shape[1]
    expected_shape = (combined_dim,) + TEST_ARRAY.shape[2:]
    assert tensor.dimensions == expected_shape, "Shape not updated correctly after combining indices"
    
    # Check data manipulation
    reshaped_data = TEST_ARRAY.reshape(expected_shape)
    assert (tensor.data == reshaped_data).all(), "Data not reshaped correctly"

def test_tensor_to_matrix():
    tensor = Tensor(TEST_ARRAY, ["I1", "I2", "I3", "I4"], ["Label1"])
    tensor.tensor_to_matrix(["I1", "I2"], ["I3", "I4"])
    
    # Check rank
    assert tensor.rank == 2, "Tensor rank not reduced to 2 for matrix"

    # Check indices
    assert tensor.indices == ["O1", "I1"], "Matrix indices not set correctly"

    # Check shape
    input_dim = TEST_ARRAY.shape[0] * TEST_ARRAY.shape[1] 
    output_dim = TEST_ARRAY.shape[2] * TEST_ARRAY.shape[3] 
    expected_shape = (output_dim, input_dim)
    assert tensor.dimensions == expected_shape, "Shape not updated correctly for matrix"

    # Check data manipulation
    reshaped_data = np.moveaxis(TEST_ARRAY, [0, 1], [2, 3])
    reshaped_data = reshaped_data.reshape(expected_shape)
    tensor_data = tensor.data.todense() 
    assert np.allclose(tensor_data, reshaped_data), "Data not reshaped correctly"

def test_multiply_by_constant():
    tensor = Tensor(TEST_ARRAY, ["I1", "I2", "I3", "I4"], ["Label1"])
    tensor.multiply_by_constant(2)

    assert np.allclose(tensor.data.todense(), TEST_ARRAY * 2)
