from tn4qa.tensor import Tensor 
import numpy as np 
import sparse

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

def test_from_array():
    return 

def test_from_qiskit_gate():
    return 

def test_rank_3_copy():
    return 

def test_rank_4_copy():
    return 

def test_rank_3_copy_open():
    return 

def test_rank_4_copy_open():
    return 

def test_rank_3_qiskit_gate():
    return

def test_rank_4_qiskit_gate():
    return 

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
