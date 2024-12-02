from tn4qa.tensor import Tensor
from tn4qa.tn import TensorNetwork

import numpy as np
from qiskit import QuantumCircuit

np.random.seed(1001)

TEST_ARRAY_1 = np.array([[0,1],[1,0]], dtype=complex)
TEST_ARRAY_2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
TEST_ARRAY_3 = np.asmatrix(TEST_ARRAY_1) @ np.asmatrix(TEST_ARRAY_2)
TEST_ARRAY_4 = TEST_ARRAY_1.reshape((2,2,1))

def test_constructor():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], ["T1"])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], ["T2"])

    tn = TensorNetwork([t1,t2], "TEST_TN")

    # Test all properties have been assigned correctly
    assert tn.name == "TEST_TN", "TN name does not match input."
    assert len(tn.tensors) == 2, "The number of tensors in the TN should match the number of input arrays."
    assert np.allclose(tn.tensors[0].data.todense(), TEST_ARRAY_1) and np.allclose(tn.tensors[1].data.todense(), TEST_ARRAY_2), "The tensors in the TN should appear in input order."
    assert len(tn.indices) == 3, "The total number of indices should match the input arrays."
    assert "A" in tn.indices and "B" in tn.indices and "C" in tn.indices, "The TN indices should match the input values."

    return 

def test_from_qiskit_layer():
    num_qubits = 3
    layer_number = 3
    qc = QuantumCircuit(num_qubits)
    qc.x(0) 
    qc.y(1)
    tn = TensorNetwork.from_qiskit_layer(qc, layer_number)

    assert len(tn.tensors) == num_qubits, "The number of tensors should match the number of qubits."
    assert len(tn.indices) == 2*num_qubits, "Incorrect number of indices in the TN."
    assert f"QW0N{layer_number-1}" in tn.indices and f"QW2N{layer_number}" in tn.indices, "Incorrect index names in the TN."
    assert np.allclose(tn.tensors[0].data.todense(), TEST_ARRAY_1) and np.allclose(tn.tensors[1].data.todense(), TEST_ARRAY_2), "Incorrect tensor data in the TN."
    assert np.allclose(tn.tensors[2].data.todense(), [[1,0],[0,1]]), "Incorrect handling of idle qubits."

    return 

def test_from_qiskit_circuit():
    num_qubits = 3
    num_layers = 2
    qc = QuantumCircuit(num_qubits)
    qc.x(0)
    qc.y(1)
    qc.cx(1,2)
    tn = TensorNetwork.from_qiskit_circuit(qc) 

    assert len(tn.tensors) == 5, "The number of tensors should match the number of qubits."
    assert len(tn.indices) == (num_layers+1) * num_qubits, "Incorrect number of indices in the TN."
    assert np.allclose(tn.tensors[3].data.todense().reshape(4,4), [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]), "Incorrect tensor data in the TN."
    assert "L2" in tn.tensors[3].labels, "Incorrect labels in the TN."
    assert "Q1" in tn.tensors[3].labels and "Q2" in tn.tensors[3].labels, "Incorrect labels in the TN."
    assert "QW1N1" in tn.tensors[3].indices and "QW2N2" in tn.tensors[3].indices, "Incorrect indices in the TN."

    return 

def test_get_index_to_tensor_dict():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], ["T1"])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], ["T2"])

    tn = TensorNetwork([t1,t2], "TEST_TN")
    tn_dict = tn.get_index_to_tensor_dict()

    assert len(list(tn_dict.keys())) == 3, "There should be three indices in the dict."
    assert len(tn_dict["A"]) == 1 and len(tn_dict["B"]) == 2 and len(tn_dict["C"]) == 1, "Indices do not map to the right number of tensors."
    assert tn_dict["A"][0].labels[0] == "T1" and tn_dict["C"][0].labels[0] == "T2", "Indices do not map to the correct tensors."

    return 

def test_get_label_to_tensor_dict():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], ["TEST", "T1"])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], ["TEST", "T2"])

    tn = TensorNetwork([t1,t2], "TEST_TN")
    tn_dict = tn.get_label_to_tensor_dict()

    assert len(list(tn_dict.keys())) == 5, "There should be five labels in the dict."
    assert len(tn_dict["TN_T1"]) == 1 and len(tn_dict["TN_T2"]) == 1 and len(tn_dict["TEST"]) == 2, "Labels do not map to the right number of tensors."
    assert tn_dict["T1"][0].indices[0] == "A" and tn_dict["TEST"][1].indices[0] == "B", "Labels do not map to the correct tensors."

    return 

def test_get_dimension_of_index():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], ["T1"])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], ["T2"])

    tn = TensorNetwork([t1,t2], "TEST_TN")
    dim_A = tn.get_dimension_of_index("A")
    dim_B = tn.get_dimension_of_index("B")
    dim_C = tn.get_dimension_of_index("C")

    assert dim_A == 2, "Incorrect dimension for index A."
    assert dim_B == 2, "Incorrect dimension for index B."
    assert dim_C == 2, "Incorrect dimension for index C."

    return 

def test_get_internal_indices():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], [])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], [])
    t3 = Tensor(TEST_ARRAY_1, ["C", "D"], [])
    t4 = Tensor(TEST_ARRAY_2, ["D", "E"], [])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    internal_indices = tn.get_internal_indices()

    assert "B" in internal_indices and "C" in internal_indices and "D" in internal_indices, "Did not find all internal indices."
    assert "A" not in internal_indices and "E" not in internal_indices, "Found the wrong internal indices."

    return 

def test_get_external_indices():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], [])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], [])
    t3 = Tensor(TEST_ARRAY_1, ["C", "D"], [])
    t4 = Tensor(TEST_ARRAY_2, ["D", "E"], [])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    external_indices = tn.get_external_indices()

    assert "B" not in external_indices and "C" not in external_indices and "D" not in external_indices, "Found the wrong external indices."
    assert "A" in external_indices and "E" in external_indices, "Did not find all external indices."

    return 

def test_get_all_indices():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], [])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], [])
    t3 = Tensor(TEST_ARRAY_1, ["C", "D"], [])
    t4 = Tensor(TEST_ARRAY_2, ["D", "E"], [])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    all_indices = tn.get_all_indices()

    assert "B" in all_indices and "C" in all_indices and "D" in all_indices, "Did not find all indices."
    assert "A" in all_indices and "E" in all_indices, "Did not find all indices."

    return 

def test_get_all_labels():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], ["TEST1"])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], [])
    t3 = Tensor(TEST_ARRAY_1, ["C", "D"], ["TEST2"])
    t4 = Tensor(TEST_ARRAY_2, ["D", "E"], [])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    all_labels = tn.get_all_labels()

    assert len(all_labels) == 6, "Did not find all labels."
    assert "TN_T1" in all_labels and "TN_T4" in all_labels, "Did not find all labels."
    assert "TEST1" in all_labels and "TEST2" in all_labels, "Did not find all labels."

    return 

def test_get_new_label():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], ["T2"])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], ["T8"])
    t3 = Tensor(TEST_ARRAY_1, ["C", "D"], ["T101"])
    t4 = Tensor(TEST_ARRAY_2, ["D", "E"], ["F100001"])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    new_label_1 = tn.get_new_label("T")
    new_label_2 = tn.get_new_label()

    assert new_label_1 == "T102", "New custom label generated incorrectly."
    assert new_label_2 == "TN_T5", "New default label generated incorrectly."

    return 

def test_get_tensors_from_index_name():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], [])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], [])
    t3 = Tensor(TEST_ARRAY_1, ["C", "D"], [])
    t4 = Tensor(TEST_ARRAY_2, ["D", "E"], [])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    tensors = tn.get_tensors_from_index_name("B")

    assert len(tensors) == 2, "Did not find all tensors."
    assert np.allclose(tensors[0].data.todense(), TEST_ARRAY_1), "Did not find the correct first tensor."
    assert np.allclose(tensors[1].data.todense(), TEST_ARRAY_2), "Did not find the correct second tensor."
    assert "TN_T1" in tensors[0].labels, "Tensors should be found in order."
    assert "TN_T2" in tensors[1].labels, "Tensors should be found in order."

    return

def test_get_tensors_from_label():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], ["TEST"])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], [])
    t3 = Tensor(TEST_ARRAY_1, ["C", "D"], ["TEST"])
    t4 = Tensor(TEST_ARRAY_2, ["D", "E"], ["TEST"])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    tensors = tn.get_tensors_from_label("TEST")

    assert len(tensors) == 3, "Did not find all tensors."
    assert np.allclose(tensors[1].data.todense(), TEST_ARRAY_1), "Did not find correct tensors."
    assert "E" in tensors[2].indices, "Tensors should be found in order."

    return 

def test_contract_index():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], [])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], [])
    t3 = Tensor(TEST_ARRAY_1, ["C", "D"], [])
    t4 = Tensor(TEST_ARRAY_2, ["D", "E"], [])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    tn.contract_index("B")

    all_labels = tn.get_all_labels()
    all_indices = tn.get_all_indices()

    # Test TN structure after contraction
    assert len(all_labels) == 3, "Should only be 3 tensors after contraction."
    assert "TN_T1" not in all_labels and "TN_T2" not in all_labels, "Old labels should be removed."
    assert "TN_T5" in all_labels, "New label generated incorrectly."
    assert len(all_indices) == 4, "Should only be 4 indices after contraction."
    assert "B" not in all_indices, "Old index should be removed."
    assert "A" in all_indices and "C" in all_indices, "Uncontracted indices should persist."

    tensor = tn.get_tensors_from_label("TN_T5")[0]

    # Test contraction manipulates data correctly
    assert np.allclose(tensor.data.todense(), TEST_ARRAY_3), "Contracted tensor data incorrect."
    assert "A" in tensor.indices and "C" in tensor.indices, "Contracted tensor has the wrong indices."

    return 

def test_compress_index():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], [])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], [])

    tn = TensorNetwork([t1,t2], "TEST_TN")

    tn.compress_index("B", max_bond=1)
    bond_size_A = tn.get_dimension_of_index("A")
    bond_size_B = tn.get_dimension_of_index("B")
    bond_size_C = tn.get_dimension_of_index("C")

    assert bond_size_A == 2 and bond_size_B == 1 and bond_size_C == 2, "Bond dimensions incorrect after compression."

    return 

def test_pop_tensors_by_label():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], ["TEST_LABEL"])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], [])
    t3 = Tensor(TEST_ARRAY_1, ["C", "D"], [])
    t4 = Tensor(TEST_ARRAY_2, ["D", "E"], [])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    popped_tensor = tn.pop_tensors_by_label(["TEST_LABEL"])

    assert len(popped_tensor) == 1, "Did not find tensor."
    assert np.allclose(popped_tensor[0].data.todense(), TEST_ARRAY_1), "Found incorrect tensor."
    assert len(tn.tensors) == 3, "TN should only contain 3 tensors after pop."

    return 

def test_add_tensor():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], [])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], [])
    t3 = Tensor(TEST_ARRAY_1, ["C", "D"], [])
    t4 = Tensor(TEST_ARRAY_2, ["D", "E"], ["NEW_TENSOR"])

    tn = TensorNetwork([t1,t2,t3], "TEST_TN")
    tn.add_tensor(t4)

    assert len(tn.tensors) == 4, "TN should contain 4 tensors after add."
    assert "NEW_TENSOR" in tn.tensors[3].labels, "New tensor added incorrectly."
    assert len(tn.indices) == 5, "New tensor indices should be added to TN."

    return 

def test_contract_entire_network():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], [])
    t2 = Tensor(TEST_ARRAY_1, ["B", "C"], [])
    t3 = Tensor(TEST_ARRAY_1, ["C", "D"], [])
    t4 = Tensor(TEST_ARRAY_2, ["D", "E"], [])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    t = tn.contract_entire_network()

    assert np.allclose(t.data.todense(), TEST_ARRAY_3), "Contraction manipulates data incorrectly."
    assert "A" in t.indices and "E" in t.indices, "Uncontracted indices should appear in the output."

    return 

def test_compute_environment_tensor_by_label():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], [])
    t2 = Tensor(TEST_ARRAY_1, ["B", "C"], [])
    t3 = Tensor(TEST_ARRAY_2, ["C", "D"], ["TEST_LABEL"])
    t4 = Tensor(TEST_ARRAY_1, ["D", "A"], [])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    env = tn.compute_environment_tensor_by_label(["TEST_LABEL"])

    assert np.allclose(env.data.todense(), TEST_ARRAY_1), "Environment tensor calculated incorrectly."
    assert "C" in env.indices and "D" in env.indices, "Uncontracted indices should appear in the output."

    return 

def test_new_index_name():
    t1 = Tensor(TEST_ARRAY_1, ["E1", "E2"], [])
    t2 = Tensor(TEST_ARRAY_1, ["E2", "E3"], [])
    t3 = Tensor(TEST_ARRAY_2, ["E3", "E4"], [])
    t4 = Tensor(TEST_ARRAY_1, ["E4", "E5"], [])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    new_index_name = tn.new_index_name("E") 

    assert new_index_name == "E6", "New index name generated incorrectly."

    return 

def test_combine_indices():
    t1 = Tensor(TEST_ARRAY_4, ["A", "B", "C"], [])
    t2 = Tensor(TEST_ARRAY_4, ["B", "D", "C"], [])

    tn = TensorNetwork([t1,t2], "TEST_TN")
    tn.combine_indices(["B", "C"], new_index_name="TEST")

    assert "TEST" in tn.indices and "B" not in tn.indices and "C" not in tn.indices, "Index naming not handled correctly."
    assert np.allclose(tn.tensors[0].data.todense(), TEST_ARRAY_1), "Data manipulation incorrect."
    assert np.allclose(tn.tensors[1].data.todense(), TEST_ARRAY_1), "Data manipulation incorrect."
    assert tn.tensors[0].data.todense().shape == (2,2), "New shape incorrect."
    assert tn.tensors[1].data.todense().shape == (2,2), "New shape incorrect."

    return 

def test_svd():
    t1 = Tensor(np.random.rand(6,6), ["A", "B"], [])
    t2 = Tensor(np.random.rand(6,6), ["B", "C"], [])
    t3 = Tensor(np.random.rand(6,6), ["C", "D"], [])

    tn = TensorNetwork([t1,t2,t3], "TEST_TN")
    original_output = tn.contract_entire_network().data.todense()

    tn.svd(t2, ["C"], ["B"], new_index_name="TEST_INDEX")
    new_bond_dim = tn.get_dimension_of_index("TEST_INDEX")

    # Test TN structure after SVD
    assert "TEST_INDEX" in tn.indices, "New index name should appear in TN after SVD."
    assert len(tn.tensors) == 4, "SVD should add one tensor to the TN."
    assert new_bond_dim == 6, "New bond dimension should default to the TN max_bond."

    new_output = tn.contract_entire_network().data.todense()

    # Test data manipulation
    assert np.allclose(original_output, new_output), "Data manipulation incorrect after SVD."

    return 

def test_compress():
    t1 = Tensor(np.random.rand(10,10), ["A", "B"], [])
    t2 = Tensor(np.random.rand(10,10), ["B", "C"], [])
    t3 = Tensor(np.random.rand(10,10), ["C", "A"], [])

    tn = TensorNetwork([t1,t2,t3], "TEST_TN")
    original_output = tn.contract_entire_network()

    tn.compress(9)

    new_output = tn.contract_entire_network()

    assert np.isclose(original_output, new_output, atol=0.1), "TN compression does not match input."

    return 