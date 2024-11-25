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

    assert tn.name == "TEST_TN"
    assert len(tn.tensors) == 2
    assert np.allclose(tn.tensors[0].data.todense(), TEST_ARRAY_1) and np.allclose(tn.tensors[1].data.todense(), TEST_ARRAY_2)
    assert len(tn.indices) == 3
    assert "A" in tn.indices and "B" in tn.indices and "C" in tn.indices

    return 

def test_from_qiskit_layer():
    num_qubits = 3
    layer_number = 3
    qc = QuantumCircuit(num_qubits)
    qc.x(0) 
    qc.y(1)
    tn = TensorNetwork.from_qiskit_layer(qc, layer_number)

    assert len(tn.tensors) == num_qubits 
    assert len(tn.indices) == 2*num_qubits 
    assert f"QW0N{layer_number-1}" in tn.indices and f"QW2N{layer_number}" in tn.indices 
    assert np.allclose(tn.tensors[0].data.todense(), TEST_ARRAY_1) and np.allclose(tn.tensors[1].data.todense(), TEST_ARRAY_2)
    assert np.allclose(tn.tensors[2].data.todense(), [[1,0],[0,1]])

    return 

def test_from_qiskit_circuit():
    num_qubits = 3
    num_layers = 2
    qc = QuantumCircuit(num_qubits)
    qc.x(0)
    qc.y(1)
    qc.cx(1,2)
    tn = TensorNetwork.from_qiskit_circuit(qc) 

    assert len(tn.tensors) == 5
    assert len(tn.indices) == (num_layers+1) * num_qubits
    assert np.allclose(tn.tensors[3].data.todense().reshape(4,4), [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    assert "L2" in tn.tensors[3].labels 
    assert "Q1" in tn.tensors[3].labels and "Q2" in tn.tensors[3].labels 
    assert "QW1N1" in tn.tensors[3].indices and "QW2N2" in tn.tensors[3].indices

    return 

def test_get_index_to_tensor_dict():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], ["T1"])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], ["T2"])

    tn = TensorNetwork([t1,t2], "TEST_TN")
    tn_dict = tn.get_index_to_tensor_dict()

    assert len(list(tn_dict.keys())) == 3
    assert len(tn_dict["A"]) == 1 and len(tn_dict["B"]) == 2 and len(tn_dict["C"]) == 1
    assert tn_dict["A"][0].labels[0] == "T1" and tn_dict["C"][0].labels[0] == "T2"

    return 

def test_get_label_to_tensor_dict():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], ["TEST", "T1"])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], ["TEST", "T2"])

    tn = TensorNetwork([t1,t2], "TEST_TN")
    tn_dict = tn.get_label_to_tensor_dict()

    assert len(list(tn_dict.keys())) == 5
    assert len(tn_dict["TN_T1"]) == 1 and len(tn_dict["TN_T2"]) == 1 and len(tn_dict["TEST"]) == 2
    assert tn_dict["T1"][0].indices[0] == "A" and tn_dict["TEST"][1].indices[0] == "B"

    return 

def test_get_dimension_of_index():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], ["T1"])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], ["T2"])

    tn = TensorNetwork([t1,t2], "TEST_TN")
    dim_A = tn.get_dimension_of_index("A")
    dim_B = tn.get_dimension_of_index("B")
    dim_C = tn.get_dimension_of_index("C")

    assert dim_A == 2
    assert dim_B == 2
    assert dim_C == 2

    return 

def test_get_internal_indices():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], [])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], [])
    t3 = Tensor(TEST_ARRAY_1, ["C", "D"], [])
    t4 = Tensor(TEST_ARRAY_2, ["D", "E"], [])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    internal_indices = tn.get_internal_indices()

    assert "B" in internal_indices and "C" in internal_indices and "D" in internal_indices
    assert "A" not in internal_indices and "E" not in internal_indices

    return 

def test_get_external_indices():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], [])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], [])
    t3 = Tensor(TEST_ARRAY_1, ["C", "D"], [])
    t4 = Tensor(TEST_ARRAY_2, ["D", "E"], [])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    external_indices = tn.get_external_indices()

    assert "B" not in external_indices and "C" not in external_indices and "D" not in external_indices
    assert "A" in external_indices and "E" in external_indices

    return 

def test_get_all_indices():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], [])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], [])
    t3 = Tensor(TEST_ARRAY_1, ["C", "D"], [])
    t4 = Tensor(TEST_ARRAY_2, ["D", "E"], [])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    all_indices = tn.get_all_indices()

    assert "B" in all_indices and "C" in all_indices and "D" in all_indices
    assert "A" in all_indices and "E" in all_indices

    return 

def test_get_all_labels():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], ["TEST1"])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], [])
    t3 = Tensor(TEST_ARRAY_1, ["C", "D"], ["TEST2"])
    t4 = Tensor(TEST_ARRAY_2, ["D", "E"], [])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    all_labels = tn.get_all_labels()

    assert len(all_labels) == 6
    assert "TN_T1" in all_labels and "TN_T4" in all_labels 
    assert "TEST1" in all_labels and "TEST2" in all_labels

    return 

def test_get_new_label():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], ["T2"])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], ["T8"])
    t3 = Tensor(TEST_ARRAY_1, ["C", "D"], ["T101"])
    t4 = Tensor(TEST_ARRAY_2, ["D", "E"], ["F100001"])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    new_label_1 = tn.get_new_label("T")
    new_label_2 = tn.get_new_label()

    assert new_label_1 == "T102"
    assert new_label_2 == "TN_T5"

    return 

def test_get_tensors_from_index_name():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], [])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], [])
    t3 = Tensor(TEST_ARRAY_1, ["C", "D"], [])
    t4 = Tensor(TEST_ARRAY_2, ["D", "E"], [])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    tensors = tn.get_tensors_from_index_name("B")

    assert len(tensors) == 2
    assert np.allclose(tensors[0].data.todense(), TEST_ARRAY_1)
    assert np.allclose(tensors[1].data.todense(), TEST_ARRAY_2)
    assert "TN_T1" in tensors[0].labels 
    assert "TN_T2" in tensors[1].labels

    return

def test_get_tensors_from_label():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], ["TEST"])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], [])
    t3 = Tensor(TEST_ARRAY_1, ["C", "D"], ["TEST"])
    t4 = Tensor(TEST_ARRAY_2, ["D", "E"], ["TEST"])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    tensors = tn.get_tensors_from_label("TEST")

    assert len(tensors) == 3
    assert np.allclose(tensors[1].data.todense(), TEST_ARRAY_1)
    assert "E" in tensors[2].indices

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

    assert len(all_labels) == 3
    assert "TN_T1" not in all_labels and "TN_T2" not in all_labels 
    assert "TN_T5" in all_labels
    assert len(all_indices) == 4
    assert "B" not in all_indices 
    assert "A" in all_indices and "C" in all_indices

    tensor = tn.get_tensors_from_label("TN_T5")[0]

    assert np.allclose(tensor.data.todense(), TEST_ARRAY_3)
    assert "A" in tensor.indices and "C" in tensor.indices

    return 

def test_compress_index():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], [])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], [])

    tn = TensorNetwork([t1,t2], "TEST_TN")
    tn.compress_index("B", max_bond=2)

    assert np.allclose(tn.tensors[0].data.todense(), TEST_ARRAY_1)
    assert np.allclose(tn.tensors[1].data.todense(), TEST_ARRAY_2)

    tn.compress_index("B", max_bond=1)
    bond_size_A = tn.get_dimension_of_index("A")
    bond_size_B = tn.get_dimension_of_index("B")
    bond_size_C = tn.get_dimension_of_index("C")

    assert bond_size_A == 2 and bond_size_B == 1 and bond_size_C == 2

    return 

def test_pop_tensors_by_label():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], ["TEST_LABEL"])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], [])
    t3 = Tensor(TEST_ARRAY_1, ["C", "D"], [])
    t4 = Tensor(TEST_ARRAY_2, ["D", "E"], [])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    popped_tensor = tn.pop_tensors_by_label(["TEST_LABEL"])

    assert len(popped_tensor) == 1
    assert np.allclose(popped_tensor[0].data.todense(), TEST_ARRAY_1)
    assert len(tn.tensors) == 3

    return 

def test_add_tensor():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], [])
    t2 = Tensor(TEST_ARRAY_2, ["B", "C"], [])
    t3 = Tensor(TEST_ARRAY_1, ["C", "D"], [])
    t4 = Tensor(TEST_ARRAY_2, ["D", "E"], ["NEW_TENSOR"])

    tn = TensorNetwork([t1,t2,t3], "TEST_TN")
    tn.add_tensor(t4)

    assert len(tn.tensors) == 4
    assert "NEW_TENSOR" in tn.tensors[3].labels
    assert len(tn.indices) == 5

    return 

def test_contract_entire_network():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], [])
    t2 = Tensor(TEST_ARRAY_1, ["B", "C"], [])
    t3 = Tensor(TEST_ARRAY_1, ["C", "D"], [])
    t4 = Tensor(TEST_ARRAY_2, ["D", "E"], [])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    t = tn.contract_entire_network()

    assert np.allclose(t.data.todense(), TEST_ARRAY_3)
    assert "A" in t.indices and "E" in t.indices

    return 

def test_compute_environment_tensor_by_label():
    t1 = Tensor(TEST_ARRAY_1, ["A", "B"], [])
    t2 = Tensor(TEST_ARRAY_1, ["B", "C"], [])
    t3 = Tensor(TEST_ARRAY_2, ["C", "D"], ["TEST_LABEL"])
    t4 = Tensor(TEST_ARRAY_1, ["D", "A"], [])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    env = tn.compute_environment_tensor_by_label(["TEST_LABEL"])

    assert np.allclose(env.data.todense(), TEST_ARRAY_1)
    assert "C" in env.indices and "D" in env.indices

    return 

def test_new_index_name():
    t1 = Tensor(TEST_ARRAY_1, ["E1", "E2"], [])
    t2 = Tensor(TEST_ARRAY_1, ["E2", "E3"], [])
    t3 = Tensor(TEST_ARRAY_2, ["E3", "E4"], [])
    t4 = Tensor(TEST_ARRAY_1, ["E4", "E5"], [])

    tn = TensorNetwork([t1,t2,t3,t4], "TEST_TN")
    new_index_name = tn.new_index_name("E") 

    assert new_index_name == "E6"

    return 

def test_combine_indices():
    t1 = Tensor(TEST_ARRAY_4, ["A", "B", "C"], [])
    t2 = Tensor(TEST_ARRAY_4, ["B", "D", "C"], [])

    tn = TensorNetwork([t1,t2], "TEST_TN")
    tn.combine_indices(["B", "C"], new_index_name="TEST")

    assert "TEST" in tn.indices and "B" not in tn.indices and "C" not in tn.indices
    assert np.allclose(tn.tensors[0].data.todense(), TEST_ARRAY_1)
    assert np.allclose(tn.tensors[1].data.todense(), TEST_ARRAY_1)
    assert tn.tensors[0].data.todense().shape == (2,2)
    assert tn.tensors[1].data.todense().shape == (2,2)

    return 

def test_svd():
    t1 = Tensor(np.random.rand(6,6), ["A", "B"], [])
    t2 = Tensor(np.random.rand(6,6), ["B", "C"], [])
    t3 = Tensor(np.random.rand(6,6), ["C", "D"], [])

    tn = TensorNetwork([t1,t2,t3], "TEST_TN")
    original_output = tn.contract_entire_network().data.todense()

    tn.svd(t2, ["C"], ["B"], new_index_name="TEST_INDEX")
    new_bond_dim = tn.get_dimension_of_index("TEST_INDEX")

    assert "TEST_INDEX" in tn.indices
    assert len(tn.tensors) == 4
    assert new_bond_dim == 6

    new_output = tn.contract_entire_network().data.todense()

    assert np.allclose(original_output, new_output)

    return 

def test_compress():
    t1 = Tensor(np.random.rand(10,10), ["A", "B"], [])
    t2 = Tensor(np.random.rand(10,10), ["B", "C"], [])
    t3 = Tensor(np.random.rand(10,10), ["C", "A"], [])

    tn = TensorNetwork([t1,t2,t3], "TEST_TN")
    original_output = tn.contract_entire_network()

    tn.compress(9)

    new_output = tn.contract_entire_network()

    assert np.isclose(original_output, new_output, atol=0.1)

    return 