from typing import List, Union

# Underlying tensor objects can either be NumPy arrays or Sparse arrays
import numpy as np
from numpy.linalg import svd
import sparse
from tn4qa.tensor import Tensor

# Contraction path finding is offloaded to Cotengra
import cotengra as ctg

# Qiskit quantum circuit integration
from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel
from qiskit.converters import circuit_to_dag, dag_to_circuit

# Visualisation purposes
import re
import matplotlib.pyplot as plt
import networkx as nx

class TensorNetwork:
    def __init__(self, tensors : List[Tensor], name : str="TN") -> "TensorNetwork":
        """
        Constructor for the TensorNetwork class.
        
        Args:
            tensors: A list of Tensor objects.
            name (optional): A name for the tensor network.
        
        Returns:
            A tensor network.
        """
        self.name = name
        i = 1
        for t in tensors:
            t.labels.append(f"TN_T{i}")
            i += 1
        self.tensors = tensors
        self.indices = self.get_all_indices()

    def __str__(self) -> str:
        """
        Defines output of print.
        """
        output = f"Tensor Network containing: \n"
        for t in self.tensors:
            shape = str(t.dimensions)
            indices = str(t.indices)
            output += f"Tensor with shape {shape} and indices {indices} \n"
        return output
    
    def __add__(self, other : "TensorNetwork") -> "TensorNetwork":
        """
        Defines addition for tensor networks.
        """
        all_tensors = self.tensors + other.tensors 
        tn = TensorNetwork(all_tensors, name=self.name)
        return tn
    
    @classmethod
    def from_qiskit_layer(cls, layer : QuantumCircuit, layer_number : int=1) -> "TensorNetwork":
        """
        Construct a tensor network from a Qiskit QuantumCircuit object (single layer).
        
        Args:
            layer: The QuantumCircuit layer.
            layer_number (optional): The layer number within a larger circuit. Default to 1.
            
        Returns:
            A tensor network.
        """
        num_qubits = layer.num_qubits 
        index_prefixes = [f"QW{x}" for x in range(num_qubits)]
        wire_counts = {str(x) : layer_number-1 for x in range(num_qubits)}
        tensors = []
        
        tensor_number = 1
        for inst in layer.data:
            inst_num_qubits = inst.operation.num_qubits
            qidxs = [inst.qubits[i]._index for i in range(inst_num_qubits)]
            if inst_num_qubits == 1:
                indices = [
                    index_prefixes[qidxs[0]]+"N"+str(wire_counts[str(qidxs[0])]+1), 
                    index_prefixes[qidxs[0]]+"N"+str(wire_counts[str(qidxs[0])])
                ]
                labels = [f"L{layer_number}", f"Q{qidxs[0]}"]
                wire_counts[str(qidxs[0])] += 1
            else:
                indices = [
                    index_prefixes[qidxs[0]]+"N"+str(wire_counts[str(qidxs[0])]+1),
                    index_prefixes[qidxs[1]]+"N"+str(wire_counts[str(qidxs[1])]+1),
                    index_prefixes[qidxs[0]]+"N"+str(wire_counts[str(qidxs[0])]),
                    index_prefixes[qidxs[1]]+"N"+str(wire_counts[str(qidxs[1])])
                ]
                labels = [f"L{layer_number}", f"Q{qidxs[0]}", f"Q{qidxs[1]}"]
                wire_counts[str(qidxs[0])] += 1
                wire_counts[str(qidxs[1])] += 1
            
            inst_tensor = Tensor.from_qiskit_gate(inst, indices, labels)
            tensors.append(inst_tensor)
            tensor_number += 1
        
        unused_qubits = [x for x in wire_counts if wire_counts[x] == layer_number-1]
        for qidx in unused_qubits:
            array = np.array([[1,0],[0,1]], dtype=complex).reshape(2,2)
            indices = [f"QW{qidx}"+"N"+str(layer_number), f"QW{qidx}"+"N"+str(layer_number-1)]
            labels = [f"L{layer_number}", f"Q{qidx}"]
            tensor = Tensor(array, indices, labels)
            tensors.append(tensor)
            tensor_number += 1
        
        tn = TensorNetwork(tensors, name="QuantumCircuit")
        return tn

    @classmethod
    def from_qiskit_circuit(cls, qc : QuantumCircuit) -> "TensorNetwork":
        """
        Construct a tensor network from a Qiskit QuantumCircuit object.
        
        Args:
            gc: The QuantumCircuit object.

        Returns:
            A tensor network.
        """
        dag = circuit_to_dag(qc)
        all_layers = [x for x in dag.layers()]
        first_layer = all_layers[0]
        first_layer_as_circ = dag_to_circuit(first_layer['graph'])
        tn = TensorNetwork.from_qiskit_layer(first_layer_as_circ, layer_number=1)
        layer_number = 2
        for layer in all_layers[1:]:
            layer_as_circ = dag_to_circuit(layer['graph'])
            tn = tn + TensorNetwork.from_qiskit_layer(layer_as_circ, layer_number)
        return tn

    @classmethod
    def tnqem_construction(cls, qc : QuantumCircuit, noise_model : NoiseModel) -> "TensorNetwork":
        # TODO: Will likely need some helper functions to convert the noise model into tensors etc.
        return

    def get_index_to_tensor_dict(self) -> dict:
        """
        Build a dictionary mapping indices to their tensors.
        
        Returns:
            A dictionary of the form {idx : [tensor1,...]}
        """
        tn_dict = {}
        for t in self.tensors:
            for idx in t.indices:
                if idx in tn_dict:
                    tn_dict[idx].append(t)
                else:
                    tn_dict[idx] = [t]
        return tn_dict
    
    def get_label_to_tensor_dict(self) -> dict:
        """
        Build a dictionary mapping labels to their tensors.
        
        Returns:
            A dictionary of the form {label : [tensor1,...]}
        """
        tn_dict = {}
        for t in self.tensors:
            for label in t.labels:
                if label in tn_dict:
                    tn_dict[label].append(t)
                else:
                    tn_dict[label] = [t]
        return tn_dict
    
    def get_dimension_of_index(self, idx : str) -> int:
        """
        Get the dimension of an index.
        
        Args:
            idx: The index name.
            
        Returns:
            The dimension of idx.
        """
        tn_dict = self.get_index_to_tensor_dict()
        t = tn_dict[idx][0]
        dim = t.get_dimension_of_index(idx)
        return dim

    def get_internal_indices(self) -> List[str]:
        """
        Get the internal indices of the tensor network.
        
        Returns:
            Indices that are connected to 2 tensors in the network.
        """
        tn_dict = self.get_index_to_tensor_dict()
        internal_bonds = [idx for idx in tn_dict.keys() if len(tn_dict[idx]) == 2]
        return internal_bonds

    def get_external_indices(self) -> List[str]:
        """
        Get the external bonds of the tensor network.
        
        Returns:
            Indices that are connected to 1 tensor in the network.
        """
        tn_dict = self.get_index_to_tensor_dict()
        external_bonds = [idx for idx in tn_dict.keys() if len(tn_dict[idx]) == 1]
        return external_bonds

    def get_all_indices(self) -> List[str]:
        """
        Get all indices in the tensor network.
        
        Returns:
            A list of all index names.
        """
        tn_dict = self.get_index_to_tensor_dict()
        return list(tn_dict.keys())

    def get_all_labels(self) -> List[str]:
        """
        Get all labels in the tensor network.
        
        Returns:
            A list of all label names.
        """
        tn_dict = self.get_label_to_tensor_dict()
        return list(tn_dict.keys())

    def get_new_label(self, tensor_prefix : str="TN_T") -> str:
        """
        Get a new tensor label with the specified prefix.
        
        Args:
            tensor_prefix (optional): Defaults to "TN_T".
        
        Returns:
            A new label that doesn't already appear in the network starting with tensor_prefix. 
        """
        all_labels = [x for x in self.get_all_labels() if len(x) > len(tensor_prefix)]

        current_vals = []
        for l in all_labels:
            if l[:len(tensor_prefix)] == tensor_prefix and l[len(tensor_prefix):].isdigit():
                current_vals.append(int(l[len(tensor_prefix):]))
        if len(current_vals) > 0:
            max_current_val = max(current_vals)
        else:
            max_current_val = 0
        new_label = tensor_prefix + str(max_current_val+1)
        
        return new_label

    def get_tensors_from_index_name(self, idx : str) -> List[Tensor]:
        """
        Get all tensors connected to a given index.
        
        Args:
            idx: The index name.
            
        Returns
            A list of tensors with idx as one of their indices.
        """
        tensors = []
        for t in self.tensors:
            if idx in t.indices:
                tensors.append(t)
        return tensors

    def get_tensors_from_label(self, label : str) -> List[Tensor]:
        """
        Get all tensors connected to a given label.
        
        Args:
            label: The label name.
            
        Returns
            A list of tensors with label as one of their labels.
        """
        tensors = []
        for t in self.tensors:
            if label in t.labels:
                tensors.append(t)
        return tensors

    def contract_index(self, idx : str) -> None:
        """
        Contract an index in the tensor network.
        
        Args:
            idx: The name of the index to contract.
        """
        tensors = self.get_tensors_from_index_name(idx)

        array0, array1 = tensors[0].data, tensors[1].data
        indices0, indices1 = tensors[0].indices, tensors[1].indices

        output_indices = [i for i in indices0 if i != idx] + [i for i in indices1 if i != idx]
        new_data = ctg.array_contract(arrays=[array0, array1], inputs=[indices0, indices1], output=output_indices, cache_expression=False)
        new_labels = [self.get_new_label()]
        if len(new_data.shape) > len(output_indices):
            new_data = new_data.reshape(new_data.shape[1:])
        new_tensor = Tensor(new_data, output_indices, new_labels)

        self.tensors.remove(tensors[0])
        self.tensors.remove(tensors[1])
        self.indices.remove(idx)

        self.tensors.append(new_tensor)

        return

    def compress_index(self, idx : str, max_bond : int, reverse_direction : bool=False) -> None:
        """
        Compress a given index using SVD.
        
        Args:
            idx: The index to compress.
            max_bond: The maximum bond dimension for this index.
        """
        if reverse_direction:
            tensors = self.get_tensors_from_index_name(idx)[::-1]
        else:
            tensors = self.get_tensors_from_index_name(idx)

        array0, array1 = tensors[0].data, tensors[1].data
        indices0, indices1 = tensors[0].indices, tensors[1].indices
        dims0, dims1 = tensors[0].dimensions, tensors[1].dimensions

        output_indices = [i for i in indices0 if i != idx] + [i for i in indices1 if i != idx]
        
        new_data = ctg.array_contract(arrays=[array0, array1], inputs=[indices0, indices1], output=output_indices, cache_expression=False)
        temp_tensor = Tensor(new_data, output_indices, ["TEMP"])

        input_idxs = [i for i in indices0 if i != idx]
        output_idxs = [i for i in indices1 if i != idx]
        temp_tensor.tensor_to_matrix(input_idxs, output_idxs)
        u, s, vh = svd(temp_tensor.data.todense(), full_matrices=True)    
        bond_dim = min([max_bond, temp_tensor.data.shape[0], temp_tensor.data.shape[1]])  
        new_data0 = sparse.COO.from_numpy(vh[:bond_dim, :])
        new_data1 = sparse.COO.from_numpy(u[:, :bond_dim] * s[:bond_dim])

        idx_pos0 = indices0.index(idx)
        idx_pos1 = indices1.index(idx)
        new_dims0 = (bond_dim,) + dims0[:idx_pos0] + dims0[idx_pos0+1:]
        new_dims1 = dims1[:idx_pos1] + dims1[idx_pos1+1:] + (bond_dim,)
        new_indices0 = [idx] + indices0[:idx_pos0] + indices0[idx_pos0+1:] 
        new_indices1 = indices1[:idx_pos1] + indices1[idx_pos1+1:] + [idx]

        new_data0 = new_data0.reshape(new_dims0)
        new_data1 = new_data1.reshape(new_dims1)

        tensors[0].data = new_data0
        tensors[1].data = new_data1
        tensors[0].indices = new_indices0
        tensors[1].indices = new_indices1
        tensors[0].dimensions = new_dims0
        tensors[1].dimensions = new_dims1

        tensors[0].reorder_indices(indices0)
        tensors[1].reorder_indices(indices1)

        return

    def pop_tensors_by_label(self, labels : List[str]) -> List[Tensor]:
        """
        Remove tensors from the network given a set of labels.
        
        Args:
            labels: The list of labels to search for.

        Returns:
            The (possibly empty) list of removed tensors.
        """
        tensors = []
        for tensor in self.tensors:
            has_all_labels = True
            for label in labels:
                if label not in tensor.labels:
                    has_all_labels = False
            if has_all_labels:
                tensors.append(tensor)

        for t in tensors:
            self.tensors.remove(t)

        tn_dict = self.get_index_to_tensor_dict()
        for idx in tn_dict:
            still_exists = False
            if len(tn_dict[idx]) > 0:
                still_exists = True
            if not still_exists:
                self.indices.remove(idx)
        
        return tensors

    def add_tensor(self, tensor : Tensor, position : int=None) -> None:
        """
        Add a tensor to the network.
        
        Args:
            tensor: The tensor to add.
        """
        unique_label = self.get_new_label("TN_T")
        tensor.labels.append(unique_label)
        if not position:
            self.tensors.append(tensor)
        else:
            self.tensors.insert(position, tensor)
        for idx in tensor.indices:
            if idx not in self.indices:
                self.indices.append(idx)
        return
    
    def contract_entire_network(self) -> Union[Tensor, complex]:
        """
        Contracts all internal indices in the network.
        
        Returns:
            A tensor whose indices were the external indices of the network, or a float if there were no external indices.
        """
        output_indices = self.get_external_indices()
        output_labels = [self.get_new_label("TN_T")]
        arrays = []
        input_indices = []
        for t in self.tensors:
            arrays.append(t.data)
            input_indices.append(t.indices)

        output_tensor_data = ctg.array_contract(arrays=arrays, inputs=input_indices, output=output_indices, cache_expression=False)
        if len(output_indices) == 0:
            return complex(output_tensor_data.flatten()[0])
        else:
            output_tensor = Tensor(output_tensor_data, output_indices, output_labels)
            return output_tensor
    
    def compute_environment_tensor_by_label(self, labels : List[str], replace_tensor : bool=False) -> Union[Tensor, None]:
        """
        Compute the environment of a tensor in the network given a set of labels.
        
        Args:
            labels: The labels to look for.
            replace_tensor (optional): When True replaces the original tensor in the network by its environment. Default is False. 

        Returns:
            If replace_tensor is True, return type is None. Otherwise, returns the environment tensor.
        """
        popped_tensor = self.pop_tensors_by_label(labels)[0]
        output_tensor = self.contract_entire_network()

        if replace_tensor:
            self.add_tensor(output_tensor)
        else:
            self.add_tensor(popped_tensor)
            return output_tensor

        return 
    
    def new_index_name(self, index_prefix : str="B", num_new_indices : int=1) -> Union[str, List[str]]:
        """
        Generate a new index name not already in use.
        
        Args:
            index_prefix (optional): Default is "B".
            num_new_indices (optional): Number of new names required. Default is 1.
        
        Returns:
            The new index name. Returned as a str if num_new_indices=1, otherwise returned as List[str].
        """
        current_indices = [x for x in self.indices if len(x) > len(index_prefix)]
        current_vals = []
        for idx in current_indices:
            if idx[:len(index_prefix)] == index_prefix and idx[len(index_prefix):].isdigit():
                current_vals.append(int(idx[len(index_prefix):]))
        if len(current_vals) > 0:
            max_current_val = max(current_vals)
        else:
            max_current_val = 0
        new_indices = [index_prefix + str(max_current_val+i) for i in range(1, num_new_indices+1)]

        if num_new_indices == 1:
            return new_indices[0]
        return new_indices

    def combine_indices(self, idxs : List[str], new_index_name : str=None) -> None:
        """
        Combine two or more indices within the network. Only valid when all indices are between the same two tensors.
        
        Args:
            idxs: The indices to combine.
            new_index_name (optional): What to call the resulting combined index.
        """
        tensors = self.get_tensors_from_index_name(idxs[0])
        for t in tensors:
            t.combine_indices(idxs, new_index_name)
        self.indices = self.get_all_indices()
        return
    
    def svd(self, tensor : Tensor, input_indices : List[str], output_indices : List[str], max_bond : int=None, new_index_name : str=None, new_labels : List[List[str]]=None) -> None:
        """
        Perform an SVD on a tensor.
        
        Args:
            tensor: The tensor.
            input_indices: Indices to be treated as one side of SVD.
            output_indices: Indices to be treated as other side of SVD. 
            max_bond: The maximum bond dimension allwoed. 
            new_index_name (optional): What to call the resulting new index.
        """
        original_position = self.tensors.index(tensor)
        original_labels = tensor.labels
        original_input_dims = [tensor.get_dimension_of_index(x) for x in input_indices]
        original_output_dims = [tensor.get_dimension_of_index(x) for x in output_indices]

        tensor.tensor_to_matrix(input_indices, output_indices)
        if not max_bond:
            bond_dim = min([tensor.dimensions[0], tensor.dimensions[1]])
        else:
            bond_dim = min([max_bond, tensor.dimensions[0], tensor.dimensions[1]])
        u, s, vh = svd(tensor.data.todense(), full_matrices=True)
        tensor0_data = sparse.COO.from_numpy(vh[:max_bond, :])
        tensor1_data = sparse.COO.from_numpy(u[:, :max_bond] * s[:max_bond])

        if not new_index_name:
            new_index_name = self.new_index_name()

        tensor0_labels = [self.get_new_label("TN_T")]
        tensor1_labels = [self.get_new_label("TN_T")]
        if new_labels:
            tensor0_labels = tensor0_labels + new_labels[0]
            tensor1_labels = tensor1_labels + new_labels[1]
        
        tensor0_indices = [new_index_name] + input_indices
        tensor1_indices = output_indices + [new_index_name]

        tensor0_dims = [bond_dim] + original_input_dims
        tensor1_dims = original_output_dims + [bond_dim]

        tensor0_data = sparse.reshape(tensor0_data, tensor0_dims)
        tensor1_data = sparse.reshape(tensor1_data, tensor1_dims)

        tensor0 = Tensor(tensor0_data, tensor0_indices, tensor0_labels)
        tensor1 = Tensor(tensor1_data, tensor1_indices, tensor1_labels)

        _ = self.pop_tensors_by_label(original_labels)
        self.add_tensor(tensor0, original_position)
        self.add_tensor(tensor1, original_position+1)
        return
    
    def compress(self, max_bond : int) -> None:
        """
        Compress the tensor network using SVD.
        
        Args:
            max_bond: The maximum bond dimension allowed.
        """
        internal_indices = self.get_internal_indices()
        for index in internal_indices:
            self.compress_index(index, max_bond)
        return

    def parse_tn_output(self, tn_output):
        """
        Parse the textual representation of the tensor networks.
        Extracts tensors and their connections.
        Args:
            TensorNetwork 
        Returns:
            list: List of tensors and their indices
        """
        tensors = self.tensors
        # tensors = []
        # for line in tn_output.split("\n"):
        #     if line.startswith("Tensor with shape"):
        #         # Extract tensor indices
        #         indices_match = re.search(r"indices\s\[(.*?)\]", line)
        #         if indices_match:
        #             indices = indices_match.group(1).replace("'", "").split(", ")
        #             tensors.append(indices)
        return tensors

    def build_graph_from_tensors(self, parsed_tensors):
        """
        Build a directed graph from the list of tensors and their indices.
        Args:
            tensors (list): List of tensors and their indices
        Returns:
            networkx.DiGraph: Directed graph representing the tensor network
        """
        G = nx.DiGraph()
        for i, tensor in enumerate(parsed_tensors):
            indices = tensor.indices
            tensor_name = f"Tensor_{i + 1}"
            G.add_node(tensor_name)
        
            # Add edges for bottom connections and dangling indices
            for idx in indices:
                connected_tensors = self.get_tensors_from_index_name(idx)
                if len(connected_tensors) == 2:
                    i, j = [self.tensors.index(t) for t in connected_tensors]
                    first_tensor = f"Tensor_{i + 1}"
                    second_tensor = f"Tensor_{j + 1}"
                    G.add_edge(first_tensor, second_tensor, label=idx)
                else:
                    first_tensor = tensor_name
                    G.add_edge(first_tensor, idx, label=idx)

        return G

    def draw(self, node_size, x_len, y_len):
        """
        Visualise the tensor network using matplotlib and networkx.
        Args:
            tn (TensorNetwork)
            node_size (int): Size of the nodes in the plot
            x_len (int): Length of the x-axis
            y_len (int): Length of the y-axis
        """
        # Define the tensor output
        tn_output = f"""\{self}
        """
        # Parse the output
        parsed_tensors = self.parse_tn_output(tn_output)

        # Build the graph
        G = self.build_graph_from_tensors(parsed_tensors)

        # Define positions for tensors and dangling indices
        pos = {}
        vertical_spacing = 1.0
        horizontal_spacing = 1.0

        # Assign positions for tensor nodes
        nodes = [node for node in G.nodes if node.startswith("Tensor")]
        for i, node in enumerate(nodes):
            tensor_labels = parsed_tensors[i].labels 
            layer_number = #### There will be a label Lx where x is the layer number
            qubit_wire = #### There will be a label Qx where x is the qubit number
            pos[node] = (layer_number * horizontal_spacing, qubit_wire * vertical_spacing)

        # Assign positions for dangling indices
        for edge in G.edges(data=True):
            if not edge[1].startswith("Tensor"):
                if edge[1] not in pos:
                    pos[edge[1]] = (pos[edge[0]][0] + horizontal_spacing, pos[edge[0]][1])
        # Draw the graph
        plt.figure(figsize=(x_len, y_len))

        # Separate tensor and index nodes
        tensor_nodes = [node for node in G.nodes if node.startswith("Tensor")]

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=tensor_nodes, node_size=node_size, node_color="hotpink", label="Tensors")

        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=False)

        # Add edge labels
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", font_size=8)

        # Title and axis
        plt.title("Tensor Network Visualisation", fontsize=14)
        plt.show()
