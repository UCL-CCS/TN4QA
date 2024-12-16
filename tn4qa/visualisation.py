# Visualisation purposes
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx import DiGraph

from .tensor import Tensor


def build_graph_from_tensor_network(tensors: list[Tensor]) -> DiGraph:  # type: ignore
    """
    Build a directed graph from the list of tensors and their indices.

    Args:
        tn: A TensorNetwork object.

    Returns:
        networkx.DiGraph: Directed graph representing the tensor network
    """

    G = nx.DiGraph()
    for _, tensor in enumerate(tensors):
        indices = tensor.indices
        tensor_name = [l for l in tensor.labels if l[:2] == "TN"][0]
        G.add_node(tensor_name)

        # Add edges for bottom connections and dangling indices
        for idx in indices:
            connected_tensors = [t for t in tensors if idx in t.indices]
            if len(connected_tensors) == 2:
                t1 = connected_tensors[0]
                t2 = connected_tensors[1]
                first_tensor = [l for l in t1.labels if l[:2] == "TN"][0]
                second_tensor = [l for l in t2.labels if l[:2] == "TN"][0]
                if (first_tensor, second_tensor) not in G.edges and (
                    second_tensor,
                    first_tensor,
                ) not in G.edges:
                    G.add_edge(first_tensor, second_tensor, label=idx)
            else:
                first_tensor = tensor_name
                G.add_edge(first_tensor, idx, label=idx)

    return G


def build_graph_from_MPO(tensors: list[Tensor]) -> DiGraph:  # type: ignore
    """
    Build a directed graph from the list of tensors and their indices.

    Args:
        mpo: MPO object containing tensors with indices.

    Returns:
        networkx.DiGraph: Directed graph representing the tensor network
    """
    G = nx.DiGraph()
    for i, tensor in enumerate(tensors):
        indices = tensor.indices
        tensor_name = f"Tensor_{i + 1}"
        G.add_node(tensor_name)

        # Add edges for connections and dangling indices
        for idx in indices:
            if idx.startswith("B") and i < len(tensors) - 1:
                # Connect to the next tensor
                next_tensor = f"Tensor_{i + 2}"
                G.add_edge(tensor_name, next_tensor, label=idx)
            elif idx.startswith(("R", "L")):
                # Connect dangling indices
                G.add_edge(tensor_name, idx, label=idx)
                G.add_node(idx)  # Ensure dangling index is a node
    return G


def build_graph_from_MPS(tensors: list[Tensor]) -> DiGraph:  # type: ignore
    """
    Build a directed graph from the list of tensors and their indices.

    Args:
        mps: MPS object containing tensors with indices.

    Returns:
        networkx.DiGraph: Directed graph representing the tensor network
    """
    G = nx.DiGraph()
    for i, tensor in enumerate(tensors):
        indices = tensor.indices
        tensor_name = f"Tensor_{i + 1}"
        G.add_node(tensor_name)

        # Add edges for connections and dangling indices
        for idx in indices:
            print(idx)
            if idx.startswith("B") and i < len(tensors) - 1:
                # Connect to the next tensor
                next_tensor = f"Tensor_{i + 2}"
                G.add_edge(tensor_name, next_tensor, label=idx)
            elif idx.startswith("P"):
                # Connect dangling indices
                G.add_edge(tensor_name, idx, label=idx)
                G.add_node(idx)  # Ensure dangling index is a node
    return G


def draw_quantum_circuit(
    tensors: list[Tensor],  # type: ignore
    node_size: int | None = None,
    x_len: int | None = None,
    y_len: int | None = None,
):
    """
    Visualise a tensor network representing a quantum circuit using matplotlib and networkx.

    Args:
        qc_tn: The TensorNetwork built from a quantum circuit
        node_size (int): Size of the nodes in the plot
        x_len (int): Length of the x-axis
        y_len (int): Length of the y-axis
    """
    if not x_len:
        x_len = int(np.sqrt(len(tensors))) * 5
    if not y_len:
        y_len = x_len / 2
    if not node_size:
        node_size = x_len * 5

    # Build the graph
    G = build_graph_from_tensor_network(tensors)

    # Define positions for tensors and dangling indices
    pos = {}
    vertical_spacing = 1.0
    horizontal_spacing = 1.0

    # Assign positions for tensor nodes
    nodes = [node for node in G.nodes if node.startswith("TN")]
    for node in nodes:
        tensor = [t for t in tensors if node in t.labels][0]
        tensor_labels = tensor.labels
        layer_number = int([l for l in tensor_labels if l[0] == "L"][0][1:])
        qubit_wire = int([l for l in tensor_labels if l[0] == "Q"][0][1:])
        pos[node] = (layer_number * horizontal_spacing, -qubit_wire * vertical_spacing)

    # Assign positions for dangling indices
    for edge in G.edges(data=True):
        if not edge[1].startswith("TN"):
            if edge[1] not in pos:
                if edge[1][-1] == "0":
                    pos[edge[1]] = (
                        pos[edge[0]][0] - horizontal_spacing,
                        pos[edge[0]][1],
                    )
                else:
                    pos[edge[1]] = (
                        pos[edge[0]][0] + horizontal_spacing,
                        pos[edge[0]][1],
                    )

    # Draw the graph
    plt.figure(figsize=(x_len, y_len))

    # Separate tensor and index nodes
    tensor_nodes = [node for node in G.nodes if node.startswith("TN")]

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=tensor_nodes,
        node_size=node_size,
        node_color="hotpink",
        label="Tensors",
    )

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=False)

    # Add edge labels
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_color="red", font_size=8
    )

    # Title and axis
    plt.title("Tensor Network Visualisation", fontsize=14)
    plt.show()


def draw_mpo(
    tensors: list[Tensor],  # type: ignore
    node_size: int | None = None,
    x_len: int | None = None,
    y_len: int | None = None,
):
    """
    Visualise the MPO.

    Args:
        mpo: MPO object containing tensors with indices.
        node_size (int): Size of the nodes in the plot.
        x_len (int): Length of the x-axis.
        y_len (int): Length of the y-axis.
    """
    if not x_len:
        x_len = int(np.sqrt(len(tensors))) * 5
    if not y_len:
        y_len = x_len / 2
    if not node_size:
        node_size = x_len * 5

    # Build the graph
    G = build_graph_from_MPO(tensors)

    # Define positions for tensors and dangling indices
    pos = {}
    vertical_spacing = 1.0
    horizontal_spacing = 1.0

    # Assign positions for tensor nodes
    tensors = [node for node in G.nodes if node.startswith("Tensor")]
    for i, tensor in enumerate(tensors):
        pos[tensor] = (0, -i * vertical_spacing)

    # Assign positions for dangling indices
    for edge in G.edges(data=True):
        if edge[1].startswith("R"):
            pos[edge[1]] = (pos[edge[0]][0] + horizontal_spacing, pos[edge[0]][1])
        elif edge[1].startswith("L"):
            pos[edge[1]] = (pos[edge[0]][0] - horizontal_spacing, pos[edge[0]][1])

    # Draw the graph
    plt.figure(figsize=(x_len, y_len))

    # Separate tensor and index nodes
    tensor_nodes = [node for node in G.nodes if node.startswith("Tensor")]

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=tensor_nodes,
        node_size=node_size,
        node_color="hotpink",
        label="Tensors",
    )

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=False)

    # Add edge labels
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_color="red", font_size=8
    )

    # Title and axis
    plt.title("MPO Visualisation", fontsize=14)
    plt.show()


def draw_mps(
    tensors: list[Tensor],  # type: ignore
    node_size: int | None = None,
    x_len: int | None = None,
    y_len: int | None = None,
):
    """
    Visualise the MPS.

    Args:
        mps: MPS object containing tensors with indices.
        node_size (int): Size of the nodes in the plot.
        x_len (int): Length of the x-axis.
        y_len (int): Length of the y-axis.
    """
    if not x_len:
        x_len = int(np.sqrt(len(tensors))) * 5
    if not y_len:
        y_len = x_len / 2
    if not node_size:
        node_size = x_len * 5

    # Build the graph
    G = build_graph_from_MPS(tensors)

    # Define positions for tensors and dangling indices
    pos = {}
    vertical_spacing = 1.0
    horizontal_spacing = 1.0

    # Assign positions for tensor nodes
    tensors = [node for node in G.nodes if node.startswith("Tensor")]
    for i, tensor in enumerate(tensors):
        pos[tensor] = (0, -i * vertical_spacing)

    # Assign positions for dangling indices
    for edge in G.edges(data=True):
        if edge[1].startswith("P"):
            pos[edge[1]] = (pos[edge[0]][0] + horizontal_spacing, pos[edge[0]][1])

    # Draw the graph
    plt.figure(figsize=(x_len, y_len))

    # Separate tensor and index nodes
    tensor_nodes = [node for node in G.nodes if node.startswith("Tensor")]

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=tensor_nodes,
        node_size=node_size,
        node_color="skyblue",
        label="Tensors",
    )

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=False)

    # Add edge labels
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_color="red", font_size=8
    )

    # Title and axis
    plt.title("MPS Visualisation", fontsize=14)
    plt.show()


def draw_arbitrary_tn(
    tensors: list[Tensor],  # type: ignore
    node_size: int | None = None,
    x_len: int | None = None,
    y_len: int | None = None,
):
    """
    Visualise an arbitrary tensor network using a default layout from NetworkX.

    Args:
        tn: A TensorNetwork object containing tensors and indices.
        node_size (int): Size of the nodes in the plot.
        x_len (int): Length of the x-axis.
        y_len (int): Length of the y-axis.
    """
    # Build the graph from the tensor network
    G = build_graph_from_tensor_network(tensors)

    # Use default spring layout for positioning
    pos = nx.spring_layout(G)

    # Draw the graph
    plt.figure(figsize=(x_len, y_len))

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_size=node_size, node_color="hotpink", label="Tensors"
    )

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=False)

    # Add node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

    # Add edge labels
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_color="red", font_size=8
    )

    # Title and axis
    plt.title("Arbitrary Tensor Network Visualisation", fontsize=14)
    plt.show()
