# Generate all possible edges for a given regularity
# Code was lovingly inspired by https://math.stackexchange.com/questions/142112/how-to-construct-a-k-regular-graph
import networkx as nx
from matplotlib import pyplot as plt


def make_edge_for_each_vertex_n_steps_away(edges, num_nodes, n):
    for i in range(num_nodes):
        u, v = i, (i + n) % num_nodes
        if (u, v) not in edges and (v, u) not in edges:
            edges.add((u, v, 1.0))
    return edges


def generate_regular_graph_edges(n, num_nodes):
    edges = set()
    if n == 1:
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 4)
    elif n == 2:
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 1)
    elif n == 3:
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 4)
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 3)
    elif n == 4:
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 1)
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 2)
    elif n == 5:
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 4)
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 3)
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 2)
    elif n == 6:
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 1)
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 2)
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 3)
    elif n == 7:
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 4)
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 3)
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 2)
        edges = make_edge_for_each_vertex_n_steps_away(edges, num_nodes, 1)
    return edges


if __name__ == "__main__":
    edges = generate_regular_graph_edges(3, 8)
    G = nx.Graph()
    nodes = list(range(8))
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True, node_color='white', edge_color='black', node_size=500, font_size=16)
    plt.show()
