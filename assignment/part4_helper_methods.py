import networkx as nx
import numpy as np

def load_graph(filename):
    """
        Load a directed graph from a file with integer nodes.
    """
    G = nx.DiGraph()
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            src = int(parts[0])
            for dst in parts[1:]:
                G.add_edge(src, int(dst))
    return G

def load_graph2(filename):
    """
        Load a directed graph from a file with string nodes.
    """
    G = nx.DiGraph()
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            src = str(parts[0])
            for dst in parts[1:]:
                G.add_edge(src, str(dst))
    return G

def load_graph_adj_matrix(filename):
    """
        Loads the graph from an adjacency list file with integer nodes.
    """
    G = nx.DiGraph()
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            src = int(parts[0])
            for dst_str in parts[1:]:
                dst = int(dst_str)
                G.add_edge(src, dst)
    return G

def load_titles(filename):
    """
        Load mapping: integer index => title (string).
    """
    titles = {}
    with open(filename, 'r', encoding='utf-8') as file:
        next(file)  # Skip the first line (it's the header)
        for line in file:
            idx, title = line.strip().split('\t')
            titles[int(idx)] = title
    return titles


def build_google_matrix(G, alpha=0.85):
    """
        Build the row-stochastic Google matrix G_matrix
        from the directed graph G.

        Warning: This method does not copy the original graph G!
    """
    nodes = sorted(G.nodes())
    N = len(nodes)

    # Create adjacency matrix A where A[i,j] = 1 if edge (i->j) else 0
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=float)

    # Out-degrees for each node
    out_degree = A.sum(axis=1)

    # Build row-stochastic transition matrix H
    H = np.zeros((N, N), dtype=float)
    for i in range(N):
        if out_degree[i] > 0:
            H[i, :] = A[i, :] / out_degree[i]
        else:
            # Dangling node => entire row is 1/N
            H[i, :] = 1.0 / N

    # Google matrix:
    G_matrix = alpha * H + (1.0 - alpha) / N * np.ones((N, N))

    return G_matrix, nodes

def power_iteration(G, alpha=0.85, tol=1e-6, max_iter=100):
    """
        Compute PageRank via power iteration using row-stochastic Google matrix.
        pi is treated as a row vector: pi_{new} = pi_{old} * G_matrix.
    """
    G_matrix, nodes = build_google_matrix(G, alpha)
    N = len(nodes)

    # Initialize pi as uniform row-vector
    pi = np.ones(N) / N

    for _ in range(max_iter):
        pi_new = pi @ G_matrix
        if np.linalg.norm(pi_new - pi, ord=1) < tol:
            break
        pi = pi_new

    # Build dict mapping node -> page rank
    page_rank_dict = {node: pi[i] for i, node in enumerate(nodes)}
    return page_rank_dict

def game_pagerank(G, target_node, edges_budget=300, alpha=0.85):
    """
        1) Get baseline PageRank
        2) Add up to edges_budget new edges from top-ranked nodes to target_node
        3) Recompute PageRank
    """
    # Baseline PageRank
    initial_scores = power_iteration(G, alpha)
    sorted_nodes = sorted(initial_scores.items(), key=lambda x: x[1], reverse=True)

    # Add edges to target_node
    edges_added = 0
    for node, _ in sorted_nodes:
        if edges_added >= edges_budget:
            break
        if node != target_node and not G.has_edge(node, target_node):
            G.add_edge(node, target_node)
            edges_added += 1

    # Recompute PageRank
    improved_scores = power_iteration(G, alpha)
    return initial_scores[target_node], improved_scores[target_node]


