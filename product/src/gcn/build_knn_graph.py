from sklearn.neighbors import kneighbors_graph
import numpy as np
import torch
from torch_geometric.utils import from_scipy_sparse_matrix

def build_knn_graph(
    features: np.ndarray,
    k: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Builds a K-Nearest Neighbors (KNN) graph from feature vectors.

    This construction is fully inductive, meaning each participant's neighbors 
    are determined solely by their own feature vectors. Unlike population-based 
    graphs, this does not utilise phenotypic information or population-level 
    statistics.

    The function uses cosine similarity to define connectivity and symmetrises 
    the resulting adjacency matrix to ensure an undirected graph.

    Args:
        features: Feature matrix of shape (N, C), typically representing 
            latent embeddings from an Autoencoder.
        k: Number of nearest neighbors to identify for each node.

    Returns:
        A tuple of (edge_index, edge_attr):
            edge_index: Long tensor of shape (2, E) representing graph edges.
            edge_attr: Float tensor of shape (E,) representing similarity weights 
                derived from (1 - cosine_distance).
    """
    # Build KNN graph using cosine similarity
    A = kneighbors_graph(
        features,
        n_neighbors=k,
        metric='cosine',
        mode='distance',
        include_self=False,
    )

    # Symmetrise: if node A is a neighbor of B, or B is a neighbor of A,
    # they are connected in the undirected graph.
    A = A.maximum(A.T)

    # Convert distance to similarity: 1.0 - (1.0 - cos_sim) = cos_sim
    A.data = 1.0 - A.data

    edge_index, edge_attr = from_scipy_sparse_matrix(A)
    return edge_index, edge_attr.float()