import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from torch_geometric.utils import dense_to_sparse

from gcn.build_population_graph import build_population_graph, threshold_graph


class PopulationGraphTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that constructs a population graph from node features and metadata.

    This class computes a similarity matrix based on feature correlation and 
    phenotypic metadata, thresholds the connections to enforce sparsity.
    
    Converts the result into a PyTorch Geometric compatible format.

    Attributes:
        graph_data_ (dict): A dictionary containing:
            - 'x': Node feature tensor of shape (N, D).
            - 'edge_index': Graph connectivity in COO format (2, E).
            - 'edge_attr': Sparse edge weights (E,).
    """
    def __init__(self, phenotypic_scores, threshold_percentile=70.0, id_col='ID', age_threshold=2.0, sigma=None):
        self.phenotypic_scores = phenotypic_scores
        self.threshold_percentile = threshold_percentile
        self.id_col = id_col
        self.age_threshold = age_threshold
        self.sigma = sigma

    def fit(self, X, y=None, **kwargs):

        metadata = kwargs.get('metadata')
        if metadata is None:
            raise ValueError("Metadata required for GCN graph construction.")

        train_mask = kwargs.get('train_mask')
        if train_mask is not None:
            if isinstance(train_mask, torch.Tensor):
                train_mask_np = train_mask.cpu().numpy()
            else:
                train_mask_np = np.asarray(train_mask, dtype=bool)
            train_features = X[train_mask_np]
        else:
            train_features = X

        # Compute sigma from training nodes only
        if self.sigma is None:
            dist = cdist(train_features, train_features, metric='correlation')
            sigma_fit = float(np.mean(dist))
        else:
            sigma_fit = self.sigma

        # Build graph over all nodes using training-derived sigma
        W = build_population_graph(
            features=X,
            metadata=metadata,
            phenotypic_scores=self.phenotypic_scores,
            id_col=self.id_col,
            age_threshold=self.age_threshold,
            sigma=sigma_fit,
        )
        W_thresh = threshold_graph(W, percentile=self.threshold_percentile)
        edge_index, edge_attr = dense_to_sparse(torch.tensor(W_thresh, dtype=torch.float32))

        self.graph_data_ = {
            "x": torch.as_tensor(X, dtype=torch.float32),
            "edge_index": edge_index,
            "edge_attr": edge_attr.clamp(min=0.0),
        }
        return self

    def transform(self, X):
        """Return the cached population graph built during fit."""
        check_is_fitted(self, 'graph_data_')
        return self.graph_data_
