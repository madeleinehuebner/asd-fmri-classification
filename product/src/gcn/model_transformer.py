import torch
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from gcn.model import ResidualGCN
from gcn.train_model import train_gcn

class GCNTransformer(BaseEstimator, ClassifierMixin):
    """
    An sklearn-compatible wrapper for the ResidualGCN model.

    This estimator allows a PyTorch Geometric-based GCN to be used within 
    standard sklearn workflows, such as Pipelines or Cross-Validation. 
    It supports transductive learning where node features and graph 
    connectivity are provided during the fit and predict phases.

    Attributes:
        model_ (ResidualGCN): The underlying PyTorch model, initialized during fit.
        classes_ (ndarray): The unique class labels found during fit.
        best_auc_ (float): The peak validation AUC achieved during training.
        history_ (dict): Training logs including loss and metric evolution.
    """
    def __init__(
        self,
        hidden_dim=64,
        dropout=0.5,
        lr=0.005,
        weight_decay=0.0005,
        epochs=300,
        patience=50,
        factor=0.5,
        scheduler_patience=20,
        device='cpu',
        verbose=False
    ):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.factor = factor
        self.scheduler_patience = scheduler_patience
        self.device = device
        self.verbose = verbose

    def fit(self, X, y, train_mask=None, val_mask=None, test_mask=None, **kwargs):
        """
        Fits the ResidualGCN model to the provided graph data.

        Args:
            X (dict or torch.Tensor): The input data. 
                If a dict, must contain x (features) and edge_index. 
                edge_attr is optional.
                If a Tensor, represents the node features; edge_index
                must be provided in kwargs.
            y (array-like): Target node labels of shape (n_nodes,).
            train_mask (torch.Tensor, optional): Boolean mask for training nodes.
            val_mask (torch.Tensor, optional): Boolean mask for validation nodes.
            test_mask (torch.Tensor, optional): Boolean mask for test nodes.
            **kwargs: Additional graph parameters if X is a raw feature tensor.

        Returns:
            self: The fitted estimator.
        """
        if isinstance(X, dict):
            node_features = X['x'].to(self.device)
            edge_index = X['edge_index'].to(self.device)
            edge_attr = X['edge_attr'].to(self.device) if 'edge_attr' in X else None
        else:
            node_features = torch.as_tensor(X, dtype=torch.float32).to(self.device)
            edge_index = kwargs.get('edge_index').to(self.device)
            edge_attr = kwargs.get('edge_attr')

        self.edge_index_ = edge_index
        self.edge_attr_ = edge_attr

        y_tensor = torch.as_tensor(y, dtype=torch.long).to(self.device)

        # Handle Masks
        N = node_features.shape[0]
        if train_mask is None:
            train_mask = torch.ones(N, dtype=torch.bool, device=self.device)
        if val_mask is None:
            val_mask = train_mask # Fallback
        if test_mask is None:
            test_mask = train_mask # Fallback

        # Initialise model
        self.model_ = ResidualGCN(
            input_dim=node_features.shape[1],
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        ).to(self.device)

        self.best_epoch_, self.stopped_epoch_, self.best_auc_, self.history_ = train_gcn(
            model=self.model_,
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            labels=y_tensor,
            train_mask=train_mask,
            val_mask=val_mask,
            lr=self.lr,
            weight_decay=self.weight_decay,
            epochs=self.epochs,
            patience=self.patience,
            factor=self.factor,
            scheduler_patience=self.scheduler_patience,
            device=torch.device(self.device)
        )
        
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        """
        Returns class probabilities for the nodes in X.

        Args:
            X (dict or torch.Tensor): Input data structured similarly to fit(). 
                If X is a Tensor and the model was fitted with a graph, it 
                reuses the fitted 'edge_index'.

        Returns:
            ndarray: Array of shape (n_nodes, n_classes) with class probabilities.
        """
        check_is_fitted(self)
        self.model_.eval()
        
        # Unpack X
        with torch.no_grad():
            if isinstance(X, dict):
                feat, ei, ea = X['x'], X['edge_index'], X.get('edge_attr')
            else:
                feat, ei, ea = X, self.edge_index_, self.edge_attr_
            
            feat = torch.as_tensor(feat, dtype=torch.float32).to(self.device)
            ei = ei.to(self.device)
            ea = ea.to(self.device) if ea is not None else None
            
            out = self.model_(feat, ei, ea)
            return torch.exp(out).cpu().numpy()

    def predict(self, X):
        """
        Predicts the class label for each node in X.

        Args:
            X (dict or torch.Tensor): Input data structured similarly to fit().

        Returns:
            ndarray: Predicted class labels of shape (n_nodes,).
        """
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)