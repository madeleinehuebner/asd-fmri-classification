"""
Training logic for ResidualGCN.

This module contains the training loop for the Residual Graph Convolutional
Network. It handles the transductive learning process where the model performs
message passing over the entire graph while restricting loss computation and
evaluation to specific node masks.
"""

import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from torchmetrics.functional import auroc

from gcn.model import ResidualGCN


def train_gcn(
    model: ResidualGCN,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    factor: float,
    scheduler_patience: int,
    device: torch.device = torch.device('cpu'),
) -> tuple:
    """
    Trains a ResidualGCN.

    Args:
        model: The ResidualGCN architecture to be trained.
        node_features: Node feature matrix with shape (N, in_dim).
        edge_index: Graph connectivity in COO format with shape (2, E).
        edge_attr: Edge weight/attribute tensor with shape (E,).
        labels: Ground truth labels of shape (N,).
        train_mask: Boolean mask indicating nodes used for backpropagation.
        val_mask: Boolean mask indicating nodes used for model selection.
        lr: Initial learning rate for the Adam optimiser.
        weight_decay: L2 regularisation coefficient.
        epochs: Maximum number of training iterations.
        patience: Number of epochs to wait for AUC improvement before 
            terminating.
        factor: Multiplicative factor by which the learning rate will 
            be reduced via ReduceLROnPlateau.
        scheduler_patience: Number of epochs with no AUC improvement after 
            which learning rate will be reduced.
        device: The torch device to perform computations on.

    Returns:
        A tuple containing:
            - best_epoch (int): The epoch index where peak validation AUC was achieved.
            - stopped_epoch (int): The epoch where training actually terminated.
            - best_auc (float): The maximum validation AUC achieved.
            - history (dict): A dictionary containing 'train_loss', 'val_loss', 
              and 'val_auc' sequences.
    """
    optimiser = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='max', factor=factor, patience=scheduler_patience,
    )

    best_auc         = 0.0
    best_state       = None
    best_epoch       = 0
    patience_counter = 0

    # Calculate class weights
    n_train = train_mask.sum()
    n_pos = labels[train_mask].sum()
    n_neg = n_train - n_pos
    weights = torch.tensor(
        [n_train / (2 * n_neg), n_train / (2 * n_pos)],
        dtype=torch.float32, device=device,
    )
    
    train_idx = torch.where(train_mask)[0]
    val_idx = torch.where(val_mask)[0]
    
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    
    # Restrict training edges to avoid leakage from validation set
    train_edges, train_attr = subgraph(
        train_mask, edge_index, edge_attr, relabel_nodes=False
    )
    # Use both train and val nodes for message passing during evaluation
    eval_mask = train_mask | val_mask
    eval_edges, eval_attr = subgraph(
        eval_mask, edge_index, edge_attr, relabel_nodes=False
    )

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        optimiser.zero_grad()
        out = model(node_features, train_edges, train_attr)
        loss = F.nll_loss(out[train_idx], labels[train_idx], weight=weights)
        loss.backward()
        optimiser.step()
        train_loss = loss.item()

        improved = False
        # Evaluation phase
        if epoch % 2 == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                out_eval = model(node_features, eval_edges, eval_attr)
                val_logits = out_eval[val_idx]

                probs_val = val_logits.exp()[:, 1]
                auc = auroc(probs_val, labels[val_idx], task='binary').item()

                scheduler.step(auc)

                if auc > best_auc:
                    best_auc = auc
                    best_epoch = epoch
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    improved = True
                    
        # Early stopping
        if improved:
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break
    
    if best_state is None:
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    stopped_epoch = epoch

    return best_epoch, stopped_epoch, best_auc, history