"""
Training logic for Supervised Denoising Autoencoder.
"""

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from autoencoder.model import SupervisedDenoisingAutoencoder


def train_phase1(
    model: SupervisedDenoisingAutoencoder,
    train_loader: DataLoader,
    epochs: int,
    alpha: float,
    lr: float,
    weight_decay: float,
    clf_weight_decay: float,
    verbose: bool = True,
    device: torch.device = torch.device('cpu'),
) -> SupervisedDenoisingAutoencoder:
    """
    Supervised and unsupervised training.

    The model learns to reconstruct corrupted inputs (MSE) while simultaneously 
    minimising classification error. This phase forces the 
    bottleneck to capture features that are both structurally representative 
    and class-discriminative.

    Args:
        model: The Autoencoder instance to train.
        train_loader: DataLoader providing features, labels.
        epochs: Number of training iterations.
        alpha: Weight for the classification loss component.
        lr: Learning rate for the Adam optimizer.
        weight_decay: L2 penalty for encoder and decoder.
        clf_weight_decay: Separate L2 penalty for the classification head.
        verbose: If True, prints epoch-wise loss statistics.
        device: Torch device for computation.

    Returns:
        The trained model with optimised weights.
    """
    model = model.to(device)
    model.alpha = alpha

    optimiser = optim.Adam([
        {'params': model.encoder.parameters(),    'weight_decay': weight_decay},
        {'params': model.decoder.parameters(),    'weight_decay': weight_decay},
        {'params': model.classifier.parameters(), 'weight_decay': clf_weight_decay},
    ], lr=lr)

    if verbose:
        print("\nPhase 1: supervised training")
        print(f"{'Epoch':>6} | {'Train Total':>11} | {'Train Recon':>11} | {'Train Clf':>9}")
        print("-" * 48)

    for epoch in range(1, epochs + 1):
        model.train()
        totals, recons, clfs = [], [], []

        for x_batch, y_batch in train_loader:
            optimiser.zero_grad()
            total, recon, clf = model.loss(x_batch, y_batch)
            total.backward()
            optimiser.step()
            totals.append(total.item())
            recons.append(recon.item())
            clfs.append(clf.item())

        if verbose and epoch % 5 == 0:
            print(f"{epoch:>6} | {sum(totals)/len(totals):>11.6f} "
                  f"| {sum(recons)/len(recons):>11.6f} "
                  f"| {sum(clfs)/len(clfs):>9.6f}")

    return model


def train_phase2(
    model: SupervisedDenoisingAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    factor: int,
    scheduler_patience: int,
    verbose: bool = True,
    device: torch.device = torch.device('cpu'),
) -> SupervisedDenoisingAutoencoder:
    """
    Unsupervised fine-tuning on reconstruction loss.

    The classifier is frozen. The encoder and decoder are fine-tuned with 
    early stopping on validation reconstruction loss. This allows the encoder 
    to converge without the classifier continuing to overfit.

    Args:
        model: The DAE instance, pre-trained from phase 1.
        train_loader: DataLoader for training features.
        val_loader: DataLoader for validation features.
        epochs: Maximum number of fine-tuning iterations.
        patience: Epochs to wait for validation improvement before stopping.
        lr: Learning rate for the Adam optimiser.
        weight_decay: L2 penalty for encoder and decoder.
        factor: Learning rate reduction factor on plateau.
        scheduler_patience: Epochs to wait before reducing the learning rate.
        verbose: If True, prints reconstruction loss metrics.
        device: Torch device for computation.

    Returns:
        A tuple containing (trained_model, stopped_epoch), where the model 
        has the best validation weights loaded.
    """
    # Freeze classifier and disable supervised loss
    for param in model.classifier.parameters():
        param.requires_grad = False
    model.alpha = 0.0

    optimiser = optim.Adam([
        {'params': model.encoder.parameters(), 'weight_decay': weight_decay},
        {'params': model.decoder.parameters(), 'weight_decay': weight_decay},
    ], lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=factor, patience=scheduler_patience,
    )

    if verbose:
        print("\nPhase 2: unsupervised fine-tuning")
        print(f"{'Epoch':>6} | {'Train Recon':>11} | {'Val Recon':>9}")
        print("-" * 32)

    best_val_recon   = float('inf')
    best_state       = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        running_train_recon = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimiser.zero_grad()
            _, recon, _ = model.loss(x_batch, y_batch)
            recon.backward()
            optimiser.step()
            running_train_recon += recon.item()

        # Validate
        if epoch % 2 == 0 or epoch == epochs:
            model.eval()
            running_val_recon = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    _, recon, _ = model.loss(x_batch, y_batch)
                    running_val_recon += recon.item()

            train_recon = running_train_recon / len(train_loader)
            val_recon = running_val_recon / len(val_loader)
            scheduler.step(val_recon)

            if verbose and epoch % 10 == 0:
                print(f"{epoch:>6} | {train_recon:>11.6f} | {val_recon:>9.6f}")

            # Early stopping
            if val_recon < best_val_recon:
                best_val_recon   = val_recon
                best_state       = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"\n  Early stopping at epoch {epoch}. "
                            f" Best val recon: {best_val_recon:.6f}")
                    break

    model.load_state_dict(best_state)
    return model, epoch, best_val_recon, train_recon