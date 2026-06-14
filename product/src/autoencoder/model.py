import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedDenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder for discriminative latent feature extraction.

    Architecture:
        - Encoder: Corrupting Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear -> BN.
        - Decoder: Linear -> BN -> ReLU -> Dropout -> Linear.
        - Classifier: Linear layer mapping latent space to class logits.

    Args:
        input_dim: Dimensionality of raw input features.
        hidden_dim: Size of the intermediate hidden layers.
        latent_dim: Size of the bottleneck representation.
        input_dropout: Corruption rate applied to raw inputs.
        hidden_dropout: Dropout rate applied within encoder/decoder layers.
        alpha: Weighting factor for classification loss: Total = MSE + alpha * CE.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        input_dropout: float,
        hidden_dropout: float,
        alpha: float,
    ):
        super().__init__()
        self.alpha = alpha

        self.encoder = nn.Sequential(
            nn.Dropout(p=input_dropout),
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=hidden_dropout),
            nn.Linear(hidden_dim, latent_dim, bias=False),
            nn.BatchNorm1d(latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=hidden_dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        self.classifier = nn.Linear(latent_dim, 2)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialises weights using Xavier uniform and biases to zero."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass through encoder, decoder, and classifier.

        Returns:
            z: Latent representation, shape (N, latent_dim).
            x_hat: Reconstruction, shape (N, input_dim).
            logits: Class logits, shape (N, 2).
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        logits = self.classifier(z)
        return z, x_hat, logits

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Combined reconstruction and classification loss.

        total = MSE(x, x_hat) + alpha * CrossEntropy(logits, y)

        Returns:
            total: Scalar total loss.
            recon_loss: MSE reconstruction loss (for logging).
            clf_loss: Cross-entropy classification loss (for logging).
        """
        z, x_hat, logits = self(x)
        recon_loss = F.mse_loss(x_hat, x)
        clf_loss = F.cross_entropy(logits, y)
        total = recon_loss + self.alpha * clf_loss
        return total, recon_loss, clf_loss

    @torch.no_grad()
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Note: Ensure model is in .eval() mode to disable input corruption.
        """
        return self.encoder(x)
