import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from autoencoder.model import SupervisedDenoisingAutoencoder
from autoencoder.train_model import train_phase1, train_phase2

class SDAETransformer(BaseEstimator, TransformerMixin):
    """
    An sklearn-compatible transformer for Supervised Denoising Autoencoders (SDAE).

    This transformer wraps the two-phase SDAE training logic, allowing for joint 
    reconstruction-classification learning followed by unsupervised refinement. 
    It transforms high-dimensional input features into a lower-dimensional, 
    class-discriminative latent space.

    Attributes:
        model (SupervisedDenoisingAutoencoder): The underlying PyTorch model 
            instance, initialised during fit.
        params (dict): A compiled dictionary of hyperparameters used for 
            model initialisation and optimisation.
    """
    def __init__(
        self,
        ae_hidden_dim: int = 512,
        ae_latent_dim: int = 128,
        ae_input_dropout: float = 0.1,
        ae_hidden_dropout: float = 0.1,
        alpha: float = 0.1,
        ae1_epochs: int = 50,
        ae1_lr: float = 1e-3,
        ae1_weight_decay: float = 1e-4,
        ae1_clf_weight_decay: float = 0.007,
        ae2_epochs: int = 200,
        ae2_patience: int = 15,
        ae2_lr: float = 5e-4,
        ae2_weight_decay: float = 0.0001,
        ae2_factor: float = 0.5,
        ae2_scheduler_patience: int = 10,
        device: str = "mps",
        verbose: bool = False,
    ):
        self.ae_hidden_dim = ae_hidden_dim
        self.ae_latent_dim = ae_latent_dim
        self.ae_input_dropout = ae_input_dropout
        self.ae_hidden_dropout = ae_hidden_dropout
        self.alpha = alpha
        self.ae1_epochs = ae1_epochs
        self.ae1_lr = ae1_lr
        self.ae1_weight_decay = ae1_weight_decay
        self.ae1_clf_weight_decay = ae1_clf_weight_decay
        self.ae2_epochs = ae2_epochs
        self.ae2_patience = ae2_patience
        self.ae2_lr = ae2_lr
        self.ae2_weight_decay = ae2_weight_decay
        self.ae2_factor = ae2_factor
        self.ae2_scheduler_patience = ae2_scheduler_patience
        self.device = device
        self.verbose = verbose
        
        self._rebuild_params_dict()
    
    def _rebuild_params_dict(self):
        """
        Synchronises internal class attributes into a parameter dictionary 
        for the model and training modules.
        """
        self.params = {
            "ae_hidden_dim": self.ae_hidden_dim,
            "ae_latent_dim": self.ae_latent_dim,
            "ae_input_dropout": self.ae_input_dropout,
            "ae_hidden_dropout": self.ae_hidden_dropout,
            "alpha": self.alpha,
            "ae1_epochs": self.ae1_epochs,
            "ae1_lr": self.ae1_lr,
            "ae1_weight_decay": self.ae1_weight_decay,
            "ae1_clf_weight_decay": self.ae1_clf_weight_decay,
            "ae2_epochs": self.ae2_epochs,
            "ae2_patience": self.ae2_patience,
            "ae2_lr": self.ae2_lr,
            "ae2_weight_decay": self.ae2_weight_decay,
            "ae2_factor": self.ae2_factor,
            "ae2_scheduler_patience": self.ae2_scheduler_patience,
        }

    def fit(self, X, y=None, **kwargs):
        train_mask = kwargs.get('train_mask')
        if train_mask is not None:
            if isinstance(train_mask, torch.Tensor):
                train_mask = train_mask.cpu().numpy()
            train_mask = np.asarray(train_mask, dtype=bool)
            X = X[train_mask]
            if y is not None:
                y = y[train_mask]
        
        hpo_mode = kwargs.get('hpo_mode', False)

        self._rebuild_params_dict()

        self.model = SupervisedDenoisingAutoencoder(
            input_dim=X.shape[1],
            hidden_dim=self.params['ae_hidden_dim'],
            latent_dim=self.params['ae_latent_dim'],
            input_dropout=self.params['ae_input_dropout'],
            hidden_dropout=self.params['ae_hidden_dropout'],
            alpha=self.params['alpha'],
        ).to(self.device)
        
        phase1_epochs = max(10, self.params['ae1_epochs'] // 3) if hpo_mode else self.params['ae1_epochs']
        phase2_epochs = max(30, self.params['ae2_epochs'] // 3) if hpo_mode else self.params['ae2_epochs']
        
        phase1_args = {
            'epochs':           phase1_epochs,
            'alpha':            self.params['alpha'],
            'lr':               self.params['ae1_lr'],
            'weight_decay':     self.params['ae1_weight_decay'],
            'clf_weight_decay': self.params['ae1_clf_weight_decay'],
            'device':           self.device,
            'verbose':          self.verbose,
        }
        
        phase2_args = {
            'epochs':             phase2_epochs,
            'patience':           self.params['ae2_patience'],
            'lr':                 self.params['ae2_lr'],
            'weight_decay':       self.params['ae2_weight_decay'],
            'factor':             self.params['ae2_factor'],
            'scheduler_patience': self.params['ae2_scheduler_patience'],
            'device':             self.device,
            'verbose':            self.verbose,
        }
        
        
        X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
        t_loader = self._make_loader(X_t, y_t)
        v_loader = self._make_loader(X_v, y_v)

        self.model = train_phase1(self.model, t_loader, **phase1_args)
        self.model, _, _, _ = train_phase2(self.model, t_loader, v_loader, **phase2_args)
        return self

    def transform(self, X):
        self.model.eval()
        with torch.no_grad():
            X_torch = torch.as_tensor(X, dtype=torch.float32).to(self.device)
            latent = self.model.encoder(X_torch)
        return latent.cpu().numpy()

    def _make_loader(self, X, y, shuffle=True):
        """
        Converts Scikit-Learn input (NumPy) into PyTorch DataLoader.
        """
        X_tensor = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.as_tensor(y, dtype=torch.long).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        
        return DataLoader(
            dataset, 
            batch_size=self.params.get('batch_size', 32), 
            shuffle=shuffle,
        )