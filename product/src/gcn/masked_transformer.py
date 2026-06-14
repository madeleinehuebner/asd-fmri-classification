import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin, clone


class MaskedTransformer(BaseEstimator, TransformerMixin):
    """
    Wraps any sklearn transformer so it fits only on training-mask nodes
    when used in a transductive GCN pipeline.

    In GCN pipelines, the full dataset X is passed to Pipeline.fit so that
    the graph can be built over all nodes. Without masking, supervised steps
    like SelectKBest would see test-set labels during feature selection, which is
    a form of data leakage. This wrapper intercepts the train_mask fit_param
    and restricts fitting to training nodes only, while transform still
    operates on the full array.

    Args:
        transformer: Any sklearn-compatible transformer, e.g. SelectKBest.
    """

    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None, train_mask=None):
        if train_mask is not None:
            if isinstance(train_mask, torch.Tensor):
                train_mask = train_mask.cpu().numpy()
            mask = np.asarray(train_mask, dtype=bool)
            y_fit = y[mask] if y is not None else None
            self.transformer_ = clone(self.transformer).fit(X[mask], y_fit)
        else:
            self.transformer_ = clone(self.transformer).fit(X, y)
        return self

    def transform(self, X):
        return self.transformer_.transform(X)

    def get_params(self, deep=True):
        params = {'transformer': self.transformer}
        if deep:
            for key, val in self.transformer.get_params(deep=True).items():
                params[f'transformer__{key}'] = val
        return params

    def set_params(self, **params):
        transformer_params = {}
        own_params = {}
        for key, val in params.items():
            if key.startswith('transformer__'):
                transformer_params[key[len('transformer__'):]] = val
            else:
                own_params[key] = val
        if transformer_params:
            self.transformer.set_params(**transformer_params)
        return super().set_params(**own_params)
