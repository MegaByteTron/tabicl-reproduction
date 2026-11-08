"""Inference-only sklearn-like wrapper; fit never trains the neural network."""

from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from tabicl_reproduction.model.model import NanoTabICLv2


class Classifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        model: NanoTabICLv2,
        device: str = "cpu",
        batch_size: int = 256,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size

    @staticmethod
    def _features(X: Any) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            result = X.detach().to(device="cpu", dtype=torch.float32).clone()
        else:
            result = torch.tensor(np.asarray(X, dtype=np.float32))
        if result.ndim != 2 or result.shape[1] == 0:
            raise ValueError("X must be a 2D numeric table with at least one feature")
        if torch.isinf(result).any():
            raise ValueError("X contains infinity; clean it before prediction")
        return result

    def fit(self, X_train: Any, y_train: Any):
        X = self._features(X_train)
        labels = (
            y_train.detach().cpu().numpy()
            if isinstance(y_train, torch.Tensor)
            else np.asarray(y_train)
        )
        if labels.ndim != 1 or len(labels) != len(X) or len(X) < 2:
            raise ValueError("Need at least two training rows and one label per row")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.classes_, encoded = np.unique(labels, return_inverse=True)
        capacity = self.model.y_embed_in.in_features
        output_size = self.model.out_mlp[-1].out_features
        if len(self.classes_) > min(capacity, output_size):
            raise ValueError(
                f"{len(self.classes_)} classes exceed this model's capacity "
                f"({min(capacity, output_size)}); many-class wrappers are not implemented"
            )
        self.n_features_in_ = X.shape[1]
        self.imputation_ = torch.nanmean(X, dim=0).nan_to_num(0.0)
        self.X_train_ = torch.where(X.isnan(), self.imputation_, X)
        self.y_train_ = torch.from_numpy(encoded).long()
        return self

    # RoPE caches tensors inside the model. inference_mode would create cached
    # tensors that cannot later participate in training's backward pass.
    @torch.no_grad()
    def predict_proba(self, X_test: Any) -> np.ndarray:
        check_is_fitted(self, ["X_train_", "classes_"])
        X = self._features(X_test)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Train and test feature counts differ")
        if len(X) == 0:
            return np.empty((0, len(self.classes_)))
        if len(self.classes_) == 1:
            return np.ones((len(X), 1))
        X = torch.where(X.isnan(), self.imputation_, X)
        model = self.model.to(self.device)
        was_training = model.training
        model.eval()
        parts = []
        try:
            train = self.X_train_.to(self.device)
            labels = self.y_train_.to(self.device).unsqueeze(0)
            for test in X.split(self.batch_size):
                inputs = torch.cat([train, test.to(self.device)]).unsqueeze(0)
                logits = model(inputs, labels)[0, :, : len(self.classes_)]
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Model produced non-finite logits")
                parts.append(logits.softmax(-1).cpu().numpy())
        finally:
            model.train(was_training)
        return np.concatenate(parts)

    def predict(self, X_test: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(X_test).argmax(axis=-1)]
