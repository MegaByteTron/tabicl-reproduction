"""scikit-learn style wrapper around NanoTabICLv2.

Adapted from ``NanoTabPFNClassifier`` in nanoTabPFN
(https://github.com/automl/nanoTabPFN/blob/main/model.py, Apache-2.0),
Pfefferle et al., "nanoTabPFN: A Lightweight and Educational
Reimplementation of TabPFN", arXiv:2511.03634, 2025.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin

from .model import NanoTabICLv2


class NanoTabICLv2Classifier(ClassifierMixin, BaseEstimator):
    """In-context classifier: ``fit`` only stores the training rows, the
    pretrained model's weights are never changed.

    Args:
        model: A pretrained classification model (``max_classes > 0``).
        device: If given, ``fit`` moves the model to this device (e.g.
            ``"cuda"``). ``None`` leaves the model where it is; use that for
            the live model during training, so the classifier never moves it
            away from the optimizer.
    """

    def __init__(self, model: NanoTabICLv2, device: str | torch.device | None = None):
        self.model = model
        self.device = device

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Store the training rows. Labels must already be integers 0..k-1
        (done in preprocessing)."""
        self.num_classes = max(set(y_train)) + 1
        if self.device is not None:
            self.model.to(self.device)
        self.X_train_ = np.asarray(X_train, dtype=np.float32)
        self.y_train_ = np.asarray(y_train, dtype=np.int64)
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Class probabilities, one column per class 0..num_classes-1."""
        device = next(self.model.parameters()).device
        X_train_test = np.concatenate(
            (self.X_train_, np.asarray(X_test, dtype=np.float32))
        )
        X_train_test = torch.from_numpy(X_train_test).unsqueeze(0).to(device)
        y_train = torch.from_numpy(self.y_train_).unsqueeze(0).to(device, torch.long)

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                # squeeze removes the batch dimension of size 1
                out = self.model(X_train_test, y_train).squeeze(0)
        finally:
            self.model.train(was_training)

        # the model has max_classes outputs; cut off the classes not in y_train
        out = out[:, : self.num_classes]
        return F.softmax(out, dim=1).cpu().numpy()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Most likely class index."""
        return self.predict_proba(X_test).argmax(axis=1)
