"""Our pretrained model behind the scikit-learn classifier interface.

    clf = TabICLClassifier.from_checkpoint("outputs/runs/<date>/<time>")
    clf.fit(X_train, y_train)          # only stores the rows (the context)
    proba = clf.predict_proba(X_test)  # one forward pass per column order
    pred = clf.predict(X_test)

"Fitting" here is in-context learning: no weights change. The weights are
learned once, by pretraining on the prior (see training/). Anything that
takes a scikit-learn classifier (the experiment, show()'s DecisionBoundary)
takes this one too.
"""

from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array, column_or_1d
from sklearn.utils.validation import check_is_fitted
from torch import nn

from tabicl_reproduction.data.preprocessing import impute


class UnsupportedTableError(ValueError):
    """A table the model can't predict, e.g. more classes than it supports."""


class TabICLClassifier(ClassifierMixin, BaseEstimator):
    """Scikit-learn classifier around a pretrained NanoTabICLv2.

    Args:
        model: the pretrained network. It is used as is, not copied, so
            wrapping a model during pretraining always scores its current
            weights. The device is the one the model is on.
        max_classes: the most classes the model was pretrained for; a table
            with more raises UnsupportedTableError in fit().
        n_orders: the model depends on the column order, so the
            probabilities are averaged over this many column orders (the
            official TabICL code uses 8). The first is the original order,
            so n_orders=1 is a plain forward pass.
        random_state: seed for the extra column orders.
    """

    # The training step of the loaded weights (set by from_checkpoint).
    checkpoint_step: int | None = None

    def __init__(
        self,
        model: nn.Module | None = None,
        max_classes: int = 10,
        n_orders: int = 8,
        random_state: int = 0,
    ):
        self.model = model
        self.max_classes = max_classes
        self.n_orders = n_orders
        self.random_state = random_state

    @classmethod
    def from_checkpoint(
        cls,
        run_dir: str | Path,
        weights: str = "best",
        device: str = "cpu",
        **kwargs: Any,
    ) -> Self:
        """Load <run_dir>/<weights>.pt, built exactly as the run trained it.

        The run's saved config (.hydra/config.yaml) gives the model size.
        The returned classifier has checkpoint_step: the step of the weights.
        """
        # Imported here: training imports this module for its evaluator.
        from hydra_zen import instantiate, load_from_yaml

        from tabicl_reproduction.training.checkpoint import load_model

        config = load_from_yaml(Path(run_dir) / ".hydra" / "config.yaml")
        model = instantiate(config.model)
        step = load_model(model, run_dir, weights)
        clf = cls(
            model=model.to(device).eval(),
            max_classes=config.model.max_classes,
            **kwargs,
        )
        clf.checkpoint_step = step
        return clf

    def fit(self, X, y) -> Self:
        """Store the training rows; no weights change."""
        if self.model is None:
            raise ValueError("TabICLClassifier needs a pretrained model")
        X = check_array(X, ensure_all_finite="allow-nan", dtype=np.float32)
        y = column_or_1d(y, warn=True)
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows, y has {len(y)}")

        # Labels become 0..n_classes-1; classes_ keeps the original ones.
        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)
        if not 2 <= n_classes <= self.max_classes:
            raise UnsupportedTableError(
                f"{n_classes} classes, the model needs 2 to {self.max_classes}"
            )
        self.X_train_ = torch.from_numpy(X)
        self.y_train_ = torch.from_numpy(self.label_encoder_.transform(y)).long()
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Probabilities [n_rows, n_classes]; column k is classes_[k]."""
        check_is_fitted(self)
        assert self.model is not None  # fit() checked it
        X_test = torch.from_numpy(
            check_array(X, ensure_all_finite="allow-nan", dtype=np.float32)
        )
        if X_test.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_test.shape[1]} features, fit() saw {self.n_features_in_}"
            )
        # Missing values get the training-row mean, as the data pipeline does.
        X_train, X_test = impute(self.X_train_, X_test)
        X_all = torch.cat(
            [X_train, X_test]
        )  # training rows first, as the model expects
        n_classes = len(self.classes_)
        device = next(self.model.parameters()).device

        generator = torch.Generator().manual_seed(self.random_state)
        n_features = X_all.shape[1]
        orders = [torch.arange(n_features)]
        orders += [
            torch.randperm(n_features, generator=generator)
            for _ in range(self.n_orders - 1)
        ]

        proba = torch.zeros(len(X_test), n_classes)
        y_train = self.y_train_.unsqueeze(0).to(device)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                # One order at a time: a batch of 8 big tables needs a lot of memory.
                for order in orders:
                    x = X_all[:, order].unsqueeze(0).to(device)
                    logits = self.model(x, y_train)[0, :, :n_classes]
                    proba += torch.softmax(logits.float(), dim=-1).cpu()
        finally:
            # Called mid-pretraining, the model must go back to training mode.
            self.model.train(was_training)
        return (proba / len(orders)).numpy()

    def predict(self, X) -> np.ndarray:
        """The most likely class of each row, as an original label."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
