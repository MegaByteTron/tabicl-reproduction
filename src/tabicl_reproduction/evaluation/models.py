"""Every model the benchmark compares, registered by name.

    @register_model()
    def random_forest() -> ClassifierMixin: ...

Each factory returns a new, unfitted scikit-learn classifier. Our model is
registered the same way, with needs_checkpoint=True: its factory gets the
training run to load, because a TabICLClassifier without pretrained weights
can't predict.
"""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tabicl_reproduction.model import TabICLClassifier


@dataclass(frozen=True)
class ModelSpec:
    factory: Callable[..., ClassifierMixin]
    needs_checkpoint: bool = False


MODEL_REGISTRY: dict[str, ModelSpec] = {}


def register_model(
    name: str | None = None, *, needs_checkpoint: bool = False
) -> Callable[[Callable[..., ClassifierMixin]], Callable[..., ClassifierMixin]]:
    def decorator(
        factory: Callable[..., ClassifierMixin],
    ) -> Callable[..., ClassifierMixin]:
        model_name = name or getattr(factory, "__name__", repr(factory))
        if model_name in MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name}' already registered")
        MODEL_REGISTRY[model_name] = ModelSpec(factory, needs_checkpoint)
        return factory

    return decorator


@register_model()
def knn() -> ClassifierMixin:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5),
    )


@register_model()
def logistic_regression() -> ClassifierMixin:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    )


@register_model()
def random_forest() -> ClassifierMixin:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=300, random_state=42),
    )


@register_model()
def hist_gradient_boosting() -> ClassifierMixin:
    return HistGradientBoostingClassifier(random_state=42)


@register_model(needs_checkpoint=True)
def nano_tabicl_v2(checkpoint: str | Path, **kwargs: Any) -> TabICLClassifier:
    """Our pretrained model; kwargs: weights ("best"/"model"), device, n_orders."""
    return TabICLClassifier.from_checkpoint(checkpoint, **kwargs)


def get_models(
    names: Iterable[str] | None = None,
    checkpoint: str | Path | None = None,
    **checkpoint_kwargs: Any,
) -> Iterator[tuple[str, ClassifierMixin]]:
    """Yield (name, new unfitted model) for the chosen (default: all) models.

    Models that need a checkpoint are left out when none is given, unless
    asked for by name, which raises.
    """
    wanted = list(MODEL_REGISTRY) if names is None else list(names)
    unknown = sorted(set(wanted) - set(MODEL_REGISTRY))
    if unknown:
        raise ValueError(
            f"Unknown models {unknown}; registered: {list(MODEL_REGISTRY)}"
        )
    for name in wanted:
        spec = MODEL_REGISTRY[name]
        if not spec.needs_checkpoint:
            yield name, spec.factory()
        elif checkpoint is not None:
            yield name, spec.factory(checkpoint, **checkpoint_kwargs)
        elif names is not None:
            raise ValueError(
                f"Model '{name}' needs a checkpoint (a training run folder)"
            )
