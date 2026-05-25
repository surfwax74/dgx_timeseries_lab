from . import layered, trivial  # noqa: F401  side-effect: register both
from .layered import LayeredSyntheticDataset
from .trivial import TrivialSyntheticDataset

__all__ = ["LayeredSyntheticDataset", "TrivialSyntheticDataset", "layered", "trivial"]
