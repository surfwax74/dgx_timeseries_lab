"""Sparse Autoencoders for interpreting Sat-TSFM (and other) activations.

The core object is `TopKSAE`, which learns a wide over-complete dictionary
of features whose activations are sparsified with a hard top-k. Trained
on frozen encoder activations, the SAE's dictionary atoms serve as
human-interpretable features — you can name them by looking at the
inputs that maximally activate each atom.

See models/sae/README.md for the sprint design + the follow-up hook /
activation-capture package that will feed this trainer.
"""

from .sae import TopKSAE
from .train import train_sae

__all__ = ["TopKSAE", "train_sae"]
