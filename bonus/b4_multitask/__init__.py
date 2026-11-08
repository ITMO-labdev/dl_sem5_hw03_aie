from __future__ import annotations

from bonus.b4_multitask._model_utils import strip_classification_head
from bonus.b4_multitask._ndvi import compute_pseudo_ndvi
from bonus.b4_multitask._train_utils import evaluate_multitask, set_seed
from bonus.b4_multitask.multitask_dataset import EuroSATMultiTaskDataset
from bonus.b4_multitask.multitask_loss import MultiTaskLoss
from bonus.b4_multitask.multitask_model import MultiTaskViT
from bonus.b4_multitask.train_multitask import train_multitask

__all__ = [
    "EuroSATMultiTaskDataset",
    "MultiTaskViT",
    "MultiTaskLoss",
    "train_multitask",
    "compute_pseudo_ndvi",
    "strip_classification_head",
    "set_seed",
    "evaluate_multitask",
]