"""Цикл обучения multi-task модели.

Функции:
    train_multitask: обучает модель с ``MultiTaskLoss`` и возвращает
        историю ``{train_loss, val_cls_acc, val_reg_mse}`` по эпохам.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bonus.b4_multitask._train_utils import evaluate_multitask
from bonus.b4_multitask.multitask_loss import MultiTaskLoss


# ===========================================================================
# B4.4. Обучение multi-task модели
# ===========================================================================
def train_multitask(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float = 1e-3,
    alpha: float = 0.5,
    device: str = "cpu",
) -> dict[str, list[float]]:
    """Обучает multi-task модель и возвращает историю по эпохам.

    Recipe:
        - Оптимизатор: ``Adam(model.parameters(), lr=lr)``.
        - Loss: ``MultiTaskLoss(alpha=alpha)``.
        - Каждую эпоху: train-loop → ``evaluate_multitask`` на val.

    Args:
        model: Модель с ``forward(x) → {"cls_logits": ..., "reg_pred": ...}``.
        train_loader: DataLoader с train-сплитом; батч — словарь с ключами
            ``"image"``, ``"cls_target"``, ``"reg_target"``.
        val_loader: DataLoader с val-сплитом (для мониторинга).
        epochs: Число эпох (>= 1).
        lr: Learning rate (> 0).
        alpha: Вес CE в multi-task loss, передаётся в ``MultiTaskLoss``.
        device: ``"cpu"`` или ``"cuda"``.

    Returns:
        Словарь:
            - ``"train_loss"``: ``list[float]`` длины ``epochs``,
            - ``"val_cls_acc"``: ``list[float]`` длины ``epochs``,
            - ``"val_reg_mse"``: ``list[float]`` длины ``epochs``.

    Raises:
        ValueError: Если ``epochs < 1`` или ``lr <= 0``.

    Hint:
        1. ``model.to(device)``.
        2. Создать ``MultiTaskLoss(alpha=alpha)`` и ``Adam``.
        3. Цикл по эпохам:
           a. ``model.train()``.
           b. Для каждого батча: forward → loss → backward → step.
              Не забудьте ``optimizer.zero_grad(set_to_none=True)`` перед backward.
              ``running_loss += loss.item() * images.size(0)``.
           c. ``epoch_loss = running_loss / total_samples``.
           d. ``val_metrics = evaluate_multitask(model, val_loader, device)``.
           e. Записать в историю.
        4. Вернуть историю.
    """
    pass