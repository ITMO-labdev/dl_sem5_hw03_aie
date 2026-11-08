"""Циклы обучения для дистилляции и baseline-сравнения.

Функции:
    train_student_with_distillation — обучение student через дистилляцию
                                       от teacher (задание B2.3a).
    train_student_baseline           — обучение того же student на hard
                                       labels без дистилляции (задание B2.3b).

Baseline использует тот же optimizer, scheduler, seed и число эпох,
что и дистилляция — чтобы разницу в финальном качестве можно было
приписать именно дистилляции, а не отличиям в training-recipe.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from bonus.b2_distillation._train_utils import evaluate, set_seed
from bonus.b2_distillation.distillation_loss import DistillationLoss


# ======================================================================
# Задание B2.3a. Обучение с дистилляцией
# ======================================================================

def train_student_with_distillation(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    alpha: float = 0.5,
    temperature: float = 4.0,
    device: str = "cuda",
    seed: int = 42,
) -> dict[str, list[float]]:
    """Обучает ``student`` через knowledge distillation от ``teacher``.

    Схема эпохи:
        1. Перевести teacher в ``eval()`` и отключить градиенты
           (``requires_grad = False``). Teacher **не** обучается.
        2. В цикле по батчам:
           a) Прогнать images через teacher в ``torch.no_grad()`` →
              ``teacher_logits``.
           b) Прогнать images через student → ``student_logits``.
           c) Вычислить ``DistillationLoss(student_logits,
              teacher_logits, labels)``.
           d) Шаг оптимизатора.
        3. После эпохи — ``scheduler.step()`` и
           ``evaluate(student, val_loader, nn.CrossEntropyLoss(), device)``
           (валидация проводится обычным CE, т.к. нас интересует
           качество классификации, а не величина training-loss).

    Подсказки:
        - ``set_seed(seed)`` в самом начале для воспроизводимости.
        - ``optimizer = AdamW(student.parameters(), lr=lr, weight_decay=0.01)``.
        - ``scheduler = CosineAnnealingLR(optimizer, T_max=epochs)``.
        - Для накопления train loss используйте взвешенное среднее:
          ``running_loss += loss.item() * batch_size``.
        - ``images.to(device, non_blocking=True)`` ускоряет копирование
          при ``pin_memory=True`` в DataLoader.

    Args:
        student: Модель student (обучаемая).
        teacher: Предобученная teacher-модель (замороженная).
        train_loader, val_loader: DataLoader'ы для train/val.
        epochs: Число эпох.
        lr: Начальный learning rate для AdamW.
        alpha: Вес KL-члена в ``DistillationLoss``.
        temperature: Температура в ``DistillationLoss``.
        device: ``"cuda"`` или ``"cpu"``.
        seed: Seed для воспроизводимости.

    Returns:
        Словарь ``history``:
            - ``"train_loss"``: list[float], длина ``epochs``.
            - ``"val_loss"``:   list[float], длина ``epochs``.
            - ``"val_acc"``:    list[float], длина ``epochs``.

    Raises:
        ValueError: Если ``epochs < 1``.
    """
    pass


# ======================================================================
# Задание B2.3b. Baseline-обучение (без дистилляции)
# ======================================================================

def train_student_baseline(
    student: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda",
    seed: int = 42,
) -> dict[str, list[float]]:
    """Обучает ``student`` обычным CrossEntropy — baseline для сравнения.

    Полностью повторяет training-recipe ``train_student_with_distillation``
    (тот же AdamW с ``weight_decay=0.01``, тот же ``CosineAnnealingLR``,
    тот же ``seed``), но заменяет ``DistillationLoss`` на
    ``nn.CrossEntropyLoss``. Это позволяет честно сравнить вклад
    **именно** дистилляционного сигнала, изолировав его от всех остальных
    факторов training-пайплайна.

    Схема эпохи аналогична ``train_student_with_distillation``,
    но без teacher и без KL-члена.

    Args:
        student: Модель student (обучаемая).
        train_loader, val_loader: DataLoader'ы для train/val.
        epochs: Число эпох.
        lr: Начальный learning rate.
        device: ``"cuda"`` или ``"cpu"``.
        seed: Seed для воспроизводимости.

    Returns:
        Словарь ``history`` того же формата, что и у
        ``train_student_with_distillation``.

    Raises:
        ValueError: Если ``epochs < 1``.
    """
    pass