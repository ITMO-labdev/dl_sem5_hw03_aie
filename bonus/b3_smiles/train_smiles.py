"""Цикл обучения для ESOL-регрессоров.

Функции:
    train_regressor — обучение с MSE-loss и Adam, с историей по эпохам
                      (задание B3.3).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bonus.b3_smiles._train_utils import evaluate_mse


# ======================================================================
# Задание B3.3. Цикл обучения
# ======================================================================

def train_regressor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict[str, list[float]]:
    """Обучает регрессор с MSE-loss и Adam, собирая историю по эпохам.

    Функция универсальна: она ничего не знает о внутреннем устройстве
    модели и применяется и к ``SmilesTransformerRegressor``, и к
    ``BagOfTokensBaseline``. Благодаря этому обе модели обучаются по
    абсолютно одинаковому рецепту, и разницу в итоговом качестве можно
    приписать именно архитектуре.

    Схема эпохи:
        1. ``model.train()``, итерация по ``train_loader``:
           - перенести ``input_ids``, ``attention_mask``, ``target``
             на ``device``;
           - прогнать модель и привести выход ``(B, 1)`` к ``(B,)``
             через ``.squeeze(-1)`` — иначе MSE между ``(B, 1)`` и
             ``(B,)`` молча broadcast'нется в ``(B, B)`` и посчитает
             полную матрицу попарных разностей;
           - посчитать ``nn.MSELoss()``, сделать шаг оптимизатора;
           - накапливать взвешенную сумму:
             ``running_sq += loss.item() * target.numel()``.
        2. ``train_mse = running_sq / running_n``.
        3. ``val_mse = evaluate_mse(model, val_loader, device=device)``.
        4. Добавить оба значения в ``history``.

    Подсказки:
        - ``optimizer = torch.optim.Adam(model.parameters(), lr=lr)``.
        - ``optimizer.zero_grad(set_to_none=True)`` перед ``backward()``.
        - Не забудьте ``model.to(device)`` до создания оптимизатора.

    Args:
        model: Модель с сигнатурой
            ``forward(input_ids, attention_mask) -> (B, 1)``.
        train_loader: DataLoader обучающей выборки; отдаёт словари
            с ключами ``input_ids``, ``attention_mask``, ``target``.
        val_loader: DataLoader валидационной выборки того же формата.
        epochs: Число эпох обучения.
        lr: Learning rate для Adam.
        device: ``"cpu"`` или ``"cuda"``.

    Returns:
        Словарь ``history``:
            - ``"train_loss"``: list[float] длины ``epochs`` — MSE
              на обучающей выборке;
            - ``"val_loss"``: list[float] длины ``epochs`` — MSE
              на валидационной выборке.

    Raises:
        ValueError: Если ``epochs < 1`` или ``lr <= 0``.
    """
    pass