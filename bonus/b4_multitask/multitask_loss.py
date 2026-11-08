"""Взвешенная multi-task loss: CE (классификация) + MSE (регрессия).

Функции и классы:
    MultiTaskLoss: модуль, считающий α·CE + (1−α)·MSE.

Формула:

    L = α · CE(cls_logits, cls_target) + (1 − α) · MSE(reg_pred, reg_target)

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# B4.3. Multi-task loss
# ===========================================================================
class MultiTaskLoss(nn.Module):
    """Взвешенная комбинация cross-entropy и MSE.

    .. math::

        L = \\alpha \\cdot \\text{CE}(\\text{cls\\_logits}, \\text{cls\\_target})
          + (1 - \\alpha) \\cdot \\text{MSE}(\\text{reg\\_pred}, \\text{reg\\_target})

    Args:
        alpha: Вес классификационного loss'а, ``[0, 1]``.
            ``alpha=1`` — чистая классификация, ``alpha=0`` — чистая регрессия.

    Raises:
        ValueError: Если ``alpha`` вне ``[0, 1]``.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha

    def forward(
        self,
        cls_logits: torch.Tensor,
        reg_pred: torch.Tensor,
        cls_target: torch.Tensor,
        reg_target: torch.Tensor,
    ) -> torch.Tensor:
        """Считает взвешенную multi-task loss.

        Args:
            cls_logits: ``(B, C)`` FloatTensor — сырые логиты классификатора.
            reg_pred: ``(B, 1)`` FloatTensor — предсказание регрессионной головы.
            cls_target: ``(B,)`` LongTensor — метки классов.
            reg_target: ``(B,)`` FloatTensor — целевые значения pseudo-NDVI.

        Returns:
            0-мерный FloatTensor — скалярный loss.

        Hint:
            - ``F.cross_entropy(cls_logits, cls_target)`` — CE по сырым логитам.
            - ``F.mse_loss(reg_pred.squeeze(-1), reg_target)`` — MSE;
              ``squeeze(-1)`` нужен, чтобы привести ``(B, 1)`` к ``(B,)``,
              иначе broadcast ``(B, 1)`` vs ``(B,)`` даст матрицу ``(B, B)``.
            - Итог: ``alpha * ce + (1 - alpha) * mse``.
        """
        pass