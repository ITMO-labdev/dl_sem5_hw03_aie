"""Multi-task ViT: общий backbone → классификационная + регрессионная головы.

Функции и классы:
    MultiTaskViT: обёртка над backbone'ом с двумя задачными головами.

Архитектура:

    image → backbone → features (B, D)
                          ├→ cls_head: Linear(D, num_classes) → (B, C)
                          └→ reg_head: Linear(D, 1)           → (B, 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ===========================================================================
# B4.2. Multi-task модель
# ===========================================================================
class MultiTaskViT(nn.Module):
    """Multi-task обёртка над общим vision backbone.

    Backbone извлекает фичи ``(B, d_model)``, затем две независимые
    линейные головы делают предсказания для разных задач.

    Args:
        backbone: Feature extractor, ``forward(x) → (B, d_model)``.
            Получается через ``strip_classification_head`` из ViT.
        d_model: Размерность выхода backbone.
        num_classes: Число классов для классификационной головы (default 10).

    Raises:
        ValueError: Если ``d_model < 1`` или ``num_classes < 1``.

    Hint:
        ``__init__``:
            - Вызвать ``super().__init__()``.
            - Валидация: ``d_model >= 1``, ``num_classes >= 1``.
            - Сохранить ``backbone`` как ``self.backbone``.
            - ``self.cls_head = nn.Linear(d_model, num_classes)``
            - ``self.reg_head = nn.Linear(d_model, 1)``

        ``forward``:
            - ``features = self.backbone(x)``   # (B, d_model)
            - Вернуть словарь с ``"cls_logits"`` и ``"reg_pred"``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_model: int,
        num_classes: int = 10,
    ) -> None:
        pass

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Прямой проход multi-task модели.

        Args:
            x: ``(B, 3, H, W)`` FloatTensor — батч изображений.

        Returns:
            Словарь:
                - ``"cls_logits"``: ``(B, num_classes)`` FloatTensor,
                - ``"reg_pred"``: ``(B, 1)`` FloatTensor.
        """
        pass