"""Утилиты для извлечения backbone из timm ViT.

Выдаётся заполненным. **Не изменяйте этот файл.**

Функции и классы:
    strip_classification_head: удаляет классификационную голову timm ViT
        и оборачивает в ``_BackboneWrapper``, возвращающий CLS-pooled фичи.

Функции и классы (внутренние):
    _BackboneWrapper: обёртка над ``forward_features`` timm ViT, возвращающая
        CLS-токен ``(B, d_model)`` вместо полной карты ``(B, N+1, D)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _BackboneWrapper(nn.Module):
    """Обёртка timm ViT: ``forward`` возвращает CLS-pooled фичи ``(B, d_model)``.

    timm'овские ViT отдают из ``forward_features`` тензор
    ``(B, num_patches + 1, embed_dim)``. Эта обёртка берёт CLS-токен
    (индекс 0) и возвращает ``(B, embed_dim)``.

    Args:
        vit: timm Vision Transformer **без** классификационной головы
            (голова заменена на ``nn.Identity`` в ``strip_classification_head``).
    """

    def __init__(self, vit: nn.Module) -> None:
        super().__init__()
        self.vit = vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """CLS-pooled фичи.

        Args:
            x: ``(B, 3, 224, 224)`` FloatTensor — батч изображений.

        Returns:
            ``(B, d_model)`` FloatTensor — CLS-эмбеддинг каждого изображения.
        """
        features = self.vit.forward_features(x)  # (B, N+1, D)
        return features[:, 0]  # CLS token


def strip_classification_head(
    vit: nn.Module,
) -> tuple[nn.Module, int]:
    """Удаляет классификационную голову и возвращает CLS-pooled backbone.

    Заменяет ``vit.head`` на ``nn.Identity()`` и оборачивает модель в
    ``_BackboneWrapper``. Исходный объект ``vit`` **модифицируется
    in-place** (голова заменяется).

    Args:
        vit: timm ViT (например, ``vit_tiny_patch16_224``). Модель должна
            иметь атрибуты ``head`` (с ``in_features``) и ``forward_features``.

    Returns:
        Кортеж ``(backbone, d_model)``:
            - ``backbone``: ``nn.Module`` с ``forward(x) → (B, d_model)``,
            - ``d_model``: размерность выходных фичей.

    Raises:
        AttributeError: Если ``vit`` не имеет ожидаемой timm-структуры
            (нет ``head`` или ``forward_features``).
    """
    if not hasattr(vit, "head"):
        raise AttributeError(
            "Model has no 'head' attribute — expected a timm ViT."
        )
    if not hasattr(vit, "forward_features"):
        raise AttributeError(
            "Model has no 'forward_features' — expected a timm ViT."
        )
    d_model = vit.head.in_features
    vit.head = nn.Identity()
    backbone = _BackboneWrapper(vit)
    return backbone, d_model