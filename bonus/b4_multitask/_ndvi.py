"""Вычисление pseudo-NDVI из RGB-изображений.

Выдаётся заполненным. **Не изменяйте этот файл.**

Функции:
    compute_pseudo_ndvi: считает приближённый NDVI по видимым каналам
        (зелёный и красный), возвращает скаляр в [-1, 1].

Настоящий NDVI требует ближнего ИК-канала (NIR), которого в RGB нет.
Для RGB используется proxy:

    NDVI_proxy = mean( (G - R) / (G + R + eps) )

Диапазон: [-1, 1]. Положительные значения соответствуют растительности
(зелёный > красного), отрицательные — воде или снегу.
"""

from __future__ import annotations

import torch


def compute_pseudo_ndvi(image: torch.Tensor, eps: float = 1e-6) -> float:
    """Вычисляет pseudo-NDVI скаляр из RGB-тензора.

    Args:
        image: ``(3, H, W)`` FloatTensor с пиксельными значениями в ``[0, 1]``.
            Каналы: ``[R, G, B]`` (стандартный порядок torchvision).
        eps: Малая константа для избежания деления на ноль.

    Returns:
        Среднее pseudo-NDVI по всем пикселям, ``float`` в ``[-1, 1]``.

    Raises:
        ValueError: Если ``image`` не имеет ровно 3 канала.
    """
    if image.shape[0] != 3:
        raise ValueError(
            f"Expected 3-channel image, got {image.shape[0]} channels."
        )
    r = image[0]  # (H, W)
    g = image[1]  # (H, W)
    ndvi_map = (g - r) / (g + r + eps)
    return float(ndvi_map.mean().item())