"""EuroSATMultiTaskDataset — lazy-датасет для классификации + pseudo-NDVI.

Функции и классы:
    EuroSATMultiTaskDataset: обёртка над image-classification датасетом
        (HuggingFace или list-of-dicts), добавляющая pseudo-NDVI как
        регрессионный таргет.

Датасет хранит **ссылки** на исходный источник данных и вычисляет
transform / NDVI «лениво» в ``__getitem__``, чтобы не материализовать
десятки тысяч тензоров в RAM.

Важно: pseudo-NDVI всегда считается от **ненормализованного** тензора
``[0, 1]`` (результат ``torchvision.transforms.functional.to_tensor``),
а не от тензора после ImageNet Normalize. Если ``transform`` содержит
``Normalize``, он применяется к изображению **после** вычисления NDVI —
модель получит нормализованный вход, а NDVI останется корректным.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from bonus.b4_multitask._ndvi import compute_pseudo_ndvi


# ===========================================================================
# B4.1. Multi-task датасет для EuroSAT
# ===========================================================================
class EuroSATMultiTaskDataset(Dataset):
    """Lazy multi-task обёртка над image-classification датасетом.

    Каждый элемент — словарь:
        - ``"image"``: ``FloatTensor (3, H, W)`` — изображение после ``transform``,
        - ``"cls_target"``: ``LongTensor ()`` — метка класса (0-мерный скаляр),
        - ``"reg_target"``: ``FloatTensor ()`` — pseudo-NDVI в ``[-1, 1]`` (0-мерный).

    Args:
        dataset: Любой индексируемый объект, ``dataset[idx]`` возвращает
            словарь с ключами ``"image"`` (``PIL.Image``) и ``"label"``
            (``int``). HuggingFace ``Dataset`` подходит из коробки.
        indices: Какие строки ``dataset`` входят в этот сплит.
        transform: Применяется к PIL-изображению и должен возвращать
            ``(3, H, W)`` FloatTensor. Может содержать ``Normalize`` —
            NDVI вычисляется **до** нормализации (от raw ``[0, 1]``
            тензора), поэтому на его корректность transform не влияет.
            Если ``None`` — используется ``torchvision.transforms.functional.to_tensor``.

    Raises:
        ValueError: Если ``indices`` пуст.

    Hint:
        Порядок в ``__getitem__``:
        1. Извлечь PIL-изображение и метку из ``self._dataset[self._indices[idx]]``.
        2. Конвертировать PIL → тензор ``[0, 1]`` через ``TF.to_tensor(image_pil)``.
        3. Вычислить pseudo-NDVI от этого «сырого» тензора.
        4. Если задан ``self._transform`` — применить его к **PIL**-изображению
           (результат уже содержит Resize / ToTensor / Normalize).
           Иначе использовать тензор из шага 2.
        5. Вернуть словарь с тремя ключами.
    """

    def __init__(
        self,
        dataset: Any,
        indices: Sequence[int],
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        if len(indices) == 0:
            raise ValueError("Dataset requires at least one sample.")
        self._dataset = dataset
        self._indices = list(indices)
        self._transform = transform

    def __len__(self) -> int:
        """Возвращает число изображений в этом сплите."""
        pass

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Возвращает словарь ``{image, cls_target, reg_target}`` для индекса ``idx``.

        Returns:
            Словарь:
                - ``"image"``: FloatTensor ``(3, H, W)``,
                - ``"cls_target"``: LongTensor ``()`` (0-мерный),
                - ``"reg_target"``: FloatTensor ``()`` (0-мерный, pseudo-NDVI).
        """
        pass