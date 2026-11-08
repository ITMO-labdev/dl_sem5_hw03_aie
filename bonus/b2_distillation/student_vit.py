"""Student ViT для knowledge distillation.

Функции:
    build_student_vit — обрезанная ViT-Tiny без предобученных весов
                        (задание B2.1).
"""

from __future__ import annotations

import timm
from torch import nn

_BASE_MODEL = "vit_tiny_patch16_224"
_MAX_BLOCKS = 12  # vit_tiny_patch16_224 содержит 12 transformer-блоков


# ======================================================================
# Задание B2.1. Архитектура student ViT
# ======================================================================

def build_student_vit(
    num_classes: int,
    num_blocks: int = 3,
) -> nn.Module:
    """Собирает «мелкую» ViT-модель для обучения через дистилляцию.

    Алгоритм:
        1. Создать ``vit_tiny_patch16_224`` через ``timm.create_model``
           с ``pretrained=False`` и ``num_classes=num_classes``.
           Флаг ``pretrained=False`` критичен: student обучается
           **с нуля** через дистилляцию — использование предобученных
           весов смешало бы вклад дистилляции и вклад предобучения.
           Параметр ``num_classes`` сам создаст голову ``nn.Linear``
           нужной размерности.
        2. Обрезать стек transformer-блоков до ``num_blocks`` штук:
           в timm блоки лежат в ``model.blocks`` (``nn.Sequential``
           из 12 ``Block``-модулей). Замените его на новый
           ``nn.Sequential`` из первых ``num_blocks`` блоков.

    Подсказка:
        ``model.blocks = nn.Sequential(*list(model.blocks)[:num_blocks])``

    Args:
        num_classes: Число классов (размер выхода).
        num_blocks: Сколько первых transformer-блоков оставить.
            Полная ViT-Tiny имеет 12 блоков.

    Returns:
        Модель-student, готовая к обучению через дистилляцию.

    Raises:
        ValueError: Если ``num_classes < 1`` или ``num_blocks``
            вне ``[1, 12]``.
    """
    pass