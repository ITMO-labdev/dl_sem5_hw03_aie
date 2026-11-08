"""Извлечение эмбеддингов из TransformerEncoder с пулингом.

Функции:
    extract_embeddings — извлечение эмбеддингов с mean/cls пулингом
                         (задание B1.2).
"""

from __future__ import annotations

import torch

from modules.transformer import TransformerEncoder

_VALID_POOLING: frozenset[str] = frozenset({"mean", "cls"})


# ======================================================================
# Задание B1.2. Извлечение эмбеддингов
# ======================================================================

def extract_embeddings(
    model: TransformerEncoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str = "mean",
) -> torch.Tensor:
    """Извлекает пулированные эмбеддинги из TransformerEncoder.

    Паттерн, стандартный для sentence-transformers:
        1. Перевести модель в ``eval()`` режим.
        2. Без градиентов (``torch.no_grad()``) прогнать ``input_ids``
           и ``attention_mask`` через модель → ``hidden (B, L, d_model)``.
        3. Применить пулинг:
           - ``"mean"``: средневзвешенное по непаддинговым позициям.
             Используйте ``attention_mask`` как float-маску:
             ``mask = attention_mask.to(hidden.dtype).unsqueeze(-1)``.
             Делите сумму на ``counts.clamp_min(1.0)`` для защиты
             от деления на 0 при полностью замаскированных строках.
           - ``"cls"``: вектор позиции 0 (``hidden[:, 0, :]``).
        4. Восстановить исходный режим модели (``train``/``eval``).

    Args:
        model: Экземпляр ``TransformerEncoder`` из ``modules.transformer``.
        input_ids: Индексы токенов, ``(B, L)``.
        attention_mask: Булева маска, ``(B, L)``. ``True`` — реальный
            токен, ``False`` — паддинг.
        pooling: Стратегия пулинга — ``"mean"`` или ``"cls"``.

    Returns:
        Тензор ``(B, d_model)`` с ``requires_grad=False``.

    Raises:
        ValueError: Если ``pooling`` не из ``{"mean", "cls"}``.
    """
    pass