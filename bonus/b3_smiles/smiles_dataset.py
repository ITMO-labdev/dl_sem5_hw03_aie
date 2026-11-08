"""Датасет токенизированных SMILES со скалярными таргетами.

Классы:
    ESOLDataset — пары (токенизированная SMILES, скалярный таргет)
                  переменной длины (задание B3.1).
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch.utils.data import Dataset


# Вспомогательный класс для type hinting — реализовывать не нужно.
class _TokenizerProtocol(Protocol):
    """Минимальный интерфейс, который должен поддерживать токенизатор."""

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str, add_special: bool = True) -> list[int]: ...


# ======================================================================
# Задание B3.1. Датасет ESOL
# ======================================================================

class ESOLDataset(Dataset):
    """Токенизированные SMILES переменной длины и скалярные таргеты.

    Элемент датасета — словарь с ключами:
        - ``input_ids``: ``LongTensor`` формы ``(L,)`` — последовательность
          **без паддинга**, обёрнутая в ``<bos> ... <eos>``;
        - ``target``: 0-мерный ``FloatTensor`` — нормализованный таргет.

    Паддинг выполняется не здесь, а в ``esol_collate_fn`` на уровне батча:
    так длина паддинга определяется самой длинной молекулой **в батче**,
    а не во всём датасете, что экономит вычисления.

    Args:
        smiles: Список SMILES-строк.
        targets: Список нормализованных скалярных таргетов
            (по одному на SMILES).
        tokenizer: Объект с методом
            ``encode(text, add_special=True) -> list[int]``.

    Raises:
        ValueError: Если длины ``smiles`` и ``targets`` не совпадают
            или датасет пуст.
    """

    def __init__(
        self,
        smiles: list[str],
        targets: list[float],
        tokenizer: _TokenizerProtocol,
    ) -> None:
        if len(smiles) != len(targets):
            raise ValueError(
                f"smiles and targets length mismatch: "
                f"{len(smiles)} vs {len(targets)}"
            )
        if len(smiles) == 0:
            raise ValueError("ESOLDataset requires at least one sample.")
        self._smiles = list(smiles)
        self._targets = list(targets)
        self._tokenizer = tokenizer

    def __len__(self) -> int:
        """Возвращает число молекул в датасете.

        Returns:
            Количество примеров.
        """
        pass

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Возвращает один пример датасета.

        Порядок действий:
            1. Токенизировать SMILES по индексу ``idx`` вызовом
               ``self._tokenizer.encode(..., add_special=True)`` —
               специальные токены ``<bos>``/``<eos>`` обязательны.
            2. Если токенизатор вернул пустой список — бросить
               ``ValueError`` с указанием проблемной SMILES и индекса.
            3. Завернуть результат в тензоры нужных типов.

        Внимание к dtype: ``input_ids`` должен быть ``torch.long``
        (индексы для ``nn.Embedding``), ``target`` — ``torch.float``
        и **0-мерным** (``torch.tensor(scalar)``, не ``(1,)``).

        Args:
            idx: Индекс примера.

        Returns:
            Словарь ``{"input_ids": (L,) long, "target": () float}``.

        Raises:
            ValueError: Если токенизатор вернул пустую последовательность.
        """
        pass