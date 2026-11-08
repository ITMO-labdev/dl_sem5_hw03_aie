"""Transformer-регрессор и bag-of-tokens baseline для ESOL.

Классы:
    SmilesTransformerRegressor — TransformerEncoder + masked mean pool
                                 + Linear (задание B3.2a).
    BagOfTokensBaseline        — усреднение эмбеддингов токенов
                                 + Linear (задание B3.2b).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from modules.transformer import TransformerEncoder


# ======================================================================
# Задание B3.2a. Transformer-регрессор
# ======================================================================

class SmilesTransformerRegressor(nn.Module):
    """Регрессор молекулярных свойств на базе Transformer encoder.

    Архитектура: ``TransformerEncoder`` → masked mean pooling → ``Linear``.

    Обратите внимание: ``TransformerEncoder`` из задания 3.2 **уже
    содержит внутри** ``TokenEmbeddingPipeline`` (token-эмбеддинг +
    позиционное кодирование + LayerNorm + dropout). Отдельный слой
    эмбеддингов создавать не нужно — на вход подаются сырые индексы
    токенов.

    Masked mean pooling нужен потому, что encoder возвращает по вектору
    на каждую позицию, включая паддинг. Усреднять надо только по реальным
    токенам, иначе длина паддинга начнёт влиять на предсказание.

    Args:
        vocab_size: Размер словаря токенизатора (включая спецтокены).
        d_model: Размерность скрытого состояния.
        num_heads: Число голов внимания.
        d_ff: Размерность feed-forward слоя.
        num_layers: Число encoder-блоков.
        max_len: Максимальная длина SMILES (с учётом ``<bos>``/``<eos>``).
        dropout: Вероятность dropout.
        pad_idx: Индекс паддинг-токена (совпадает с позицией ``<pad>``
            в ``SmilesTokenizer.SPECIAL_TOKENS``, то есть 0).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 256,
        num_layers: int = 2,
        max_len: int = 128,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ) -> None:
        """Создаёт encoder и регрессионную голову.

        Подсказки:
            - ``self.encoder``: ``TransformerEncoder(...)`` со всеми
              параметрами, переданными в конструктор.
            - ``self.head``: ``nn.Linear(d_model, 1)`` — предсказываем
              один скаляр, поэтому никакой активации на выходе быть
              не должно (это регрессия, не классификация).

        Не забудьте вызвать ``super().__init__()``.
        """
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Предсказывает скаляр для каждой молекулы в батче.

        Порядок операций:
            1. Прогнать ``input_ids`` и ``attention_mask`` через
               ``self.encoder`` → ``(B, L, d_model)``.
            2. Masked mean pooling: привести маску к типу скрытых
               состояний и форме ``(B, L, 1)``, умножить, просуммировать
               по ``dim=1`` и поделить на число реальных токенов
               (``.clamp(min=1.0)`` защищает от деления на ноль).
            3. Применить ``self.head`` → ``(B, 1)``.

        Args:
            input_ids: ``(B, L)``, ``torch.long``.
            attention_mask: ``(B, L)``, ``torch.bool``. ``True`` —
                реальный токен, ``False`` — паддинг.

        Returns:
            Тензор ``(B, 1)``, ``torch.float``.
        """
        pass


# ======================================================================
# Задание B3.2b. Bag-of-tokens baseline
# ======================================================================

class BagOfTokensBaseline(nn.Module):
    """Baseline, игнорирующий порядок токенов: mean(embeddings) → Linear.

    Модель не знает ничего о последовательности: молекулы ``CCO`` и
    ``OCC`` дадут одинаковый выход. Фактически это обучаемая линейная
    регрессия по «мешку атомов» — средним частотам токенов в формуле.

    Зачем он нужен: любое качество выше тривиального предсказания
    среднего (``R² > 0``) означает, что сигнал несут уже **отдельные
    атомы и функциональные группы**, безотносительно их взаимного
    расположения. Transformer с его на порядок большим числом параметров
    обязан обойти этот baseline — иначе его структурный inductive bias
    не оправдывает своей стоимости.

    Args:
        vocab_size: Размер словаря токенизатора.
        embed_dim: Размерность эмбеддинга токена.
        pad_idx: Индекс паддинг-токена; его эмбеддинг зафиксирован нулём
            через ``padding_idx``. Маска всё равно исключает паддинг
            из усреднения — это двойная защита.

    Raises:
        ValueError: Если ``vocab_size < 1`` или ``embed_dim < 1``.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        pad_idx: int = 0,
    ) -> None:
        """Создаёт слой эмбеддингов и линейную голову.

        Подсказки:
            - ``self.embed``:
              ``nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)``.
            - ``self.head``: ``nn.Linear(embed_dim, 1)``.

        Не забудьте вызвать ``super().__init__()``.

        Raises:
            ValueError: Если ``vocab_size < 1`` или ``embed_dim < 1``.
        """
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Предсказывает скаляр для каждого примера в батче.

        Порядок операций полностью повторяет пулинг из
        ``SmilesTransformerRegressor``, но вместо скрытых состояний
        encoder'а усредняются сырые эмбеддинги токенов:
            1. ``self.embed(input_ids)`` → ``(B, L, embed_dim)``.
            2. Masked mean pooling по ``dim=1`` → ``(B, embed_dim)``.
            3. ``self.head(pooled)`` → ``(B, 1)``.

        Args:
            input_ids: ``(B, L)``, ``torch.long``.
            attention_mask: ``(B, L)``, ``torch.bool``.

        Returns:
            Тензор ``(B, 1)``, ``torch.float``.
        """
        pass