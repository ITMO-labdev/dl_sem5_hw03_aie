"""Загрузка и подготовка датасета ESOL для задачи регрессии log-растворимости.

Функции:
    load_esol: читает CSV, делает train/val/test split, z-нормализует таргет.
    esol_collate_fn: паддинг батча переменной длины до максимума в батче.

Формат CSV: две колонки — ``smiles`` и ``measured log solubility in mols per litre``.
Размер датасета: 1128 молекул, диапазон log-solubility примерно [-11.6, 1.6].
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from bonus.b3_smiles.smiles_dataset import ESOLDataset


# ===========================================================================
# B3.0. Загрузка ESOL и сплит
# ===========================================================================
def load_esol(
    csv_path: str | Path,
    tokenizer=None,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[ESOLDataset, ESOLDataset, ESOLDataset, float, float]:
    """Читает ESOL CSV, делит на train/val/test, z-нормализует таргет по train.

    Args:
        csv_path: Путь к CSV-файлу с колонками ``smiles`` и
            ``measured log solubility in mols per litre``.
        val_frac: Доля валидационной выборки от общего числа молекул.
        test_frac: Доля тестовой выборки от общего числа молекул.
        seed: Сид перестановки индексов (детерминированный split).

    Returns:
        Кортеж ``(train_ds, val_ds, test_ds, target_mean, target_std)``.
        Статистики ``mean``/``std`` вычисляются **только по train**, чтобы
        избежать утечки из val/test; применяются ко всем трём сплитам.

    Raises:
        ValueError: Если ``val_frac`` или ``test_frac`` вне (0, 1) или их сумма
            >= 1, либо если на какой-то из сплитов приходится 0 примеров.
        FileNotFoundError: Если ``csv_path`` не существует.

    Notes:
        Test-сплит используется только для финальной оценки в
        ``scripts/train_smiles.py``; на нём не производится подбор гиперпараметров
        и ранняя остановка, поэтому метрики на test корректно отражают
        обобщающую способность модели.
    """
    if not (0.0 < val_frac < 1.0) or not (0.0 < test_frac < 1.0):
        raise ValueError(
            f"val_frac и test_frac должны быть в (0, 1), получено "
            f"val_frac={val_frac}, test_frac={test_frac}"
        )
    if val_frac + test_frac >= 1.0:
        raise ValueError(
            f"val_frac + test_frac должны быть < 1, получено {val_frac + test_frac}"
        )

    smiles_list, targets = _read_csv(Path(csv_path))
    n_total = len(smiles_list)

    n_test = round(n_total * test_frac)
    n_val = round(n_total * val_frac)
    n_train = n_total - n_val - n_test
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Пустой сплит: n_train={n_train}, n_val={n_val}, n_test={n_test} "
            f"при n_total={n_total}"
        )

    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n_total, generator=generator).tolist()

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    train_targets = torch.tensor([targets[i] for i in train_idx], dtype=torch.float)
    target_mean = float(train_targets.mean().item())
    target_std = float(train_targets.std(unbiased=False).item())
    if target_std == 0.0:
        target_std = 1.0

    def _subset(indices):
        # Нормализуем таргеты ДО передачи в ESOLDataset
        norm_targets = [
            (targets[i] - target_mean) / target_std for i in indices
        ]
        return ESOLDataset(
            smiles=[smiles_list[i] for i in indices],
            targets=norm_targets,
            tokenizer=tokenizer,
        )
    return _subset(train_idx), _subset(val_idx), _subset(test_idx), target_mean, target_std


# ===========================================================================
# B3.0. Collate для переменной длины
# ===========================================================================
def esol_collate_fn(batch, pad_token_id: int):
    """Паддит батч (input_ids, target) до максимальной длины в батче.

    Args:
        batch: Список словарей ``{"input_ids": LongTensor [L_i], "target": FloatTensor []}``.
        pad_token_id: ID padding-токена в словаре SMILES-токенизатора.

    Returns:
        Словарь с ключами:
            - ``input_ids``: LongTensor ``[B, L_max]``,
            - ``attention_mask``: LongTensor ``[B, L_max]`` (1 — токен, 0 — паддинг),
            - ``targets``: FloatTensor ``[B]`` (z-нормализованные).
    """
    input_ids = [item["input_ids"] for item in batch]
    targets = torch.stack([item["target"] for item in batch])
    padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    #attention_mask = (padded != pad_token_id).long()
    attention_mask = (padded != pad_token_id).bool()
    return {"input_ids": padded, "attention_mask": attention_mask, "target": targets}


def _read_csv(path: Path):
    """Читает CSV и возвращает списки SMILES и таргетов."""
    if not path.exists():
        raise FileNotFoundError(f"ESOL CSV не найден: {path}")
    smiles_list: list[str] = []
    targets: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles_list.append(row["smiles"])
            targets.append(float(row["measured log solubility in mols per litre"]))
    return smiles_list, targets