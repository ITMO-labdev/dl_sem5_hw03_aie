"""Метрики расстояния между распределениями для детекции дрифта.

Реализуемые метрики:
    mmd_rbf                    — квадрат MMD с RBF-ядром (задание B1.1a).
    cosine_centroid_distance   — косинусное расстояние между центроидами (задание B1.1b).
    sliced_wasserstein_distance — Sliced Wasserstein-1 (задание B1.1c).

Вспомогательные функции (уже реализованы):
    _pairwise_squared_distances — матрица квадратов евклидовых расстояний.
    _quantile_resample          — линейная интерполяция по квантильной сетке.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ======================================================================
# Вспомогательные функции (НЕ ИЗМЕНЯТЬ)
# ======================================================================

def _pairwise_squared_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Матрица квадратов евклидовых расстояний между строками ``a`` и ``b``.

    Использует тождество ||x − y||² = ||x||² + ||y||² − 2x·y
    и зажимает результат снизу нулём для защиты от ошибок округления.

    Args:
        a: Тензор формы ``(N, D)``.
        b: Тензор формы ``(M, D)``.

    Returns:
        Тензор формы ``(N, M)``.
    """
    a_sq = (a * a).sum(dim=1, keepdim=True)       # (N, 1)
    b_sq = (b * b).sum(dim=1, keepdim=True).T     # (1, M)
    cross = a @ b.T                                # (N, M)
    return (a_sq + b_sq - 2.0 * cross).clamp_min(0.0)


# ======================================================================
# Задание B1.1a. MMD с RBF-ядром
# ======================================================================

def mmd_rbf(
    X: torch.Tensor,
    Y: torch.Tensor,
    gamma: float | None = None,
) -> torch.Tensor:
    """Квадрат Maximum Mean Discrepancy с гауссовым (RBF) ядром.

    Формула:
        MMD²(X, Y) = E[k(x,x')] − 2·E[k(x,y)] + E[k(y,y')],
        где k(a, b) = exp(−γ·||a − b||²).

    Median heuristic (при ``gamma is None``):
        1. Объединить X и Y: ``Z = concat(X, Y)``.
        2. Вычислить попарные квадраты расстояний в Z.
        3. Взять медиану **верхнего треугольника** (без диагонали).
        4. ``γ = 1 / (2 · median)``. Если медиана равна 0 — ``γ = 1.0``.

    Подсказка: используйте ``_pairwise_squared_distances`` для вычисления
    матриц расстояний.

    Args:
        X: Выборка формы ``(N, D)``.
        Y: Выборка формы ``(M, D)``.
        gamma: Ширина ядра. ``None`` — использовать median heuristic.

    Returns:
        Скаляр (0-d тензор).

    Raises:
        ValueError: Если X или Y не 2D, размерности не совпадают,
            или выборка пуста.
    """
    pass


# ======================================================================
# Задание B1.1b. Косинусное расстояние между центроидами
# ======================================================================

def cosine_centroid_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Косинусное расстояние между центроидами (средними) выборок.

    Формула:
        d(X, Y) = 1 − cos(μ_X, μ_Y),

    где μ_X = mean(X, dim=0), μ_Y = mean(Y, dim=0).

    Подсказка: используйте ``F.cosine_similarity`` с ``eps=1e-12``.

    Args:
        X: Выборка формы ``(N, D)``.
        Y: Выборка формы ``(M, D)``.

    Returns:
        Скаляр (0-d тензор) в диапазоне ``[0, 2]``.

    Raises:
        ValueError: Если X или Y не 2D, размерности не совпадают,
            или выборка пуста.
    """
    pass


# ======================================================================
# Задание B1.1c. Sliced Wasserstein Distance
# ======================================================================

def sliced_wasserstein_distance(
    X: torch.Tensor,
    Y: torch.Tensor,
    n_projections: int = 100,
    seed: int | None = None,
) -> torch.Tensor:
    """Монте-Карло оценка Sliced Wasserstein-1 расстояния.

    Алгоритм:
        1. Сгенерировать ``n_projections`` случайных направлений
           на единичной сфере S^{D−1} (через нормализацию N(0, I)).
           Если ``seed is not None`` — создать отдельный
           ``torch.Generator`` с этим seed (не затрагивая глобальный RNG).
        2. Спроецировать X и Y на каждое направление (матричное
           умножение) → одномерные выборки.
        3. Отсортировать проекции вдоль оси выборок.
        4. Если размеры выборок различаются — привести обе к общей
           квантильной сетке размера ``max(N, M)`` через
           ``_quantile_resample``.
        5. Wasserstein-1 между одномерными отсортированными выборками
           одного размера = среднее абсолютной разности.
        6. Усреднить по всем направлениям.

    Args:
        X: Выборка формы ``(N, D)``.
        Y: Выборка формы ``(M, D)``.
        n_projections: Число случайных направлений.
        seed: Seed для генератора направлений (опционально).

    Returns:
        Неотрицательный скаляр (0-d тензор).

    Raises:
        ValueError: Если X или Y не 2D, размерности не совпадают,
            выборка пуста или ``n_projections < 1``.
    """
    pass


def _quantile_resample(
    sorted_values: torch.Tensor,
    quantile_grid: torch.Tensor,
) -> torch.Tensor:
    """Линейная интерполяция отсортированных значений на квантильной сетке.

    Используется в ``sliced_wasserstein_distance`` для выравнивания
    выборок разного размера.

    Args:
        sorted_values: Тензор ``(S, K)`` — каждый столбец отсортирован.
        quantile_grid: Тензор ``(T,)`` со значениями в ``[0, 1]``.

    Returns:
        Тензор ``(T, K)``.
    """
    s = sorted_values.size(0)
    if s == 1:
        return sorted_values.expand(quantile_grid.size(0), -1)

    source_positions = torch.linspace(
        0.0, 1.0, steps=s, device=sorted_values.device, dtype=sorted_values.dtype
    )

    indices = torch.searchsorted(source_positions, quantile_grid)
    indices = indices.clamp(min=1, max=s - 1)

    left = indices - 1
    right = indices

    left_pos = source_positions[left]
    right_pos = source_positions[right]

    weights = (quantile_grid - left_pos) / (right_pos - left_pos)

    left_values = sorted_values[left]    # (T, K)
    right_values = sorted_values[right]  # (T, K)
    return left_values + weights.unsqueeze(1) * (right_values - left_values)