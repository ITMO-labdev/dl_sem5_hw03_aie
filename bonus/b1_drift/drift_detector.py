"""Детектор дрифта на основе скользящего окна.

Классы:
    DriftDetector — скользящее окно + порог для обнаружения дрифта
                    (задание B1.3).
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from bonus.b1_drift.drift_metrics import (
    cosine_centroid_distance,
    mmd_rbf,
    sliced_wasserstein_distance,
)

_METRICS: dict[str, Callable[..., torch.Tensor]] = {
    "mmd": mmd_rbf,
    "cosine": cosine_centroid_distance,
    "swd": sliced_wasserstein_distance,
}


class DriftDetector:
    """Обнаруживает дрифт распределения методом скользящего окна.

    Принцип работы:
        - ``fit`` сохраняет референсную выборку и вычисляет порог.
        - ``detect`` скользит окном по потоку, считает расстояние
          каждого окна до референса и сообщает о первом превышении
          порога.

    Args:
        metric: Метрика — ``"mmd"``, ``"cosine"`` или ``"swd"``.
        window_size: Размер скользящего окна.
        threshold: Фиксированный порог. Если ``None`` — вычисляется
            адаптивно в ``fit``.
        step: Шаг скользящего окна.
        metric_kwargs: Дополнительные аргументы для метрики
            (например, ``{"gamma": 0.1}`` для ``"mmd"``).
    """

    def __init__(
        self,
        metric: str = "mmd",
        window_size: int = 50,
        threshold: float | None = None,
        step: int = 1,
        metric_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if metric not in _METRICS:
            raise ValueError(
                f"metric must be one of {sorted(_METRICS)}, got {metric!r}"
            )
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        if threshold is not None and threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")

        self.metric = metric
        self.window_size = window_size
        self.threshold = threshold
        self.step = step
        self.metric_kwargs: dict[str, Any] = dict(metric_kwargs or {})

        self._metric_fn = _METRICS[metric]
        self._reference: torch.Tensor | None = None
        self._fitted_threshold: float | None = None

    def _compute_distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Вычисляет расстояние между двумя выборками (скаляр)."""
        return float(self._metric_fn(a, b, **self.metric_kwargs).item())

    # ------------------------------------------------------------------
    # Задание B1.3a — fit
    # ------------------------------------------------------------------

    def fit(self, reference: torch.Tensor) -> None:
        """Сохраняет референсную выборку и вычисляет порог.

        Если ``self.threshold`` задан явно — просто сохраняет его.
        Иначе вычисляет **адаптивный порог** по референсу:
            1. Разбить референс на ``n_windows = N // window_size``
               непересекающихся окон.
            2. Для каждой пары окон ``(i, j)`` (``i < j``) вычислить
               расстояние через ``_compute_distance``.
            3. Порог = ``mean + 3 * std`` расстояний (``std`` с
               ``unbiased=False``; если расстояние одно — ``std = 0``).

        Args:
            reference: Референсная выборка ``(N_ref, D)``.

        Raises:
            ValueError: Если ``reference`` не 2D, содержит меньше
                ``window_size`` сэмплов, или для адаптивного порога
                нужно ``>= 2 * window_size`` сэмплов.
        """
        pass

    # ------------------------------------------------------------------
    # Задание B1.3b — detect
    # ------------------------------------------------------------------

    def detect(self, stream: torch.Tensor) -> dict[str, Any]:
        """Скользит окном по потоку и находит первый дрифт.

        Алгоритм:
            1. Для каждой позиции ``start`` от 0 до
               ``T − window_size`` с шагом ``step``:
               - Вырезать окно ``stream[start : start + window_size]``.
               - Вычислить расстояние до **всего** референса
                 через ``_compute_distance``.
               - Если расстояние **строго больше** порога и это первое
                 превышение — запомнить ``start`` как ``drift_index``.
            2. Вернуть словарь с результатами.

        Args:
            stream: Входной поток ``(T, D)``.

        Returns:
            Словарь с ключами:
                - ``"drift_detected"``: ``bool``.
                - ``"drift_index"``: ``int | None`` — индекс начала
                  первого окна с дрифтом.
                - ``"distances"``: ``list[float]`` — расстояния
                  для каждого окна.
                - ``"threshold"``: ``float`` — использованный порог.

        Raises:
            RuntimeError: Если ``fit`` не был вызван.
            ValueError: Если ``stream`` не 2D или размерности
                не совпадают с референсом.
        """
        pass