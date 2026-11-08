"""Генератор синтетических потоков эмбеддингов с контролируемым дрифтом.

Выдаётся заполненным как testing/debugging утилита. **Не изменяйте этот файл.**

Поток генерируется из гауссовой смеси (GMM) с диагональной ковариацией.
Точка дрифта делит поток на два режима:
    - до дрифта: сэмплы из базового GMM,
    - после дрифта: сэмплы из возмущённого GMM (сдвиг средних,
      изменение дисперсии или перераспределение весов компонент).

Классы:
    SyntheticDriftStream — генератор потока с заданным типом дрифта.
"""

from __future__ import annotations

import torch

_VALID_DRIFT_TYPES: frozenset[str] = frozenset(
    {"mean_shift", "variance_change", "component_swap"}
)


class SyntheticDriftStream:
    """Генерирует потоки эмбеддингов с контролируемым распределённым дрифтом.

    Базовое распределение — GMM с ``n_components`` компонентами,
    диагональной ковариацией и случайными весами. При инициализации
    параметры GMM (средние, стандартные отклонения, веса) фиксируются
    по ``seed`` и не меняются между вызовами ``generate``.

    Поддерживаемые типы дрифта:
        - ``"mean_shift"``: средние всех компонент сдвигаются в случайном
          направлении на величину ``drift_intensity``.
        - ``"variance_change"``: стандартные отклонения всех компонент
          умножаются на ``drift_intensity``.
        - ``"component_swap"``: веса компонент разворачиваются
          (``weights.flip(0)``) — минорная компонента становится
          мажорной и наоборот. ``drift_intensity`` не используется.

    Args:
        d: Размерность эмбеддингов.
        n_components: Число компонент в GMM.
        seed: Seed для воспроизводимости параметров GMM.

    Raises:
        ValueError: Если ``d < 1`` или ``n_components < 1``.

    Example:
        >>> gen = SyntheticDriftStream(d=16, n_components=3, seed=42)
        >>> stream, drift_point = gen.generate(
        ...     n_before=200, n_after=200, drift_type="mean_shift"
        ... )
        >>> stream.shape, drift_point
        (torch.Size([400, 16]), 200)
    """

    def __init__(self, d: int, n_components: int = 3, seed: int = 42) -> None:
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")

        self.d = d
        self.n_components = n_components
        self.seed = seed

        generator = torch.Generator().manual_seed(seed)

        # Базовые параметры GMM.
        self._means = torch.randn(n_components, d, generator=generator) * 2.0
        self._stds = (
            torch.rand(n_components, d, generator=generator) * 0.5 + 0.5
        )  # в [0.5, 1.0]
        weights = torch.rand(n_components, generator=generator) + 0.1
        self._weights = weights / weights.sum()

    def _sample_gmm(
        self,
        n: int,
        means: torch.Tensor,
        stds: torch.Tensor,
        weights: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Сэмплирует ``n`` точек из GMM с диагональной ковариацией.

        Для каждой точки:
            1. Выбирается индекс компоненты по весам (``multinomial``).
            2. Генерируется шум ``N(0, I)`` и масштабируется на ``stds``
               выбранной компоненты.
            3. Прибавляется среднее выбранной компоненты.

        Args:
            n: Число сэмплов.
            means: Средние компонент, ``(n_components, d)``.
            stds: Стандартные отклонения, ``(n_components, d)``.
            weights: Веса компонент (нормированные), ``(n_components,)``.
            generator: Генератор случайных чисел для воспроизводимости.

        Returns:
            Тензор формы ``(n, d)``.
        """
        component_ids = torch.multinomial(
            weights, n, replacement=True, generator=generator
        )
        noise = torch.randn(n, self.d, generator=generator)
        return means[component_ids] + noise * stds[component_ids]

    def generate(
        self,
        n_before: int,
        n_after: int,
        drift_type: str = "mean_shift",
        drift_intensity: float = 3.0,
    ) -> tuple[torch.Tensor, int]:
        """Генерирует поток с дрифтом, начинающимся в позиции ``n_before``.

        Первые ``n_before`` сэмплов берутся из базового GMM, последующие
        ``n_after`` — из возмущённого GMM согласно ``drift_type``.
        При ``n_after == 0`` возвращается стационарный поток
        (эквивалент отсутствия дрифта).

        Args:
            n_before: Число сэмплов из базового распределения.
            n_after: Число сэмплов из возмущённого распределения.
            drift_type: Тип возмущения — ``"mean_shift"``,
                ``"variance_change"`` или ``"component_swap"``.
            drift_intensity: Магнитуда возмущения. Игнорируется при
                ``drift_type="component_swap"``.

        Returns:
            Кортеж ``(stream, drift_point)``:
                - ``stream``: тензор формы ``(n_before + n_after, d)``.
                - ``drift_point``: индекс начала дрифта (равен ``n_before``).

        Raises:
            ValueError: Если ``n_before < 1``, ``n_after < 0``,
                или ``drift_type`` не из допустимого набора.
        """
        if n_before < 1:
            raise ValueError(f"n_before must be >= 1, got {n_before}")
        if n_after < 0:
            raise ValueError(f"n_after must be >= 0, got {n_after}")
        if drift_type not in _VALID_DRIFT_TYPES:
            raise ValueError(
                f"drift_type must be one of {sorted(_VALID_DRIFT_TYPES)}, "
                f"got {drift_type!r}"
            )

        generator = torch.Generator().manual_seed(self.seed + 1)
        before = self._sample_gmm(
            n_before, self._means, self._stds, self._weights, generator
        )

        if n_after == 0:
            return before, n_before

        # Применяем возмущение распределения.
        if drift_type == "mean_shift":
            shift_dir = torch.randn(self.d, generator=generator)
            shift_dir = shift_dir / shift_dir.norm().clamp_min(1e-12)
            new_means = self._means + shift_dir * drift_intensity
            new_stds = self._stds
            new_weights = self._weights
        elif drift_type == "variance_change":
            new_means = self._means
            new_stds = self._stds * drift_intensity
            new_weights = self._weights
        else:  # "component_swap"
            new_means = self._means
            new_stds = self._stds
            # Разворот весов: мажорная компонента <-> минорная.
            new_weights = self._weights.flip(0)

        after = self._sample_gmm(
            n_after, new_means, new_stds, new_weights, generator
        )
        stream = torch.cat([before, after], dim=0)
        return stream, n_before