"""Loss для knowledge distillation.

Классы:
    DistillationLoss — комбинация KL-divergence на softened logits teacher
                       и CrossEntropy на hard labels (задание B2.2).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# ======================================================================
# Задание B2.2. Distillation Loss
# ======================================================================

class DistillationLoss(nn.Module):
    """Комбинированный loss для дистилляции знаний (Hinton et al., 2015).

    Формула:
        L = α · T² · KL(σ(z_s / T) ‖ σ(z_t / T)) + (1 − α) · CE(z_s, y)

    где:
        - z_s, z_t — логиты student и teacher;
        - σ — softmax, T — температура;
        - y — истинные метки;
        - KL-divergence усредняется по батчу (``reduction="batchmean"``).

    Множитель T² выравнивает масштаб градиентов KL-члена с градиентами
    CE-члена: при повышении температуры мягкие вероятности становятся
    более размытыми, а их градиенты масштабируются как 1/T². Умножение
    на T² возвращает им прежний масштаб (Hinton et al., §2.1).

    Ответственность вызывающего кода: передавать ``teacher_logits``
    без градиентов (через ``torch.no_grad()`` при прогоне teacher
    или через ``.detach()``), чтобы градиенты не текли в teacher.

    Атрибуты:
        alpha: Вес KL-члена в ``[0, 1]``.
            - ``alpha=0`` → чистый CrossEntropy (student учится только
              по hard labels);
            - ``alpha=1`` → чистый KL (student учится только по soft
              targets teacher).
        temperature: Температура softmax, ``>= 1``.
    """

    def __init__(self, alpha: float = 0.5, temperature: float = 4.0) -> None:
        """Инициализирует гиперпараметры loss'а.

        Args:
            alpha: Вес KL-члена в ``[0, 1]``.
            temperature: Температура softmax, ``>= 1``.

        Raises:
            ValueError: Если ``alpha`` вне ``[0, 1]`` или
                ``temperature < 1``.
        """
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if temperature < 1.0:
            raise ValueError(f"temperature must be >= 1.0, got {temperature}")

        self.alpha = float(alpha)
        self.temperature = float(temperature)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Вычисляет комбинированный loss.

        Подсказки по реализации:
            - KL-часть удобно считать через ``F.kl_div`` с параметром
              ``reduction="batchmean"`` (математически корректная
              нормировка). Первым аргументом в ``F.kl_div`` идёт
              **log-probabilities** (``F.log_softmax(... / T)``),
              вторым — **probabilities** (``F.softmax(... / T)``).
            - CE-часть: ``F.cross_entropy(student_logits, labels)``
              работает с сырыми логитами (без softmax) и int-метками.
            - Не забудьте множитель ``T²`` перед KL-членом.

        Args:
            student_logits: Логиты student, форма ``(B, C)``,
                с градиентами.
            teacher_logits: Логиты teacher, форма ``(B, C)``,
                должны быть detached (без градиентов).
            labels: Истинные метки, ``torch.long``, форма ``(B,)``.

        Returns:
            Скалярный (0-d) тензор с общим loss'ом.

        Raises:
            ValueError: Если формы ``student_logits`` и
                ``teacher_logits`` не совпадают, или ``labels`` не 1D,
                или размер батча в ``labels`` не совпадает с ``B``.
        """
        pass