"""Скрипт обучения student-ViT через knowledge distillation на EuroSAT.

Обучает shallow student (3 блока) двумя способами:

1. С distillation от fine-tuned teacher (ViT-Tiny, task 4.2).
2. Без distillation (baseline, обычное обучение на CE).

Сравнивает качество обоих подходов на test-сплите и сохраняет
артефакты в ``results/bonus/b2_distillation/``.

Использование
-------------
    python -m scripts.train_distillation --epochs 10 --batch-size 64

Требования
----------
* Чекпоинт учителя в ``results/model.pt`` (результат task 4.2).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

# Student-written code
from bonus.b2_distillation.student_vit import build_student_vit
from bonus.b2_distillation.train_distill import (
    train_student_baseline,
    train_student_with_distillation,
)

# Provided utilities
from bonus.b2_distillation._teacher_loader import load_teacher
from bonus.b2_distillation._train_utils import evaluate

# Reuse data pipeline from main task 4.2
from scripts.train_vit import get_dataloaders


NUM_CLASSES = 10
NUM_STUDENT_BLOCKS = 3
ARTIFACT_DIR = Path("results/bonus/b2_distillation")


def _parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--teacher-ckpt",
        type=str,
        default="results/model.pt",
        help="Path to teacher checkpoint from task 4.2.",
    )
    return parser.parse_args()


def _plot_curves(
    hist_distill: dict[str, list[float]],
    hist_baseline: dict[str, list[float]],
    out_dir: Path,
) -> None:
    epochs = range(1, len(hist_distill["train_loss"]) + 1)

    # Loss
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, hist_distill["train_loss"], "o-", label="distill train")
    ax[0].plot(epochs, hist_baseline["train_loss"], "s-", label="baseline train")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].set_title("Train loss")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(epochs, hist_distill["val_loss"], "o-", label="distill val")
    ax[1].plot(epochs, hist_baseline["val_loss"], "s-", label="baseline val")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("loss")
    ax[1].set_title("Validation loss")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=120)
    plt.close(fig)

    # Accuracy
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, hist_distill["val_acc"], "o-", label="distill")
    ax.plot(epochs, hist_baseline["val_acc"], "s-", label="baseline")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Validation accuracy: distillation vs baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_curves.png", dpi=120)
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Loading teacher from: {args.teacher_ckpt}")
    teacher = load_teacher(args.teacher_ckpt, num_classes=NUM_CLASSES, device=device)

    print("Preparing dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    criterion_eval = nn.CrossEntropyLoss()

    print(f"\n{'=' * 60}\n  Training student with distillation\n{'=' * 60}")
    student_distill = build_student_vit(
        num_classes=NUM_CLASSES, num_blocks=NUM_STUDENT_BLOCKS
    )
    n_params = sum(p.numel() for p in student_distill.parameters())
    print(f"Student parameters: {n_params:,}")

    t0 = time.time()
    history_distill = train_student_with_distillation(
        student=student_distill,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        alpha=args.alpha,
        temperature=args.temperature,
        device=device,
        seed=args.seed,
    )
    time_distill = time.time() - t0
    test_loss_distill, test_acc_distill = evaluate(
        student_distill, test_loader, criterion_eval, device
    )

    print(f"\n{'=' * 60}\n  Training student baseline (no distillation)\n{'=' * 60}")
    student_baseline = build_student_vit(
        num_classes=NUM_CLASSES, num_blocks=NUM_STUDENT_BLOCKS
    )

    t0 = time.time()
    history_baseline = train_student_baseline(
        student=student_baseline,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        seed=args.seed,
    )
    time_baseline = time.time() - t0
    test_loss_baseline, test_acc_baseline = evaluate(
        student_baseline, test_loader, criterion_eval, device
    )

    metrics = {
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "alpha": args.alpha,
            "temperature": args.temperature,
            "num_blocks": NUM_STUDENT_BLOCKS,
            "seed": args.seed,
        },
        "student_num_parameters": n_params,
        "distill": {
            "test_loss": test_loss_distill,
            "test_accuracy": test_acc_distill,
            "train_time_sec": time_distill,
            "train_history": history_distill,
        },
        "baseline": {
            "test_loss": test_loss_baseline,
            "test_accuracy": test_acc_baseline,
            "train_time_sec": time_baseline,
            "train_history": history_baseline,
        },
    }

    (ARTIFACT_DIR / "metrics_distill.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save(student_distill.state_dict(), ARTIFACT_DIR / "student_model.pt")
    _plot_curves(history_distill, history_baseline, ARTIFACT_DIR)

    print(f"\n{'=' * 60}\n  Results\n{'=' * 60}")
    print(f"Distillation test acc: {test_acc_distill:.4f}  (loss {test_loss_distill:.4f})")
    print(f"Baseline     test acc: {test_acc_baseline:.4f}  (loss {test_loss_baseline:.4f})")
    print(f"\nArtifacts saved to {ARTIFACT_DIR}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())