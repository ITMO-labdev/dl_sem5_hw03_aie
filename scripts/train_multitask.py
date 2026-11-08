"""CLI-скрипт multi-task обучения на EuroSAT.

Выдаётся заполненным. **Не изменяйте этот файл.**

Обучает ``MultiTaskViT`` (общий backbone ViT-Tiny + голова классификации +
голова pseudo-NDVI регрессии) по одному рецепту и сохраняет артефакты
в ``results/bonus/b4_multitask/``:

- ``metrics.json``           — classification accuracy и regression MSE
                               на test-сплите, конфиг запуска,
                               история обучения;
- ``multitask_model.pt``     — ``state_dict`` обученной модели;
- ``loss_curves.png``        — кривая train-loss и валидационные метрики.

Сплит: train ≈ 80 %, val ≈ 10 %, test ≈ 10 %. Val-сплит используется
для мониторинга сходимости, test-сплит — только для финальной оценки
в ``metrics.json``.

Backbone инициализируется весами fine-tuned ViT из задания 4.2
(``results/model.pt``), после чего классификационная голова заменяется
двумя новыми линейными головами (cls и reg), обучаемыми с нуля.

Пример запуска
--------------
    python -m scripts.train_multitask --epochs 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import timm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from bonus.b4_multitask._model_utils import strip_classification_head
from bonus.b4_multitask._train_utils import evaluate_multitask, set_seed
from bonus.b4_multitask.multitask_dataset import EuroSATMultiTaskDataset
from bonus.b4_multitask.multitask_model import MultiTaskViT
from bonus.b4_multitask.train_multitask import train_multitask


def _parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="results/model.pt",
                   help="Путь к fine-tuned ViT из задания 4.2.")
    p.add_argument("--out", default="results/bonus/b4_multitask")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _get_transform() -> transforms.Compose:
    """Стандартный препроцессинг для ImageNet-pretrained ViT.

    Returns:
        Compose, возвращающий тензор ``(3, 224, 224)`` после ImageNet
        нормализации. NDVI в датасете считается **до** нормализации
        (от raw ``[0, 1]`` тензора), поэтому Normalize здесь безопасен.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def _collate(batch):
    """Собирает батч словарей в один словарь тензоров."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "cls_target": torch.stack([b["cls_target"] for b in batch]),
        "reg_target": torch.stack([b["reg_target"] for b in batch]),
    }


def main() -> None:
    """Обучает multi-task модель, считает метрики и сохраняет артефакты."""
    args = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(args.seed)

    # ── Dataset (lazy — no RAM explosion) ─────────────────────────────────
    from datasets import load_dataset

    hf_ds = load_dataset("timm/eurosat-rgb", split="train")
    n_total = len(hf_ds)
    n_test = max(1, round(n_total * 0.1))
    n_val = max(1, round(n_total * 0.1))
    n_train = n_total - n_val - n_test

    rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_total, generator=rng).tolist()
    train_indices = perm[:n_train]
    val_indices = perm[n_train : n_train + n_val]
    test_indices = perm[n_train + n_val :]

    transform = _get_transform()
    train_ds = EuroSATMultiTaskDataset(hf_ds, train_indices, transform)
    val_ds = EuroSATMultiTaskDataset(hf_ds, val_indices, transform)
    test_ds = EuroSATMultiTaskDataset(hf_ds, test_indices, transform)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=_collate, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=_collate, num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Чекпоинт teacher'а не найден: {ckpt_path}\n"
            f"Сначала выполните задание 4.2 и сохраните results/model.pt."
        )
    vit = timm.create_model(
        "vit_tiny_patch16_224", pretrained=False, num_classes=10
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    vit.load_state_dict(ckpt)

    backbone, d_model = strip_classification_head(vit)
    model = MultiTaskViT(backbone, d_model=d_model, num_classes=10)

    # ── Train ─────────────────────────────────────────────────────────────
    history = train_multitask(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, alpha=args.alpha, device=device,
    )

    # Финальная оценка — на test, а не на val.
    test_metrics = evaluate_multitask(model, test_loader, device=device)

    # ── Artifacts ─────────────────────────────────────────────────────────
    metrics = {
        "cls_accuracy": test_metrics["cls_accuracy"],
        "reg_mse": test_metrics["reg_mse"],
        "alpha": args.alpha,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "alpha": args.alpha,
            "seed": args.seed,
        },
        "train_history": history,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8",
    )
    torch.save(model.state_dict(), out_dir / "multitask_model.pt")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="train_loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Multi-task training loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["val_cls_acc"], label="val cls accuracy")
    ax2.plot(history["val_reg_mse"], label="val reg MSE")
    ax2.set_xlabel("epoch")
    ax2.set_title("Validation metrics")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=120)
    plt.close(fig)

    print(
        f"Multi-task model ({metrics['num_parameters']:,} params): "
        f"test cls_accuracy={test_metrics['cls_accuracy']:.4f}, "
        f"test reg_mse={test_metrics['reg_mse']:.6f}"
    )


if __name__ == "__main__":
    main()