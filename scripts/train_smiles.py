"""CLI-скрипт обучения ESOL-регрессоров.

Выдаётся заполненным. **Не изменяйте этот файл.**

Обучает две модели по одинаковому рецепту — Transformer-регрессор
и bag-of-tokens baseline — и сохраняет артефакты в
``results/bonus/b3_smiles/``:

- ``metrics.json``           — MSE, RMSE, MAE, R² и история обучения
                               для обеих моделей (на test-сплите);
- ``transformer_model.pt``   — ``state_dict`` обученного трансформера;
- ``loss_curves.png``        — кривые train/val MSE для обеих моделей.

Сплит: train ≈ 80 %, val ≈ 10 %, test ≈ 10 %.  Метрики в ``metrics.json``
считаются на **test**-сплите, который не участвовал ни в обучении, ни в
мониторинге сходимости.  Кривые на графике — train/val, чтобы было видно
динамику обучения.

Пример запуска
--------------
    python -m scripts.train_smiles --epochs 50
"""

from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from bonus.b3_smiles._esol_data import esol_collate_fn, load_esol
from bonus.b3_smiles._train_utils import evaluate_mse, set_seed
from bonus.b3_smiles.smiles_model import (
    BagOfTokensBaseline,
    SmilesTransformerRegressor,
)
from bonus.b3_smiles.train_smiles import train_regressor
from modules.tokenization import SmilesTokenizer


def _parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="bonus/b3_smiles/data/esol.csv")
    p.add_argument("--out", default="results/bonus/b3_smiles")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _r2_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Коэффициент детерминации R².

    R² инвариантен к общему аффинному преобразованию предсказаний
    и таргетов, поэтому его можно считать прямо в нормализованных
    z-единицах — значение совпадёт с R² в исходных единицах.

    Args:
        preds: Предсказания модели, ``(N,)``.
        targets: Истинные значения, ``(N,)``.

    Returns:
        ``1 - SS_res / SS_tot``. Значение ``0`` соответствует модели,
        всегда предсказывающей среднее.
    """
    ss_res = ((targets - preds) ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().item()
    return 1.0 - ss_res / max(ss_tot, 1e-12)


@torch.no_grad()
def _collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Собирает предсказания и таргеты по всему loader'у.

    Args:
        model: Обученная модель.
        loader: DataLoader.
        device: Устройство для инференса.

    Returns:
        Кортеж ``(preds, targets)``, оба тензора формы ``(N,)`` на CPU.
    """
    was_training = model.training
    model.eval()
    preds, tgts = [], []
    try:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            preds.append(model(input_ids, mask).squeeze(-1).cpu())
            tgts.append(batch["target"])
    finally:
        if was_training:
            model.train()
    return torch.cat(preds), torch.cat(tgts)


def _compute_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    target_std: float,
    prefix: str,
) -> dict[str, float]:
    """Считает MSE, RMSE, MAE и R² для одной модели.

    MSE возвращается в нормализованных z-единицах (в них же считался
    loss при обучении), а RMSE и MAE дополнительно пересчитываются
    в исходные единицы log-solubility умножением на ``target_std`` —
    так их можно интерпретировать химически.

    Args:
        model: Обученная модель.
        loader: DataLoader для оценки (обычно test).
        device: Устройство.
        target_std: Стандартное отклонение train-таргетов.
        prefix: Префикс ключей в словаре (``"transformer"`` или ``"baseline"``).

    Returns:
        Словарь с ключами ``{prefix}_mse``, ``{prefix}_r2``,
        ``{prefix}_rmse_logS``, ``{prefix}_mae_logS``.
    """
    preds, targets = _collect_predictions(model, loader, device)
    mse = evaluate_mse(model, loader, device=device)
    return {
        f"{prefix}_mse": mse,
        f"{prefix}_r2": _r2_score(preds, targets),
        f"{prefix}_rmse_logS": (mse ** 0.5) * target_std,
        f"{prefix}_mae_logS": (preds - targets).abs().mean().item() * target_std,
    }


def main() -> None:
    """Обучает обе модели, считает метрики и сохраняет артефакты."""
    args = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(args.seed)

    # Словарь строится по полному корпусу SMILES (см. docstring модуля).
    tokenizer = SmilesTokenizer()
    with open(args.csv, newline="", encoding="utf-8") as f:
        corpus = [row["smiles"].strip() for row in csv.DictReader(f)]
    tokenizer.fit(corpus)
    pad_id = tokenizer.vocab["<pad>"]

    train_ds, val_ds, test_ds, mean, std = load_esol(
        args.csv, tokenizer, seed=args.seed,
    )

    # partial, а не lambda: lambda не пиклится и сломает num_workers > 0.
    collate = partial(esol_collate_fn, pad_token_id=pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
    )

    # ── Transformer ───────────────────────────────────────────────────────
    # set_seed перед каждой моделью: инициализация весов и порядок
    # перемешивания батчей должны стартовать из одного состояния RNG,
    # иначе сравнение архитектур нечестное.
    set_seed(args.seed)
    transformer = SmilesTransformerRegressor(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        max_len=args.max_len,
        pad_idx=pad_id,
    )
    tf_params = sum(p.numel() for p in transformer.parameters())
    tf_history = train_regressor(
        transformer, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=device,
    )
    # Финальная оценка — на test, а не на val (val использовался для
    # мониторинга сходимости, поэтому метрики на нём были бы оптимистичны).
    tf_metrics = _compute_metrics(
        transformer, test_loader, device, std, prefix="transformer",
    )

    # ── Baseline ──────────────────────────────────────────────────────────
    set_seed(args.seed)
    baseline = BagOfTokensBaseline(
        vocab_size=tokenizer.vocab_size, pad_idx=pad_id,
    )
    bl_params = sum(p.numel() for p in baseline.parameters())
    bl_history = train_regressor(
        baseline, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=device,
    )
    bl_metrics = _compute_metrics(
        baseline, test_loader, device, std, prefix="baseline",
    )

    # ── Artifacts ─────────────────────────────────────────────────────────
    metrics = {
        **tf_metrics,
        **bl_metrics,
        "transformer_num_parameters": tf_params,
        "baseline_num_parameters": bl_params,
        "target_mean": mean,
        "target_std": std,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "d_ff": args.d_ff,
            "num_layers": args.num_layers,
            "max_len": args.max_len,
            "seed": args.seed,
        },
        "train_history": {"transformer": tf_history, "baseline": bl_history},
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8",
    )
    torch.save(transformer.state_dict(), out_dir / "transformer_model.pt")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tf_history["train_loss"], label="transformer/train")
    ax.plot(tf_history["val_loss"], label="transformer/val")
    ax.plot(bl_history["train_loss"], label="baseline/train", linestyle="--")
    ax.plot(bl_history["val_loss"], label="baseline/val", linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (normalised units)")
    ax.set_title("ESOL training curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=120)
    plt.close(fig)

    print(
        f"Transformer ({tf_params:,} params): "
        f"MSE={tf_metrics['transformer_mse']:.4f}, "
        f"R²={tf_metrics['transformer_r2']:.4f}, "
        f"RMSE={tf_metrics['transformer_rmse_logS']:.4f} logS"
    )
    print(
        f"Baseline    ({bl_params:,} params): "
        f"MSE={bl_metrics['baseline_mse']:.4f}, "
        f"R²={bl_metrics['baseline_r2']:.4f}, "
        f"RMSE={bl_metrics['baseline_rmse_logS']:.4f} logS"
    )


if __name__ == "__main__":
    main()