"""Real-time loss logging and post-training convergence diagnostics.

Generated inside the active Lightning version directory:
  loss_batches.csv
  loss_validation_epochs.csv
  loss_epochs.csv
  loss_summary.json
  loss_curve.png

Regenerate the curve after an interrupted run:
  python tools/loss_monitor.py --log-dir lightning_logs/.../version_0
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import pytorch_lightning as pl
except ImportError:
    pl = None


BATCH_FIELDS = [
    "epoch",
    "global_step",
    "batch_idx",
    "loss",
    "learning_rate",
    "elapsed_seconds",
]
VAL_FIELDS = ["epoch", "val_loss", "val_F1", "val_mIoU", "val_OA"]
EPOCH_FIELDS = [
    "epoch",
    "train_loss_mean",
    "train_loss_min",
    "train_loss_max",
    "epoch_batches",
    "learning_rate",
    "val_loss",
    "val_F1",
    "val_mIoU",
    "val_OA",
]


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
        try:
            value = value.mean()
        except (RuntimeError, ValueError):
            return None
        return float(value.cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _output_loss(outputs: Any, keys: tuple[str, ...]) -> float | None:
    if isinstance(outputs, dict):
        for key in keys:
            value = _number(outputs.get(key))
            if value is not None:
                return value
        for value in outputs.values():
            scalar = _number(value)
            if scalar is not None:
                return scalar
        return None
    return _number(outputs)


def _metric(trainer: Any, name: str) -> float | None:
    return _number(getattr(trainer, "callback_metrics", {}).get(name))


def _learning_rate(trainer: Any) -> float | None:
    try:
        return float(trainer.optimizers[0].param_groups[0]["lr"])
    except (AttributeError, IndexError, KeyError, TypeError):
        return None


def _find_log_dir(trainer: Any) -> Path:
    log_dir = getattr(trainer, "log_dir", None)
    if log_dir:
        return Path(log_dir)
    for logger in getattr(trainer, "loggers", []):
        log_dir = getattr(logger, "log_dir", None)
        if log_dir:
            return Path(log_dir)
    return Path(getattr(trainer, "default_root_dir", ".")) / "loss_monitor"


def _open_csv(path: Path, fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    handle = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    if new_file:
        writer.writeheader()
        handle.flush()
    return handle, writer


def _append_row(handle, writer, row: dict[str, Any]) -> None:
    writer.writerow(row)
    handle.flush()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _epoch(value: Any) -> int | None:
    scalar = _number(value)
    return int(scalar) if scalar is not None else None


def _moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    window = max(1, min(window, len(values)))
    queue: list[float] = []
    total = 0.0
    result: list[float] = []
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.pop(0)
        result.append(total / len(queue))
    return result


def _slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 1e-12:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def _build_epoch_rows(
    batch_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    train_by_epoch: dict[int, list[dict[str, Any]]] = {}
    for row in batch_rows:
        epoch = _epoch(row.get("epoch"))
        loss = _number(row.get("loss"))
        if epoch is None or loss is None:
            continue
        train_by_epoch.setdefault(epoch, []).append(
            {
                "loss": loss,
                "step": _epoch(row.get("global_step")),
                "lr": _number(row.get("learning_rate")),
            }
        )

    val_by_epoch: dict[int, dict[str, str]] = {}
    for row in val_rows:
        epoch = _epoch(row.get("epoch"))
        if epoch is not None:
            val_by_epoch[epoch] = row

    result: list[dict[str, Any]] = []
    for epoch in sorted(set(train_by_epoch) | set(val_by_epoch)):
        train_values = train_by_epoch.get(epoch, [])
        losses = [row["loss"] for row in train_values]
        val = val_by_epoch.get(epoch, {})
        result.append(
            {
                "epoch": epoch,
                "train_loss_mean": mean(losses) if losses else None,
                "train_loss_min": min(losses) if losses else None,
                "train_loss_max": max(losses) if losses else None,
                "epoch_batches": len(losses),
                "learning_rate": train_values[-1]["lr"] if train_values else None,
                "val_loss": _number(val.get("val_loss")),
                "val_F1": _number(val.get("val_F1")),
                "val_mIoU": _number(val.get("val_mIoU")),
                "val_OA": _number(val.get("val_OA")),
            }
        )
    return result


def _diagnose(epoch_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_rows = [row for row in epoch_rows if row["train_loss_mean"] is not None]
    val_rows = [row for row in epoch_rows if row["val_loss"] is not None]
    summary: dict[str, Any] = {
        "epochs_logged": len(epoch_rows),
        "last_epoch": epoch_rows[-1]["epoch"] if epoch_rows else None,
        "note": (
            "Loss-based diagnostics only. Use validation mIoU/F1 for model "
            "selection and held-out test metrics for final capability."
        ),
    }
    if not train_rows:
        summary["status"] = "No training loss data"
        summary["recommendation"] = "Check that LossMonitorCallback is attached."
        return summary

    summary["last_train_loss"] = train_rows[-1]["train_loss_mean"]
    summary["best_train_loss"] = min(row["train_loss_mean"] for row in train_rows)
    summary["best_train_epoch"] = min(
        train_rows, key=lambda row: row["train_loss_mean"]
    )["epoch"]

    if not val_rows:
        summary["status"] = "Training loss only"
        summary["recommendation"] = "Validation loss is unavailable."
        return summary

    best_val = min(val_rows, key=lambda row: row["val_loss"])
    best_f1 = max(
        (row for row in val_rows if row["val_F1"] is not None),
        key=lambda row: row["val_F1"],
        default=None,
    )
    summary.update(
        {
            "last_val_loss": val_rows[-1]["val_loss"],
            "best_val_loss": best_val["val_loss"],
            "best_val_epoch": best_val["epoch"],
            "epochs_since_best_val": val_rows[-1]["epoch"] - best_val["epoch"],
        }
    )
    if best_f1 is not None:
        summary["best_val_F1"] = best_f1["val_F1"]
        summary["best_val_F1_epoch"] = best_f1["epoch"]

    recent = val_rows[-min(10, len(val_rows)) :]
    val_change = recent[-1]["val_loss"] - recent[0]["val_loss"]
    val_change_pct = 100.0 * val_change / max(abs(recent[0]["val_loss"]), 1e-12)
    val_slope = _slope(
        [float(row["epoch"]) for row in recent],
        [row["val_loss"] for row in recent],
    )
    train_lookup = {row["epoch"]: row["train_loss_mean"] for row in train_rows}
    gaps = [
        row["val_loss"] - train_lookup[row["epoch"]]
        for row in recent
        if row["epoch"] in train_lookup
    ]
    train_recent = train_rows[-min(10, len(train_rows)) :]
    train_slope = _slope(
        [float(row["epoch"]) for row in train_recent],
        [row["train_loss_mean"] for row in train_recent],
    )
    gap_change = gaps[-1] - gaps[0] if len(gaps) >= 2 else 0.0

    summary["recent_val_change_pct"] = val_change_pct
    summary["recent_mean_gap"] = mean(gaps) if gaps else None

    epochs_since_best = summary["epochs_since_best_val"]
    if train_slope < 0 and val_slope > 0 and gap_change > 0:
        status = "Possible overfitting"
        advice = (
            "Training loss is still falling while validation loss is rising. "
            "Try earlier stopping, stronger regularization, or a smaller model."
        )
    elif epochs_since_best >= 10 and val_slope >= 0:
        status = "Validation plateau reached"
        advice = (
            f"Best validation loss was {epochs_since_best} epochs ago. "
            "More epochs are unlikely to help without changing data, loss, "
            "learning-rate schedule, or model capacity."
        )
    elif epochs_since_best <= 4 and val_change_pct < -1.0:
        status = "Still improving"
        advice = "Validation loss is still decreasing; a longer run may help."
    elif abs(val_change_pct) <= 1.0 and epochs_since_best <= 10:
        status = "Likely converged"
        advice = "Recent validation loss is close to its best value."
    elif train_slope >= 0 and val_slope >= 0:
        status = "Loss not decreasing"
        advice = (
            "Both losses have stopped decreasing. Check underfitting, learning "
            "rate, augmentation strength, and task metrics."
        )
    else:
        status = "Mixed or unstable"
        advice = (
            "Loss trends are inconsistent. Inspect schedule, batch composition, "
            "and validation variance before tuning."
        )
    summary["status"] = status
    summary["recommendation"] = advice
    return summary


def render_loss_curve(
    log_dir: str | Path,
    smooth_window: int = 101,
) -> dict[str, Path]:
    log_dir = Path(log_dir)
    batch_path = log_dir / "loss_batches.csv"
    val_path = log_dir / "loss_validation_epochs.csv"
    epoch_path = log_dir / "loss_epochs.csv"
    summary_path = log_dir / "loss_summary.json"
    curve_path = log_dir / "loss_curve.png"

    batch_rows = _read_csv(batch_path)
    val_rows = _read_csv(val_path)
    if not batch_rows and not val_rows:
        raise FileNotFoundError(f"No loss CSV files found in {log_dir}")

    epoch_rows = _build_epoch_rows(batch_rows, val_rows)
    if not epoch_rows:
        raise ValueError("Loss CSV files contain no valid rows.")

    with epoch_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPOCH_FIELDS)
        writer.writeheader()
        writer.writerows(epoch_rows)

    summary = _diagnose(epoch_rows)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps: list[float] = []
    losses: list[float] = []
    for row in batch_rows:
        loss = _number(row.get("loss"))
        if loss is None:
            continue
        step = _number(row.get("global_step"))
        steps.append(step if step is not None else float(len(steps)))
        losses.append(loss)

    train_x = [
        float(row["epoch"])
        for row in epoch_rows
        if row["train_loss_mean"] is not None
    ]
    train_y = [
        row["train_loss_mean"]
        for row in epoch_rows
        if row["train_loss_mean"] is not None
    ]
    val_x = [float(row["epoch"]) for row in epoch_rows if row["val_loss"] is not None]
    val_y = [row["val_loss"] for row in epoch_rows if row["val_loss"] is not None]

    fig = plt.figure(figsize=(15, 12), dpi=150, facecolor="white")
    grid = fig.add_gridspec(4, 1, height_ratios=[2.4, 1.5, 1.0, 0.65], hspace=0.42)
    ax_batch = fig.add_subplot(grid[0])
    ax_epoch = fig.add_subplot(grid[1])
    ax_gap = fig.add_subplot(grid[2])
    ax_info = fig.add_subplot(grid[3])

    if losses:
        ax_batch.plot(steps, losses, color="#94A3B8", linewidth=0.7, alpha=0.35)
        ax_batch.plot(
            steps,
            _moving_average(losses, smooth_window),
            color="#D97706",
            linewidth=2.0,
            label="Moving average",
        )
        ax_batch.legend(loc="upper right", frameon=False)
    ax_batch.set_title("Batch-level training loss")
    ax_batch.set_xlabel("Global training step")
    ax_batch.set_ylabel("Loss")
    ax_batch.grid(alpha=0.18)

    if train_y:
        ax_epoch.plot(
            train_x,
            train_y,
            marker="o",
            markersize=3,
            color="#D97706",
            linewidth=2,
            label="Train loss (epoch mean)",
        )
    if val_y:
        ax_epoch.plot(
            val_x,
            val_y,
            marker="o",
            markersize=3,
            color="#0F766E",
            linewidth=2,
            label="Validation loss",
        )
        ax_epoch.axvline(
            summary["best_val_epoch"],
            color="#B42318",
            linestyle="--",
            linewidth=1.2,
            label=f"Best val epoch {summary['best_val_epoch']}",
        )
    ax_epoch.set_title("Epoch-level loss")
    ax_epoch.set_xlabel("Epoch")
    ax_epoch.set_ylabel("Loss")
    ax_epoch.legend(loc="best", frameon=False)
    ax_epoch.grid(alpha=0.18)

    if train_y and val_y:
        gap_x = []
        gap_y = []
        train_lookup = dict(zip(train_x, train_y))
        for x, y in zip(val_x, val_y):
            if x in train_lookup:
                gap_x.append(x)
                gap_y.append(y - train_lookup[x])
        ax_gap.plot(gap_x, gap_y, color="#3B6F95", linewidth=1.8)
        ax_gap.axhline(0, color="#64748B", linewidth=1)
        ax_gap.fill_between(
            gap_x,
            gap_y,
            0,
            where=[value >= 0 for value in gap_y],
            color="#FCA5A5",
            alpha=0.22,
        )
    ax_gap.set_title("Generalization gap (validation - train)")
    ax_gap.set_xlabel("Epoch")
    ax_gap.set_ylabel("Loss gap")
    ax_gap.grid(alpha=0.18)

    ax_info.axis("off")
    info_lines = [
        f"Last epoch: {summary.get('last_epoch')}    "
        f"Last train loss: {summary.get('last_train_loss')}",
        f"Best val loss: {summary.get('best_val_loss')} at epoch "
        f"{summary.get('best_val_epoch')}    "
        f"Best val F1: {summary.get('best_val_F1')} at epoch "
        f"{summary.get('best_val_F1_epoch')}",
        f"Recent val loss change: {summary.get('recent_val_change_pct', 0.0):.2f}%    "
        f"Status: {summary['status']}",
        summary["recommendation"],
        summary["note"],
    ]
    ax_info.text(
        0.01,
        0.95,
        "\n".join(info_lines),
        ha="left",
        va="top",
        fontsize=9.5,
        color="#1F2937",
        linespacing=1.45,
        transform=ax_info.transAxes,
    )
    fig.suptitle(
        "Training loss diagnostics",
        fontsize=16,
        fontweight="bold",
        color="#17324D",
        y=0.985,
    )
    fig.savefig(curve_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "curve": curve_path,
        "epoch_csv": epoch_path,
        "summary_json": summary_path,
    }


_CallbackBase = pl.Callback if pl is not None else object


class LossMonitorCallback(_CallbackBase):
    """Log each batch loss and render a curve when training ends."""

    def __init__(self, smooth_window: int = 101):
        super().__init__()
        self.smooth_window = smooth_window
        self._log_dir: Path | None = None
        self._batch_handle = None
        self._batch_writer = None
        self._val_handle = None
        self._val_writer = None
        self._val_losses: list[float] = []
        self._start_time = 0.0
        self._closed = False

    def on_train_start(self, trainer, pl_module):
        self._close_files()
        self._log_dir = _find_log_dir(trainer)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._batch_handle, self._batch_writer = _open_csv(
            self._log_dir / "loss_batches.csv",
            BATCH_FIELDS,
        )
        self._val_handle, self._val_writer = _open_csv(
            self._log_dir / "loss_validation_epochs.csv",
            VAL_FIELDS,
        )
        self._start_time = time.monotonic()
        self._val_losses = []
        self._closed = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = _output_loss(outputs, ("loss", "train_loss"))
        if self._batch_writer is None or loss is None:
            return
        _append_row(
            self._batch_handle,
            self._batch_writer,
            {
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "batch_idx": batch_idx,
                "loss": loss,
                "learning_rate": _learning_rate(trainer),
                "elapsed_seconds": time.monotonic() - self._start_time,
            },
        )

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if getattr(trainer, "sanity_checking", False):
            return
        loss = _output_loss(outputs, ("loss_val", "val_loss"))
        if loss is not None:
            self._val_losses.append(loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            self._val_losses = []
            return
        if self._val_writer is None or not self._val_losses:
            return
        _append_row(
            self._val_handle,
            self._val_writer,
            {
                "epoch": trainer.current_epoch,
                "val_loss": mean(self._val_losses),
                "val_F1": _metric(trainer, "val_F1"),
                "val_mIoU": _metric(trainer, "val_mIoU"),
                "val_OA": _metric(trainer, "val_OA"),
            },
        )
        self._val_losses = []

    def on_fit_end(self, trainer, pl_module):
        self._close_files()
        if self._log_dir is None:
            return
        try:
            outputs = render_loss_curve(self._log_dir, self.smooth_window)
            print(f"[LossMonitor] curve: {outputs['curve']}")
            print(f"[LossMonitor] summary: {outputs['summary_json']}")
        except Exception as exc:
            print(f"[LossMonitor] WARNING: failed to render loss curve: {exc}")

    def on_exception(self, trainer, pl_module, exception):
        self._close_files()

    def _close_files(self):
        if self._closed:
            return
        for handle in (self._batch_handle, self._val_handle):
            if handle is not None and not handle.closed:
                handle.flush()
                handle.close()
        self._batch_handle = None
        self._batch_writer = None
        self._val_handle = None
        self._val_writer = None
        self._closed = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--smooth-window", type=int, default=101)
    args = parser.parse_args()
    outputs = render_loss_curve(args.log_dir, args.smooth_window)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
