"""Real-time training loss logging and post-training diagnostics.

The callback writes:
  - loss_batches.csv: one row per training batch, flushed in real time
  - loss_validation_epochs.csv: one row per validation epoch

At the end of training it writes:
  - loss_epochs.csv: joined train/validation epoch table
  - loss_summary.json: loss-based convergence diagnostics
  - loss_curve.png: batch loss, epoch losses, and generalization gap

If training was interrupted, the curve can be regenerated with:
    python tools/loss_monitor.py --log-dir lightning_logs/.../version_0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    "epoch_batch_index",
    "loss",
    "learning_rate",
    "elapsed_seconds",
]
VALIDATION_FIELDS = [
    "epoch",
    "val_loss",
    "val_F1",
    "val_mIoU",
    "val_OA",
]
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


def _as_float(value: Any) -> float | None:
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


def _extract_scalar(value: Any, preferred_keys: tuple[str, ...]) -> float | None:
    if isinstance(value, dict):
        for key in preferred_keys:
            if key in value:
                scalar = _as_float(value[key])
                if scalar is not None:
                    return scalar
        for nested in value.values():
            scalar = _as_float(nested)
            if scalar is not None:
                return scalar
        return None
    return _as_float(value)


def _metric(trainer: Any, name: str) -> float | None:
    metrics = getattr(trainer, "callback_metrics", {})
    return _as_float(metrics.get(name))


def _learning_rate(trainer: Any) -> float | None:
    optimizers = getattr(trainer, "optimizers", [])
    if not optimizers:
        return None
    try:
        return float(optimizers[0].param_groups[0]["lr"])
    except (AttributeError, IndexError, KeyError, TypeError):
        return None


def _resolve_log_dir(trainer: Any) -> Path:
    log_dir = getattr(trainer, "log_dir", None)
    if log_dir:
        return Path(log_dir)

    for logger in getattr(trainer, "loggers", []):
        candidate = getattr(logger, "log_dir", None)
        if candidate:
            return Path(candidate)

    return Path(getattr(trainer, "default_root_dir", ".")) / "loss_monitor"


def _open_csv(path: Path, fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    handle = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    if needs_header:
        writer.writeheader()
        handle.flush()
    return handle, writer


def _write_row(handle, writer, row: dict[str, Any]) -> None:
    writer.writerow(row)
    handle.flush()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_int(value: Any) -> int | None:
    scalar = _as_float(value)
    return None if scalar is None else int(scalar)


def _to_float(value: Any) -> float | None:
    return _as_float(value)


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 1e-12:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def _moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    window = max(1, min(window, len(values)))
    result: list[float] = []
    running_sum = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        running_sum += value
        if len(queue) > window:
            running_sum -= queue.pop(0)
        result.append(running_sum / len(queue))
    return result


def _build_epoch_rows(
    batch_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    train_by_epoch: dict[int, list[dict[str, Any]]] = {}
    for row in batch_rows:
        epoch = _to_int(row.get("epoch"))
        loss = _to_float(row.get("loss"))
        if epoch is None or loss is None:
            continue
        train_by_epoch.setdefault(epoch, []).append(
            {
                "loss": loss,
                "global_step": _to_int(row.get("global_step")),
                "lr": _to_float(row.get("learning_rate")),
            }
        )

    val_by_epoch: dict[int, dict[str, Any]] = {}
    for row in validation_rows:
        epoch = _to_int(row.get("epoch"))
        if epoch is None:
            continue
        val_by_epoch[epoch] = {
            "val_loss": _to_float(row.get("val_loss")),
            "val_F1": _to_float(row.get("val_F1")),
            "val_mIoU": _to_float(row.get("val_mIoU")),
            "val_OA": _to_float(row.get("val_OA")),
        }

    epochs = sorted(set(train_by_epoch) | set(val_by_epoch))
    result: list[dict[str, Any]] = []
    for epoch in epochs:
        train_values = train_by_epoch.get(epoch, [])
        losses = [item["loss"] for item in train_values]
        val = val_by_epoch.get(epoch, {})
        result.append(
            {
                "epoch": epoch,
                "train_loss_mean": mean(losses) if losses else None,
                "train_loss_min": min(losses) if losses else None,
                "train_loss_max": max(losses) if losses else None,
                "epoch_batches": len(losses),
                "learning_rate": (
                    train_values[-1]["lr"] if train_values else None
                ),
                "val_loss": val.get("val_loss"),
                "val_F1": val.get("val_F1"),
                "val_mIoU": val.get("val_mIoU"),
                "val_OA": val.get("val_OA"),
            }
        )
    return result


def _write_epoch_csv(path: Path, epoch_rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPOCH_FIELDS)
        writer.writeheader()
        writer.writerows(epoch_rows)


def _summarize(epoch_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_rows = [row for row in epoch_rows if row["train_loss_mean"] is not None]
    val_rows = [row for row in epoch_rows if row["val_loss"] is not None]

    summary: dict[str, Any] = {
        "epochs_logged": len(epoch_rows),
        "last_epoch": epoch_rows[-1]["epoch"] if epoch_rows else None,
        "note": (
            "Loss-based convergence diagnostics only. Use validation mIoU/F1 "
            "for model selection; final capability requires held-out test data."
        ),
    }
    if not train_rows:
        summary["convergence_status"] = "No training loss data"
        summary["recommendation"] = "Check that the callback is attached to Trainer."
        return summary

    summary["last_train_loss"] = train_rows[-1]["train_loss_mean"]
    summary["best_train_loss"] = min(row["train_loss_mean"] for row in train_rows)
    summary["best_train_loss_epoch"] = min(
        train_rows, key=lambda row: row["train_loss_mean"]
    )["epoch"]

    if not val_rows:
        summary["convergence_status"] = "Training loss only"
        summary["recommendation"] = (
            "Validation loss is unavailable. Add a validation loader before "
            "using the curve for overfitting decisions."
        )
        return summary

    best_val = min(val_rows, key=lambda row: row["val_loss"])
    best_f1_row = (
        max(
            (row for row in val_rows if row["val_F1"] is not None),
            key=lambda row: row["val_F1"],
            default=None,
        )
    )
    summary.update(
        {
            "last_val_loss": val_rows[-1]["val_loss"],
            "best_val_loss": best_val["val_loss"],
            "best_val_loss_epoch": best_val["epoch"],
            "epochs_since_best_val_loss": (
                val_rows[-1]["epoch"] - best_val["epoch"]
            ),
        }
    )
    if best_f1_row is not None:
        summary["best_val_F1"] = best_f1_row["val_F1"]
        summary["best_val_F1_epoch"] = best_f1_row["epoch"]

    recent_n = min(10, len(val_rows))
    recent_val = val_rows[-recent_n:]
    val_change = recent_val[-1]["val_loss"] - recent_val[0]["val_loss"]
    val_change_pct = (
        100.0 * val_change / max(abs(recent_val[0]["val_loss"]), 1e-12)
    )
    recent_epochs = [float(row["epoch"]) for row in recent_val]
    recent_val_slope = _linear_slope(
        recent_epochs, [row["val_loss"] for row in recent_val]
    )
    summary["recent_val_loss_change_pct"] = val_change_pct

    recent_train_by_epoch = {
        row["epoch"]: row for row in train_rows
    }
    gaps = []
    for row in recent_val:
        train_row = recent_train_by_epoch.get(row["epoch"])
        if train_row is not None:
            gaps.append(row["val_loss"] - train_row["train_loss_mean"])
    if gaps:
        summary["recent_mean_generalization_gap"] = mean(gaps)
        summary["recent_gap_change"] = gaps[-1] - gaps[0]

    epochs_since_best = summary["epochs_since_best_val_loss"]
    train_recent = train_rows[-recent_n:]
    train_slope = _linear_slope(
        [float(row["epoch"]) for row in train_recent],
        [row["train_loss_mean"] for row in train_recent],
    )
    gap_change = summary.get("recent_gap_change", 0.0)

    if train_slope < 0 and recent_val_slope > 0 and gap_change > 0:
        status = "Possible overfitting"
        recommendation = (
            "Training loss is still falling while validation loss is rising. "
            "Consider earlier stopping, stronger regularization, or a smaller model."
        )
    elif epochs_since_best >= 10 and recent_val_slope >= 0:
        status = "Validation plateau reached"
        recommendation = (
            f"Best validation loss was at epoch {best_val['epoch']}, "
            f"{epochs_since_best} epochs ago. More epochs are unlikely to help "
            "without a data, loss, schedule, or capacity change."
        )
    elif epochs_since_best <= 4 and val_change_pct < -1.0:
        status = "Still improving"
        recommendation = (
            "Validation loss is still decreasing. A longer run may help, but "
            "confirm with validation mIoU/F1 rather than loss alone."
        )
    elif abs(val_change_pct) <= 1.0 and epochs_since_best <= 10:
        status = "Likely converged"
        recommendation = (
            "Recent validation loss is close to its best value. The current "
            "epoch count is likely sufficient for this configuration."
        )
    elif train_slope >= 0 and recent_val_slope >= 0:
        status = "Loss not decreasing"
        recommendation = (
            "Both losses have stopped decreasing. Check underfitting, learning "
            "rate, augmentation strength, and whether the task metric is still moving."
        )
    else:
        status = "Mixed or unstable"
        recommendation = (
            "Loss trends are inconsistent. Inspect the curve, learning-rate "
            "schedule, batch composition, and validation variance before tuning."
        )

    summary["convergence_status"] = status
    summary["recommendation"] = recommendation
    return summary


def render_loss_diagnostics(
    log_dir: str | Path,
    curve_name: str = "loss_curve.png",
    smooth_window: int = 101,
) -> dict[str, Path]:
    """Build diagnostics from the CSVs written by LossMonitorCallback."""

    log_dir = Path(log_dir)
    batch_path = log_dir / "loss_batches.csv"
    validation_path = log_dir / "loss_validation_epochs.csv"
    epoch_path = log_dir / "loss_epochs.csv"
    summary_path = log_dir / "loss_summary.json"
    curve_path = log_dir / curve_name

    batch_rows = _read_csv(batch_path)
    validation_rows = _read_csv(validation_path)
    if not batch_rows and not validation_rows:
        raise FileNotFoundError(
            f"No loss CSV files found in {log_dir}. Expected loss_batches.csv "
            "or loss_validation_epochs.csv."
        )

    epoch_rows = _build_epoch_rows(batch_rows, validation_rows)
    if not epoch_rows:
        raise ValueError("Loss CSV files were found, but no valid rows could be parsed.")

    _write_epoch_csv(epoch_path, epoch_rows)
    summary = _summarize(epoch_rows)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    batch_epochs: list[int] = []
    batch_losses: list[float] = []
    batch_steps: list[int] = []
    for row in batch_rows:
        epoch = _to_int(row.get("epoch"))
        loss = _to_float(row.get("loss"))
        global_step = _to_int(row.get("global_step"))
        if epoch is None or loss is None:
            continue
        batch_epochs.append(epoch)
        batch_losses.append(loss)
        batch_steps.append(global_step if global_step is not None else len(batch_steps))

    train_x = [float(row["epoch"]) for row in epoch_rows if row["train_loss_mean"] is not None]
    train_y = [row["train_loss_mean"] for row in epoch_rows if row["train_loss_mean"] is not None]
    val_x = [float(row["epoch"]) for row in epoch_rows if row["val_loss"] is not None]
    val_y = [row["val_loss"] for row in epoch_rows if row["val_loss"] is not None]

    fig = plt.figure(figsize=(15, 12), dpi=150, facecolor="white")
    grid = fig.add_gridspec(4, 1, height_ratios=[2.4, 1.5, 1.1, 0.55], hspace=0.42)
    ax_batch = fig.add_subplot(grid[0])
    ax_epoch = fig.add_subplot(grid[1])
    ax_gap = fig.add_subplot(grid[2])
    ax_text = fig.add_subplot(grid[3])

    if batch_losses:
        ax_batch.plot(
            batch_steps,
            batch_losses,
            color="#94A3B8",
            linewidth=0.7,
            alpha=0.35,
            label="Batch loss",
        )
        smoothed = _moving_average(batch_losses, smooth_window)
        ax_batch.plot(
            batch_steps,
            smoothed,
            color="#D97706",
            linewidth=2.0,
            label=f"Moving average ({max(1, min(smooth_window, len(batch_losses)))})",
        )
        ax_batch.set_xlabel("Global training step")
        ax_batch.set_ylabel("Training loss")
        ax_batch.set_title("Batch-level training loss")
        ax_batch.legend(loc="upper right", frameon=False)
        ax_batch.grid(alpha=0.18)

    if train_y:
        ax_epoch.plot(
            train_x,
            train_y,
            marker="o",
            markersize=3,
            linewidth=2.0,
            color="#D97706",
            label="Train loss (epoch mean)",
        )
    if val_y:
        ax_epoch.plot(
            val_x,
            val_y,
            marker="o",
            markersize=3,
            linewidth=2.0,
            color="#0F766E",
            label="Validation loss",
        )
        best_epoch = summary.get("best_val_loss_epoch")
        if best_epoch is not None:
            ax_epoch.axvline(
                best_epoch,
                color="#B42318",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label=f"Best val loss epoch {best_epoch}",
            )
    ax_epoch.set_xlabel("Epoch")
    ax_epoch.set_ylabel("Loss")
    ax_epoch.set_title("Epoch-level loss and best validation checkpoint")
    ax_epoch.legend(loc="best", frameon=False)
    ax_epoch.grid(alpha=0.18)

    gap_x: list[float] = []
    gap_y: list[float] = []
    train_lookup = {
        row["epoch"]: row["train_loss_mean"]
        for row in epoch_rows
        if row["train_loss_mean"] is not None
    }
    for row in epoch_rows:
        if row["val_loss"] is None or row["epoch"] not in train_lookup:
            continue
        gap_x.append(float(row["epoch"]))
        gap_y.append(row["val_loss"] - train_lookup[row["epoch"]])
    if gap_y:
        ax_gap.plot(gap_x, gap_y, color="#3B6F95", linewidth=1.8)
        ax_gap.axhline(0, color="#64748B", linewidth=1.0)
        ax_gap.fill_between(
            gap_x,
            gap_y,
            0,
            where=[value >= 0 for value in gap_y],
            color="#FCA5A5",
            alpha=0.22,
            interpolate=True,
        )
    ax_gap.set_xlabel("Epoch")
    ax_gap.set_ylabel("Validation - train")
    ax_gap.set_title("Generalization gap")
    ax_gap.grid(alpha=0.18)

    ax_text.axis("off")
    text_lines = [
        f"Last epoch: {summary.get('last_epoch')}    "
        f"Last train loss: {summary.get('last_train_loss')}",
        f"Best val loss: {summary.get('best_val_loss')} at epoch "
        f"{summary.get('best_val_loss_epoch')}    "
        f"Best val F1: {summary.get('best_val_F1')} at epoch "
        f"{summary.get('best_val_F1_epoch')}",
        f"Recent val loss change: {summary.get('recent_val_loss_change_pct', 0.0):.2f}%    "
        f"Status: {summary['convergence_status']}",
        summary["recommendation"],
        summary["note"],
    ]
    ax_text.text(
        0.01,
        0.95,
        "\n".join(text_lines),
        ha="left",
        va="top",
        fontsize=9.5,
        color="#1F2937",
        linespacing=1.45,
        transform=ax_text.transAxes,
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
    """Lightning callback that logs batch/validation loss and renders a curve."""

    def __init__(self, smooth_window: int = 101):
        super().__init__()
        self.smooth_window = smooth_window
        self._batch_handle = None
        self._batch_writer = None
        self._validation_handle = None
        self._validation_writer = None
        self._log_dir: Path | None = None
        self._start_time = 0.0
        self._epoch_batch_index = 0
        self._validation_losses: list[float] = []
        self._closed = False

    def on_train_start(self, trainer, pl_module):
        self._close_files()
        self._log_dir = _resolve_log_dir(trainer)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._batch_handle, self._batch_writer = _open_csv(
            self._log_dir / "loss_batches.csv", BATCH_FIELDS
        )
        self._validation_handle, self._validation_writer = _open_csv(
            self._log_dir / "loss_validation_epochs.csv", VALIDATION_FIELDS
        )
        self._start_time = time.monotonic()
        self._epoch_batch_index = 0
        self._validation_losses = []
        self._closed = False

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_batch_index = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._batch_writer is None:
            return
        loss = _extract_scalar(outputs, ("loss", "train_loss"))
        if loss is None:
            return
        _write_row(
            self._batch_handle,
            self._batch_writer,
            {
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "batch_idx": batch_idx,
                "epoch_batch_index": self._epoch_batch_index,
                "loss": loss,
                "learning_rate": _learning_rate(trainer),
                "elapsed_seconds": time.monotonic() - self._start_time,
            },
        )
        self._epoch_batch_index += 1

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
        loss = _extract_scalar(outputs, ("loss_val", "val_loss"))
        if loss is not None:
            self._validation_losses.append(loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            self._validation_losses = []
            return
        if self._validation_writer is None or not self._validation_losses:
            return
        _write_row(
            self._validation_handle,
            self._validation_writer,
            {
                "epoch": trainer.current_epoch,
                "val_loss": mean(self._validation_losses),
                "val_F1": _metric(trainer, "val_F1"),
                "val_mIoU": _metric(trainer, "val_mIoU"),
                "val_OA": _metric(trainer, "val_OA"),
            },
        )
        self._validation_losses = []

    def on_fit_end(self, trainer, pl_module):
        self._close_files()
        if self._log_dir is None:
            return
        try:
            outputs = render_loss_diagnostics(
                self._log_dir,
                smooth_window=self.smooth_window,
            )
            print(f"[LossMonitor] curve: {outputs['curve']}")
            print(f"[LossMonitor] summary: {outputs['summary_json']}")
        except Exception as exc:
            print(f"[LossMonitor] WARNING: failed to render loss curve: {exc}")

    def on_exception(self, trainer, pl_module, exception):
        self._close_files()

    def _close_files(self):
        if self._closed:
            return
        for handle in (self._batch_handle, self._validation_handle):
            if handle is not None and not handle.closed:
                handle.flush()
                handle.close()
        self._batch_handle = None
        self._batch_writer = None
        self._validation_handle = None
        self._validation_writer = None
        self._closed = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        required=True,
        help="Lightning version directory containing loss CSV files.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=101,
        help="Moving-average window for batch loss.",
    )
    args = parser.parse_args()
    outputs = render_loss_diagnostics(
        args.log_dir,
        smooth_window=args.smooth_window,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
