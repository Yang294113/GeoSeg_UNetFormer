import argparse
import csv
import html
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


METRIC_ALIASES = {
    "train_loss": ("train_loss", "train_loss_step", "loss"),
    "val_loss": ("val_loss", "val_loss_step"),
    "train_miou": ("train_mIoU",),
    "val_miou": ("val_mIoU",),
    "train_f1": ("train_F1",),
    "val_f1": ("val_F1",),
    "train_oa": ("train_OA",),
    "val_oa": ("val_OA",),
    "lr": ("lr",),
}


@dataclass
class EpochRecord:
    epoch: int
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    train_miou: Optional[float] = None
    val_miou: Optional[float] = None
    train_f1: Optional[float] = None
    val_f1: Optional[float] = None
    train_oa: Optional[float] = None
    val_oa: Optional[float] = None
    lr: Optional[float] = None


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _values_for_epoch(rows: Iterable[dict], aliases: tuple[str, ...]) -> list[float]:
    for column in aliases:
        values = [_as_float(row.get(column)) for row in rows]
        numeric = [value for value in values if value is not None]
        if numeric:
            return numeric
    return []


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def load_metrics(csv_path: Path) -> list[EpochRecord]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    grouped: dict[int, list[dict]] = {}
    for row in rows:
        epoch = _as_float(row.get("epoch"))
        if epoch is not None:
            grouped.setdefault(int(epoch), []).append(row)

    records = []
    for epoch in sorted(grouped):
        epoch_rows = grouped[epoch]
        records.append(
            EpochRecord(
                epoch=epoch,
                train_loss=_mean(_values_for_epoch(epoch_rows, METRIC_ALIASES["train_loss"])),
                val_loss=_mean(_values_for_epoch(epoch_rows, METRIC_ALIASES["val_loss"])),
                train_miou=_mean(_values_for_epoch(epoch_rows, METRIC_ALIASES["train_miou"])),
                val_miou=_mean(_values_for_epoch(epoch_rows, METRIC_ALIASES["val_miou"])),
                train_f1=_mean(_values_for_epoch(epoch_rows, METRIC_ALIASES["train_f1"])),
                val_f1=_mean(_values_for_epoch(epoch_rows, METRIC_ALIASES["val_f1"])),
                train_oa=_mean(_values_for_epoch(epoch_rows, METRIC_ALIASES["train_oa"])),
                val_oa=_mean(_values_for_epoch(epoch_rows, METRIC_ALIASES["val_oa"])),
                lr=_mean(_values_for_epoch(epoch_rows, METRIC_ALIASES["lr"])),
            )
        )
    return records


def find_latest_metrics(root: Path) -> Optional[Path]:
    candidates = list(root.rglob("metrics.csv")) if root.exists() else []
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _pairs(records: list[EpochRecord], field: str) -> list[tuple[float, float]]:
    return [
        (float(record.epoch), float(getattr(record, field)))
        for record in records
        if getattr(record, field) is not None
    ]


def _linear_slope(points: list[tuple[float, float]]) -> Optional[float]:
    if len(points) < 2:
        return None
    x_mean = sum(x for x, _ in points) / len(points)
    y_mean = sum(y for _, y in points) / len(points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator == 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator


def analyze_convergence(
    records: list[EpochRecord],
    window: int,
    patience: int,
    min_epochs: int,
    slope_tolerance_pp: float,
) -> tuple[str, list[str]]:
    if not records:
        return "暂无数据", ["metrics.csv 中还没有可解析的 epoch。"]

    val_miou_points = _pairs(records, "val_miou")
    val_loss_points = _pairs(records, "val_loss")
    train_loss_points = _pairs(records, "train_loss")
    last_epoch = records[-1].epoch
    lines = [f"当前轮数：{last_epoch + 1}（最后记录 epoch={last_epoch}）"]

    val_loss_slope = None
    train_loss_slope = None
    if val_loss_points:
        recent_val_loss = val_loss_points[-max(window, 2):]
        val_loss_slope = _linear_slope(recent_val_loss)
        lines.append(f"当前验证 loss：{val_loss_points[-1][1]:.6f}")
    if train_loss_points:
        recent_train_loss = train_loss_points[-max(window, 2):]
        train_loss_slope = _linear_slope(recent_train_loss)
        lines.append(f"当前训练 loss：{train_loss_points[-1][1]:.6f}")
    loss_overfitting = (
        train_loss_slope is not None
        and val_loss_slope is not None
        and train_loss_slope < 0
        and val_loss_slope > 0
    )
    if loss_overfitting:
        lines.append("近期训练 loss 继续下降，但验证 loss 已转为上升。")

    if val_miou_points:
        best_epoch, best_value = max(val_miou_points, key=lambda point: point[1])
        age = int(last_epoch - best_epoch)
        recent = val_miou_points[-max(window, 2):]
        recent_slope = _linear_slope(recent)
        slope_pp = None if recent_slope is None else recent_slope * 100
        lines.extend(
            [
                f"最高验证 mIoU：{best_value * 100:.3f}%，epoch={int(best_epoch)}",
                f"距最佳验证结果：{age} 轮",
            ]
        )
        if slope_pp is not None:
            lines.append(f"最近 {len(recent)} 轮验证 mIoU 斜率：{slope_pp:+.4f} 个百分点/epoch")

        gap_pairs = [
            (record.epoch, (record.train_miou - record.val_miou) * 100)
            for record in records
            if record.train_miou is not None and record.val_miou is not None
        ]
        if gap_pairs:
            lines.append(f"当前训练-验证 mIoU 差距：{gap_pairs[-1][1]:.3f} 个百分点")

        if len(records) < min_epochs:
            status = "训练轮数偏少"
            lines.append(f"尚不足 {min_epochs} 轮，不建议此时判断收敛。")
        elif age < patience or (slope_pp is not None and slope_pp > slope_tolerance_pp):
            status = "仍在改善"
            lines.append("验证指标仍有上升趋势，或最佳结果刚出现不久。")
        elif gap_pairs and len(gap_pairs) >= max(4, window // 2):
            recent_gap = gap_pairs[-max(window // 2, 2):]
            gap_delta = recent_gap[-1][1] - recent_gap[0][1]
            if (
                slope_pp is not None
                and slope_pp <= slope_tolerance_pp
                and (gap_delta > 1.0 or loss_overfitting)
            ):
                status = "可能过拟合"
                lines.append("验证指标趋于平台，同时泛化差距或验证 loss 正在恶化。")
            else:
                status = "基本收敛"
                lines.append(f"连续 {age} 轮未刷新最高验证 mIoU，近期斜率已小于阈值。")
        else:
            status = "基本收敛" if age >= patience else "趋势放缓，接近收敛"
        lines.append(f"按当前 patience={patience}，最早可考虑停止区间约在 epoch={int(best_epoch + patience)} 之后。")
        return status, lines

    if val_loss_points:
        best_epoch, best_value = min(val_loss_points, key=lambda point: point[1])
        age = int(last_epoch - best_epoch)
        recent = val_loss_points[-max(window, 2):]
        slope = _linear_slope(recent)
        lines.extend(
            [
                f"最低验证 loss：{best_value:.6f}，epoch={int(best_epoch)}",
                f"距最佳验证 loss：{age} 轮",
            ]
        )
        if len(records) < min_epochs:
            status = "训练轮数偏少"
        elif age < patience or (slope is not None and slope < -1e-5):
            status = "仍在改善"
        elif age >= patience:
            status = "基本收敛"
        else:
            status = "趋势放缓，接近收敛"
        return status, lines

    return "缺少验证指标", ["未找到 val_mIoU 或 val_loss，无法判断收敛。"]


def _scale(value: float, source_min: float, source_max: float, target_min: float, target_max: float):
    if math.isclose(source_min, source_max):
        return (target_min + target_max) / 2
    ratio = (value - source_min) / (source_max - source_min)
    return target_min + ratio * (target_max - target_min)


def _axis_domain(values: list[float], log_values: bool = False) -> tuple[float, float]:
    if log_values:
        values = [math.log10(value) for value in values if value > 0]
    if not values:
        return 0.0, 1.0
    low, high = min(values), max(values)
    if math.isclose(low, high):
        padding = max(abs(high) * 0.05, 0.01)
    else:
        padding = (high - low) * 0.08
    return low - padding, high + padding


def _svg_panel(
    values: list[tuple[float, float]],
    x_domain: tuple[float, float],
    y_domain: tuple[float, float],
    box: tuple[float, float, float, float],
    title: str,
    y_label: str,
    color: str,
    canvas_parts: list[str],
):
    x0, y0, x1, y1 = box
    canvas_parts.extend(
        [
            f'<text x="{x0}" y="{y0 - 20}" class="panel-title">{html.escape(title)}</text>',
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="axis"/>',
            f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="axis"/>',
            f'<text x="{x0 - 12}" y="{(y0 + y1) / 2}" class="axis-label" '
            f'transform="rotate(-90 {x0 - 12} {(y0 + y1) / 2})">{html.escape(y_label)}</text>',
        ]
    )

    for index in range(6):
        fraction = index / 5
        y = y1 - fraction * (y1 - y0)
        value = y_domain[0] + fraction * (y_domain[1] - y_domain[0])
        canvas_parts.append(
            f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" class="grid"/>'
        )
        canvas_parts.append(
            f'<text x="{x0 - 10}" y="{y + 4:.2f}" text-anchor="end" class="tick">'
            f'{value:.3f}</text>'
        )

    for index in range(5):
        fraction = index / 4
        epoch = x_domain[0] + fraction * (x_domain[1] - x_domain[0])
        x = x0 + fraction * (x1 - x0)
        canvas_parts.append(
            f'<line x1="{x:.2f}" y1="{y1}" x2="{x:.2f}" y2="{y1 + 6}" class="axis"/>'
        )
        canvas_parts.append(
            f'<text x="{x:.2f}" y="{y1 + 24}" text-anchor="middle" class="tick">'
            f'{epoch:.0f}</text>'
        )

    if values:
        points = []
        for epoch, value in values:
            x = _scale(epoch, x_domain[0], x_domain[1], x0, x1)
            y = _scale(value, y_domain[0], y_domain[1], y1, y0)
            points.append(f"{x:.2f},{y:.2f}")
        canvas_parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
            'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )
    else:
        canvas_parts.append(
            f'<text x="{(x0 + x1) / 2}" y="{(y0 + y1) / 2}" text-anchor="middle" class="empty">'
            "No epoch-level data available.</text>"
        )


def _legend_item(x: float, y: float, color: str, label: str) -> str:
    return (
        f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" stroke="{color}" stroke-width="3"/>'
        f'<text x="{x + 36}" y="{y + 4}" class="legend">{html.escape(label)}</text>'
    )


def plot_records(records: list[EpochRecord], output_path: Path, status: str, log_loss: bool):
    if not records:
        return

    epochs = [record.epoch for record in records]
    x_domain = (float(min(epochs)), float(max(epochs)))
    canvas_width, canvas_height = 1200, 820
    canvas_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" '
        f'viewBox="0 0 {canvas_width} {canvas_height}">',
        "<style>"
        "text{font-family:Arial,'Microsoft YaHei',sans-serif;fill:#1f2937}"
        ".title{font-size:24px;font-weight:700}"
        ".subtitle{font-size:15px;fill:#475569}"
        ".panel-title{font-size:17px;font-weight:700}"
        ".axis{stroke:#64748b;stroke-width:1.2}"
        ".axis-label{font-size:13px;text-anchor:middle}"
        ".grid{stroke:#dbe3ea;stroke-width:1}"
        ".tick{font-size:12px;fill:#64748b}"
        ".legend{font-size:13px}"
        ".empty{font-size:15px;fill:#64748b}"
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="70" y="45" class="title">Training convergence monitor</text>',
        f'<text x="70" y="70" class="subtitle">{html.escape(status)} | '
        f'current epoch {epochs[-1]}</text>',
    ]

    loss_series = []
    all_loss_values = []
    for field, label, color in (
        ("train_loss", "Train loss", "#2563EB"),
        ("val_loss", "Validation loss", "#DC2626"),
    ):
        points = _pairs(records, field)
        transformed = []
        for epoch, value in points:
            if log_loss and value > 0:
                value = math.log10(value)
            transformed.append((epoch, value))
        loss_series.append((label, color, transformed))
        all_loss_values.extend(value for _, value in transformed)

    loss_y_domain = _axis_domain(all_loss_values)
    for label, color, points in loss_series:
        _svg_panel(
            values=points,
            x_domain=x_domain,
            y_domain=loss_y_domain,
            box=(90.0, 130.0, 1140.0, 390.0),
            title="Loss curve",
            y_label="log10(loss)" if log_loss else "Loss",
            color=color,
            canvas_parts=canvas_parts,
        )
    canvas_parts.append(_legend_item(850, 108, "#2563EB", "Train loss"))
    canvas_parts.append(_legend_item(1000, 108, "#DC2626", "Validation loss"))

    metric_series = []
    all_metric_values = []
    for field, label, color in (
        ("train_miou", "Train mIoU", "#0F766E"),
        ("val_miou", "Validation mIoU", "#D97706"),
    ):
        points = [(epoch, value * 100) for epoch, value in _pairs(records, field)]
        metric_series.append((label, color, points))
        all_metric_values.extend(value for _, value in points)

    metric_y_domain = _axis_domain(all_metric_values)
    for label, color, points in metric_series:
        _svg_panel(
            values=points,
            x_domain=x_domain,
            y_domain=metric_y_domain,
            box=(90.0, 500.0, 1140.0, 760.0),
            title="mIoU curve",
            y_label="mIoU (%)",
            color=color,
            canvas_parts=canvas_parts,
        )
    canvas_parts.append(_legend_item(850, 478, "#0F766E", "Train mIoU"))
    canvas_parts.append(_legend_item(1000, 478, "#D97706", "Validation mIoU"))
    canvas_parts.append('<text x="615" y="795" text-anchor="middle" class="tick">Epoch</text>')
    canvas_parts.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(canvas_parts), encoding="utf-8")


def write_summary(path: Path, csv_path: Path, status: str, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = [
        "Training convergence report",
        f"Source: {csv_path}",
        f"Status: {status}",
        *lines,
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def generate_training_report(
    csv_path: Path,
    output_dir: Optional[Path],
    window: int,
    patience: int,
    min_epochs: int,
    slope_tolerance_pp: float,
    log_loss: bool,
):
    records = load_metrics(csv_path)
    status, lines = analyze_convergence(
        records=records,
        window=window,
        patience=patience,
        min_epochs=min_epochs,
        slope_tolerance_pp=slope_tolerance_pp,
    )
    destination = output_dir or csv_path.parent
    plot_path = destination / "training_monitor.svg"
    summary_path = destination / "training_summary.txt"
    plot_records(records, plot_path, status, log_loss)
    write_summary(summary_path, csv_path, status, lines)
    print(f"\n状态：{status}")
    for line in lines:
        print(f"- {line}")
    print(f"曲线：{plot_path}")
    print(f"摘要：{summary_path}")
    return plot_path, summary_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Monitor Lightning metrics.csv and estimate whether more epochs are useful."
    )
    parser.add_argument("--csv", type=Path, help="Path to a Lightning metrics.csv file.")
    parser.add_argument("--metrics-root", type=Path, default=Path("lightning_logs"))
    parser.add_argument("--output-dir", type=Path, help="Directory for PNG and summary output.")
    parser.add_argument("--watch", action="store_true", help="Refresh the report when metrics.csv changes.")
    parser.add_argument("--interval", type=float, default=15.0, help="Watch polling interval in seconds.")
    parser.add_argument("--window", type=int, default=10, help="Recent epochs used for trend estimation.")
    parser.add_argument("--patience", type=int, default=10, help="Epochs allowed without a new best result.")
    parser.add_argument("--min-epochs", type=int, default=15, help="Do not judge convergence before this epoch.")
    parser.add_argument(
        "--slope-tolerance",
        type=float,
        default=0.03,
        help="Validation mIoU slope tolerance in percentage points per epoch.",
    )
    parser.add_argument("--log-loss", action="store_true", help="Use a logarithmic loss axis.")
    return parser


def main():
    args = build_parser().parse_args()
    csv_path = args.csv
    last_signature = None

    while True:
        active_csv = csv_path or find_latest_metrics(args.metrics_root)
        if active_csv is None or not active_csv.exists():
            if not args.watch:
                raise SystemExit(
                    f"未找到 metrics.csv。请用 --csv 指定，或在 {args.metrics_root} 下开始训练。"
                )
            print(f"等待 metrics.csv（目录：{args.metrics_root}）...", flush=True)
            time.sleep(max(args.interval, 1.0))
            continue

        signature = (active_csv.stat().st_mtime_ns, active_csv.stat().st_size)
        if signature != last_signature:
            try:
                generate_training_report(
                    csv_path=active_csv,
                    output_dir=args.output_dir,
                    window=max(args.window, 2),
                    patience=max(args.patience, 1),
                    min_epochs=max(args.min_epochs, 1),
                    slope_tolerance_pp=args.slope_tolerance,
                    log_loss=args.log_loss,
                )
                last_signature = signature
            except PermissionError:
                # Lightning may be writing the file while it is being read.
                if not args.watch:
                    raise
                print("metrics.csv 正在写入，稍后重试。", flush=True)

        if not args.watch:
            break
        time.sleep(max(args.interval, 1.0))


if __name__ == "__main__":
    main()
