"""Validation-time diagnostics for GL-VSS gate weights.

The diagnostics cache only the first few validation images from the first
validation batch. Full weight maps are never retained across training batches,
and the cache is detached from autograd before it is stored.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


STAGES = ("b4", "b3", "b2")
RESAMPLING = getattr(Image, "Resampling", Image)
DEFAULT_CLASSES = ("ImSurf", "Building", "LowVeg", "Tree", "Car", "Clutter")
DEFAULT_PALETTE = (
    (255, 255, 255),
    (0, 0, 255),
    (0, 255, 255),
    (0, 255, 0),
    (255, 204, 0),
    (255, 0, 0),
)

CSV_FIELDS = (
    "epoch",
    "val_F1",
    "is_best_so_far",
    "stage",
    "group",
    "images",
    "pixels",
    "sample_ids",
    "Wg_mean",
    "Wg_std",
    "Wg_within_image_std_mean",
    "Wg_gt_0.7_ratio",
    "Wl_mean",
    "Wl_std",
    "Wl_within_image_std_mean",
    "Wl_gt_0.7_ratio",
    "A1_mean",
    "A1_std",
    "A1_within_image_std_mean",
    "A1_dominant_ratio",
    "A2_mean",
    "A2_std",
    "A2_within_image_std_mean",
    "A2_dominant_ratio",
    "A3_mean",
    "A3_std",
    "A3_within_image_std_mean",
    "A3_dominant_ratio",
    "A_no_clear_dominant_ratio",
    "A_entropy_mean",
)


def _to_numpy(tensor):
    if hasattr(tensor, "detach"):
        return tensor.detach().float().cpu().numpy()
    return np.asarray(tensor)


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "sample"


def _resize_labels(labels, height, width):
    resized = []
    for label in labels:
        image = Image.fromarray(label.astype(np.uint8), mode="L")
        image = image.resize((width, height), resample=RESAMPLING.NEAREST)
        resized.append(np.asarray(image, dtype=np.uint8))
    return np.stack(resized)


def _morphology(mask, size, is_max):
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    image_filter = ImageFilter.MaxFilter(size) if is_max else ImageFilter.MinFilter(size)
    return np.asarray(image.filter(image_filter), dtype=np.uint8) > 0


def _decode_image(tensor):
    image = _to_numpy(tensor).transpose(1, 2, 0)
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    image = image * std + mean
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def _decode_mask(mask, palette):
    label = mask.astype(np.int64)
    output = np.zeros((*label.shape, 3), dtype=np.uint8)
    for index, color in enumerate(palette):
        output[label == index] = color
    return output


def _colorize(values):
    values = np.clip(values.astype(np.float32), 0.0, 1.0)
    anchors = np.asarray(
        (
            (20, 36, 120),
            (35, 115, 180),
            (90, 190, 170),
            (250, 220, 95),
            (220, 80, 55),
            (120, 20, 45),
        ),
        dtype=np.float32,
    )
    positions = values * (len(anchors) - 1)
    lower = np.floor(positions).astype(np.int32)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    fraction = (positions - lower)[..., None]
    rgb = anchors[lower] * (1.0 - fraction) + anchors[upper] * fraction
    return rgb.astype(np.uint8)


def _dominant_color(values):
    ordered = np.sort(values, axis=0)
    clear = ordered[-1] - ordered[-2] >= 0.02
    dominant = values.argmax(axis=0)
    colors = np.asarray(((45, 105, 210), (45, 175, 100), (220, 85, 65)), dtype=np.uint8)
    output = colors[dominant]
    output[~clear] = (145, 145, 145)
    return output


def _resize_rgb(array, height, width, nearest=False):
    resample = RESAMPLING.NEAREST if nearest else RESAMPLING.BILINEAR
    image = Image.fromarray(array.astype(np.uint8), mode="RGB")
    return np.asarray(image.resize((width, height), resample=resample), dtype=np.uint8)


def _overlay(base, weight_map, categorical=False):
    heat = _dominant_color(weight_map) if categorical else _colorize(weight_map)
    heat = _resize_rgb(heat, base.shape[0], base.shape[1], nearest=categorical)
    return np.clip(0.35 * base.astype(np.float32) + 0.65 * heat.astype(np.float32), 0, 255).astype(np.uint8)


def _panel(image, title, size):
    image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    image = image.resize((size, size), resample=RESAMPLING.BILINEAR)
    panel = Image.new("RGB", (size, size + 42), "white")
    panel.paste(image, (0, 42))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 5), title, fill=(25, 25, 25))
    return panel


class GateWeightDiagnostics:
    def __init__(
        self,
        enabled=True,
        num_samples=3,
        heatmap_interval=5,
        threshold=0.7,
        class_names=None,
        palette=None,
        edge_radius=2,
        dominant_margin=0.02,
    ):
        self.enabled = bool(enabled)
        self.num_samples = max(1, int(num_samples))
        self.heatmap_interval = max(1, int(heatmap_interval))
        self.threshold = float(threshold)
        self.class_names = tuple(class_names or DEFAULT_CLASSES)
        self.palette = tuple(palette or DEFAULT_PALETTE)
        self.edge_radius = max(1, int(edge_radius))
        self.dominant_margin = max(0.0, float(dominant_margin))
        self._best_val_f1 = -math.inf
        self._fixed_sample_ids = None
        self._clear_epoch()

    def _clear_epoch(self):
        self._viz_cache = None
        self._capture_complete = False
        self._epoch_rows = []

    def begin_epoch(self):
        self._clear_epoch()

    def should_capture(self, batch_idx):
        return self.enabled and not self._capture_complete and batch_idx == 0

    def record_batch(self, model, batch):
        if self._capture_complete:
            return
        cache = model.get_gate_weight_cache()
        images = batch["img"]
        masks = batch["gt_semantic_seg"]
        count = min(self.num_samples, int(images.shape[0]))
        if count <= 0:
            return

        stages = {}
        for stage in STAGES:
            fusion = cache.get(stage, {}).get("fusion")
            scales = cache.get(stage, {}).get("scales")
            if fusion is None or scales is None:
                raise RuntimeError(f"missing cached gate weights for stage {stage}")
            fusion = _to_numpy(fusion[:count])
            scales = _to_numpy(scales[:count])
            if fusion.shape[1] != 2 or scales.shape[1] != 3:
                raise RuntimeError(
                    f"unexpected gate shapes at {stage}: fusion={fusion.shape}, scales={scales.shape}"
                )
            stages[stage] = {
                "fusion": fusion.astype(np.float32),
                "scales": scales.astype(np.float32),
            }

        img_ids = batch.get("img_id")
        if isinstance(img_ids, str):
            img_ids = [img_ids]
        if not img_ids:
            img_ids = [f"val_{index:04d}" for index in range(count)]
        img_ids = [str(value) for value in img_ids[:count]]
        if self._fixed_sample_ids is None:
            self._fixed_sample_ids = img_ids
        elif img_ids != self._fixed_sample_ids:
            raise RuntimeError(
                "validation sample order changed; keep val_loader shuffle=False"
            )

        self._viz_cache = {
            "images": _to_numpy(images[:count]).astype(np.float16),
            "masks": _to_numpy(masks[:count]).astype(np.uint8),
            "sample_ids": img_ids,
            "stages": stages,
        }
        self._capture_complete = True

    def finish_epoch(self, epoch, val_f1, log_dir):
        if not self.enabled:
            return
        try:
            self._finish_epoch_impl(int(epoch), float(val_f1), Path(log_dir))
        except Exception as exc:
            print(f"[gate diagnostics] failed at epoch {epoch}: {exc}")

    def _finish_epoch_impl(self, epoch, val_f1, log_dir):
        if self._viz_cache is None:
            print(f"[gate diagnostics] no weights captured at epoch {epoch}")
            return

        output_dir = log_dir / "gate_diagnostics"
        output_dir.mkdir(parents=True, exist_ok=True)
        is_best = val_f1 > self._best_val_f1
        self._epoch_rows = self._build_epoch_rows(epoch, val_f1, is_best)
        self._append_statistics(output_dir / "gate_statistics.csv")
        self._write_readme(output_dir)
        self._write_interpretation(output_dir)

        if (epoch + 1) % self.heatmap_interval == 0:
            self._save_heatmaps(output_dir / "heatmaps" / f"epoch_{epoch + 1:04d}", epoch, val_f1)

        if is_best:
            self._best_val_f1 = val_f1
            best_dir = output_dir / "heatmaps" / "best_val_f1"
            self._save_heatmaps(best_dir, epoch, val_f1)
            (best_dir / "best_info.txt").write_text(
                f"monitor=val_F1\nepoch={epoch}\nval_F1={val_f1:.8f}\n",
                encoding="utf-8",
            )
            print(f"[gate diagnostics] new best val_F1={val_f1:.6f} at epoch {epoch}")

        print(f"[gate diagnostics] statistics: {output_dir / 'gate_statistics.csv'}")

    def _build_epoch_rows(self, epoch, val_f1, is_best):
        rows = []
        masks = self._viz_cache["masks"]
        sample_ids = ";".join(self._viz_cache["sample_ids"])

        building_index = next(
            (index for index, name in enumerate(self.class_names) if name.lower() == "building"),
            None,
        )
        building_edge = None
        building_interior = None
        if building_index is not None:
            building_mask = masks == building_index
            building_edge = np.stack(
                [
                    _morphology(item, 2 * self.edge_radius + 1, True)
                    & ~_morphology(item, 2 * self.edge_radius + 1, False)
                    for item in building_mask
                ]
            )
            building_interior = building_mask & ~building_edge

        for stage in STAGES:
            stage_cache = self._viz_cache["stages"][stage]
            fusion = stage_cache["fusion"].astype(np.float32)
            scales = stage_cache["scales"].astype(np.float32)
            height, width = fusion.shape[-2:]
            resized_masks = _resize_labels(masks, height, width)
            stage_groups = [("all", np.ones_like(resized_masks, dtype=bool))]
            stage_groups.extend(
                (f"class:{name}", resized_masks == index)
                for index, name in enumerate(self.class_names)
            )
            if building_edge is not None:
                stage_building = resized_masks == building_index
                edge = _resize_labels(building_edge.astype(np.uint8), height, width) > 0
                interior = _resize_labels(building_interior.astype(np.uint8), height, width) > 0
                interior &= stage_building & ~edge
                stage_groups.extend((("edge:Building", edge), ("interior:Building", interior)))

            for group_name, group_mask in stage_groups:
                normalized_mask = group_mask.astype(bool)
                summary = self._summarize_group(fusion, scales, normalized_mask)
                if summary is None:
                    continue
                row = {
                    "epoch": epoch,
                    "val_F1": f"{val_f1:.8f}",
                    "is_best_so_far": str(is_best),
                    "stage": stage,
                    "group": group_name,
                    "images": summary["images"],
                    "pixels": summary["pixels"],
                    "sample_ids": sample_ids,
                }
                for key, value in summary.items():
                    if key not in {"images", "pixels"}:
                        row[key] = "" if value is None else f"{value:.8f}"
                rows.append(row)
        return rows

    def _summarize_group(self, fusion, scales, region):
        images = int(np.count_nonzero(region.reshape(region.shape[0], -1).sum(axis=1)))
        pixels = int(np.count_nonzero(region))
        if pixels == 0:
            return None

        wg = fusion[:, 0][region]
        wl = fusion[:, 1][region]
        scale_values = scales.transpose(0, 2, 3, 1)[region]
        entropy = -(scale_values * np.log(np.clip(scale_values, 1e-8, 1.0))).sum(axis=1)
        entropy = entropy / math.log(3.0)
        ordered = np.sort(scale_values, axis=1)
        clear_dominant = ordered[:, -1] - ordered[:, -2] >= self.dominant_margin
        dominant = scale_values.argmax(axis=1)

        summary = {
            "images": images,
            "pixels": pixels,
            "Wg_mean": float(wg.mean()),
            "Wg_std": float(wg.std()),
            "Wg_within_image_std_mean": self._within_image_std(fusion[:, 0], region),
            "Wg_gt_0.7_ratio": float((wg > self.threshold).mean()),
            "Wl_mean": float(wl.mean()),
            "Wl_std": float(wl.std()),
            "Wl_within_image_std_mean": self._within_image_std(fusion[:, 1], region),
            "Wl_gt_0.7_ratio": float((wl > self.threshold).mean()),
            "A_entropy_mean": float(entropy.mean()),
            "A_no_clear_dominant_ratio": float((~clear_dominant).mean()),
        }
        for index in range(3):
            summary[f"A{index + 1}_mean"] = float(scale_values[:, index].mean())
            summary[f"A{index + 1}_std"] = float(scale_values[:, index].std())
            summary[f"A{index + 1}_within_image_std_mean"] = self._within_image_std(
                scales[:, index], region
            )
            summary[f"A{index + 1}_dominant_ratio"] = float(
                ((dominant == index) & clear_dominant).mean()
            )
        return summary

    @staticmethod
    def _within_image_std(values, region):
        standard_deviations = []
        for image_index in range(values.shape[0]):
            selected = values[image_index][region[image_index]]
            if selected.size:
                standard_deviations.append(float(selected.std()))
        return float(np.mean(standard_deviations)) if standard_deviations else None

    def _append_statistics(self, path):
        import csv

        file_exists = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerows(self._epoch_rows)

    def _save_heatmaps(self, target_dir, epoch, val_f1):
        target_dir.mkdir(parents=True, exist_ok=True)
        images = self._viz_cache["images"]
        masks = self._viz_cache["masks"]
        sample_ids = self._viz_cache["sample_ids"]

        for sample_index, sample_id in enumerate(sample_ids):
            base = _decode_image(images[sample_index])
            gt = _decode_mask(masks[sample_index], self.palette)
            sample_dir = target_dir / _safe_name(sample_id)
            sample_dir.mkdir(parents=True, exist_ok=True)

            for stage in STAGES:
                fusion = self._viz_cache["stages"][stage]["fusion"][sample_index].astype(np.float32)
                scales = self._viz_cache["stages"][stage]["scales"][sample_index].astype(np.float32)
                summaries = {
                    row["group"]: row for row in self._epoch_rows
                    if row["stage"] == stage and row["sample_ids"] == ";".join(sample_ids)
                }
                overall = summaries.get("all", {})
                panels = (
                    _panel(base, "Validation image", 360),
                    _panel(gt, "Ground truth", 360),
                    _panel(
                        _overlay(base, fusion[0]),
                        f"Wg mean={float(overall.get('Wg_mean', 0.0)):.3f}",
                        360,
                    ),
                    _panel(
                        _overlay(base, fusion[1]),
                        f"Wl mean={float(overall.get('Wl_mean', 0.0)):.3f}",
                        360,
                    ),
                    _panel(
                        _overlay(base, scales[0]),
                        f"A1 d=1 mean={float(overall.get('A1_mean', 0.0)):.3f}",
                        360,
                    ),
                    _panel(
                        _overlay(base, scales[1]),
                        f"A2 d=2 mean={float(overall.get('A2_mean', 0.0)):.3f}",
                        360,
                    ),
                    _panel(
                        _overlay(base, scales[2]),
                        f"A3 d=3 mean={float(overall.get('A3_mean', 0.0)):.3f}",
                        360,
                    ),
                    _panel(
                        _overlay(base, scales, categorical=True),
                        "Dominant scale: A1 blue / A2 green / A3 red / tie gray",
                        360,
                    ),
                )
                contact = self._contact_sheet(
                    panels,
                    f"sample={sample_id} epoch={epoch} val_F1={val_f1:.6f} stage={stage}",
                )
                contact.save(sample_dir / f"{stage}_gate_maps.png")

    @staticmethod
    def _contact_sheet(panels, header):
        margin = 12
        gap = 12
        width = panels[0].width
        height = panels[0].height
        canvas = Image.new(
            "RGB",
            (margin * 2 + width * 4 + gap * 3, margin * 2 + 48 + height * 2 + gap),
            (245, 245, 245),
        )
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, margin + 4), header, fill=(20, 20, 20))
        for index, panel in enumerate(panels):
            row, column = divmod(index, 4)
            x = margin + column * (width + gap)
            y = margin + 48 + row * (height + gap)
            canvas.paste(panel, (x, y))
        return canvas

    def _write_readme(self, output_dir):
        text = f"""GL-VSS gate diagnostics

Statistics are computed from the first {self.num_samples} validation images.
The validation loader is not shuffled, so the same image IDs are used in every epoch.
Only the first validation batch is cached; full maps are never written per training batch.

Fields:
- Wg/Wl: global and local fusion weights from FeatureEmbedding.
- A1/A2/A3: local scale weights for dilation 1/2/3 from LocalFeature.
- *_std: population standard deviation over all selected pixels.
- *_within_image_std_mean: mean spatial standard deviation inside each image.
- *_gt_{self.threshold:.2f}_ratio: fraction of pixels above the threshold.
- *_dominant_ratio: fraction where that scale is largest and leads the runner-up by at least the dominant margin.
- A_no_clear_dominant_ratio: fraction where the top-two scale weights differ by less than {self.dominant_margin:.2f}.
- A_entropy_mean: normalized scale entropy, 1 means uniform and 0 means one-hot.
- group=class:* uses ground-truth class pixels.
- group=edge:Building/interior:Building uses a {2 * self.edge_radius + 1}x{2 * self.edge_radius + 1} morphological band around the Building boundary at the original validation resolution.

Outputs:
- gate_statistics.csv: one row per epoch, stage, and region group.
- gate_interpretation.txt: latest automatic interpretation.
- heatmaps/epoch_XXXX: saved every {self.heatmap_interval} epochs.
- heatmaps/best_val_f1: replaced when a higher validation F1 is reached.
"""
        (output_dir / "README.txt").write_text(text, encoding="utf-8")

    def _write_interpretation(self, output_dir):
        latest_epoch = max(int(row["epoch"]) for row in self._epoch_rows)
        latest = [row for row in self._epoch_rows if int(row["epoch"]) == latest_epoch]
        lines = [
            f"Latest validation epoch: {latest_epoch}",
            "Interpret with both the statistics and the saved heatmaps.",
            "",
        ]
        for stage in STAGES:
            rows = {row["group"]: row for row in latest if row["stage"] == stage}
            overall = rows.get("all")
            if not overall:
                continue
            wg_std = float(overall["Wg_std"])
            entropy = float(overall["A_entropy_mean"])
            dominance = max(float(overall[f"A{index}_dominant_ratio"]) for index in range(1, 4))
            if wg_std < 0.02:
                wg_text = "near-constant global/local gate"
            elif wg_std < 0.05:
                wg_text = "weak global/local spatial variation"
            else:
                wg_text = "clear global/local spatial variation"
            if entropy > 0.95:
                scale_text = "local scales are close to uniform"
            elif entropy > 0.80:
                scale_text = "local scales show moderate differentiation"
            else:
                scale_text = "local scales show strong differentiation"
            lines.append(
                f"{stage}: Wg_mean={float(overall['Wg_mean']):.4f}, "
                f"Wg_std={wg_std:.4f}, dominant_ratio_max={dominance:.4f}, "
                f"A_entropy={entropy:.4f}; {wg_text}; {scale_text}."
            )

            edge = rows.get("edge:Building")
            interior = rows.get("interior:Building")
            if edge and interior:
                delta = float(edge["Wl_mean"]) - float(interior["Wl_mean"])
                if delta > 0.03:
                    preference = "Building edges prefer Local"
                elif delta < -0.03:
                    preference = "Building edges prefer Global"
                else:
                    preference = "Building edges and interiors are similar"
                lines.append(
                    f"{stage}: Wl(edge)-Wl(interior)={delta:+.4f}; {preference}."
                )

        lines.extend(
            (
                "",
                "If both Wg_std and within-image Wg_std are very small, the global/local gate is not making spatial choices.",
                "If A_entropy is near 1 and all dominant ratios are near 1/3, the three-scale selector is effectively uniform.",
                "Use class:* rows to compare road, vegetation, car, and other category preferences.",
            )
        )
        (output_dir / "gate_interpretation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
