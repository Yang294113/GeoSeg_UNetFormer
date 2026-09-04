
# ==========================================
# Potsdam Test Script
# 增加 SpatialWS 权重提取与可视化功能
# ==========================================

# Test Time Augmentation
import ttach as tta

# 多进程保存预测图片
import multiprocessing.pool as mpp
import multiprocessing as mp

import time
from train_supervision import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ==========================================
# 把类别 mask 转换为彩色图
# ==========================================

def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]

    # 创建 RGB 图像
    mask_rgb = np.zeros(
        shape=(h, w, 3),
        dtype=np.uint8
    )

    mask_convert = mask[np.newaxis, :, :]

    mask_rgb[
        np.all(mask_convert == 3, axis=0)
    ] = [0, 255, 0]

    mask_rgb[
        np.all(mask_convert == 0, axis=0)
    ] = [255, 255, 255]

    mask_rgb[
        np.all(mask_convert == 1, axis=0)
    ] = [255, 0, 0]

    mask_rgb[
        np.all(mask_convert == 2, axis=0)
    ] = [255, 255, 0]

    mask_rgb[
        np.all(mask_convert == 4, axis=0)
    ] = [0, 204, 255]

    mask_rgb[
        np.all(mask_convert == 5, axis=0)
    ] = [0, 0, 255]

    return mask_rgb


# ==========================================
# 保存预测结果
# ==========================================

def img_writer(inp):

    (mask, mask_id, rgb) = inp

    if rgb:

        # 保存为 RGB
        mask_name_tif = mask_id + '.png'

        mask_tif = label2rgb(mask)

        cv2.imwrite(
            mask_name_tif,
            mask_tif
        )

    else:

        # 保存为灰度图
        mask_png = mask.astype(np.uint8)

        mask_name_png = mask_id + '.png'

        cv2.imwrite(
            mask_name_png,
            mask_png
        )


# ==========================================
# 读取命令行参数
# ==========================================

def get_args():

    parser = argparse.ArgumentParser()

    arg = parser.add_argument

    # 配置文件
    arg(
        "-c",
        "--config_path",
        type=Path,
        required=True,
        help="Path to config"
    )

    # 结果保存位置
    arg(
        "-o",
        "--output_path",
        type=Path,
        required=True,
        help="Path where to save resulting masks."
    )

    # 测试增强
    arg(
        "-t",
        "--tta",
        help="Test time augmentation.",
        default=None,
        choices=[None, "d4", "lr"]
    )

    # 是否输出彩色 mask
    arg(
        "--rgb",
        help="whether output rgb images",
        action='store_true'
    )

    return parser.parse_args()


# ==========================================
# 权重统计函数
# ==========================================

def print_weight_statistics(name, weights):

    """
    weights:
        [B, 2, H, W]

    weights[:, 0]
        Encoder 权重

    weights[:, 1]
        Decoder 权重
    """

    encoder_weight = weights[:, 0]
    decoder_weight = weights[:, 1]

    print()
    print("==========================================")
    print(f"Weight module: {name}")
    print("==========================================")

    print("Shape:", tuple(weights.shape))

    print()
    print("[Encoder Weight]")

    print(
        f"min  = {encoder_weight.min().item():.6f}"
    )

    print(
        f"max  = {encoder_weight.max().item():.6f}"
    )

    print(
        f"mean = {encoder_weight.mean().item():.6f}"
    )

    print(
        f"std  = {encoder_weight.std().item():.6f}"
    )

    print()
    print("[Decoder Weight]")

    print(
        f"min  = {decoder_weight.min().item():.6f}"
    )

    print(
        f"max  = {decoder_weight.max().item():.6f}"
    )

    print(
        f"mean = {decoder_weight.mean().item():.6f}"
    )

    print(
        f"std  = {decoder_weight.std().item():.6f}"
    )

    # 检查 Softmax 后两个权重是否约等于 1
    weight_sum = encoder_weight + decoder_weight

    max_error = torch.abs(
        weight_sum - 1.0
    ).max().item()

    print()
    print(
        f"Max |Encoder + Decoder - 1| = "
        f"{max_error:.8f}"
    )


# ==========================================
# 保存权重 Heatmap
# ==========================================

def save_weight_heatmap(
        weight_map,
        save_path,
        title):

    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        weight_map,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0
    )

    plt.colorbar(
        fraction=0.046,
        pad=0.04
    )

    plt.title(title)

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================
# 主函数
# ==========================================

def main():

    # --------------------------------------
    # 1. 读取命令行参数
    # --------------------------------------

    args = get_args()

    seed_everything(42)

    # --------------------------------------
    # 2. 读取配置
    # --------------------------------------

    config = py2cfg(
        args.config_path
    )

    # --------------------------------------
    # 3. 创建输出目录
    # --------------------------------------

    args.output_path.mkdir(
        exist_ok=True,
        parents=True
    )

    # 权重分析目录
    weight_save_dir = (
        args.output_path / "weights"
    )

    weight_save_dir.mkdir(
        exist_ok=True,
        parents=True
    )

    # --------------------------------------
    # 4. 加载训练好的模型
    # --------------------------------------

    checkpoint_path = os.path.join(
        config.weights_path,
        config.test_weights_name + '.ckpt'
    )

    print()
    print("Loading checkpoint:")
    print(checkpoint_path)
    print()

    model = Supervision_Train.load_from_checkpoint(
        checkpoint_path,
        config=config
    )

    model.cuda()

    # --------------------------------------
    # 5. 进入推理模式
    # --------------------------------------

    model.eval()

    # ======================================
    # 6. 注册 hook
    # ======================================

    # 用来保存 weight_gen 的输出
    weight_outputs = {}

    # 用来保存 hook 句柄
    hook_handles = []

    print()
    print("==========================================")
    print("Searching for SpatialWS weight_gen...")
    print("==========================================")

    def save_weight_hook(name):

        def hook(module, inputs, output):

            # output:
            # [B, 2, H, W]
            #
            # output 是 logits
            # 所以这里再做一次 softmax
            weights = torch.softmax(
                output.detach(),
                dim=1
            )

            weight_outputs[name] = (
                weights.cpu()
            )

        return hook

    # 搜索所有名字以 weight_gen 结尾的模块
    for name, module in model.named_modules():

        if name.endswith("weight_gen"):

            print(
                f"Register hook: {name}"
            )

            handle = module.register_forward_hook(
                save_weight_hook(name)
            )

            hook_handles.append(handle)

    print()

    if len(hook_handles) == 0:

        print(
            "WARNING: No weight_gen was found!"
        )

        print(
            "Please check whether "
            "SpatialWS is correctly loaded."
        )

    else:

        print(
            f"Found {len(hook_handles)} "
            f"weight_gen module(s)."
        )

    print()

    # --------------------------------------
    # 7. Evaluator
    # --------------------------------------

    evaluator = Evaluator(
        num_class=config.num_classes
    )

    evaluator.reset()

    # ======================================
    # 8. TTA
    # ======================================

    # 注意：
    # 如果你要分析 SpatialWS 权重，
    # 推荐不要使用 TTA。
    #
    # 因为 TTA 会对同一张图进行多次 forward，
    # hook 中的 weight_outputs 会不断被覆盖。
    #
    # 所以建议：
    #
    # python potsdam_test.py \
    # -c config/potsdam/unetformer.py \
    # -o fig_results/potsdam/spatialws_analysis \
    # --rgb
    #
    # 不加 -t lr / -t d4

    if args.tta == "lr":

        print(
            "WARNING: TTA=lr is enabled."
        )

        print(
            "The saved weights will correspond "
            "to the last TTA forward pass."
        )

        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )

        model = tta.SegmentationTTAWrapper(
            model,
            transforms
        )

    elif args.tta == "d4":

        print(
            "WARNING: TTA=d4 is enabled."
        )

        print(
            "The saved weights will correspond "
            "to the last TTA forward pass."
        )

        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),

                # tta.Rotate90(angles=[90]),

                tta.Scale(
                    scales=[
                        0.75,
                        1.0,
                        1.25,
                        1.5
                    ],
                    interpolation='bicubic',
                    align_corners=False
                )
            ]
        )

        model = tta.SegmentationTTAWrapper(
            model,
            transforms
        )

    # --------------------------------------
    # 9. Test Dataset
    # --------------------------------------

    test_dataset = config.test_dataset

    # --------------------------------------
    # 10. DataLoader
    # --------------------------------------

    with torch.no_grad():

        test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        results = []

        # ----------------------------------
        # 只分析第一批
        # ----------------------------------

        saved_weight_once = False

        # ----------------------------------
        # 11. 开始测试
        # ----------------------------------

        for input in tqdm(test_loader):

            # ==================================
            # 模型推理
            # ==================================

            raw_predictions = model(
                input['img'].cuda()
            )

            # ==================================
            # 第一次 forward 后
            # 分析 SpatialWS 权重
            # ==================================

            if not saved_weight_once:

                print()
                print()
                print(
                    "########################################"
                )

                print(
                    "# SpatialWS Weight Analysis"
                )

                print(
                    "########################################"
                )

                # ------------------------------
                # 检查有没有捕获到 weight
                # ------------------------------

                if len(weight_outputs) == 0:

                    print()
                    print(
                        "ERROR:"
                    )

                    print(
                        "No weight output was captured."
                    )

                    print(
                        "Please check the model."
                    )

                else:

                    # ==========================
                    # 遍历 p3 / p2 / p1
                    # ==========================

                    for name, weights in (
                        weight_outputs.items()
                    ):

                        # ----------------------
                        # 打印统计信息
                        # ----------------------

                        print_weight_statistics(
                            name,
                            weights
                        )

                        # ----------------------
                        # 当前批次第一张图片
                        # ----------------------

                        encoder_map = (
                            weights[0, 0]
                            .numpy()
                        )

                        decoder_map = (
                            weights[0, 1]
                            .numpy()
                        )

                        # ----------------------
                        # 文件名安全处理
                        # ----------------------

                        safe_name = (
                            name
                            .replace(".", "_")
                            .replace("/", "_")
                        )

                        # ----------------------
                        # 保存 Encoder 权重
                        # ----------------------

                        encoder_npy_path = (
                            weight_save_dir
                            /
                            f"{safe_name}"
                            "_encoder.npy"
                        )

                        np.save(
                            encoder_npy_path,
                            encoder_map
                        )

                        encoder_png_path = (
                            weight_save_dir
                            /
                            f"{safe_name}"
                            "_encoder.png"
                        )

                        save_weight_heatmap(
                            encoder_map,
                            encoder_png_path,
                            f"{name} - Encoder Weight"
                        )

                        # ----------------------
                        # 保存 Decoder 权重
                        # ----------------------

                        decoder_npy_path = (
                            weight_save_dir
                            /
                            f"{safe_name}"
                            "_decoder.npy"
                        )

                        np.save(
                            decoder_npy_path,
                            decoder_map
                        )

                        decoder_png_path = (
                            weight_save_dir
                            /
                            f"{safe_name}"
                            "_decoder.png"
                        )

                        save_weight_heatmap(
                            decoder_map,
                            decoder_png_path,
                            f"{name} - Decoder Weight"
                        )

                        print()
                        print(
                            "Saved:"
                        )

                        print(
                            encoder_png_path
                        )

                        print(
                            decoder_png_path
                        )

                print()
                print(
                    "########################################"
                )

                print(
                    "# End SpatialWS Weight Analysis"
                )

                print(
                    "########################################"
                )

                print()

                # 只保存一次
                saved_weight_once = True

            # ==================================
            # 下面完全保持原 test 逻辑
            # ==================================

            image_ids = input["img_id"]

            masks_true = input["gt_semantic_seg"]

            # ----------------------------------
            # Softmax
            # ----------------------------------

            raw_predictions = nn.Softmax(
                dim=1
            )(raw_predictions)

            # ----------------------------------
            # Argmax
            # ----------------------------------

            predictions = raw_predictions.argmax(
                dim=1
            )

            # ----------------------------------
            # 计算指标
            # ----------------------------------

            for i in range(
                raw_predictions.shape[0]
            ):

                mask = (
                    predictions[i]
                    .cpu()
                    .numpy()
                )

                evaluator.add_batch(
                    pre_image=mask,
                    gt_image=(
                        masks_true[i]
                        .cpu()
                        .numpy()
                    )
                )

                mask_name = image_ids[i]

                results.append(
                    (
                        mask,
                        str(
                            args.output_path
                            / mask_name
                        ),
                        args.rgb
                    )
                )

    # ======================================
    # 12. 输出测试指标
    # ======================================

    iou_per_class = (
        evaluator.Intersection_over_Union()
    )

    f1_per_class = (
        evaluator.F1()
    )

    OA = evaluator.OA()

    print()
    print(
        "=========================================="
    )

    print(
        "Test Results"
    )

    print(
        "=========================================="
    )

    for (
        class_name,
        class_iou,
        class_f1
    ) in zip(
        config.classes,
        iou_per_class,
        f1_per_class
    ):

        print(
            'F1_{}:{}, IOU_{}:{}'.format(
                class_name,
                class_f1,
                class_name,
                class_iou
            )
        )

    print(
        'F1:{}, mIOU:{}, OA:{}'.format(
            np.nanmean(
                f1_per_class[:-1]
            ),
            np.nanmean(
                iou_per_class[:-1]
            ),
            OA
        )
    )

    # ======================================
    # 13. 保存预测图
    # ======================================

    t0 = time.time()

    mpp.Pool(
        processes=mp.cpu_count()
    ).map(
        img_writer,
        results
    )

    t1 = time.time()

    img_write_time = (
        t1 - t0
    )

    print(
        'images writing spends: {} s'.format(
            img_write_time
        )
    )

    # ======================================
    # 14. 删除 hooks
    # ======================================

    for handle in hook_handles:

        handle.remove()

    print()
    print(
        "All hooks removed."
    )


if __name__ == "__main__":

    main()
