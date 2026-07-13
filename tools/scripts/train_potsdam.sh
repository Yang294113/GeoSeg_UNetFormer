#!/bin/bash
# ============================================================
# UNetFormer Potsdam 一键训练脚本（从 patch split 到训练）
# 适用于 Linux 算力服务器
# ============================================================
# 用法:
#   bash tools/scripts/train_potsdam.sh
#
# 前提条件:
#   1. 本地已运行 prepare_potsdam_data.py (Step 1-2)
#   2. 预训练权重在 pretrain_weights/resnet18_imagenet.pth
#   3. Python 环境已配置 (conda activate airs)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo " UNetFormer - Potsdam 一键训练"
echo "=============================================="

# ===== Step 1: 检查并执行 patch split =====

if [ -d "data/potsdam/train/images_1024" ]; then
    echo "[OK] 切块数据已存在，跳过 patch split"
else
    echo ""
    echo "[Step 1] 检查原始 tile 数据..."
    MISSING=0
    for DIR in train_images train_masks val_images val_masks test_images test_masks_eroded; do
        if [ ! -d "data/potsdam/$DIR" ]; then
            echo "  [ERROR] 缺少: data/potsdam/$DIR"
            MISSING=1
        fi
    done
    if [ $MISSING -eq 1 ]; then
        echo ""
        echo "  ________________________________________________________________"
        echo "  请先在本地运行数据预处理:"
        echo "    python tools/scripts/prepare_potsdam_data.py --data-root <路径>"
        echo "  ________________________________________________________________"
        exit 1
    fi

    echo "  [OK] tile 目录存在，patch split 开始..."
    python tools/potsdam_patch_split.py --img-dir data/potsdam/train_images --mask-dir data/potsdam/train_masks --output-img-dir data/potsdam/train/images_1024 --output-mask-dir data/potsdam/train/masks_1024 --mode train --split-size 1024 --stride 1024 --rgb-image
    python tools/potsdam_patch_split.py --img-dir data/potsdam/val_images --mask-dir data/potsdam/val_masks --output-img-dir data/potsdam/val/images_1024 --output-mask-dir data/potsdam/val/masks_1024 --mode val --split-size 1024 --stride 1024 --rgb-image
    python tools/potsdam_patch_split.py --img-dir data/potsdam/test_images --mask-dir data/potsdam/test_masks_eroded --output-img-dir data/potsdam/test/images_1024 --output-mask-dir data/potsdam/test/masks_1024 --mode val --split-size 1024 --stride 1024 --rgb-image --eroded
    echo "  [OK] Patch split 完成！"
fi

# ===== Step 2: 检查预训练权重 =====
if [ ! -f "pretrain_weights/resnet18_imagenet.pth" ]; then
    echo "[WARNING] 未找到预训练权重，训练将从头开始"
fi

mkdir -p model_weights/potsdam lightning_logs

# ===== Step 3: 训练 =====
echo ""
echo "开始训练..."
python train_supervision.py -c config/potsdam/unetformer.py
echo ""
echo "训练完成！测试命令:"
echo "  python potsdam_test.py -c config/potsdam/unetformer.py -o fig_results/potsdam/unetformer -t 'lr' --rgb"
echo ""
