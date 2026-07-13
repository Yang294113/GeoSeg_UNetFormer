#!/bin/bash
# ============================================================
# UNetFormer Potsdam 一键训练脚本（从 patch split 到训练）
# 适用于 Linux 算力服务器
# ============================================================
# 用法:
#   bash tools/scripts/train_potsdam.sh
#
# 前提条件:
#   1. 本地已运行 prepare_potsdam_data.py 的 Step 1-2
#      (即 data/potsdam/{train_images,train_masks,val_images,
#       val_masks,test_images,test_masks_eroded} 已存在)
#   2. 预训练权重在 pretrain_weights/resnet18_imagenet.pth
#   3. Python 环境已配置 (conda activate airs)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo " UNetFormer - Potsdam 一键训练"
echo " 工作目录: $(pwd)"
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
        echo "  _______________________________________________________________"
        echo "  请先在本地 Windows 上运行数据预处理:"
        echo "    python tools/scripts/prepare_potsdam_data.py --data-root ..."
        echo "  然后将整个 GeoSeg-main 文件夹上传到服务器。"
        echo "  _______________________________________________________________"
        exit 1
    fi

    echo "  [OK] 所有 tile 目录存在，开始执行 patch split..."
    echo ""

    echo "  --- 训练集 patch split (22 tiles -> 1024x1024 patches) ---"
    python tools/potsdam_patch_split.py \
        --img-dir data/potsdam/train_images \
        --mask-dir data/potsdam/train_masks \
        --output-img-dir data/potsdam/train/images_1024 \
        --output-mask-dir data/potsdam/train/masks_1024 \
        --mode train --split-size 1024 --stride 1024 --rgb-image

    echo "  --- 验证集 patch split (1 tile: 2_10) ---"
    python tools/potsdam_patch_split.py \
        --img-dir data/potsdam/val_images \
        --mask-dir data/potsdam/val_masks \
        --output-img-dir data/potsdam/val/images_1024 \
        --output-mask-dir data/potsdam/val/masks_1024 \
        --mode val --split-size 1024 --stride 1024 --rgb-image

    echo "  --- 测试集 patch split (14 tiles, eroded) ---"
    python tools/potsdam_patch_split.py \
        --img-dir data/potsdam/test_images \
        --mask-dir data/potsdam/test_masks_eroded \
        --output-img-dir data/potsdam/test/images_1024 \
        --output-mask-dir data/potsdam/test/masks_1024 \
        --mode val --split-size 1024 --stride 1024 --rgb-image --eroded

    echo ""
    echo "  [OK] Patch split 完成！"
fi

# ===== Step 2: 检查预训练权重 =====
if [ ! -f "pretrain_weights/resnet18_imagenet.pth" ]; then
    echo "[WARNING] 未找到预训练权重: pretrain_weights/resnet18_imagenet.pth"
    echo "训练将从头开始（可能影响最终精度）"
fi

mkdir -p model_weights/potsdam
mkdir -p lightning_logs

# ===== Step 3: 训练 =====
echo ""
echo "=============================================="
echo " [Step 2] 开始训练..."
echo "=============================================="
echo ""

python train_supervision.py -c config/potsdam/unetformer.py

echo ""
echo "=============================================="
echo " 训练完成！"
echo "=============================================="
echo ""
echo "模型权重: model_weights/potsdam/unetformer-r18-768crop-ms-e45/"
echo "训练日志: lightning_logs/potsdam/"
echo ""
echo "测试命令:"
echo "  python potsdam_test.py -c config/potsdam/unetformer.py -o fig_results/potsdam/unetformer -t 'lr' --rgb"
echo ""
