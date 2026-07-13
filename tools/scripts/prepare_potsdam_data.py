import os
import os.path as osp
import shutil
import zipfile
import glob
from pathlib import Path
import subprocess
import sys

"""
Potsdam 数据预处理脚本
========================
按照 UNetFormer 论文（Table 9）的划分方式准备 Potsdam 数据集。

论文划分:
  - 训练集 (22张): 排除测试14张 + 验证1张(2_10) + 有标注错误的1张(7_10) 后的剩余图片
  - 验证集 (1张):  2_10
  - 测试集 (14张): 2_13, 2_14, 3_13, 3_14, 4_13, 4_14, 4_15, 5_13, 5_14, 
                    5_15, 6_13, 6_14, 6_15, 7_13

用法:
  python tools/scripts/prepare_potsdam_data.py --data-root /path/to/Potsdam数据集
"""

# ========== 论文规定的划分 ==========
TEST_IDS = [
    '2_13', '2_14', '3_13', '3_14', '4_13', '4_14', '4_15',
    '5_13', '5_14', '5_15', '6_13', '6_14', '6_15', '7_13'
]
VAL_IDS = ['2_10']
EXCLUDED_IDS = ['7_10']  # 论文指出 7_10 有标注错误


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Prepare Potsdam dataset following UNetFormer paper split')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to the raw Potsdam dataset directory (containing zip files)')
    parser.add_argument('--output-dir', type=str, default='data/potsdam',
                        help='Output directory for processed data (relative to GeoSeg-main)')
    parser.add_argument('--skip-extract', action='store_true',
                        help='Skip zip extraction if already done')
    parser.add_argument('--skip-patch-split', action='store_true',
                        help='Skip patch splitting if already done')
    return parser.parse_args()


def extract_zip(zip_path, extract_dir):
    """Extract a zip file to the target directory."""
    print(f'[Extract] {zip_path} -> {extract_dir}')
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
    tif_files = list(Path(extract_dir).rglob('*.tif'))
    print(f'  -> Found {len(tif_files)} .tif files')


def get_tile_ids(all_files):
    """Extract tile IDs from Path objects or filenames."""
    ids = set()
    for f in all_files:
        name = Path(f).name
        if name.startswith('top_potsdam_'):
            parts = name.replace('top_potsdam_', '').split('_')
            if len(parts) >= 2:
                tile_id = f'{parts[0]}_{parts[1]}'
                ids.add(tile_id)
    return sorted(ids)


def find_source_file(source_dir, tile_id, suffix):
    """Find and return path of file matching tile_id + suffix (recursive)."""
    pattern = f'*{tile_id}{suffix}'
    matches = list(Path(source_dir).rglob(pattern))
    return str(matches[0]) if matches else None


def main():
    args = parse_args()
    data_root = args.data_root
    output_dir = args.output_dir
    geoseg_root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
    
    if not osp.isabs(output_dir):
        output_dir = osp.join(geoseg_root, output_dir)

    print('=' * 60)
    print('UNetFormer Potsdam 数据预处理')
    print('=' * 60)
    print(f'原始数据目录: {data_root}')
    print(f'输出目录:     {output_dir}')
    print(f'GeoSeg 根目录: {geoseg_root}')
    print()

    required_zips = {
        'rgb': osp.join(data_root, '2_Ortho_RGB.zip'),
        'labels_all': osp.join(data_root, '5_Labels_all.zip'),
        'labels_no_boundary': osp.join(data_root, '5_Labels_all_noBoundary.zip'),
    }
    for name, path in required_zips.items():
        if not osp.exists(path):
            print(f'[ERROR] 找不到 {name}: {path}')
            sys.exit(1)
        print(f'[OK] {name}: {path}')

    tmp_dir = osp.join(output_dir, '_tmp_extracted')
    
    if not args.skip_extract:
        print('\n[Step 1/4] 提取压缩包...')
        extract_zip(required_zips['rgb'], osp.join(tmp_dir, 'rgb'))
        extract_zip(required_zips['labels_all'], osp.join(tmp_dir, 'labels_all'))
        extract_zip(required_zips['labels_no_boundary'], osp.join(tmp_dir, 'labels_all_nb'))
        print('提取完成。')
    else:
        print('\n[Step 1/4] 跳过提取 (--skip-extract)')

    print('\n[Step 2/4] 按论文划分整理文件...')
    
    rgb_dir = osp.join(tmp_dir, 'rgb')
    label_dir = osp.join(tmp_dir, 'labels_all')
    label_nb_dir = osp.join(tmp_dir, 'labels_all_nb')

    rgb_tifs = list(Path(rgb_dir).rglob('*.tif'))
    tile_ids = get_tile_ids([str(f) for f in rgb_tifs])
    print(f'  总 tile 数: {len(tile_ids)}')
    
    train_ids = [tid for tid in tile_ids 
                 if tid not in TEST_IDS and tid not in VAL_IDS and tid not in EXCLUDED_IDS]
    
    print(f'  训练集: {len(train_ids)} tiles')
    print(f'  验证集: {len(VAL_IDS)} tiles (2_10)')
    print(f'  测试集: {len(TEST_IDS)} tiles')
    print(f'  排除:   {len(EXCLUDED_IDS)} tiles (7_10, 论文指出标注错误)')
    
    dirs = {
        'train_images': osp.join(output_dir, 'train_images'),
        'train_masks': osp.join(output_dir, 'train_masks'),
        'val_images': osp.join(output_dir, 'val_images'),
        'val_masks': osp.join(output_dir, 'val_masks'),
        'test_images': osp.join(output_dir, 'test_images'),
        'test_masks_eroded': osp.join(output_dir, 'test_masks_eroded'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print('\n  复制训练集 (22 tiles)...')
    for tid in train_ids:
        src_img = find_source_file(rgb_dir, tid, '_RGB.tif')
        src_lbl = find_source_file(label_dir, tid, '_label.tif')
        if src_img:
            shutil.copy2(src_img, osp.join(dirs['train_images'], f'top_potsdam_{tid}_RGB.tif'))
        if src_lbl:
            shutil.copy2(src_lbl, osp.join(dirs['train_masks'], f'top_potsdam_{tid}_label.tif'))

    print('  复制验证集 (1 tile: 2_10)...')
    for vid in VAL_IDS:
        src_img = find_source_file(rgb_dir, vid, '_RGB.tif')
        src_lbl = find_source_file(label_dir, vid, '_label.tif')
        if src_img:
            shutil.copy2(src_img, osp.join(dirs['val_images'], f'top_potsdam_{vid}_RGB.tif'))
        if src_lbl:
            shutil.copy2(src_lbl, osp.join(dirs['val_masks'], f'top_potsdam_{vid}_label.tif'))

    print('  复制测试集 (14 tiles)...')
    for tid in TEST_IDS:
        src_img = find_source_file(rgb_dir, tid, '_RGB.tif')
        src_lbl = find_source_file(label_nb_dir, tid, '_label_noBoundary.tif')
        if src_img:
            shutil.copy2(src_img, osp.join(dirs['test_images'], f'top_potsdam_{tid}_RGB.tif'))
        if src_lbl:
            shutil.copy2(src_lbl, osp.join(dirs['test_masks_eroded'], f'top_potsdam_{tid}_label_noBoundary.tif'))

    print(f'\n  训练集 tiles: {sorted(train_ids)}')
    print(f'  验证集 tiles: {VAL_IDS}')
    print(f'  测试集 tiles: {TEST_IDS}')

    if not args.skip_patch_split:
        print('\n[Step 3/4] 执行 Patch Split...')
        
        # Check if required packages are available
        try:
            import albumentations
            import cv2
            import torch
        except ImportError as e:
            print(f'  [WARNING] 缺少依赖包: {e}')
            print('  本地 Windows 可能没有安装 GPU 相关包，跳过 patch split。')
            print('  上传到服务器后再运行 patch split 即可。')
            args.skip_patch_split = True
        
    if not args.skip_patch_split:
        patch_split_script = osp.join(geoseg_root, 'tools', 'potsdam_patch_split.py')
        
        print('\n  --- 训练集 patch split ---')
        train_cmd = [
            sys.executable, patch_split_script,
            '--img-dir', dirs['train_images'],
            '--mask-dir', dirs['train_masks'],
            '--output-img-dir', osp.join(output_dir, 'train', 'images_1024'),
            '--output-mask-dir', osp.join(output_dir, 'train', 'masks_1024'),
            '--mode', 'train',
            '--split-size', '1024',
            '--stride', '1024',
            '--rgb-image'
        ]
        print(f'  Run: {" ".join(train_cmd)}')
        subprocess.run(train_cmd, check=True)
        
        print('\n  --- 验证集 patch split ---')
        val_cmd = [
            sys.executable, patch_split_script,
            '--img-dir', dirs['val_images'],
            '--mask-dir', dirs['val_masks'],
            '--output-img-dir', osp.join(output_dir, 'val', 'images_1024'),
            '--output-mask-dir', osp.join(output_dir, 'val', 'masks_1024'),
            '--mode', 'val',
            '--split-size', '1024',
            '--stride', '1024',
            '--rgb-image'
        ]
        print(f'  Run: {" ".join(val_cmd)}')
        subprocess.run(val_cmd, check=True)
        
        print('\n  --- 测试集 patch split ---')
        test_cmd = [
            sys.executable, patch_split_script,
            '--img-dir', dirs['test_images'],
            '--mask-dir', dirs['test_masks_eroded'],
            '--output-img-dir', osp.join(output_dir, 'test', 'images_1024'),
            '--output-mask-dir', osp.join(output_dir, 'test', 'masks_1024'),
            '--mode', 'val',
            '--split-size', '1024',
            '--stride', '1024',
            '--rgb-image',
            '--eroded'
        ]
        print(f'  Run: {" ".join(test_cmd)}')
        subprocess.run(test_cmd, check=True)
        
        print('\nPatch split 完成！')
    else:
        print('\n[Step 3/4] 跳过 Patch Split（Patch split 留到服务器上做）')
        print()
        print('  ===============================================================')
        print('  数据预处理完成！目录结构:')
        print('    data/potsdam/train_images/       (22 tiles)')
        print('    data/potsdam/train_masks/        (22 tiles)')
        print('    data/potsdam/val_images/         (1 tile: 2_10)')
        print('    data/potsdam/val_masks/          (1 tile: 2_10)')
        print('    data/potsdam/test_images/        (14 tiles)')
        print('    data/potsdam/test_masks_eroded/  (14 tiles, eroded)')
        print()
        print('  上传到服务器后，一键执行全部流程:')
        print('    bash tools/scripts/train_potsdam.sh')
        print('  这将自动运行 patch split + 训练。')
        print('  ===============================================================')

    print('\n[Step 4/4] 设置预训练权重...')
    pretrain_dir = osp.join(geoseg_root, 'pretrain_weights')
    os.makedirs(pretrain_dir, exist_ok=True)
    
    possible_paths = [
        osp.join(data_root, '..', 'resnet18_imagenet.pth'),
        osp.join(geoseg_root, '..', 'resnet18_imagenet.pth'),
        osp.join(geoseg_root, 'pretrain_weights', 'resnet18_imagenet.pth'),
    ]
    found = False
    for pp in possible_paths:
        pp = osp.abspath(pp)
        if osp.exists(pp):
            dst = osp.join(pretrain_dir, 'resnet18_imagenet.pth')
            if osp.abspath(pp) != osp.abspath(dst):
                shutil.copy2(pp, dst)
                print(f'  已复制: {pp} -> {dst}')
            else:
                print(f'  权重已在目标位置: {dst}')
            found = True
            break
    
    if not found:
        print('  [WARNING] 未找到 resnet18_imagenet.pth！')
        print(f'  请手动将预训练权重文件复制到: {osp.join(pretrain_dir, "resnet18_imagenet.pth")}')
    else:
        print(f'  [OK] 预训练权重路径: {osp.join(pretrain_dir, "resnet18_imagenet.pth")}')

    if not args.skip_extract:
        print('\n  清理临时文件...')
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print('  临时文件已清理。')

    print('\n' + '=' * 60)
    print('数据预处理完成！最终目录结构:')
    print('=' * 60)
    for root, dirs_, files in os.walk(output_dir):
        if '_tmp' in root:
            continue
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        if level < 2:
            subindent = ' ' * 2 * (level + 1)
            for f in sorted(files)[:3]:
                print(f'{subindent}{f}')
            if len(files) > 3:
                print(f'{subindent}... ({len(files)} files total)')

    print('\n[完成] 数据已准备好！')
    print(f'现在你可以用以下命令运行训练:')
    print(f'  cd {geoseg_root}')
    print(f'  python train_supervision.py -c config/potsdam/unetformer.py')


if __name__ == '__main__':
    main()
