# 第一部分：导入依赖
# 第二部分：定义类别和颜色表
# 第三部分：定义训练增强
# 第四部分：定义验证增强
# 第五部分：定义PotsdamDataset
# 第六部分：普通图像读取
# 第七部分：Mosaic四图拼接
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu #图像增强库，可以对图像和分割标签同步执行几何变换
from .transform import *
import matplotlib.patches as mpatches
from PIL import Image
import random

#类别定义
CLASSES = ('ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter')
#颜色表，PALETTE通常用于将类别索引图转换成彩色标签图；预测结果可视化；保存分割图。
PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 204, 0], [255, 0, 0]]
#图像尺寸常量
ORIGIN_IMG_SIZE = (1024, 1024)#表示数据集中单张切片图像的原始尺寸
INPUT_IMG_SIZE = (1024, 1024)
TEST_IMG_SIZE = (1024, 1024)
#训练赠强入口，构建了一组Albumentations增强操作
#这里的Normalize发生在数据预处理阶段，把输入RGB像素转换到适合神经网络训练的数值分布，通常匹配ImageNet预训练权重的输入要求。
def get_training_transform():
    train_transform = [
        # albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.15),
        # albu.RandomRotate90(p=0.25),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)#按照顺序执行列表中的所有增强，当前只有Normalize

#训练增强函数，执行缩放和裁剪，归一化，将PIL数据类型转化为NumPy数组
def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),#随机尺度缩放
                        SmartCropV1(crop_size=768, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False)])#把随机缩放后的图片裁剪成768*768
    img, mask = crop_aug(img, mask)#执行缩放和裁剪（裁剪也是一种训练数据增强手段，增加样本多样性，同时降低训练显存消耗）
    img, mask = np.array(img), np.array(mask)#PIL转Numpy
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())#执行albumentations归一化
    img, mask = aug['image'], aug['mask']#albumentations返回的值是字典类型
    return img, mask

#验证增强入口
def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)

#验证增强，PIL转NumPy，归一化，无裁剪
#UNetFormer是全卷积网络，通常能够接受不同空间尺寸，只要尺寸满足网络下采样倍率等要求，1024和768都能被32整除
def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

#继承自torch.utils.data.Dataset
class PotsdamDataset(Dataset):
    #这里后面的参数只是默认值，真正使用的时候一般都会重新赋值
    def __init__(self, data_root='data/potsdam/test', mode='val', img_dir='images_1024', mask_dir='masks_1024',
                 img_suffix='.tif', mask_suffix='.png', transform=val_aug, mosaic_ratio=0.0,
                 img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        #获取全部图像编号，注意，每一个编号代表一对样本
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)
    #获取一个样本，整个dataset最核心方法，返回的是一个字典，里面的数据是tensor类型。这里获取的样本是已经经过mosaic，数据增强（如果需要）之后的数据，
    def __getitem__(self, index):
        p_ratio = random.random()
        #__getitem__获取样本时会根据这里的随机数是否大于0.25来抉择选择普通还是mosaic样本
        if p_ratio > self.mosaic_ratio or self.mode == 'val' or self.mode == 'test':
            img, mask = self.load_img_and_mask(index)
            #等价于if self.transform is not None:，判断是否需要数据增强
            if self.transform:
                img, mask = self.transform(img, mask)
        else:
            img, mask = self.load_mosaic_img_and_mask(index)
            if self.transform:
                img, mask = self.transform(img, mask)
        #把数据类型从numpy转换成tensor，注意torch.from_numpy和torch.tensor()的区别，torch.from_numpy转化数据后和原数据是共享一处内存的，也就是转换数据但不拷贝数据，省内存提升效率
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()#mask是单通道类别索引图，索引不用permute，类型为long
        img_id = self.img_ids[index]
        results = dict(img_id=img_id, img=img, gt_semantic_seg=mask)
        return results
    #len求的是样本的数量，Dataset 的一个 index 永远对应 一对 (image, GT)
    def __len__(self):
        return len(self.img_ids)
    #获取文件ID
    def get_img_ids(self, data_root, img_dir, mask_dir):
        #读取图像目录文件名
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        #读取标签目录文件名
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        #检查数量是否一致，如果图像数量和标签数量不同，程序立即报错
        assert len(img_filename_list) == len(mask_filename_list)
        #从标签文件名提取ID（从右往左执行，这里的id是个临时变量）
        img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
        return img_ids
    #普通图像读取
    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)#suffix是后缀名
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        img = Image.open(img_name).convert('RGB')#convert保证图像固定为3通道
        mask = Image.open(mask_name).convert('L')#L表示单通道8位灰度模式，转换后每个像素只有一个数值
        return img, mask
    #Mosaic增强，这个函数会选取4张图像->确定拼接中心->从每张图随机裁出不同大小区域->拼成1024*1024图像。随后标签执行完全同样操作
    def load_mosaic_img_and_mask(self, index):
        #随机选择四张图，返回的是个列表数据，四个数字，允许重复
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        #读取四组图像和标签，PIL格式
        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])
        #PIL转NumPy
        img_a, mask_a = np.array(img_a), np.array(mask_a)
        img_b, mask_b = np.array(img_b), np.array(mask_b)
        img_c, mask_c = np.array(img_c), np.array(mask_c)
        img_d, mask_d = np.array(img_d), np.array(mask_d)
        #获取目标Mosaic尺寸。注：self._size=(1024,1024)
        w = self.img_size[1]
        h = self.img_size[0]
        #拼接中心允许范围，除4是为了避免分割线过于靠近边缘
        start_x = w // 4
        strat_y = h // 4
        # 随机拼接中心
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))
        #计算四个裁剪区域尺寸
        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)
        #为每张图创建随机裁剪器，这里每张原图都从任意位置随机裁出指定尺寸
        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])
        #开始裁剪图像，并同步标签，裁剪完后输出到 croped_a里，是个字典数据类型，里面有裁剪后图像和裁剪后标签
        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())
        #取出裁剪结果
        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']
        #横向拼接顶部和底部,np.concatenate就是一个数组拼接函数
        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        #纵向拼接完整图像
        img = np.concatenate((top, bottom), axis=0)
        #对标签也执行完全相同拼接
        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        #确保拼接完成后数组内存是连续的，方便后续用torch.from_numpy()把数据转换成tensor，torch.from_numpy()  要求数据内存是连续的，np.concatenate通常拼接完以后内存已经是连续的了，这里是为了稳妥起见
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)
        #把NumPy重新转PIL
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        return img, mask
