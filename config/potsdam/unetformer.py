# 1. 导入需要的类和函数
# 2. 设置训练超参数
# 3. 创建UNetFormer模型
# 4. 创建损失函数
# 5. 创建训练集、验证集和DataLoader
# 6. 创建优化器和学习率调度器

from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.potsdam_dataset import *
from geoseg.models.UNetFormer import UNetFormer
from tools.utils import Lookahead
from tools.utils import process_model_params

# training hparam
max_epoch = 45
ignore_index = len(CLASSES)#忽略标签，不属于任何有效类别，可被用作不参与损失计算的像素
train_batch_size = 8
val_batch_size = 8
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5#通常加载ImageNet预训练权重，已经学习过，只需小幅微调
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES#类别名称序列

weights_name = "unetformer-r18-768crop-ms-e45" #实验名称，其中ms：可能表示multi-scale，多尺度训练或测试
weights_path = "model_weights/potsdam/{}".format(weights_name) #实验得到权重的存放路径
test_weights_name = "unetformer-r18-768crop-ms-e45" #使用这个名字实验的权重来测试
log_name = 'potsdam/{}'.format(weights_name) #实验日志存放路径
monitor = 'val_F1' #保存最佳模型时观察的指标是val_F1
monitor_mode = 'max' #F1越大越好
save_top_k = 1 #表示只保存指标最好的1个checkpoint
save_last = True #除了最佳模型，还额外保存最后一轮模型
check_val_every_n_epoch = 1 #表示每训练1个epoch，验证1次
pretrained_ckpt_path = None # 预训练权重路径，这个应该指的是整个模型的预训练权重，不是指resnet18的预训练权重
gpus = 'auto'  # 默认或 GPU ID：[0] 或 GPU 数量：2，更多设置可以参考 pytorch_lightning
resume_ckpt_path = None  # 是否继续使用检查点进行训练，默认值为无。断点续训：恢复整个训练现场，继续原来的epoch、优化器和学习率状态。

#  define the network
net = UNetFormer(num_classes=num_classes, pretrained=False, 
                 pretrained_path='pretrain_weights/resnet18_imagenet.pth')

# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True #启用辅助损失

# define the dataloader

train_dataset = PotsdamDataset(data_root='data/potsdam/train', mode='train',
                               mosaic_ratio=0.25, transform=train_aug)
#mosaic_ratio=0.25表示训练时有25%的概率执行Mosaic数据增强，train_aug是Albumentations的增强流水线
val_dataset = PotsdamDataset(data_root='data/potsdam/val', transform=val_aug)
test_dataset = PotsdamDataset(data_root='data/potsdam/test',
                              transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,#4个CPU进程加载数据
                          pin_memory=True,#加速CPU→GPU传输
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
#分层学习率，backbone使用特殊训练策略。
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
#把UNetFormer里的参数按照模块分类
net_params = process_model_params(net, layerwise_params=layerwise_params)
#创建AdamW，让它按照不同参数组更新网络。
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
#在AdamW外面再套一个稳定优化策略。
optimizer = Lookahead(base_optimizer)
#训练过程中动态改变学习率，不同于AdamW给每个参数调整步幅，这里是给整个模型调的学习率
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

