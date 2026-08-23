# 第一部分：导入库
# 第二部分：设置随机种子
# 第三部分：解析命令行参数
# 第四部分：定义Lightning训练模块
# 第五部分：配置训练器、日志和模型保存
# 第六部分：启动main函数
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint#用来保存模型的
from tools.cfg import py2cfg#配置加载函数
import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse#解析命令行参数
from pathlib import Path#解析命令行参数
from tools.metric import Evaluator#负责维护混淆矩阵，并计算指标
from pytorch_lightning.loggers import CSVLogger#负责把训练指标写入CSV文件
import random

#随机种子函数，确保随机序列的可重复性，随机种子让所有随机过程按照同一套随机轨迹运行，因此在相同环境下，多次训练结果应该高度一致；它的作用不是让模型性能提高，而是让实验更加可重复、更容易比较。
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #这里最好是第二个设为False，才能完美复现达到效果，等研究完了可以试一下
    torch.backends.cudnn.deterministic = True#表示优先选择确定性算法，强调结果可复现
    torch.backends.cudnn.benchmark = False#针对当前输入尺寸测试多个卷积算法，选择速度最快的算法，强调自动寻找最快算法

#就是定义启动这个脚本时如何解析配置文件
def get_args():
    parser = argparse.ArgumentParser()#创建一个命令行参数解析对象
    arg = parser.add_argument#起别名arg
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()

#定义一个封装训练过程的类
class Supervision_Train(pl.LightningModule):
    def __init__(self, config):#接受config，创建这个类时，会把配置对象传进来
        super().__init__()
        self.config = config#保存配置对象
        self.net = config.net#获取网络，就是在config中已经实例化的UNetformer对象

        self.loss = config.loss#获取损失函数

        self.metrics_train = Evaluator(num_class=config.num_classes)#创建训练指标统计器
        self.metrics_val = Evaluator(num_class=config.num_classes)#单独创建验证指标统计器，训练和验证不能用同一个指标对象

    # 8 × 3 × 768 × 768
    def forward(self, x):
        # only net is used in the prediction/inference
        seg_pre = self.net(x)
        return seg_pre
    #一个batch的训练步骤，batch_idx就是记录当前是第几个batch的
    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']#取出图像和标签
        prediction = self.net(img)#前向传播，当为训练阶段是，prediction的长度应该为2，有一个主预测和一个辅助预测
        loss = self.loss(prediction, mask)#计算损失
        #得到预测类别，假设此时输出图像尺寸：8*6*768*768
        if self.config.use_aux_loss:
            #把prediction[0]的第一个维度softmax一下，决定最后预测类别的肯定是主预测头的预测
            pre_mask = nn.Softmax(dim=1)(prediction[0])
        else:#pre_mask就是预测图的意思
            pre_mask = nn.Softmax(dim=1)(prediction)
        #对图像的第一个维度做取最大值操作，做完argmax操作后此时图像尺寸变为8*768*768
        pre_mask = pre_mask.argmax(dim=1)
        #更新评价指标
        for i in range(mask.shape[0]):#mask：(batch_size,H,W)
            #add_batch内部会统计哪些真实类别被预测成什么，然后更新混淆矩阵
            self.metrics_train.add_batch(
                mask[i].cpu().numpy(),#跟pre_mask[i]逐张比较
                pre_mask[i].cpu().numpy()
            )
        #把loss交给Lightning进行反向传播
        return {"loss": loss}
    # 训练epoch结束，核心任务：
    # 从累计混淆矩阵计算整轮训练指标
    # 打印指标
    # 清空指标状态
    # 将指标交给Lightning日志系统
    def on_train_epoch_end(self):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'potsdam' in self.config.log_name:#配置文件里有的配置
            #计算除最后一项类别外的所有类别的平均IOU和F1，nanmean表示忽略NaN后求平均
            #先计算每个类别的lOU，在对前五个类别求平均，下面的metrics_train.F1同理，[:-1]意思就是取 除最后一项外所有元素
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())
        #计算OA
        OA = np.nanmean(self.metrics_train.OA())
        #保存每类IoU
        iou_per_class = self.metrics_train.Intersection_over_Union()
        #创建整体指标字典
        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        print('train:', eval_value)
        #创建每类IoU字典
        iou_value = {}
        #zip会把第一个和第二个参数的对应位置组合起来形成一个二元组，这里的for循环是一个更加简洁的写法，可以把这个二元组的值按照位置直接赋给for循环后面的词了
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)
        #重置指标器，把混淆矩阵清零
        self.metrics_train.reset()
        log_dict = {'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}
        #记录训练指标
        self.log_dict(log_dict, prog_bar=True)
    #验证步骤，和训练步骤相似
    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.forward(img)
        pre_mask = nn.Softmax(dim=1)(prediction)
        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        loss_val = self.loss(prediction, mask)
        return {"loss_val": loss_val}
    #验证epoch结束，和训练epoch结束逻辑基本一致
    def on_validation_epoch_end(self):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())

        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()

        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        print('val:', eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)

        self.metrics_val.reset()
        log_dict = {'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]#使用列表是因为Lightning支持使用多个优化器和多个学习率调度器

    def train_dataloader(self):

        return self.config.train_loader

    def val_dataloader(self):

        return self.config.val_loader


# training，训练启动入口
def main():
    args = get_args()#获取命令行参数
    seed_everything(42)#加载种子这行应当在加载配置文件之前，可以试一下调换位置（已调换）
    config = py2cfg(args.config_path)#加载配置文件
    #Checkpoint不仅保存模型参数，还包含优化器状态，学习率调度状态，当前epoch等等
    #ModelCheckpoint是 PyTorch Lightning 提供的一个自动保存模型工具。
    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=config.weights_path,
                                          filename=config.weights_name)
    #训练过程中的各项指标，比如train_mIou，val_mIou等会写入CSV,后续可以用pandas或excel绘制训练曲线
    logger = CSVLogger('lightning_logs', name=config.log_name)
    #创建Lightning训练模块
    model = Supervision_Train(config) #model比纯UNetformer多了一层训练封装，真正做分割的是model.net
    #加载预训练checkpoint，注意，这里的预训练权重指的是整个模型，不是resnet的预训练权重
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)
    #创建训练器,配置训练设备、epoch数量、验证频率、checkpoint、日志等。
    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.max_epoch, accelerator='auto',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback], strategy='auto',
                         logger=logger)
    #正式开始训练
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)


if __name__ == "__main__":
   main()
#trainer.fit内部概念流程：
# 1. 调用model.configure_optimizers()
#    得到optimizer和scheduler
# 2. 调用model.train_dataloader()
#    得到train_loader
# 3. 调用model.val_dataloader()
#    得到val_loader
# 4. 把模型和batch移动到GPU
# 5. 开始epoch循环
# 6. 每个训练batch：
#    调用training_step()
#    获取loss
#    自动loss.backward()
#    自动optimizer.step()
#    自动optimizer.zero_grad()
# 7. epoch训练结束：
#    调用on_train_epoch_end()
# 8. 开始验证：
#    自动切换model.eval()
#    自动关闭梯度
#    每个验证batch调用validation_step()
# 9. 验证结束：
#    调用on_validation_epoch_end()
# 10. ModelCheckpoint检查val_F1
# 11. scheduler更新学习率
# 12. 进入下一epoch