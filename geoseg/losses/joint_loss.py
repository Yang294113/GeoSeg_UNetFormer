from torch import nn
from torch.nn.modules.loss import _Loss#pytorch所有损失函数常用的基础父类之一
#只导出"JointLoss", "WeightedLoss"，和训练没关系
__all__ = ["JointLoss", "WeightedLoss"]
#整个文件比较简单，本质上整个文件都可以缩写成
# class JointLoss:
#     def forward(logits, labels):
#         return (
#             first_weight * first_loss(logits, labels)
#             +
#             second_weight * second_loss(logits, labels)
#         )

#给某个loss乘一个固定权重，为了给主损失和辅助损失配权重用的
class WeightedLoss(_Loss):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight
    #*input是一个可以接受任意数量的位置参数
    def forward(self, *input):
        return self.loss(*input) * self.weight

#把两个损失函数组合成一个新的损失函数。
class JointLoss(_Loss):
    """
    Wrap two loss functions into one. This class computes a weighted sum of two losses.
    """
    #接收第一个loss跟第二个loss，第一个权重和第二个权重。提示使用者：first这个参数应该传入一个nn.Module对象
    def __init__(self, first: nn.Module, second: nn.Module, first_weight=1.0, second_weight=1.0):
        super().__init__()
        #就是套上了一层权重
        self.first = WeightedLoss(first, first_weight)
        self.second = WeightedLoss(second, second_weight)
    #假设调用loss(logits,labels)，则self.first(logits,labels)+self.second(logits,labels)
    def forward(self, *input):
        return self.first(*input) + self.second(*input)
