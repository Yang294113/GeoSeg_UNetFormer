import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from geoseg.models.umamba_v12.ssm import Mamba_Block
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
import os


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),#自动计算保持合理尺寸变化的 padding，当stride=1时，保持输出尺寸=输入尺寸的padding大小
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            #groups=in_channels，深度可分离卷积
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            #注意这里的卷积核如果是奇数，那么可以达到输入输出尺寸一致，如果是偶数，则需要对输入高宽尺寸+1才能达到一致
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

#视觉Transformer里的FFN，等价于Transformer原始的nn.Linear()，因为输入没有flatten，所以就nn.Conv2d()
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MambaBlock(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        drop_path=0.,
        d_state=16,
    ):
        super().__init__()

        assert hidden_dim % 4 == 0

        self.self_attention = Mamba_Block(
            depth=1,
            embed_dim=hidden_dim // 4,
            out_dim=hidden_dim // 4,
            bimamba_type='None'
        )

        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0.
            else nn.Identity()
        )

    def forward(self, x):

        # 64 channel -> 4 × 16 channel
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        x1 = self.drop_path(self.self_attention(x1))
        x2 = self.drop_path(self.self_attention(x2))
        x3 = self.drop_path(self.self_attention(x3))
        x4 = self.drop_path(self.self_attention(x4))

        x = torch.cat(
            [x1, x2, x3, x4],
            dim=1
        )

        return x

class CoordinateAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden = max(8, channels // reduction)

        self.conv1 = nn.Conv2d(
            channels,
            hidden,
            kernel_size=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(
            hidden,
            channels,
            kernel_size=1,
            bias=True
        )

        self.conv_w = nn.Conv2d(
            hidden,
            channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):

        B, C, H, W = x.shape

        # 沿宽方向池化，保留高度位置信息
        x_h = x.mean(dim=3, keepdim=True)      # [B,C,H,1]

        # 沿高方向池化，保留宽度位置信息
        x_w = x.mean(dim=2, keepdim=True)      # [B,C,1,W]
        x_w = x_w.permute(0, 1, 3, 2)         # [B,C,W,1]

        # [B,C,H+W,1]
        y = torch.cat([x_h, x_w], dim=2)

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        y_h, y_w = torch.split(
            y,
            [H, W],
            dim=2
        )

        y_w = y_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))

        return x * a_h * a_w

class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,#输入通道
                 num_heads=16,#多头注意力数量
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads#计算每个头维度
        self.scale = head_dim ** -0.5#用head_dim的负二分之一次方缩放Q和K转置的乘积，防止注意力分数过大，softmax变成one-hot
        self.ws = window_size
        #QKV生成
        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)#输入dim，输出3dim，然后拆qkv
        #Local分支，注意用的是Conv+BN
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        #输出投影，这里kernel_size=window_size是为了增强窗口间的信息交互
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)
        #两个平均池化，一个沿高度方向，一个沿宽度方向
        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))
        #相对位置编码，attention需要记录token之间的相对位置
        self.relative_pos_embedding = relative_pos_embedding
        #相对位置告诉模型“两个token离多远”，相对位置偏置告诉模型“这个距离关系重要不重要”，
        #相对位置一般都可以直接算出来，而相对位置偏置需要学习，是可学习的参数
        if self.relative_pos_embedding:
            # define a parameter table of relative position bias，建立相对位置偏置表
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            #获取窗口内每个token对相对位置索引，就是index
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            #先生成坐标，以便后续计算相对位置
            #meshgrid会根据两个一维数据生成对应的二维坐标，stack会根据meshgrid生成的两个tensor矩阵在新的维度上合并起来
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww，第0个通道：每个像素的高度坐标，第1个通道：每个像素的宽度坐标
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww，把coords从第一维开始压平，让每个token都能找到坐标
            #开始计算坐标差值
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            #数组索引不能为复数，转换成从0开始
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            #二维坐标转一维index
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            #生成二维relative_position_index
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            #注册buffer，区别于Parameter，它不会更新，没有梯度，只是固定的坐标关系，但是是模型的一部分
            self.register_buffer("relative_position_index", relative_position_index)
            #使用标准差为 0.02 的截断正态分布随机初始化 relative_position_bias_table
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x
    #为了输出阶段尺寸和输入保持一致做的padding
    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        #局部卷积分支
        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)
        #拆分为QKV矩阵，各参数和论文图4(b)对应为：[B,3*h*C/h,H/w*w,W/w*w]->[3，B*H/w*W/w,h,w*w,C/h]（由此也可见论文的图中少了一个h）
        #解释:若batchsize=8，每个图片的窗口数为H/w*W/w，则一共的窗口数就为B*H/w*W/w，每个窗口单独attention，对于每个window的每个head，都需要计算形状为w*w×C/h的Q和C/h×w*w的K的转置的乘积
        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)#然后python再把rearrange传回的一个tensor沿第0维度分别赋值给q，k，v
        #Attention计算，@表示矩阵乘法，等价于torch.matmul(A, B)。transpose(-2, -1)就是交换最后两个维度
        #注意：矩阵乘法只作用于最后两个维度，所以这里也就是不同窗口信息不会产生交互的原因
        #batch维度=B,head维度=h，GPU会同时并行计算每个B下的h个窗口注意力
        dots = (q @ k.transpose(-2, -1)) * self.scale#注意：经过矩阵乘法后，dots形状为[batch_window, head,w*w,w*w]
        # 加上相对位置偏置，窗口8*8，那么self.relative_position_index.shape是[64,64]
        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)#把二维矩阵拉平成一维（为了一次性取出偏置表中元素）
            ].view(
                self.ws * self.ws,
                self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH（-1就是根据剩下的数自动算形状，这里就是nH），形状统一，拉平成一维并快速查找出每一个token pair对应的位置偏置后，再转化成跟q @ k.transpose(-2, -1)一样的形状
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            #为了广播加法，增加一个batch_window维度，和dots的形状保持一致
            #注意，每个batch_window是共享一个relative_position_bias的，relative_position_bias学习的是通用空间关系，例如“相邻像素通常更相关”
            dots += relative_position_bias.unsqueeze(0)
        #得到注意力权重，沿着每一行做softmax（每一行代表着一个query应该关注哪些key。）
        attn = dots.softmax(dim=-1)
        #融合信息，加权V
        attn = attn @ v
        #恢复图片结构，恢复成了[B，C,Hp,Wp]，每个head的信息也在此时融合了
        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)
        #去padding，从Hp,Wp恢复成H，W。B和C维度保持不变，取前H行和前W列
        attn = attn[:, :, :H, :W]
        #进行十字窗口上下文交互模块，其实就是分别在水平方向和垂直方向上做平均池化，然后相加
        #一个在底部反射填充一行，一个在右边反射填充一行，确保输入输出尺寸一致
        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))
        #全局分支和局部分支结果相加
        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out

class X4GuidedComplementaryMamba(nn.Module):
    def __init__(
        self,
        encoder_channels=(64, 128, 256, 512),
        out_channels=64,
        d_state=16
    ):
        super().__init__()

        c1, c2, c3, c4 = encoder_channels
        d = out_channels

        # =================================================
        # 1. Multi-scale projection
        #
        # 将四个Encoder层全部映射到统一的64维空间
        # =================================================

        self.proj1 = ConvBNReLU(
            c1, d,
            kernel_size=1
        )

        self.proj2 = ConvBNReLU(
            c2, d,
            kernel_size=1
        )

        self.proj3 = ConvBNReLU(
            c3, d,
            kernel_size=1
        )

        self.proj4 = ConvBNReLU(
            c4, d,
            kernel_size=1
        )

        # =================================================
        # 2. Unified relation modeling
        #
        # 输入：
        #
        # z1,z2,z3,z4,
        # |z1-z4|,
        # |z2-z4|,
        # |z3-z4|
        #
        # 一共 7d = 448 channels
        #
        # 448 -> 64
        # =================================================

        self.relation_reduce = ConvBNReLU(
            7 * d,
            d,
            kernel_size=1
        )

        # =================================================
        # 3. Coordinate Attention
        #
        # 只运行一次
        # =================================================

        self.ca = CoordinateAttention(
            d,
            reduction=16
        )

        # =================================================
        # 4. Spatial gate generator
        #
        # [B,64,H4,W4]
        # ->
        # [B,3,H4,W4]
        #
        # 三张独立空间权重图
        # =================================================

        self.gate = nn.Sequential(

            # 先进行轻量局部空间建模
            nn.Conv2d(
                d,
                d,
                kernel_size=3,
                padding=1,
                groups=d,
                bias=False
            ),

            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),

            # 输出三个尺度对应的Gate
            nn.Conv2d(
                d,
                3,
                kernel_size=1,
                bias=True
            )
        )

        # =================================================
        # 5. Complementary feature fusion
        #
        # z4,c1,c2,c3
        # -> 4d
        # -> d
        # =================================================

        self.fusion = nn.Sequential(

            ConvBNReLU(
                4 * d,
                d,
                kernel_size=1
            ),

            SeparableConvBNReLU(
                d,
                d,
                kernel_size=3
            )
        )

        # =================================================
        # 6. Mamba global modeling
        # =================================================

        self.mamba = MambaBlock(
            hidden_dim=d,
            drop_path=0.,
            d_state=d_state
        )

    def forward(
        self,
        res1,
        res2,
        res3,
        res4
    ):

        # res4的空间尺寸
        target_size = res4.shape[-2:]

        # =================================================
        # Step 1
        # 空间对齐 + 通道对齐
        # =================================================

        # res1: H/4 -> H/32
        z1 = F.interpolate(
            res1,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        z1 = self.proj1(z1)

        # res2: H/8 -> H/32
        z2 = F.interpolate(
            res2,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        z2 = self.proj2(z2)

        # res3: H/16 -> H/32
        z3 = F.interpolate(
            res3,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        z3 = self.proj3(z3)

        # res4本身只需要做通道映射
        z4 = self.proj4(res4)

        # 此时：
        #
        # z1,z2,z3,z4
        # 全部为：
        # [B,64,H/32,W/32]


        # =================================================
        # Step 2
        # X4-guided discrepancy modeling
        # =================================================

        d1 = torch.abs(z1 - z4)
        d2 = torch.abs(z2 - z4)
        d3 = torch.abs(z3 - z4)

        # =================================================
        # Step 3
        # Unified relation feature
        #
        # 7 × 64 = 448 channels
        # =================================================

        relation = torch.cat(
            [
                z1,
                z2,
                z3,
                z4,
                d1,
                d2,
                d3
            ],
            dim=1
        )

        # 448 -> 64
        #
        # 将不同来源的信息融合成统一关系描述
        relation = self.relation_reduce(
            relation
        )

        # =================================================
        # Step 4
        # Coordinate-aware relation enhancement
        # =================================================

        relation = self.ca(
            relation
        )

        # =================================================
        # Step 5
        # Generate spatial gates
        # =================================================

        gate = self.gate(
            relation
        )

        # 三个尺度相互不竞争
        # 所以使用 Sigmoid 而不是 Softmax
        gate = torch.sigmoid(
            gate
        )

        # 每一个都是：
        # [B,1,H4,W4]
        g1 = gate[:, 0:1, :, :]
        g2 = gate[:, 1:2, :, :]
        g3 = gate[:, 2:3, :, :]

        # =================================================
        # Step 6
        # Complementary information selection
        #
        # Gate自动在channel维广播
        # =================================================

        c1 = g1 * z1
        c2 = g2 * z2
        c3 = g3 * z3

        # =================================================
        # Step 7
        # Multi-scale fusion
        # =================================================

        fusion = torch.cat(
            [
                z4,
                c1,
                c2,
                c3
            ],
            dim=1
        )

        # 256 -> 64
        fusion = self.fusion(
            fusion
        )

        # =================================================
        # Step 8
        # Mamba global modeling
        # =================================================

        global_feature = self.mamba(
            fusion
        )

        # =================================================
        # Step 9
        # X4 residual
        # =================================================

        out = global_feature + z4

        # [B,64,H/32,W/32]
        return out

#Transformer Block，对融合后的feature进行上下文增强
class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):#DropPath随机深度，随机丢弃某个残差分支
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)
        #若drop_path=0.1，就是训练时有10％概率丢弃Attention分支，这里是0.，执行nn.Identity()恒等映射
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #构建MLP，dim * mlp_ratio是mlp隐藏层数量，升维学习更多组合
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        #两个残差，计算完注意力后进行一次，做完mlp后进行一次，通过这两行也能看出整个Block的结构图
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class WF(nn.Module):
    #eps就是epsilon（极小正数），用于防止数值计算不稳定。
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        #可学习融合权重，初始【1，1】
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        #上采样
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        #创建nn.ReLU()，再用(self.weights)去调用它，保证权重大于零
        weights = nn.ReLU()(self.weights)
        #权重归一化，让两个权重都在0-1之间，weight是两个数，利用广播机制对每个数做归一化
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        #融合，融合前要对res做1*1conv保证通道数一致
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        #融合后重新提取特征
        x = self.post_conv(x)
        return x


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
        #注意这里不是普通卷积，这里设置了groups = decode_channels，groups是通道分组的意思，加入groups=G，则输入通道被平均分成 G 组，输出通道也被平均分成 G 组，每组完全隔绝，互不通信
        #猜想：如果将这里的groups设置为别的值，也就是不是单纯的深度卷积，是否具有创新意义？
        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),#全局平均池化
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        #轻量卷积，也就是深度卷积+1*1conv
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        #先对融合后的辅助特征进行一次特征加工
        feat = self.conv(x)
        #防止过拟合
        feat = self.drop(feat)
        #最后分类
        feat = self.conv_out(feat)
        #分类好的图进行上采样，恢复成原图尺寸
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),#encoder每一层输出特征图的通道数
                 decode_channels=64,#decoder每层保持64通道数
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()
        #用作压缩res4通道，encoder_channels[-1]为512
        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        # E1: AMMB is an additive residual. Zero initialization keeps the
        # initial X4 path equivalent to the baseline decoder.
        self.ammb_residual_scale = nn.Parameter(torch.zeros(1))
        #b4,b3,b2都是自深向浅层attention，p3-p1都分别用作融合res3-res1
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)
        #训练时，初始化这些变量用于辅助头
        if self.training:
            #分别用于4倍上采样和2倍上采样
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
            self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.aux_head = AuxHead(decode_channels, num_classes)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def fuse_x4(self, res4, res4_ammb=None):
        x4 = self.pre_conv(res4)
        if res4_ammb is not None:
            x4 = x4 + self.ammb_residual_scale * res4_ammb
        return x4

    def forward(self, res1, res2, res3, res4, h, w, res4_ammb=None):
        x4 = self.fuse_x4(res4, res4_ammb)
        if self.training:
            x = self.b4(x4)
            h4 = self.up4(x)#不是用于主干融合，而是辅助头
            #加权融合x，res3
            x = self.p3(x, res3)
            x = self.b3(x)
            h3 = self.up3(x)
            #加权融合x，res2
            x = self.p2(x, res2)
            x = self.b2(x)
            h2 = x
            #送入特征精炼头，先特征加权融合，后精炼
            x = self.p1(x, res1)
            #结果送入分割头进行最终预测
            x = self.segmentation_head(x)
            #生成逐像素分类结果后上采样，恢复到原图尺寸
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            #辅助分支开始融合特征
            ah = h4 + h3 + h2
            #生成辅助预测
            ah = self.aux_head(ah, h, w)
            #返回主预测和辅助预测
            return x, ah
        else:#验证或测试模式没有辅助头
            x = self.b4(x4)
            x = self.p3(x, res3)
            x = self.b3(x)

            x = self.p2(x, res2)
            x = self.b2(x)

            x = self.p1(x, res1)

            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            return x
    #权重初始化，实际上当前什么也没做
    def init_weight(self):
        #self.children()会遍历当前类所有第一层的子模块
        for m in self.children():
            #对当前所有子模块中的nn.Conv2d进行权重初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class UNetFormer(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name='swsl_resnet18',#指定使用骨干网络的名字
                 pretrained=True,
                 window_size=8,
                 num_classes=6,
                 pretrained_path=None#这里是为了本地加载resnet18的预训练权重自己加的
                 ):
        super().__init__()
        #初始化backbone，features_only=True表示只要中间特征，output_stride=32表示最终的feature相对于输入缩小32倍,out_indices=(1, 2, 3, 4)决定返回哪些层
        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=False)
        
        # Load local pretrained weights from file if provided
        if pretrained_path is not None and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # FeatureListNet wraps the actual backbone in .model; try both
            try:
                self.backbone.model.load_state_dict(state_dict, strict=False)
                print(f'[UNetFormer] Loaded pretrained backbone from: {pretrained_path}')
            except Exception:
                try:
                    self.backbone.load_state_dict(state_dict, strict=False)
                    print(f'[UNetFormer] Loaded pretrained backbone from: {pretrained_path}')
                except Exception as e:
                    print(f'[UNetFormer] Warning: could not load pretrained weights: {e}')
        elif pretrained:
            # Original behavior: let timm download weights
            self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                              out_indices=(1, 2, 3, 4), pretrained=True)
        else:
            print('[UNetFormer] No pretrained weights loaded, training from scratch.')

        encoder_channels = self.backbone.feature_info.channels()

        # =============================================
        # X4-guided complementary multi-scale module
        # =============================================
        self.x4_enhance = X4GuidedComplementaryMamba(
            encoder_channels=encoder_channels,
            out_channels=decode_channels,
            d_state=16
        )

        self.decoder = Decoder(
            encoder_channels,
            decode_channels,
            dropout,
            window_size,
            num_classes
        )

    def forward(self, x):
        #记录原图尺寸，便于最后预测完恢复，（提取张量 x 形状（Size）中的最后两个维度赋值给变量 h 和 w）
        h, w = x.size()[-2:]#-2代表倒数第二个位置（起始索引），: 代表一直取到末尾（结束索引）。
        #提取Backbone特征
        res1, res2, res3, res4 = self.backbone(x)

        # =============================================
        # Encoder -> Decoder之前
        # 对X4进行多尺度互补增强
        # =============================================
        res4_ammb = self.x4_enhance(
            res1,
            res2,
            res3,
            res4
        )

        if self.training:
            x, ah = self.decoder(
                res1,
                res2,
                res3,
                res4,
                h,
                w,
                res4_ammb=res4_ammb
            )
            return x, ah

        else:
            x = self.decoder(
                res1,
                res2,
                res3,
                res4,
                h,
                w,
                res4_ammb=res4_ammb
            )
            return x
