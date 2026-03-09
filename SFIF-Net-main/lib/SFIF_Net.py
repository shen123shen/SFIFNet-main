# coding:utf-8
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.InceptionNext import inceptionnext_tiny
from lib.swin import  swin_L
up_kwargs = {'mode': 'bilinear', 'align_corners': False}

from torchsummary import summary
from pytorch_wavelets import DWTForward

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    return param_size


class SFIF_Net(nn.Module):
    def __init__(self, out_planes=1, n_filts=96, encoder='inceptionnext_tiny',HFF_dp=0.):
        super(SFIF_Net, self).__init__()
        self.encoder = encoder
        if self.encoder == 'swin_L':
            mutil_channel = [192, 384, 768, 1536]
            self.backbone = swin_L()
        elif self.encoder == 'inceptionnext_tiny':
            mutil_channel = [96, 192, 384, 768]
            self.backbone = inceptionnext_tiny()
        # elif self.encoder == 'pvt':
        #     mutil_channel = [64, 128, 320, 512]
        #     self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        #     path = '/kaggle/input/pretrained/swin_base_patch4_window12_384_22k.pth'
        #     save_model = torch.load(path)
        #     model_dict = self.backbone.state_dict()
        #     state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        #     model_dict.update(state_dict)
        #     self.backbone.load_state_dict(model_dict)
        self.dropout = torch.nn.Dropout(0.3)  # 添加 Dropout
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.dmc1 = DMC1([mutil_channel[2], mutil_channel[3]], width=mutil_channel[2], up_kwargs=up_kwargs)
        # self.dmc2 = DMC2([mutil_channel[1], mutil_channel[2], mutil_channel[3]], width=mutil_channel[1],
        #                  up_kwargs=up_kwargs)
        # self.dmc3 = DMC3([mutil_channel[0], mutil_channel[1], mutil_channel[2], mutil_channel[3]],
        #                  width=mutil_channel[0], up_kwargs=up_kwargs)
        # self.DMC_fusion = DMC_fusion(mutil_channel, up_kwargs=up_kwargs)

        self.mlfc1 = LWFA(mutil_channel[0], mutil_channel[1], mutil_channel[2], mutil_channel[3], lenn=1)
        self.mlfc2 = LWFA(mutil_channel[0], mutil_channel[1], mutil_channel[2], mutil_channel[3], lenn=1)
        self.mlfc3 = LWFA(mutil_channel[0], mutil_channel[1], mutil_channel[2], mutil_channel[3], lenn=1)


        self.decoder4 = BasicConv2d(mutil_channel[3], mutil_channel[2], 3, padding=1)
        self.decoder3 = BasicConv2d(mutil_channel[2], mutil_channel[1], 3, padding=1)
        self.decoder2 = BasicConv2d(mutil_channel[1], mutil_channel[0], 3, padding=1)
        self.decoder1 = nn.Sequential(nn.Conv2d(mutil_channel[0], 64, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(64, out_planes, kernel_size=1, stride=1))

        self.fu1 = IFF(96, 192,  96)
        self.fu2 = IFF(192, 384, 192)
        self.fu3 = IFF(384, 768,  384)
    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        '''
        x1 : (2,128,56,56)
        x2 : (2,256,28,28)
        x3 : (2,512,14,14)
        x4 : (2,1024,7,7)
        '''
        # # x1 = self.dmc3(x1, x2, x3, x4)  #乘法融合
        # # x2 = self.dmc2(x2, x3, x4)
        # # x3 = self.dmc1(x3, x4)
        # # x4 = x4
        # x1, x2, x3, x4 = self.DMC_fusion(x1, x2, x3, x4)  # 整理为一个模块
        x1, x2, x3, x4 = self.mlfc1(x1, x2, x3, x4)
        # x1, x2, x3, x4 = self.mlfc2(x1, x2, x3, x4)
        # x1, x2, x3, x4 = self.mlfc3(x1, x2, x3, x4)

        x_f_3 = self.fu3(x3, x4)
        x_f_2 = self.fu2(x2, x_f_3)
        x_f_1 = self.fu1(x1, x_f_2)

        d1 = self.decoder1(x_f_1)
        d1 = self.dropout(d1)  # 在解码器阶段应用 Dropout
        d1 = F.interpolate(d1, scale_factor=4, mode='bilinear')  # (1,1,224,224)
        return d1



def channel_shuffle(x, groups): # groups:表示将输入通道分成多少组进行洗牌。
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)  # 为了将通道分组，每组包含 channels_per_group 个通道。
    # 将张量的 groups 和 channels_per_group 维度交换，这是洗牌操作的关键步骤。这样做的目的是将不同组的通道混合在一起。
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # (1,768,14,14)
        x = self.conv(x) # (1,384,14,14)
        x = self.bn(x)
        return x


class Conv2d_batchnorm(torch.nn.Module):
    """
    2D Convolutional layers
    """

    def __init__(
            self,
            num_in_filters,  # 输入特征图的通道数
            num_out_filters,  # 输出特征图的通道数，即卷积层中卷积核的数量。
            kernel_size,
            groups=8,
            stride=(1, 1),  # 表示卷积核每次移动一个像素。
            activation="LeakyReLU",
    ):
        """
        Initialization

        Args:
            num_in_filters {int} -- number of input filters
            num_out_filters {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {'LeakyReLU'})
        """
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            groups=groups
            # 使用 "same" 作为 padding 参数的值，意味着自动计算并应用适当的填充量，以保持输出特征图的空间尺寸与输入特征图相同。
        )
        self.num_in_filters = num_in_filters
        self.num_out_filters = num_out_filters
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        self.sqe = ChannelSELayer(num_out_filters)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):  # (2,1792,56,56)  1920->1792
        x = channel_shuffle(x, gcd(self.num_in_filters, self.num_out_filters))
        x = self.conv1(x)  # (2,128,56,56)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.sqe(x)


class Conv2d_channel(torch.nn.Module):
    """
    2D pointwise Convolutional layers
    """

    def __init__(self, num_in_filters, num_out_filters):
        """
        Initialization

        Args:
            num_in_filters {int} -- number of input filters
            num_out_filters {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {'LeakyReLU'})
        """
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=(1, 1),
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        self.sqe = ChannelSELayer(num_out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        return self.sqe(self.activation(x))


# 实现了通道注意力（简称 SE）机制，SE块的实现可以帮助网络集中注意力在最有信息量的特征上，抑制不太有用的特征，从而提高网络的性能和泛化能力。
# 通常放在 卷积--归一化--激活之后
class ChannelSELayer(torch.nn.Module):
    """
    Implements Squeeze and Excitation
    """

    def __init__(self, num_channels):
        """
        Initialization

        Args:
            num_channels (int): No of input channels
        """

        super(ChannelSELayer, self).__init__()
        # 自适应平均池化操作,表示 1*1 大小的输出  -
        # 使用`AdaptiveAvgPool2d`进行全局平均池化，以获取每个通道的全局空间信息。
        self.gp_avg_pool = torch.nn.AdaptiveAvgPool2d(1)  # 全局平均池化层，用于将每个通道的空间信息压缩成一个单一的全局特征。

        self.reduction_ratio = 8  # default reduction ratio

        num_channels_reduced = num_channels // self.reduction_ratio
        # 两个全连接层，用于实现降维和升维操作。第一个全连接层将通道数减少到
        # num_channels // reduction_ratio，第二个全连接层将其恢复到原始通道数。
        self.fc1 = torch.nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = torch.nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.act = torch.nn.LeakyReLU()

        self.sigmoid = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm2d(num_channels)

    def forward(self, inp):  # inp输入张量 (2,128,56,56)

        batch_size, num_channels, H, W = inp.size()
        # 将池化后得到的特征图通过view函数。将其变为形状为(batch_size, num_channels)batch_size表示输入张量的批次大小，
        # num_channels表示卷积后的特征通道数，最后再经过激活函数
        # 通过两个全连接层（`Linear`）和`LeakyReLU`激活函数来学习通道之间的关系。
        out = self.act(self.fc1(self.gp_avg_pool(inp).view(batch_size, num_channels)))
        out = self.fc2(out)
        # 使用`Sigmoid`函数生成权重，并通过逐元素乘法将这些权重应用于输入特征图。
        out = self.sigmoid(out)
        # 将输入特征图 inp 与这些权重进行逐元素乘法，以调整每个通道的贡献。
        out = torch.mul(inp, out.view(batch_size, num_channels, 1, 1))
        # 最后，使用批量归一化（`BatchNorm2d`）和`LeakyReLU`激活函数输出最终的特征图。
        out = self.bn(out)
        out = self.act(out)

        return out


def build_act_layer(act_type):
    """Build activation layer."""
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()


def build_norm_layer(norm_type, embed_dims):
    """Build normalization layer."""
    assert norm_type in ['BN', 'GN', 'LN2d', 'SyncBN']
    if norm_type == 'GN':
        return nn.GroupNorm(embed_dims, embed_dims, eps=1e-5)
    if norm_type == 'LN2d':
        return LayerNorm2d(embed_dims, eps=1e-6)
    if norm_type == 'SyncBN':
        return nn.SyncBatchNorm(embed_dims, eps=1e-5)
    else:
        return nn.BatchNorm2d(embed_dims, eps=1e-5)


class LayerNorm2d(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        assert self.data_format in ["channels_last", "channels_first"]
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        act_type (str): The type of activation. Defaults to 'GELU'.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 kernel_size=3,
                 act_type='GELU',
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = build_act_layer(act_type)

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiOrderDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel.

    Args:
        embed_dims (int): Number of input channels.
        dw_dilation (list): Dilations of three DWConv layers.
        channel_split (list): The raletive ratio of three splited channels.
    """

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3, ],
                 channel_split=[1, 3, 4, ],
                 ):
        super(MultiOrderDWConv, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1, dilation=dw_dilation[0],
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1, dilation=dw_dilation[1],
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1, dilation=dw_dilation[2],
        )
        # a channel convolution
        self.PW_conv = nn.Conv2d(  # point-wise convolution
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)
        self.conv0 = nn.Conv2d(7 * embed_dims // 8, 7 * embed_dims // 8, 5, padding=2, groups=7 * embed_dims // 8)
        self.conv_spatial = nn.Conv2d(7 * embed_dims // 8, 7 * embed_dims // 8, 7, stride=1, padding=9, groups=7 * embed_dims // 8,
                                      dilation=3)
        self.conv1 = nn.Conv2d(7 * embed_dims // 8, 3 * embed_dims // 8, 1)
        self.conv2 = nn.Conv2d(7 * embed_dims // 8, embed_dims // 2, 1)

        self.split_indexes = (embed_dims // 8, 7 * embed_dims // 8)
        self.branch1 = nn.Sequential()
        self.conv1x1 = nn.Sequential(
            # 1*1卷积通道数不变
            nn.Conv2d(in_channels=2 * embed_dims // 3, out_channels=2 * embed_dims// 3, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2 * embed_dims // 3),  # 对输出的每个通道做BN
            nn.ReLU(inplace=True))
        self.norm = nn.BatchNorm2d(embed_dims)

    def forward(self, x):
        x_0 = self.DW_conv0(x) # (1,384,14,14)
        x_id, x_k = torch.split(x_0, self.split_indexes, dim=1)  # (1,48,14,14), (1,336,14,14)
        attn1 = self.conv0(x_k)   # (1,336,14,14)
        attn2 = self.conv_spatial(attn1) # (1,336,14,14)
        attn1 = self.conv1(attn1)   # (1,144,14,14)
        attn2 = self.conv2(attn2)   # (1,192,14,14)
        x_id = self.branch1(x_id)
        x = torch.cat((x_id, attn1, attn2), dim=1)
        x = self.PW_conv(x)  # (1,384,14,14)
        return x

class SupervisedAttentionModule(nn.Module):
    def __init__(self, mid_d):
        super(SupervisedAttentionModule, self).__init__()
        self.mid_d = mid_d
        # fusion
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)
        self.conv_context = nn.Sequential(
            nn.Conv2d(2, self.mid_d, kernel_size=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        mask = self.cls(x)
        mask_f = torch.sigmoid(mask)
        mask_b = 1 - mask_f
        context = torch.cat([mask_f, mask_b], dim=1)
        context = self.conv_context(context)
        return context

class MFE(nn.Module):
    """Spatial Block with Multi-order Gated Aggregation.

    Args:
        embed_dims (int): Number of input channels.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for Spatial Block.
            Defaults to 'SiLU'.
    """

    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False,
                 ):
        super(MFE, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )
        self.proj_2 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = build_act_layer(attn_act_type)
        self.act_gate = build_act_layer(attn_act_type)

        # decompose
        self.sigma = ElementScale(
            embed_dims, init_value=1e-5, requires_grad=True)

        self.cls = nn.Conv2d(embed_dims, 1, kernel_size=1)
        self.conv_context = nn.Sequential(
            nn.Conv2d(2, embed_dims, kernel_size=1),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True)
        )
    def feat_decompose(self, x):
        mask = self.cls(x)
        mask_f = torch.sigmoid(mask)
        mask_b = 1 - mask_f
        context = torch.cat([mask_f, mask_b], dim=1)
        x = self.conv_context(context)
        # x = self.proj_1(x)    # (1,384,14,14)
        # # x_d: [B, C, H, W] -> [B, C, 1, 1]
        # x_d = F.adaptive_avg_pool2d(x, output_size=1)   # (1,384,1,1)
        # x = x + self.sigma(x - x_d)  # (1,384,1,1)
        # x = self.act_value(x)
        return x

    def forward_gating(self, g, v):
        with torch.autocast(device_type='cuda', enabled=False):
            g = g.to(torch.float32)
            v = v.to(torch.float32)
            return self.proj_2(self.act_gate(g) * self.act_gate(v))

    def forward(self, x):
        shortcut = x.clone()   # (1,384,14,14)
        # proj 1x1
        x = self.feat_decompose(x)
        # gating and value branch
        g = self.gate(x) # (1,384,14,14)
        v = self.value(x)
        # aggregation
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        x = x + shortcut  # (1,384,14,14)
        return x

class SFM(nn.Module):
    def __init__(self, in_ch, out_ch,num_heads=8, window_size=8):
        super(SFM, self).__init__()
        # self.wt：一个 DWTForward 实例，用于执行小波变换，将输入特征分解为低频和高频部分。
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*3, in_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_glb = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_local = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # 添加用于调整低频和高频特征的权重系数的卷积层
        self.weight_conv_L = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1)
        self.weight_conv_H = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()  # 使用Sigmoid函数将权重系数限制在0和1之间
    def forward(self, x,imagename=None):
        _, _, h, w = x.shape
        # 使用 self.wt 对输入特征 x 进行小波变换，得到低频部分 yL 和高频部分 yH。
        yL, yH = self.wt(x)
        # 从 yH 中提取三个方向的高频特征（水平、垂直和对角线）。
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        # 将这三个方向的高频特征拼接起来，并通过 self.conv_bn_relu 进行处理。
        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yH = self.conv_bn_relu(yH)   # (1,384,7,7)
        # 分别对低频特征 yL 和处理后的高频特征 yH 应用输出序列，得到最终的低频和高频输出。
        yL = self.outconv_bn_relu_L(yL)   # (1,384,7,7)
        yH = self.outconv_bn_relu_H(yH)   # (1,384,7,7)

        yL_up = F.interpolate(yL, size=(h, w), mode='bilinear', align_corners=True)   # (1,384,14,14)
        yH_up = F.interpolate(yH, size=(h, w), mode='bilinear', align_corners=True)

        output = x + 0.4 * yL_up + 0.6 * yH_up  # 在原始特征基础上增强边界
        return output   # (1,384,7,7)



class IFF(nn.Module):
    def __init__(self, channels, g_dim, f=2):
        super(IFF,self).__init__()
        self.f = f  # Number of layers in the shared MLP
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Shared MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // f),
            nn.ReLU(inplace=True),
            nn.Linear(channels // f, channels),
            nn.Sigmoid()
        )
        # Convolutional layers for feature fusion
        self.conv1 = nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1x1 = BasicConv2d(g_dim, channels, 1, padding=0)
        self.SFM = SFM(channels, channels)
        self.extral = MFE(channels)
        self.final_conv = Conv2d_batchnorm(channels*2, channels, 1)
        self.GAP = nn.AdaptiveAvgPool2d(1)
    def forward(self, F_l, F_h):
        # F_l: low-level features from encoder
        # F_h: high-level features from decoder
        F_h = self.extral(self.conv1x1(self.upsample(F_h)))   # (1,384,14,14)

        # Global average pooling
        F_l_pool = self.GAP(F_l).view(F_l.size(0), F_l.size(1))   # (1,384)
        F_h_pool = self.GAP(F_h).view(F_h.size(0), F_h.size(1))  # (1,384)

        # Shared MLP for contextual correlation
        l = self.mlp(F_l_pool)    # (1,384)
        h = self.mlp(F_h_pool)   # (1,384)

        # Re-weighting features
        F_l_weighted = F_l * l.unsqueeze(2).unsqueeze(3)   # (1,384,14,14)
        F_h_weighted = F_h * h.unsqueeze(2).unsqueeze(3)   # (1,384,14,14)

        F_fused = F_l_weighted + F_h_weighted
        F_fused = self.SFM(F_fused)

        # Residual connection
        F_out = self.final_conv(torch.cat([F_fused, F_h], dim=1))

        return F_out


class FeatureFusionModule(nn.Module):
    def __init__(self, fuse_d, id_d, out_d):
        super(FeatureFusionModule, self).__init__()
        self.fuse_d = fuse_d
        self.id_d = id_d
        self.out_d = out_d
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d)
        )
        self.conv_identity = nn.Conv2d(self.id_d, self.out_d, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, c_fuse, c):
        c_fuse = self.conv_fuse(c_fuse)
        c_out = self.relu(c_fuse + self.conv_identity(c))

        return c_out

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x



class HFA(nn.Module):
    def __init__(self, in_channels, width=128, up_kwargs=None):
        super(HFA, self).__init__()

        self.MFE_x4 = MFE(in_channels[-1])
        self.MFE_x3 = MFE(in_channels[-2])
        self.MFE_x2 = MFE(in_channels[-3])
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.sqe = ChannelSELayer(width)
        self.global_enhancer = nn.Sequential(
            nn.Conv2d(in_channels[-1], in_channels[-1], kernel_size=1),
            nn.BatchNorm2d(in_channels[-1]),
            nn.ReLU(inplace=True),
        )
        self.conv_x4 = Conv2d_batchnorm(in_channels[-1], in_channels[-2], 1)
        self.conv_x3 = Conv2d_batchnorm(in_channels[-2], in_channels[-3], 1)
        self.conv_out = Conv2d_batchnorm(in_channels[-3]+in_channels[-2]+in_channels[-1], width, 1)
    def forward(self, x2, x3, x4):
        # 上采样操作，使所有特征图具有相同的空间维度
        x4_1 = self.MFE_x4(x4)
        yL, _ = self.wt(x4_1)
        global_enhanced = self.global_enhancer(yL)

        # 上采样并融合
        _, _, h, w = x4_1.shape
        global_up = F.interpolate(global_enhanced, size=(h, w), mode='bilinear')
        # 将全局信息与局部特征融合
        x4_2 = x4_1 + global_up  # 残差连接

        x4_up = self.up(self.up(x4_2))  # 残差连接
        x3_1 = x3 + self.up(self.conv_x4(x4_2))
        x3_1 = self.MFE_x3(x3_1)
        x3_up = self.up(x3_1)

        x2_1 = x2 + self.up(self.conv_x3(x3_1))
        x2_1 = self.MFE_x2(x2_1)
        output = self.conv_out(torch.cat([x2_1, x3_up, x4_up], dim=1))
        return output

class LWFA(torch.nn.Module):
    """
    Implements Multi Level Feature Compilation

    """

    # lenn: 表示重复上述操作的次数，默认为1。
    def __init__(self, in_filters1, in_filters2, in_filters3, in_filters4, width=128, lenn=1):
        """
        Initialization

        Args:
            in_filters1 (int): number of channels in the first level
            in_filters2 (int): number of channels in the second level
            in_filters3 (int): number of channels in the third level
            in_filters4 (int): number of channels in the fourth level
            lenn (int, optional): number of repeats. Defaults to 1.
        """

        super().__init__()

        self.in_filters1 = in_filters1
        self.in_filters2 = in_filters2
        self.in_filters3 = in_filters3
        self.in_filters4 = in_filters4
        self.in_filters_1 = in_filters1 + width
        self.in_filters_2 = in_filters2 + width
        self.in_filters_3 = in_filters3 + width
        self.in_filters_4 = in_filters4 + width
        # 一个上采样层，使用双线性插值进行上采样。
        self.no_param_up = torch.nn.Upsample(scale_factor=2)  # used for upsampling
        # 一个平均池化层，用于下采样。
        self.no_param_down = torch.nn.AvgPool2d(2)  # used for downsampling

        '''
        四个 ModuleList，分别用于存储每个层级的卷积层 (cnv_blks1, cnv_blks2, cnv_blks3, cnv_blks4)、
        合并卷积层 (cnv_mrg1, cnv_mrg2, cnv_mrg3, cnv_mrg4)、
        批量归一化层 (bns1, bns2, bns3, bns4) 和
        合并操作后的批量归一化层 (bns_mrg1, bns_mrg2, bns_mrg3, bns_mrg4)。
        '''

        self.cnv_blks1 = torch.nn.ModuleList([])
        self.cnv_blks2 = torch.nn.ModuleList([])
        self.cnv_blks3 = torch.nn.ModuleList([])
        self.cnv_blks4 = torch.nn.ModuleList([])

        self.cnv_mrg1 = torch.nn.ModuleList([])
        self.cnv_mrg2 = torch.nn.ModuleList([])
        self.cnv_mrg3 = torch.nn.ModuleList([])
        self.cnv_mrg4 = torch.nn.ModuleList([])

        self.bns1 = torch.nn.ModuleList([])
        self.bns2 = torch.nn.ModuleList([])
        self.bns3 = torch.nn.ModuleList([])
        self.bns4 = torch.nn.ModuleList([])

        self.bns_mrg1 = torch.nn.ModuleList([])
        self.bns_mrg2 = torch.nn.ModuleList([])
        self.bns_mrg3 = torch.nn.ModuleList([])
        self.bns_mrg4 = torch.nn.ModuleList([])

        for i in range(lenn):
            # 存储每个层级的卷积层
            self.cnv_blks1.append(
                # 卷积-标准化-激活-注意力
                Conv2d_batchnorm(self.in_filters_1, in_filters1, (1, 1))
            )
            # 合并卷积层
            self.cnv_mrg1.append(Conv2d_batchnorm(in_filters1, in_filters1, (1, 1)))
            # 批量归一化层
            self.bns1.append(torch.nn.BatchNorm2d(in_filters1))
            # 合并操作后的批量归一化
            self.bns_mrg1.append(torch.nn.BatchNorm2d(in_filters1))

            self.cnv_blks2.append(
                Conv2d_batchnorm(self.in_filters2+self.in_filters1+width, in_filters2, (1, 1))
            )
            self.cnv_mrg2.append(Conv2d_batchnorm(2 * in_filters2, in_filters2, (1, 1)))
            self.bns2.append(torch.nn.BatchNorm2d(in_filters2))
            self.bns_mrg2.append(torch.nn.BatchNorm2d(in_filters2))

            self.cnv_blks3.append(
                Conv2d_batchnorm(self.in_filters3+self.in_filters2+width, in_filters3, (1, 1))
            )
            self.cnv_mrg3.append(Conv2d_batchnorm(2 * in_filters3, in_filters3, (1, 1)))
            self.bns3.append(torch.nn.BatchNorm2d(in_filters3))
            self.bns_mrg3.append(torch.nn.BatchNorm2d(in_filters3))

            self.cnv_blks4.append(
                Conv2d_batchnorm(self.in_filters4+self.in_filters3+width, in_filters4, (1, 1))
            )

            self.bns4.append(torch.nn.BatchNorm2d(in_filters4))
            self.bns_mrg4.append(torch.nn.BatchNorm2d(in_filters4))

        self.act = torch.nn.LeakyReLU()

        self.sqe1 = ChannelSELayer(in_filters1)
        self.sqe2 = ChannelSELayer(in_filters2)
        self.sqe3 = ChannelSELayer(in_filters3)
        self.sqe4 = ChannelSELayer(in_filters4)
        self.dropout = nn.Dropout(0.3)
        self.fuse = HFA([in_filters2, in_filters3, in_filters4], width, up_kwargs=up_kwargs)
    def forward(self, x1, x2, x3, x4):

        batch_size, _, h1, w1 = x1.shape
        _, _, h2, w2 = x2.shape
        _, _, h3, w3 = x3.shape
        _, _, h4, w4 = x4.shape
        fuse = self.fuse(x2, x3, x4)
        fuse = self.dropout(fuse)  # 在特征融合后应用 Dropout
        '''
        图（D）是第三层级的MLFC3,可知在像素上，x1=2*x2=4*x3=8*x4,调整不同层级特征图的大小后进行拼接
        再进行逐点卷积-->再与X3(MLFC3为第三层)拼接，再进行逐点卷积
        '''

        for i in range(len(self.cnv_blks1)):
            x_c1 = self.act(
                self.bns1[i](
                    self.cnv_blks1[i](
                        torch.cat(
                            [
                                x1,
                                self.no_param_up(fuse)
                            ],
                            dim=1,  # 通道  (2,96+64,56,56)
                        )
                    )
                )
            ) # x_c1 (2,96,56,56)
            x_c2 = self.act(
                self.bns2[i](
                    self.cnv_blks2[i](
                        torch.cat(
                            [
                                fuse,
                                (self.no_param_down(x_c1)),  # (2,96,28,28)
                                x2,

                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c3 = self.act(
                self.bns3[i](
                    self.cnv_blks3[i](
                        torch.cat(
                            [
                                x3,
                                self.no_param_down(fuse),
                                (self.no_param_down(x_c2)),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c4 = self.act(
                self.bns4[i](
                    self.cnv_blks4[i](
                        torch.cat(
                            [
                                x4,
                                self.no_param_down(self.no_param_down(fuse)),
                                (self.no_param_down(x_c3)),
                            ],
                            dim=1,
                        )
                    )
                )
            )

            x_c1 = self.act(
                # 合并操作后的批量归一化层
                self.bns_mrg1[i](
                    # 合并卷积层
                    torch.mul(x_c1, x1).view(batch_size, self.in_filters1, h1, w1) + x1
                )
            )
            x_c2 = self.act(
                self.bns_mrg2[i](
                    torch.mul(x_c2, x2).view(batch_size, self.in_filters2, h2, w2) + x2
                )
            )
            x_c3 = self.act(
                self.bns_mrg3[i](
                    torch.mul(x_c3, x3).view(batch_size, self.in_filters3, h3, w3) + x3
                )
            )
            x_c4 = self.act(
                self.bns_mrg4[i](
                    torch.mul(x_c4, x4).view(batch_size, self.in_filters4, h4, w4) + x4
                )
            )

        x1 = self.sqe1(x_c1)
        x2 = self.sqe2(x_c2)
        x3 = self.sqe3(x_c3)
        x4 = self.sqe4(x_c4)

        return x1, x2, x3, x4


import thop
if __name__ == '__main__':
    device = torch.device('cpu')
    # 创建一个模型实例，指定输出平面数和编码器类型
    x = torch.randn(1,3,224,224).to(device)
    model = SFIF_Net(out_planes=1, encoder='inceptionnext_tiny')
    # size_in_bytes = get_model_size(model)
    # size_in_mb = size_in_bytes / (1024*1024)
    # print(f"模型大小:{size_in_mb:.2f} MB")
    MACs,Params = thop.profile(model,inputs = (x,),verbose=False)
    FLOPs = MACs * 2
    MACs, FLOPs, Params = thop.clever_format([MACs,FLOPs,Params],"%.3f")

    print(f"MACs:{MACs}")
    print(f"FLOPs:{FLOPs}")
    print(f"Params:{Params}")
    output = model(x)
    print(output.shape)

