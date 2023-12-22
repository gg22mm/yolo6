# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from pathlib import Path
import numpy as np
from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer
from torch.nn.parameter import Parameter
######################### from yolov6.layers.common import * #######################################
class Conv(nn.Module):    
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, padding=None):
        super().__init__()        
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))

class ConvWrapper(nn.Module):    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=True):
        super().__init__()
        self.block = Conv(in_channels, out_channels, kernel_size, stride, groups, bias)
    
    def forward(self, x):
        return self.block(x)

class SimConv(nn.Module):    
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))

class SimSPPF(nn.Module):    
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = SimConv(in_channels, c_, 1, 1)
        self.cv2 = SimConv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

class SimCSPSPPF(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_channels, out_channels, kernel_size=5, e=0.5):
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = SimConv(in_channels, c_, 1, 1)
        self.cv2 = SimConv(in_channels, c_, 1, 1)
        self.cv3 = SimConv(c_, c_, 3, 1)
        self.cv4 = SimConv(c_, c_, 1, 1)
        
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.cv5 = SimConv(4 * c_, c_, 1, 1)
        self.cv6 = SimConv(c_, c_, 3, 1)
        self.cv7 = SimConv(2 * c_, out_channels, 1, 1)
    
    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y0 = self.cv2(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x1)
            y2 = self.m(y1)
            y3 = self.cv6(self.cv5(torch.cat([x1, y1, y2, self.m(y2)], 1)))
        return self.cv7(torch.cat((y0, y3), dim=1))

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):    
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=bias))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):
    #RepVGGBlock是一个基本的rep样式块，包括训练和部署状态此代码基于 https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py  
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super().__init__()

        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        assert kernel_size == 3
        assert padding == 1
        
        padding_11 = padding - kernel_size // 2
        
        self.nonlinearity = nn.ReLU()
        
        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()
        
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)
        
        else:
            self.rbr_identity = nn.BatchNorm2d(
                    num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
    
    def forward(self, inputs):        
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
        
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
    
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
    
    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

# 入口 - 可以切其它的
def get_block(mode):
    if mode == 'repvgg':
        return RepVGGBlock    
    else:
        raise NotImplementedError("未定义Repblock {}".format(mode))


######################### from yolov6.models.efficientrep import * #######################################

class BottleRep(nn.Module):    
    def __init__(self, in_channels, out_channels, basic_block=RepVGGBlock, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = Parameter(torch.ones(1))
        else:
            self.alpha = 1.0
    
    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs

class RepBlock(nn.Module):    
    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, basic_block=RepVGGBlock):
        super().__init__()

        self.conv1 = block(in_channels, out_channels)
        self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None
        if block == BottleRep:
            self.conv1 = BottleRep(in_channels, out_channels, basic_block=basic_block, weight=True)
            n = n // 2
            self.block = nn.Sequential(
                    *(BottleRep(out_channels, out_channels, basic_block=basic_block, weight=True) for _ in
                      range(n - 1))) if n > 1 else None
    
    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x

class EfficientRep(nn.Module):
    '''EfficientRep Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    '''   
    def __init__(self,in_channels=3,channels_list=None,num_repeats=None,block=RepVGGBlock,fuse_P2=False,cspsppf=False):
        super().__init__()
        
        assert channels_list is not None
        assert num_repeats is not None
        self.fuse_P2 = fuse_P2
        
        self.stem = block(
                in_channels=in_channels,
                out_channels=channels_list[0],
                kernel_size=3,
                stride=2
        )        
        self.ERBlock_2 = nn.Sequential(
                block(
                        in_channels=channels_list[0],
                        out_channels=channels_list[1],
                        kernel_size=3,
                        stride=2
                ),
                RepBlock(
                        in_channels=channels_list[1],
                        out_channels=channels_list[1],
                        n=num_repeats[1],
                        block=block,
                )
        )
        
        self.ERBlock_3 = nn.Sequential(
                block(
                        in_channels=channels_list[1],
                        out_channels=channels_list[2],
                        kernel_size=3,
                        stride=2
                ),
                RepBlock(
                        in_channels=channels_list[2],
                        out_channels=channels_list[2],
                        n=num_repeats[2],
                        block=block,
                )
        )
        
        self.ERBlock_4 = nn.Sequential(
                block(
                        in_channels=channels_list[2],
                        out_channels=channels_list[3],
                        kernel_size=3,
                        stride=2
                ),
                RepBlock(
                        in_channels=channels_list[3],
                        out_channels=channels_list[3],
                        n=num_repeats[3],
                        block=block,
                )
        )
        
        channel_merge_layer = SPPF if block == ConvWrapper else SimSPPF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvWrapper else SimCSPSPPF
        
        self.ERBlock_5 = nn.Sequential(
                block(
                        in_channels=channels_list[3],
                        out_channels=channels_list[4],
                        kernel_size=3,
                        stride=2,
                ),
                RepBlock(
                        in_channels=channels_list[4],
                        out_channels=channels_list[4],
                        n=num_repeats[4],
                        block=block,
                ),
                channel_merge_layer(
                        in_channels=channels_list[4],
                        out_channels=channels_list[4],
                        kernel_size=5
                )
        )
    
    def forward(self, x):
        
        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)
        
        return tuple(outputs)

######################### from gold_yolo.reppan import * #######################################

#----------------5-----------------

def onnx_AdaptiveAvgPool2d(x, output_size):
    stride_size = np.floor(np.array(x.shape[-2:]) / output_size).astype(np.int32)
    kernel_size = np.array(x.shape[-2:]) - (output_size - 1) * stride_size
    avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
    x = avg(x)
    return x

class SimFusion_3in(nn.Module):
    def __init__(self, in_channel_list, out_channels):
        super().__init__()
        self.cv1 = SimConv(in_channel_list[0], out_channels, 1, 1)
        self.cv_fuse = SimConv(out_channels * 3, out_channels, 1, 1)
        self.downsample = nn.functional.adaptive_avg_pool2d
    
    def forward(self, x):
        N, C, H, W = x[1].shape
        output_size = (H, W)
        
        if torch.onnx.is_in_onnx_export():
            self.downsample = onnx_AdaptiveAvgPool2d
            output_size = np.array([H, W])
        
        x0 = self.downsample(x[0], output_size)
        x1 = self.cv1(x[1])
        x2 = F.interpolate(x[2], size=(H, W), mode='bilinear', align_corners=False)
        return self.cv_fuse(torch.cat((x0, x1, x2), dim=1))

class SimFusion_4in(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d
    
    def forward(self, x):
        x_l, x_m, x_s, x_n = x
        B, C, H, W = x_s.shape
        output_size = np.array([H, W])
        
        if torch.onnx.is_in_onnx_export():
            self.avg_pool = onnx_AdaptiveAvgPool2d
        
        x_l = self.avg_pool(x_l, output_size)
        x_m = self.avg_pool(x_m, output_size)
        x_n = F.interpolate(x_n, size=(H, W), mode='bilinear', align_corners=False)
        
        out = torch.cat([x_l, x_m, x_s, x_n], 1)
        return out

def get_avg_pool():
    if torch.onnx.is_in_onnx_export():
        avg_pool = onnx_AdaptiveAvgPool2d
    else:
        avg_pool = nn.functional.adaptive_avg_pool2d
    return avg_pool

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    
    def forward(self, x):
        return self.relu(x + 3) / 6

class InjectionMultiSum_Auto_pool(nn.Module):
    def __init__(self,
            inp: int,
            oup: int,
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=None,
            global_inp=None,
    ) -> None:
        super().__init__()
        self.norm_cfg = norm_cfg
        
        if not global_inp:
            global_inp = inp
        
        self.local_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_embedding = ConvModule(global_inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_act = ConvModule(global_inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()
    
    def forward(self, x_l, x_g):
        '''
        x_g: global features
        x_l: local features
        '''
        B, C, H, W = x_l.shape
        g_B, g_C, g_H, g_W = x_g.shape
        use_pool = H < g_H
        
        local_feat = self.local_embedding(x_l)
        
        global_act = self.global_act(x_g)
        global_feat = self.global_embedding(x_g)
        
        if use_pool:
            avg_pool = get_avg_pool()
            output_size = np.array([H, W])
            
            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)
        
        else:
            sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
            global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        out = local_feat * sig_act + global_feat
        return out

def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape

class PyramidPoolAgg(nn.Module):
    def __init__(self, stride, pool_mode='onnx'):
        super().__init__()
        self.stride = stride
        if pool_mode == 'torch':
            self.pool = nn.functional.adaptive_avg_pool2d
        elif pool_mode == 'onnx':
            self.pool = onnx_AdaptiveAvgPool2d
    
    def forward(self, inputs):
        B, C, H, W = get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        
        output_size = np.array([H, W])
        
        if not hasattr(self, 'pool'):
            self.pool = nn.functional.adaptive_avg_pool2d
        
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        
        out = [self.pool(inp, output_size) for inp in inputs]
        
        return torch.cat(out, dim=1)

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        
        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,attn_ratio=4,activation=None,norm_cfg=dict(type='BN', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        
        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
                self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
    
    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)
        
        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)
        
        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1)  # dim = k
        
        xx = torch.matmul(attn, vv)
        
        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):    
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    
    def forward(self, x):
        return self.relu(x + 3) / 6

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0., norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class top_Block(nn.Module):
    
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio        
        self.attn = Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, activation=act_layer,norm_cfg=norm_cfg)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,norm_cfg=norm_cfg)
    
    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1

class TopBasicLayer(nn.Module):
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=nn.ReLU6):
        super().__init__()
        self.block_num = block_num
        
        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(top_Block(
                    embedding_dim, key_dim=key_dim, num_heads=num_heads,
                    mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                    drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_cfg=norm_cfg, act_layer=act_layer))
    
    def forward(self, x):
        # token * N 
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x

class AdvPoolFusion(nn.Module):
    def forward(self, x1, x2):
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        else:
            self.pool = nn.functional.adaptive_avg_pool2d
        
        N, C, H, W = x2.shape
        output_size = np.array([H, W])
        x1 = self.pool(x1, output_size)
        
        return torch.cat([x1, x2], 1)

class RepGDNeck(nn.Module):
    def __init__(self,channels_list=None,num_repeats=None,block=RepVGGBlock,extra_cfg=None):
        super().__init__()        
        assert channels_list is not None
        assert num_repeats is not None
        
        self.low_FAM = SimFusion_4in()
        self.low_IFM = nn.Sequential(
                Conv(extra_cfg.fusion_in, extra_cfg.embed_dim_p, kernel_size=1, stride=1, padding=0),
                *[block(extra_cfg.embed_dim_p, extra_cfg.embed_dim_p) for _ in range(extra_cfg.fuse_block_num)],
                Conv(extra_cfg.embed_dim_p, sum(extra_cfg.trans_channels[0:2]), kernel_size=1, stride=1, padding=0),
        )
        
        self.reduce_layer_c5 = SimConv(
                in_channels=channels_list[4],  # 1024
                out_channels=channels_list[5],  # 512
                kernel_size=1,
                stride=1
        )
        self.LAF_p4 = SimFusion_3in(
                in_channel_list=[channels_list[3], channels_list[3]],  # 512, 256
                out_channels=channels_list[5],  # 256
        )
        self.Inject_p4 = InjectionMultiSum_Auto_pool(channels_list[5], channels_list[5], norm_cfg=extra_cfg.norm_cfg,
                                                     activations=nn.ReLU6)
        self.Rep_p4 = RepBlock(
                in_channels=channels_list[5],  # 256
                out_channels=channels_list[5],  # 256
                n=num_repeats[5],
                block=block
        )
        
        self.reduce_layer_p4 = SimConv(
                in_channels=channels_list[5],  # 256
                out_channels=channels_list[6],  # 128
                kernel_size=1,
                stride=1
        )
        self.LAF_p3 = SimFusion_3in(
                in_channel_list=[channels_list[5], channels_list[5]],  # 512, 256
                out_channels=channels_list[6],  # 256
        )
        self.Inject_p3 = InjectionMultiSum_Auto_pool(channels_list[6], channels_list[6], norm_cfg=extra_cfg.norm_cfg,
                                                     activations=nn.ReLU6)
        self.Rep_p3 = RepBlock(
                in_channels=channels_list[6],  # 128
                out_channels=channels_list[6],  # 128
                n=num_repeats[6],
                block=block
        )
        
        self.high_FAM = PyramidPoolAgg(stride=extra_cfg.c2t_stride, pool_mode=extra_cfg.pool_mode)
        dpr = [x.item() for x in torch.linspace(0, extra_cfg.drop_path_rate, extra_cfg.depths)]
        self.high_IFM = TopBasicLayer(
                block_num=extra_cfg.depths,
                embedding_dim=extra_cfg.embed_dim_n,
                key_dim=extra_cfg.key_dim,
                num_heads=extra_cfg.num_heads,
                mlp_ratio=extra_cfg.mlp_ratios,
                attn_ratio=extra_cfg.attn_ratios,
                drop=0, attn_drop=0,
                drop_path=dpr,
                norm_cfg=extra_cfg.norm_cfg
        )
        self.conv_1x1_n = nn.Conv2d(extra_cfg.embed_dim_n, sum(extra_cfg.trans_channels[2:4]), 1, 1, 0)
        
        self.LAF_n4 = AdvPoolFusion()
        self.Inject_n4 = InjectionMultiSum_Auto_pool(channels_list[8], channels_list[8],
                                                     norm_cfg=extra_cfg.norm_cfg, activations=nn.ReLU6)
        self.Rep_n4 = RepBlock(
                in_channels=channels_list[6] + channels_list[7],  # 128 + 128
                out_channels=channels_list[8],  # 256
                n=num_repeats[7],
                block=block
        )
        
        self.LAF_n5 = AdvPoolFusion()
        self.Inject_n5 = InjectionMultiSum_Auto_pool(channels_list[10], channels_list[10],
                                                     norm_cfg=extra_cfg.norm_cfg, activations=nn.ReLU6)
        self.Rep_n5 = RepBlock(
                in_channels=channels_list[5] + channels_list[9],  # 256 + 256
                out_channels=channels_list[10],  # 512
                n=num_repeats[8],
                block=block
        )
        
        self.trans_channels = extra_cfg.trans_channels
    
    def forward(self, input):

        (c2, c3, c4, c5) = input
        
        # Low-GD
        ## use conv fusion global info
        low_align_feat = self.low_FAM(input)
        low_fuse_feat = self.low_IFM(low_align_feat)
        low_global_info = low_fuse_feat.split(self.trans_channels[0:2], dim=1)
        
        ## inject low-level global info to p4
        c5_half = self.reduce_layer_c5(c5)
        p4_adjacent_info = self.LAF_p4([c3, c4, c5_half])
        p4 = self.Inject_p4(p4_adjacent_info, low_global_info[0])
        p4 = self.Rep_p4(p4)
        
        ## inject low-level global info to p3
        p4_half = self.reduce_layer_p4(p4)
        p3_adjacent_info = self.LAF_p3([c2, c3, p4_half])
        p3 = self.Inject_p3(p3_adjacent_info, low_global_info[1])
        p3 = self.Rep_p3(p3)
        
        # High-GD
        ## use transformer fusion global info
        high_align_feat = self.high_FAM([p3, p4, c5])
        high_fuse_feat = self.high_IFM(high_align_feat)
        high_fuse_feat = self.conv_1x1_n(high_fuse_feat)
        high_global_info = high_fuse_feat.split(self.trans_channels[2:4], dim=1)
        
        ## inject low-level global info to n4
        n4_adjacent_info = self.LAF_n4(p3, p4_half)
        n4 = self.Inject_n4(n4_adjacent_info, high_global_info[0])
        n4 = self.Rep_n4(n4)
        
        ## inject low-level global info to n5
        n5_adjacent_info = self.LAF_n5(n4, c5_half)
        n5 = self.Inject_n5(n5_adjacent_info, high_global_info[1])
        n5 = self.Rep_n5(n5)
        
        outputs = [p3, n4, n5]
        
        return outputs

#----------------5-----------------

# 这个可以有多个不同的，这里使用的是： effidehead
######################### 对象检测header -  from yolov6.models.effidehead import Detect, build_effidehead_layer #########################

def generate_anchors(feats, fpn_strides, grid_cell_size=5.0, grid_cell_offset=0.5,  device='cpu', is_eval=False, mode='af'):    
    anchors = []
    anchor_points = []
    stride_tensor = []
    num_anchors_list = []
    assert feats is not None
    if is_eval:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            shift_x = torch.arange(end=w, device=device) + grid_cell_offset
            shift_y = torch.arange(end=h, device=device) + grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack(
                    [shift_x, shift_y], axis=-1).to(torch.float)
            if mode == 'af': # anchor-free
                anchor_points.append(anchor_point.reshape([-1, 2]))
                stride_tensor.append(
                torch.full(
                    (h * w, 1), stride, dtype=torch.float, device=device))
            elif mode == 'ab': # anchor-based
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3,1))
                stride_tensor.append(
                    torch.full(
                        (h * w, 1), stride, dtype=torch.float, device=device).repeat(3,1))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor
    else:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            cell_half_size = grid_cell_size * stride * 0.5
            shift_x = (torch.arange(end=w, device=device) + grid_cell_offset) * stride
            shift_y = (torch.arange(end=h, device=device) + grid_cell_offset) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor = torch.stack(
                [
                    shift_x - cell_half_size, shift_y - cell_half_size,
                    shift_x + cell_half_size, shift_y + cell_half_size
                ],
                axis=-1).clone().to(feats[0].dtype)
            anchor_point = torch.stack(
                [shift_x, shift_y], axis=-1).clone().to(feats[0].dtype)

            if mode == 'af': # anchor-free
                anchors.append(anchor.reshape([-1, 4]))
                anchor_points.append(anchor_point.reshape([-1, 2]))
            elif mode == 'ab': # anchor-based
                anchors.append(anchor.reshape([-1, 4]).repeat(3,1))
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3,1))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor.append(
                torch.full(
                    [num_anchors_list[-1], 1], stride, dtype=feats[0].dtype)) 
        anchors = torch.cat(anchors)
        anchor_points = torch.cat(anchor_points).to(device)
        stride_tensor = torch.cat(stride_tensor).to(device)
        return anchors, anchor_points, num_anchors_list, stride_tensor

def dist2bbox(distance, anchor_points, box_format='xyxy'):
    '''Transform distance(ltrb) to box(xywh or xyxy).'''
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if box_format == 'xyxy':
        bbox = torch.cat([x1y1, x2y2], -1)
    elif box_format == 'xywh':
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        bbox = torch.cat([c_xy, wh], -1)
    return bbox

# 图像识别用这个，如果是图像分类，分割，那么得自己写了... 待添加其它功能.....
class Detect(nn.Module):    
    def __init__(self, num_classes=80, num_layers=3, inplace=True, head_layers=None, use_dfl=True,reg_max=16):  # detection layer
        super().__init__()
        assert head_layers is not None
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers
        self.grid = [torch.zeros(1)] * num_layers
        self.prior_prob = 1e-2
        self.inplace = inplace
        stride = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64]  # strides computed during build
        self.stride = torch.tensor(stride)
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0
        
        # Init decouple head
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        
        # Efficient decoupled head layers
        for i in range(num_layers):
            idx = i * 5
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx + 1])
            self.reg_convs.append(head_layers[idx + 2])
            self.cls_preds.append(head_layers[idx + 3])
            self.reg_preds.append(head_layers[idx + 4])
    
    def initialize_biases(self):        
        for conv in self.cls_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)
        
        for conv in self.reg_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)
        
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
                                             requires_grad=False)
    
    def forward(self, x):        
        if self.training: #True 执行的是这里
            cls_score_list = []
            reg_distri_list = []
            
            for i in range(self.nl): #循环层数
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)
                
                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))
            
            cls_score_list = torch.cat(cls_score_list, axis=1)
            reg_distri_list = torch.cat(reg_distri_list, axis=1)
            
            return x, cls_score_list, reg_distri_list
        else:
            cls_score_list = []
            reg_dist_list = []
            anchor_points, stride_tensor = generate_anchors(x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True,mode='af')
            
            for i in range(self.nl):
                b, _, h, w = x[i].shape
                l = h * w
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)
                
                if self.use_dfl:
                    reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output = self.proj_conv(F.softmax(reg_output, dim=1))
                
                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.reshape([b, self.nc, l]))
                reg_dist_list.append(reg_output.reshape([b, 4, l]))
            
            cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)
            reg_dist_list = torch.cat(reg_dist_list, axis=-1).permute(0, 2, 1)
            
            pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')
            pred_bboxes *= stride_tensor
            return torch.cat(
                    [
                            pred_bboxes,
                            torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                            cls_score_list
                    ],
                    axis=-1)


def build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max=16, num_layers=3):
    chx = [6, 8, 10] if num_layers == 3 else [8, 9, 10, 11]    
    head_layers = nn.Sequential(
            # stem0
            Conv(
                    in_channels=channels_list[chx[0]],
                    out_channels=channels_list[chx[0]],
                    kernel_size=1,
                    stride=1
            ),
            # cls_conv0
            Conv(
                    in_channels=channels_list[chx[0]],
                    out_channels=channels_list[chx[0]],
                    kernel_size=3,
                    stride=1
            ),
            # reg_conv0
            Conv(
                    in_channels=channels_list[chx[0]],
                    out_channels=channels_list[chx[0]],
                    kernel_size=3,
                    stride=1
            ),
            # cls_pred0
            nn.Conv2d(
                    in_channels=channels_list[chx[0]],
                    out_channels=num_classes * num_anchors,
                    kernel_size=1
            ),
            # reg_pred0
            nn.Conv2d(
                    in_channels=channels_list[chx[0]],
                    out_channels=4 * (reg_max + num_anchors),
                    kernel_size=1
            ),
            # stem1
            Conv(
                    in_channels=channels_list[chx[1]],
                    out_channels=channels_list[chx[1]],
                    kernel_size=1,
                    stride=1
            ),
            # cls_conv1
            Conv(
                    in_channels=channels_list[chx[1]],
                    out_channels=channels_list[chx[1]],
                    kernel_size=3,
                    stride=1
            ),
            # reg_conv1
            Conv(
                    in_channels=channels_list[chx[1]],
                    out_channels=channels_list[chx[1]],
                    kernel_size=3,
                    stride=1
            ),
            # cls_pred1
            nn.Conv2d(
                    in_channels=channels_list[chx[1]],
                    out_channels=num_classes * num_anchors,
                    kernel_size=1
            ),
            # reg_pred1
            nn.Conv2d(
                    in_channels=channels_list[chx[1]],
                    out_channels=4 * (reg_max + num_anchors),
                    kernel_size=1
            ),
            # stem2
            Conv(
                    in_channels=channels_list[chx[2]],
                    out_channels=channels_list[chx[2]],
                    kernel_size=1,
                    stride=1
            ),
            # cls_conv2
            Conv(
                    in_channels=channels_list[chx[2]],
                    out_channels=channels_list[chx[2]],
                    kernel_size=3,
                    stride=1
            ),
            # reg_conv2
            Conv(
                    in_channels=channels_list[chx[2]],
                    out_channels=channels_list[chx[2]],
                    kernel_size=3,
                    stride=1
            ),
            # cls_pred2
            nn.Conv2d(
                    in_channels=channels_list[chx[2]],
                    out_channels=num_classes * num_anchors,
                    kernel_size=1
            ),
            # reg_pred2
            nn.Conv2d(
                    in_channels=channels_list[chx[2]],
                    out_channels=4 * (reg_max + num_anchors),
                    kernel_size=1
            )
    )
    
    if num_layers == 4:
        head_layers.add_module('stem3',
                               # stem3
                               Conv(
                                       in_channels=channels_list[chx[3]],
                                       out_channels=channels_list[chx[3]],
                                       kernel_size=1,
                                       stride=1
                               )
                               )
        head_layers.add_module('cls_conv3',
                               # cls_conv3
                               Conv(
                                       in_channels=channels_list[chx[3]],
                                       out_channels=channels_list[chx[3]],
                                       kernel_size=3,
                                       stride=1
                               )
                               )
        head_layers.add_module('reg_conv3',
                               # reg_conv3
                               Conv(
                                       in_channels=channels_list[chx[3]],
                                       out_channels=channels_list[chx[3]],
                                       kernel_size=3,
                                       stride=1
                               )
                               )
        head_layers.add_module('cls_pred3',
                               # cls_pred3
                               nn.Conv2d(
                                       in_channels=channels_list[chx[3]],
                                       out_channels=num_classes * num_anchors,
                                       kernel_size=1
                               )
                               )
        head_layers.add_module('reg_pred3',
                               # reg_pred3
                               nn.Conv2d(
                                       in_channels=channels_list[chx[3]],
                                       out_channels=4 * (reg_max + num_anchors),
                                       kernel_size=1
                               )
                               )
    
    return head_layers


####################################################################################################

# 模型初始化
def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True

class Model(nn.Module):
    '''YOLOv6 model with backbone, neck and head.
    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.
    '''
    
    def __init__(self, config, channels=3, num_classes=None, fuse_ab=False,
                 distill_ns=False):  # model, input channels, number of classes
        super().__init__()
        # Build network
        num_layers = config.model.head.num_layers
        self.backbone, self.neck, self.detect = build_network(config, channels, num_classes, num_layers,
                                                              fuse_ab=fuse_ab, distill_ns=distill_ns)
        
        # Init Detect head
        self.stride = self.detect.stride
        self.detect.initialize_biases()
        
        # Init weights
        initialize_weights(self)
    
    def forward(self, x):
        export_mode = torch.onnx.is_in_onnx_export()
        x = self.backbone(x)
        x = self.neck(x)
        if export_mode == False:
            featmaps = []
            featmaps.extend(x)
        x = self.detect(x)
        return x if export_mode is True else [x, featmaps]
    
    def _apply(self, fn):
        self = super()._apply(fn)
        self.detect.stride = fn(self.detect.stride)
        self.detect.grid = list(map(fn, self.detect.grid))
        return self


def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor


def build_network(config, channels, num_classes, num_layers, fuse_ab=False, distill_ns=False):
    depth_mul = config.model.depth_multiple
    width_mul = config.model.width_multiple
    num_repeat_backbone = config.model.backbone.num_repeats
    channels_list_backbone = config.model.backbone.out_channels
    fuse_P2 = config.model.backbone.get('fuse_P2')
    cspsppf = config.model.backbone.get('cspsppf')
    num_repeat_neck = config.model.neck.num_repeats
    channels_list_neck = config.model.neck.out_channels
    use_dfl = config.model.head.use_dfl
    reg_max = config.model.head.reg_max
    num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]
    
    # 分别载入各种模型块
    block = get_block(config.training_mode)
    BACKBONE = eval(config.model.backbone.type) #EfficientRep
    NECK = eval(config.model.neck.type)         #RepGDNeck
    
    neck_extra_cfg = config.model.neck.extra_cfg if 'extra_cfg' in config.model.neck else None
    
    if 'CSP' in config.model.backbone.type: #EfficientRep
        backbone = BACKBONE(
                in_channels=channels,
                channels_list=channels_list,
                num_repeats=num_repeat,
                block=block,
                csp_e=config.model.backbone.csp_e,
                fuse_P2=fuse_P2,
                cspsppf=cspsppf
        )        
        neck = NECK(
                channels_list=channels_list,
                num_repeats=num_repeat,
                block=block,
                csp_e=config.model.neck.csp_e,
                extra_cfg=neck_extra_cfg
        )
    else: #执行的是这里
        backbone = BACKBONE(
                in_channels=channels,
                channels_list=channels_list,
                num_repeats=num_repeat,
                block=block,
                fuse_P2=fuse_P2,
                cspsppf=cspsppf
        )        
        neck = NECK(
                channels_list=channels_list,
                num_repeats=num_repeat,
                block=block,
                extra_cfg=neck_extra_cfg
        )
    
    
    # # 这个可以有多个不同的，这里使用的是： effidehead
    # from yolov6.models.effidehead import Detect, build_effidehead_layer
    head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max, num_layers=num_layers)
    head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)
    
    return backbone, neck, head #EfficientRep , RepGDNeck, Detect


def build_model(cfg, num_classes, device, fuse_ab=False, distill_ns=False):
    model = Model(cfg, channels=3, num_classes=num_classes, fuse_ab=fuse_ab, distill_ns=distill_ns).to(device)    
    return model

def load_yaml(file_path): 
    import yaml   
    if isinstance(file_path, str):
        with open(file_path, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
    return data_dict

# 模型配置参数
from munch import DefaultMunch as dictToObj #字典转对象 pip install munch -i https://pypi.tuna.tsinghua.edu.cn/simple
cfg={
'use_checkpoint': False,
'model': {
    'type': 'GoldYOLO-s',
    'pretrained': None,
    'depth_multiple': 0.33,
    'width_multiple': 0.5,
    'backbone': {'type': 'EfficientRep', 'num_repeats': [1, 6, 12, 18, 6], 'out_channels': [64, 128, 256, 512, 1024], 'fuse_P2': True, 'cspsppf': True},
    'neck': {'type': 'RepGDNeck', 'num_repeats': [12, 12, 12, 12], 'out_channels': [256, 128, 128, 256, 256, 512],            
        'extra_cfg': {
            'norm_cfg': {'type': 'SyncBN', 'requires_grad': True},
            'depths': 2, 'fusion_in': 960, 'ppa_in': 704, 'fusion_act': {'type': 'ReLU6'}, 'fuse_block_num': 3,
            'embed_dim_p': 128, 'embed_dim_n': 704, 'key_dim': 8, 'num_heads': 4, 'mlp_ratios': 1, 'attn_ratios': 2, 'c2t_stride': 2,
            'drop_path_rate': 0.1, 'trans_channels': [128, 64, 128, 256], 
            'pool_mode': 'torch'
        }
    },
    'head': {'type': 'EffiDeHead', 'in_channels': [128, 256, 512], 'num_layers': 3, 'begin_indices': 24, 'anchors': 3, 'anchors_init': [[10, 13, 19, 19, 33, 23], [30, 61, 59, 59, 59, 119], [116, 90, 185, 185, 373, 326]], 'out_indices': [17, 20, 23], 'strides': [8, 16, 32], 'atss_warmup_epoch': 0, 'iou_type': 'giou', 'use_dfl': True, 'reg_max': 16, 'distill_weight': {'class': 1.0, 'dfl': 1.0}}},
    'solver': {'optim': 'SGD', 'lr_scheduler': 'Cosine', 'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1},
    'data_aug': {'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0},
    'training_mode': 'repvgg', #附加的
}
cfg = dictToObj.fromDict(cfg) #字典转对象   


# 分类用的是这个loss:  loss = (F.binary_cross_entropy(pred_score.float(), gt_score.float(), reduction='none') * weight).sum()

# 获取数据集合 - 后观会替换掉
device='cpu'

# data_dict = load_yaml('data/coco.yaml')  #直接读取yaml

# 数据集配置参数 直接用json也行
data_dict={
    'train': 'dataset/coco-yolo/images/train2014', 
    'val': 'dataset/coco-yolo/images/val2014',
    'test': 'dataset/coco-yolo/images/test2014',
    'anno_path': 'dataset/coco-yolo/annotations/instances_val2014.json',
    'nc': 80, 'is_coco': True, 
    'names': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
}

nc = data_dict['nc'] #80分类数
model = build_model(cfg, nc, device, fuse_ab=False, distill_ns=False)
# print(model)
# exit()

# 循环训练

# 数据集
images = torch.randn(2, 3, 640, 640) #torch.Size([2, 3, 640, 640])
# print(images.shape) #torch.Size([2, 3, 640, 640])

# 识别
preds, s_featmaps = model(images)
feats, pred_scores, pred_distri = preds
# print(feats[0].shape)         #torch.Size([2, 64, 80, 80])
# print(pred_scores[0].shape)   #torch.Size([8400, 80])
# print(pred_distri[0].shape)   #torch.Size([8400, 68])

