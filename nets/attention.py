import torch
import torch.nn as nn

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class DilationConv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1,dilation=1, act=SiLU()):  # ch_in, ch_out, kernel, stride, padding, groups
        super(DilationConv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, dilation, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class DP_Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(DP_Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1)
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=s)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

# 可变形卷积模块
class DeformConv2D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, bias=None):
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size # 卷积核的大小
        self.padding = padding # 填充大小
        self.zero_padding = nn.ZeroPad2d(padding) # 用0去填充
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias) # 二维卷积

    def forward(self, x, offset):

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        offsets_index = Variable(torch.cat([torch.arange(0, 2 * N, 2), torch.arange(1, 2 * N + 1, 2)]),
                                 requires_grad=False).type_as(x).long()

        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # 根据维度dim按照索引列表index将offset重新排序，得到[x1, x2, .... y1, y2, ...]这样顺序的offset
        # ------------------------------------------------------------------------

        # 对输入x进行padding
        if self.padding:
            x = self.zero_padding(x)

        # p表示求偏置后，每个点的位置
        p = self._get_p(offset, dtype)  # (b, 2N, h, w)
        p = p.contiguous().permute(0, 2, 3, 1)  # (b,h,w,2N)

        q_lt = Variable(p.data, requires_grad=False).floor()  # floor是向下取整
        q_rb = q_lt + 1  # 上取整
        # +1相当于向上取整，这里为什么不用向上取整函数呢？是因为如果正好是整数的话，向上取整跟向下取整就重合了，这是我们不想看到的。

        # q_lt[..., :N]代表x方向坐标，大小[b,h,w,N], clamp将值限制在0~h-1
        # q_lt[..., N:]代表y方向坐标, 大小[b,h,w,N], clamp将值限制在0~w-1
        # cat后，还原成原大小[b,h,w,2N]
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()  # 将q_lt中的值控制在图像大小范围内 [b,h,w,2N]
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()  # 将q_rt中的值控制在图像大小范围内 [b,h,w,2N]
        '''
        获取采样后的点周围4个方向的像素点
        q_lt:  left_top 左上
        q_rb:  right_below 右下
        q_lb:  left_below 左下
        q_rt:  right_top 右上
        '''
        # 获得lb
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)  # [b,h,w,2N]
        # 获得rt
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)  # [b,h,w,2N]
        '''
        插值的时候需要考虑一下padding对原始索引的影响 
        p[..., :N]  采样点在x方向(h)的位置   大小[b,h,w,N]
        p[..., :N].lt(self.padding) : p[..., :N]中小于padding 的元素，对应的mask为true
        p[..., :N].gt(x.size(2)-1-self.padding): p[..., :N]中大于h-1-padding 的元素，对应的mask为true 
        
           p[..., N:]  采样点在y方向(w)的位置   大小[b,h,w,N]
        p[..., N:].lt(self.padding) : p[..., N:]中小于padding 的元素，对应的mask为true
        p[..., N:].gt(x.size(2)-1-self.padding): p[..., N:]中大于w-1-padding 的元素，对应的mask为true 
        cat之后，大小为[b,h,w,2N]
        '''
        mask = torch.cat([p[..., :N].lt(self.padding) + p[..., :N].gt(x.size(2) - 1 - self.padding),
                          p[..., N:].lt(self.padding) + p[..., N:].gt(x.size(3) - 1 - self.padding)], dim=-1).type_as(p)  #
        # mask不需要反向传播
        mask = mask.detach()
        # p - (p - torch.floor(p))相当于torch.floor(p)
        floor_p = p - (p - torch.floor(p))

        '''
        mask为1的区域就是padding的区域
        p*(1-mask) : mask为0的 非padding区域的p被保留
        floor_p*mask: mask为1的  padding区域的floor_p被保留
        '''
        p = p * (1 - mask) + floor_p * mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # 双线性插值的系数 大小均为 (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        # (1+左上角的点x - 原始采样点x)*(1+左上角的点y - 原始采样点y)
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        # (1-(右下角的点x - 原始采样点x))*(1-(右下角的点y - 原始采样点y))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        # (1+左下角的点x - 原始采样点x)*(1+左上角的点y - 原始采样点y)
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        # (1-(右上角的点x - 原始采样点x))*(1-(右上角的点y - 原始采样点y))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)  # 左上角的点在原始图片中对应的真实像素值
        x_q_rb = self._get_x_q(x, q_rb, N)  # 右下角的点在原始图片中对应的真实像素值
        x_q_lb = self._get_x_q(x, q_lb, N)  # 左下角的点在原始图片中对应的真实像素值
        x_q_rt = self._get_x_q(x, q_rt, N)  # 右上角的点在原始图片中对应的真实像素值

        # 双线性插值算法
        # x_offset : 偏移后的点再双线性插值后的值  大小(b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt
        '''
        偏置点含有九个方向的偏置，_reshape_x_offset() 把每个点9个方向的偏置转化成 3×3 的形式，
        于是就可以用 3×3 stride=3 的卷积核进行 Deformable Convolution，
        它等价于使用 1×1 的正常卷积核（包含了这个点9个方向的 context）对原特征直接进行卷积。
        '''
        x_offset = self._reshape_x_offset(x_offset, ks)  # (b,c,h*ks,w*ks)

        out = self.conv_kernel(x_offset)

        return out

    # 功能：求每个点的偏置方向
    def _get_p_n(self, N, dtype):
        # N=kernel_size*kernel_size
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                   range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                   indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2 * N, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n  # [1,2*N,1,1]

# 定义深度可分离卷积模块
class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWConv, self).__init__()
        # 深度卷积步骤，使用3x3卷积核，不改变卷积后的输出尺寸
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, groups=in_channels)
        # 逐点卷积步骤，增加通道数
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        # 先进行深度卷积
        x = self.depthwise(x)
        # 然后进行逐点卷积，增加通道数
        x = self.pointwise(x)
        return x

class Bottleneck(nn.Module):
    # 标准瓶颈结构，残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class DAS(nn.Module):
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = DWConv(c1, 2 * self.c, 1, 1)
        self.cv2 = DeformConv2D((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class SELayer(nn.Module):
    def __init__(self, c1, r=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // r, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // r, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        x = x * y.expand_as(x)

        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        # 写法二,亦可使用顺序容器
        # self.sharedMLP = nn.Sequential(
        # nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
        # nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


# class CBAMC3(nn.Module):
#     # CSP Bottleneck with 3 convolutions
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super(CBAMC3, self).__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.cv3 = Conv(2 * c_, c2, 1)
#         self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
#         self.channel_attention = ChannelAttention(c2, 16)
#         self.spatial_attention = SpatialAttention(7)
#
#         # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
#
#     def forward(self, x):
#         out = self.channel_attention(x) * x
#         print('outchannels:{}'.format(out.shape))
#         out = self.spatial_attention(out) * out
#         return out

