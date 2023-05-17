import torch
import torch.nn as nn
import torch.nn.functional as F
from Cell import Spectral

# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.act = nn.Tanh()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.act(out)
        out = out * x
        return out


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (1,3, 5, 7), 'kernel size must be 1 or 3 or 5 or 7'
        if kernel_size == 7:
            padding = 3
        if kernel_size == 5:
            padding = 2
        if kernel_size == 3:
            padding = 1
        if kernel_size == 1:
            padding = 0
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.act(out)
        out = out * x
        return out


class NonLocalBlock(nn.Module):
    """
    调用过程
    NONLocalBlock2D(in_channels=32),
    super(NONLocalBlock2D, self).__init__(in_channels,
            inter_channels=inter_channels,
            dimension=2, sub_sample=sub_sample,
            bn_layer=bn_layer)
    """

    def __init__(self,
                 in_channels,
                 inter_channels=None):
        super(NonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.r = 16  # 瓶颈率
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            # 进行压缩得到channel个数
            if self.inter_channels == 0:
                self.inter_channels = 1
        self.g1 = nn.Conv2d(in_channels=self.in_channels,
                            out_channels=self.inter_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0, bias=False)
        self.g2 = nn.Conv2d(in_channels=self.inter_channels,
                            out_channels=1,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.softmax = nn.Softmax(dim=2)
        self.transform = nn.Sequential(nn.Conv2d(self.in_channels, self.in_channels // self.r, kernel_size=1),
                                       nn.LayerNorm([self.in_channels // self.r, 1, 1]),
                                       nn.PReLU(),
                                       nn.Conv2d(self.in_channels // self.r, self.in_channels, kernel_size=1)

                                       )

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.g1(x)
        context_mask = self.g2(context_mask)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)
        return context

    def forward(self, x):
        """
        :param x: ( c,  h, w)
        :return:
        """
        # torch.unsqueeze(x.permute([2, 0, 1]), 0)
        context = self.spatial_pool(x)
        channel_mul_term = torch.sigmoid(self.transform(context))
        GC_out = x * channel_mul_term
        return GC_out


class Encoder(nn.Module):
    """docstring for ClassName"""

    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        kernel_1 = 5
        kernel_2 = 7
        self.Conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=0, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_1, kernel_1), padding=(kernel_1 - 1) // 2,
                      groups=out_channels),

            nn.Tanh()
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), padding=0, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_2, kernel_2), padding=(kernel_2 - 1) // 2,
                      groups=out_channels),

            nn.Tanh()
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, inputs):
        x = self.Conv1(inputs)
        x = self.Conv2(x)
        x_pool = self.pool(x)
        return x, x_pool


class Decoder(nn.Module):
    """docstring for ClassName"""

    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        kernel_1 = 7
        kernel_2 = 5
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.Conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=0, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_1, kernel_1), padding=(kernel_1 - 1) // 2,
                      groups=out_channels),

            nn.Tanh()
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), padding=0, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_2, kernel_2), padding=(kernel_2 - 1) // 2,
                      groups=out_channels),

            nn.Tanh()
        )
        self.spec_att = ChannelAttention(out_channels)
        self.spa_att = SpatialAttention(7)

    def forward(self, pool_input, inputs):
        x = self.up(inputs)
        # print('up_x:', x.size())
        # print('pool_size:', pool_input.size())
        h_diff = pool_input.shape[3] - x.shape[3]
        w_diff = pool_input.shape[2] - x.shape[2]
        padd_size = (h_diff // 2, h_diff - h_diff // 2, w_diff // 2, w_diff - w_diff // 2)
        x = F.pad(x, padd_size)
        # print('pad_x:', x.size())
        pool_input = self.spec_att(pool_input)
        x = torch.cat((pool_input, x), 1)
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.spa_att(x)
        return x


class UNet(nn.Module):
    """docstring for ClassName"""

    def __init__(self, in_ch, num_classes):
        super(UNet, self).__init__()

        self.down1 = Encoder(in_channels=in_ch, out_channels=32)
        self.down2 = Encoder(in_channels=32, out_channels=64)
        self.mid_conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
        )
        self.middle = NonLocalBlock(in_channels=64)
        self.mid_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3), padding=1),
            # SpatialAttention(3)
        )
        self.up2 = Decoder(in_channels=128, out_channels=64)
        self.up1 = Decoder(in_channels=64, out_channels=32)
        self.last_conv = nn.Conv2d(32, num_classes, kernel_size=(1, 1))

    def forward(self, inputs, index):
        (h, w, c) = inputs.shape
        data = torch.unsqueeze(inputs.permute([2, 0, 1]), 0)
        # down_layers
        x1, x = self.down1(data)
        x2, x = self.down2(x)

        x = self.mid_conv1(x)
        x = self.middle(x)
        x = self.mid_conv2(x)

        x = self.up2(x2, x)
        x = self.up1(x1, x)
        x = self.last_conv(x)
        out = torch.squeeze(x).permute([1, 2, 0]).reshape([h * w, -1])
        score = out[index.tolist()]
        return score

class U_GC_LSTM(nn.Module):
    """docstring for LSTM_GCN"""

    def __init__(self, bands,pca_ch,num_classes):
        super(U_GC_LSTM, self).__init__()
        self.U_GCNet = UNet(pca_ch,num_classes)
        self.Mul_LSTM = Spectral(bands, num_classes)
        # self.gamma = nn.Parameter(torch.rand(1))

    def forward(self, x_spec, x, index):
        # gamma = torch.tanh(self.gamma)
        U_res = self.U_GCNet(x, index)
        #x_spec = self.attention(x_spec)
        #print(x_spec.size())
        Mul_LSTM_res = self.Mul_LSTM(x_spec)
        # score = gamma*U_res + (1-gamma)*Mul_LSTM_res
        score = U_res +  Mul_LSTM_res
        #return [score,Mul_LSTM_res,U_res]
        return score


