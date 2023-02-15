import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        assert kernel_size in (1, 3, 5, 7), 'kernel size must be 1 or 3 or 5 or 7'
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


# MLP注意力
class View_Attention(nn.Module):
    def __init__(self, channel, dim):
        super(View_Attention, self).__init__()

        self.conv1 = nn.Conv1d(channel, channel, 1)
        self.mid = dim // 4
        self.linear_0 = nn.Conv1d(channel, self.mid, 1, bias=False)
        self.linear_1 = nn.Conv1d(self.mid, channel, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)
        self.conv2 = nn.Sequential(
            nn.Conv1d(channel, channel, 1, bias=False),
            nn.LayerNorm([channel, dim])
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        idn = x
        x = self.conv1(x)
        b, c, w = x.size()
        attn = self.linear_0(x)
        attn = F.softmax(attn, dim=-1)
        x = self.linear_1(attn)
        x = self.conv2(x)
        x = F.relu(x)
        return x


class GlobalContextlBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(GlobalContextlBlock, self).__init__()

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
        self.g2 = nn.Conv2d(in_channels=self.inter_channels, out_channels=1, kernel_size=3, padding=1)
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

    def __init__(self, in_channels, out_channels, kernel1, kernel2):
        super(Encoder, self).__init__()
        kernel_1 = kernel1
        kernel_2 = kernel2
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

    def __init__(self, in_channels, out_channels, kernel1, kernel2):
        super(Decoder, self).__init__()
        kernel_1 = kernel1
        kernel_2 = kernel2
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=1)
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
        h_diff = pool_input.shape[3] - x.shape[3]
        w_diff = pool_input.shape[2] - x.shape[2]
        padd_size = (h_diff // 2, h_diff - h_diff // 2, w_diff // 2, w_diff - w_diff // 2)
        x = F.pad(x, padd_size)
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

        self.down1 = Encoder(in_channels=in_ch, out_channels=32, kernel1=7, kernel2=5)
        self.down2 = Encoder(in_channels=32, out_channels=64, kernel1=7, kernel2=5)
        self.mid_conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
        )
        self.middle = GlobalContextlBlock(in_channels=64)
        self.mid_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),

        )
        self.up2 = Decoder(in_channels=128, out_channels=64,kernel1=5,kernel2=7)
        self.up1 = Decoder(in_channels=64, out_channels=32,kernel1=5,kernel2=7)
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


def trans(kernel=3, strd=2):
    trans = nn.Sequential(
        nn.Conv1d(1, 1, kernel_size=kernel, stride=strd, padding=(kernel - 1) // 2, bias=False),
        nn.Softmax(dim=-1)
    )
    return trans


class bwd_LSTM(nn.Module):
    """docstring for ClassName"""
    '''Long Memory'''

    def __init__(self, input_dim, output_dim):
        super(bwd_LSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.cell_1 = nn.LSTMCell(input_size=input_dim, hidden_size=output_dim)  # output_dim     *1
        self.cell_2 = nn.LSTMCell(input_size=input_dim // 2, hidden_size=output_dim // 2)  # output_dim//2  *2
        self.cell_3 = nn.LSTMCell(input_size=input_dim // 4, hidden_size=output_dim // 4)  # output_dim//4  *4
        self.cell_4 = nn.LSTMCell(input_size=input_dim // 8, hidden_size=output_dim // 8)  # output_dim//8  *8

        self.trans2_1h = trans(3, 1)
        self.trans2_1c = trans(3, 1)
        self.trans3_2h = trans(3, 1)
        self.trans3_2c = trans(3, 1)
        self.trans4_3h = trans(3, 1)
        self.trans4_3c = trans(3, 1)

        self.fc = nn.Linear(output_dim * 4, output_dim)

    def forward(self, spec):
        x1 = spec.unsqueeze(-1).contiguous()
        x2 = torch.zeros(spec.shape[0], self.input_dim // 2, 2).cuda()
        x3 = torch.zeros(spec.shape[0], self.input_dim // 4, 4).cuda()
        x4 = torch.zeros(spec.shape[0], self.input_dim // 8, 8).cuda()
        start, end = 0, self.input_dim // 2
        for i in range(2):
            x2[:, :, i] = spec[:, start:end]
            start = end
            end += self.input_dim // 2

        start, end = 0, self.input_dim // 4
        for i in range(4):
            x3[:, :, i] = spec[:, start:end]
            start = end
            end += self.input_dim // 4

        start, end = 0, self.input_dim // 8
        for i in range(8):
            x4[:, :, i] = spec[:, start:end]
            start = end
            end += self.input_dim // 8

        out1 = []
        out2 = []
        out3 = []
        out4 = []

        hx_backup = []
        cx_backup = []

        out_temp_hx = []
        out_temp_cx = []
        for i in range(x4.shape[2]):
            hx4, cx4 = self.cell_4(x4[:, :, i])
            out_temp_hx.append(hx4)
            out_temp_cx.append(cx4)
            if i % 2 == 1:
                temp_hx = self.trans4_3h(torch.cat(out_temp_hx, dim=-1).unsqueeze(1).contiguous()).squeeze()
                temp_cx = self.trans4_3c(torch.cat(out_temp_cx, dim=-1).unsqueeze(1).contiguous()).squeeze()
                hx_backup.append(temp_hx)
                cx_backup.append(temp_cx)
                out_temp_hx.clear()
                out_temp_cx.clear()
            out4.append(hx4)
        for i in range(x3.shape[2]):
            hx3, cx3 = self.cell_3(x3[:, :, i], (hx_backup[i], cx_backup[i]))
            out_temp_hx.append(hx3)
            out_temp_cx.append(cx3)
            if i % 2 == 1:
                temp_hx = self.trans3_2h(torch.cat(out_temp_hx, dim=-1).unsqueeze(1).contiguous()).squeeze()
                temp_cx = self.trans3_2c(torch.cat(out_temp_cx, dim=-1).unsqueeze(1).contiguous()).squeeze()
                hx_backup.append(temp_hx)
                cx_backup.append(temp_cx)
                out_temp_hx.clear()
                out_temp_cx.clear()
            out3.append(hx3)
        del hx_backup[:x4.shape[2] // 2]
        del cx_backup[:x4.shape[2] // 2]

        for i in range(x2.shape[2]):
            hx2, cx2 = self.cell_2(x2[:, :, i], (hx_backup[i], cx_backup[i]))
            out_temp_hx.append(hx2)
            out_temp_cx.append(cx2)
            if i % 2 == 1:
                temp_hx = self.trans2_1h(torch.cat(out_temp_hx, dim=-1).unsqueeze(1).contiguous()).squeeze()
                temp_cx = self.trans2_1c(torch.cat(out_temp_cx, dim=-1).unsqueeze(1).contiguous()).squeeze()
                hx_backup.append(temp_hx)
                cx_backup.append(temp_cx)
                out_temp_hx.clear()
                out_temp_cx.clear()
            out2.append(hx2)
        del hx_backup[:x3.shape[2] // 2]
        del cx_backup[:x3.shape[2] // 2]

        for i in range(x1.shape[2]):
            hx1, cx1 = self.cell_1(x1[:, :, i], (hx_backup[i], cx_backup[i]))
            out1.append(cx1)

        del hx_backup[:x2.shape[2] // 2]
        del cx_backup[:x2.shape[2] // 2]
        out1 = torch.cat(out1, dim=-1)
        out2 = torch.cat(out2, dim=-1)
        out3 = torch.cat(out3, dim=-1)
        out4 = torch.cat(out4, dim=-1)
        return out1


class fwd_LSTM(nn.Module):
    """docstring for fwd_LSTM"""
    '''Short Memory'''

    def __init__(self, input_dim, output_dim):
        super(fwd_LSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.cell_1 = nn.LSTMCell(input_size=input_dim, hidden_size=output_dim)  # output_dim     *1
        self.cell_2 = nn.LSTMCell(input_size=input_dim // 2, hidden_size=output_dim // 2)  # output_dim//2  *2
        self.cell_3 = nn.LSTMCell(input_size=input_dim // 4, hidden_size=output_dim // 4)  # output_dim//4  *4
        self.cell_4 = nn.LSTMCell(input_size=input_dim // 8, hidden_size=output_dim // 8)  # output_dim//8  *8

        self.trans1_2h = trans(3, 2)
        self.trans1_2c = trans(3, 2)
        self.trans2_3h = trans(3, 2)
        self.trans2_3c = trans(3, 2)
        self.trans3_4h = trans(3, 2)
        self.trans3_4c = trans(3, 2)

    def forward(self, spec):
        x1 = spec.unsqueeze(-1).contiguous()
        x2 = torch.zeros(spec.shape[0], self.input_dim // 2, 2).cuda()
        x3 = torch.zeros(spec.shape[0], self.input_dim // 4, 4).cuda()
        x4 = torch.zeros(spec.shape[0], self.input_dim // 8, 8).cuda()
        start, end = 0, self.input_dim // 2
        for i in range(2):
            x2[:, :, i] = spec[:, start:end]
            start = end
            end += self.input_dim // 2

        start, end = 0, self.input_dim // 4
        for i in range(4):
            x3[:, :, i] = spec[:, start:end]
            start = end
            end += self.input_dim // 4

        start, end = 0, self.input_dim // 8
        for i in range(8):
            x4[:, :, i] = spec[:, start:end]
            start = end
            end += self.input_dim // 8

        out1 = []
        out2 = []
        out3 = []
        out4 = []

        hx_backup = []
        cx_backup = []

        for i in range(x1.shape[2]):
            hx1, cx1 = self.cell_1(x1[:, :, i])  # [b,h,1]
            temp_hx = self.trans1_2h(hx1.unsqueeze(1).contiguous()).squeeze()
            temp_cx = self.trans1_2c(cx1.unsqueeze(1).contiguous()).squeeze()
            hx_backup.append(temp_hx)
            cx_backup.append(temp_cx)
            out1.append(cx1)
        for i in range(x2.shape[2]):  # [b,h,2]
            index = i // 2
            hx2, cx2 = self.cell_2(x2[:, :, i], (hx_backup[index], cx_backup[index]))
            temp_hx = self.trans2_3h(hx2.unsqueeze(1).contiguous()).squeeze()
            temp_cx = self.trans2_3c(cx2.unsqueeze(1).contiguous()).squeeze()
            hx_backup.append(temp_hx)
            cx_backup.append(temp_cx)
            out2.append(cx2)
        del hx_backup[:x1.shape[2]]  # hx_backup：[t(hx2),t(hx2)]
        del cx_backup[:x1.shape[2]]

        for i in range(x3.shape[2]):  # [b,h,4]
            index = i // 2
            hx3, cx3 = self.cell_3(x3[:, :, i], (hx_backup[index], cx_backup[index]))
            temp_hx = self.trans3_4h(hx3.unsqueeze(1).contiguous()).squeeze()
            temp_cx = self.trans3_4h(cx3.unsqueeze(1).contiguous()).squeeze()
            hx_backup.append(temp_hx)
            cx_backup.append(temp_cx)
            out3.append(cx3)
        del hx_backup[:x2.shape[2]]  # hx_backup：[t(hx2),t(hx2)]
        del cx_backup[:x2.shape[2]]

        for i in range(x4.shape[2]):  # [b,h,8]
            index = i // 2
            hx4, cx4 = self.cell_4(x4[:, :, i], (hx_backup[index], cx_backup[index]))
            out4.append(hx4)
        del hx_backup[:x3.shape[2]]  # hx_backup：[t(hx2),t(hx2)]
        del cx_backup[:x3.shape[2]]

        out1 = torch.cat(out1, dim=-1)
        out2 = torch.cat(out2, dim=-1)
        out3 = torch.cat(out3, dim=-1)
        out4 = torch.cat(out4, dim=-1)
        return out4


class Spectral(nn.Module):
    """docstring for ClassName"""

    def __init__(self, bands, class_count):
        super(Spectral, self).__init__()
        self.bands = bands
        self.class_count = class_count

        self.down_ch = 128
        self.pre = nn.Linear(bands, self.down_ch)
        self.pre_conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )
        self.recent = fwd_LSTM(self.down_ch // 1, self.down_ch // 2)
        self.longterm = bwd_LSTM(self.down_ch // 1, self.down_ch // 2)
        self.LSTM = nn.LSTM(self.down_ch // 1, self.down_ch // 2, num_layers=1, dropout=0, batch_first=True)
        self.Attention_hx = View_Attention(1, self.down_ch // 2, )
        self.Attention_cx = View_Attention(1, self.down_ch // 2, )
        self.midhx_conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )
        self.midcx_conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

        self.fc = nn.Linear(self.down_ch // 2, class_count)

    def forward(self, spec):
        x = self.pre(spec)  # [batch,128]
        # x = self.pre_conv(x.unsqueeze(1).contiguous()).squeeze()
        hx = self.recent(x).unsqueeze(1)  # [batch,1,64]
        cx = self.longterm(x).unsqueeze(1)

        hx = self.Attention_hx(hx).permute(1, 0, 2)
        cx = self.Attention_hx(cx).permute(1, 0, 2)
        out, (hx_out, cx_out) = self.LSTM(x.unsqueeze(1), (cx, hx))
        score = self.fc(out.squeeze())
        return score


class DSGSF(nn.Module):
    """docstring for LSTM_GCN"""

    def __init__(self, bands, pca_ch, num_classes):
        super(DSGSF, self).__init__()
        self.U_GCNet = UNet(pca_ch, num_classes)
        self.Mul_LSTM = Spectral(bands, num_classes)
        self.gamma = nn.Parameter(torch.rand(1))

        # self.LSTM = nn.LSTM(bands,bands//4,num_layers=2,batch_first=True,bidirectional=True)
        # self.fc = nn.Linear((bands//4)*2,num_classes)

    def forward(self, x_spec, x, index):
        gamma = torch.tanh(self.gamma)
        U_res = self.U_GCNet(x, index)
        Mul_LSTM_res = self.Mul_LSTM(x_spec)
        score = gamma * U_res + (1 - gamma) * Mul_LSTM_res
        #score = U_res + Mul_LSTM_res
        return score
