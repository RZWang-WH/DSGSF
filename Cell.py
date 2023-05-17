import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MLP_Attention(nn.Module):
    '''
	Arguments:
		c (int): The input and output channel number.
	'''

    def __init__(self, channel, dim, mid_dim=64):
        super(MLP_Attention, self).__init__()

        self.conv1 = nn.Conv1d(channel, channel, 1)
        self.mid = mid_dim
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

    def forward(self, x):  # [batch , channel , bands]
        idn = x
        x = self.conv1(x)  # [batch , channel , bands]

        b, c, w = x.size()  # [batch , channel , bands]

        attn = self.linear_0(x)  # [batch , kernal , bands]
        # 实际上这一步的1x1卷积就是将输入乘上一个(512x64)的可学习的卷积。
        attn = F.softmax(attn, dim=-1)  # [batch , kernal , bands]

        #attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # [batch , kernal , bands]
        ##进行两次norm的操作，即double-normalization

        x = self.linear_1(attn)  # [batch , chennel , bands]
        # 再进行第二次的矩阵乘法 从1*64*256重新变回1*512*256

        x = self.conv2(x)  # 卷积 [batch , chennel , bands]  -> [batch , chennel , bands]
        x = x + idn  # 与输入直接值相加  1*512*16*16 -> 1*512*16*16
        x = F.relu(x)
        return x


def trans(kernel=3, strd=2):
    trans = nn.Sequential(
        nn.Conv1d(1, 1, kernel_size=kernel, stride=strd, padding=(kernel - 1) // 2, bias=False),
        nn.Softmax(dim=-1)
    )
    return trans


class bwd_LSTM(nn.Module):
    """docstring for ClassName"""

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
            out1.append(hx1)

        del hx_backup[:x2.shape[2] // 2]
        del cx_backup[:x2.shape[2] // 2]
        out1 = torch.cat(out1, dim=-1)
        out2 = torch.cat(out2, dim=-1)
        out3 = torch.cat(out3, dim=-1)
        out4 = torch.cat(out4, dim=-1)
        return out1


class fwd_LSTM(nn.Module):
    """docstring for fwd_LSTM"""

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

        for i in range(x1.shape[2]):
            hx1, cx1 = self.cell_1(x1[:, :, i])  # [b,h,1]
            temp_hx = self.trans1_2h(hx1.unsqueeze(1).contiguous()).squeeze()
            temp_cx = self.trans1_2c(cx1.unsqueeze(1).contiguous()).squeeze()
            hx_backup.append(temp_hx)
            cx_backup.append(temp_cx)
            out1.append(cx1)

        #  hx_backup：[t(hx1)]
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
            out4.append(cx4)
        del hx_backup[:x3.shape[2]]  # hx_backup：[t(hx2),t(hx2)]
        del cx_backup[:x3.shape[2]]

        out1 = torch.cat(out1, dim=-1)
        out2 = torch.cat(out2, dim=-1)
        out3 = torch.cat(out3, dim=-1)
        out4 = torch.cat(out4, dim=-1)
        # out = torch.cat([out1, out2, out3, out4], dim=-1)
        # out = self.fc(out)
        return out4


##    隐状态输出： [64]  [32 * 3] [16*7] [8*15]


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
        self.Attention_hx = MLP_Attention(1, self.down_ch // 2, self.down_ch // 4)
        self.Attention_cx = MLP_Attention(1, self.down_ch // 2, self.down_ch // 4)
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
        x = self.pre_conv(x.unsqueeze(1).contiguous()).squeeze()
        hx = self.recent(x).unsqueeze(1)
        cx = self.longterm(x).unsqueeze(1)
        hx = self.midhx_conv(hx)
        cx = self.midcx_conv(cx)
        # hx = torch.cat(hx, dim=-1).unsqueeze(0)  # [num_layers * num_directions, batch, hidden_size]
        # cx = torch.cat(cx, dim=-1).unsqueeze(0)

        # out, (_, _) = self.LSTM(x.unsqueeze(1), (hx, cx))  # [ batch,seq_len, hidden_size]

        hx = self.Attention_hx(hx).permute(1, 0, 2)
        cx = self.Attention_hx(cx).permute(1, 0, 2)
        out, (hx_out, cx_out) = self.LSTM(x.unsqueeze(1), (cx, hx))
        # print(out.size())
        # print(hx_out.size())
        # print(cx_out.size())
        score = self.fc(out.squeeze())

        return score


model = MLP_Attention(16, 200)
model.eval()
print(list(model.named_children()))
image = torch.randn(32, 16, 200)
pred = model(image)
print(pred.size())

Net = Spectral(200, 16).cuda()
Net.eval()
spec_x = torch.rand(32, 200).cuda()
y = Net(spec_x)
print(y.size())
