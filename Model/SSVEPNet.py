# Designer:Pan YuDong
# Coder:God's hand
# Time:2022/3/20 13:41
import torch
from torch import nn
from Utils import Constraint
import torch.fft

#创建AFFB模块
class AFFB(nn.Module):
    def __init__(self, channels, T, num_filters=3):
        super(AFFB, self).__init__()
        self.num_filters = num_filters
        self.channels = channels
        self.T = T

        # 定义可学习的频域掩码参数 (Filters, Channels, Half_T)
        # 初始化为1，表示全通。训练过程中模型会学习将其某些频点变为0（滤波）
        self.freq_mask = nn.Parameter(torch.ones(num_filters, 1, 1, T // 2 + 1))

    def forward(self, x):
        # x shape: [Batch, 1, Channels, T]

        # 1. 快速傅里叶变换到频域
        x_fft = torch.fft.rfft(x, dim=-1)  # 输出 shape: [Batch, 1, Channels, T//2 + 1]

        # 2. 应用多个并行的自适应滤波器
        # 扩展维度进行广播相乘
        out_list = []
        for i in range(self.num_filters):
            mask = torch.sigmoid(self.freq_mask[i])  # 使用sigmoid限制增益在0-1之间
            filtered_fft = x_fft * mask
            # 3. 逆变换回时域
            out_list.append(torch.fft.irfft(filtered_fft, n=self.T, dim=-1))

        # 4. 将多个滤波后的分支拼接或相加
        # 这里建议拼接后通过一个1x1卷积降维，或者直接相加
        x_filtered = torch.mean(torch.stack(out_list, dim=1), dim=1)
        return x_filtered  # 保持原始 shape: [Batch, 1, Channels, T]

#创建CBAM模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 改为 1D 的自适应池化
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # 用 1D 卷积或者全连接层做 MLP
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [Batch, Channel, Time]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 确保 kernel_size 是奇数，保证 padding 后尺寸不变
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2

        # 改为 1D 卷积，处理时间轴
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [Batch, Channel, Time]
        # 在通道维度 (dim=1) 做池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)  # [Batch, 2, Time]
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # 1. Channel Attention
        out = x * self.ca(x)
        # 2. Spatial (Temporal) Attention
        result = out * self.sa(out)
        return result
class LSTM(nn.Module):
    '''
        Employ the Bi-LSTM to learn the reliable dependency between spatio-temporal features
    '''
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, num_layers=1)

    def forward(self, x):
        b, c, T = x.size()
        x = x.view(x.size(-1), -1, c)  # (b, c, T) -> (T, b, c)
        r_out, _ = self.rnn(x)  # r_out shape [time_step * 2, batch_size, output_size]
        out = r_out.view(b, 2 * T * c, -1)
        return out

#增加：SELayer
class SELayer(nn.Module):

    def __init__(self, channel, reduction=2, batch_first=True):
        super(SELayer, self).__init__()

        self.batch_first = batch_first
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        input.size == output.size
        '''
        if not self.batch_first:
            x = x.permute(1, 0, 2)

        b, c, _, = x.size()
        y = self.avg_pool(x).view(b, c)  # size = (batch,channel)

        y = self.fc(y).view(b, c, 1)  # size = (batch,channel,1,1)
        out = x * y.expand_as(x)  # size = (batch,channel,w,h)

        if not self.batch_first:
            out = out.permute(1, 0, 2)  # size = (channel,batch,w,h)

        return out

class ESNet(nn.Module):
    def calculateOutSize(self, model, nChan, nTime):
        '''
            Calculate the output based on input size
            model is from nn.Module and inputSize is a array
        '''
        data = torch.randn(1, 1, nChan, nTime)
        out = model(data).shape
        return out[1:]

    def spatial_block(self, nChan, dropout_level):
        '''
           Spatial filter block,assign different weight to different channels and fuse them
        '''
        block = []
        block.append(Constraint.Conv2dWithConstraint(in_channels=1, out_channels=nChan * 2, kernel_size=(nChan, 1),
                                                     max_norm=1.0))
        block.append(nn.BatchNorm2d(num_features=nChan * 2))
        block.append(nn.PReLU())
        block.append(nn.Dropout(dropout_level))
        layer = nn.Sequential(*block)
        return layer

    def enhanced_block(self, in_channels, out_channels, dropout_level, kernel_size, stride):
        '''
           Enhanced structure block,build a CNN block to absorb data and output its stable feature
        '''
        block = []
        block.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                               stride=(1, stride)))
        block.append(nn.BatchNorm2d(num_features=out_channels))
        block.append(nn.PReLU())
        block.append(nn.Dropout(dropout_level))
        layer = nn.Sequential(*block)
        return layer

    def __init__(self, num_channels, T, num_classes):
        super(ESNet, self).__init__()
        self.dropout_level = 0.5

        #修改：加入AFFB
        self.affb = AFFB(channels=num_channels, T=T, num_filters=3)
        #

        self.F = [num_channels * 2] + [num_channels * 4]
        self.K = 10
        self.S = 2

        net = []
        net.append(self.spatial_block(num_channels, self.dropout_level))
        net.append(self.enhanced_block(self.F[0], self.F[1], self.dropout_level,
                                           self.K, self.S))

        self.conv_layers = nn.Sequential(*net)

        #修改：增加SE
        #self.se = SELayer(channel=self.F[1], reduction=2)
        #

        #把原来的SE变成CBAM
        #
        feature_dim = num_channels * 4
        self.cbam = CBAM(planes=feature_dim, ratio=16, kernel_size=7)
        #

        self.fcSize = self.calculateOutSize(self.conv_layers, num_channels, T)
        self.fcUnit = self.fcSize[0] * self.fcSize[1] * self.fcSize[2] * 2
        self.D1 = self.fcUnit // 10
        self.D2 = self.D1 // 5

        self.rnn = LSTM(input_size=self.F[1], hidden_size=self.F[1])

        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fcUnit, self.D1),
            nn.PReLU(),
            nn.Linear(self.D1, self.D2),
            nn.PReLU(),
            nn.Dropout(self.dropout_level),
            nn.Linear(self.D2, num_classes))

    def forward(self, x):
        # --- DEBUG START: 打印输入形状 ---
        #print(f"\n[DEBUG] Input x shape: {x.shape}")
        #修改：第一步进行自适应滤波
        x = self.affb(x)
        #print(f"[DEBUG] After AFFB: {x.shape}")

        out = self.conv_layers(x)
        out = out.squeeze(2)
        #增加：se
        #out = self.se(out)
        #应用 CBAM
        # --- DEBUG: 检查进入 CBAM 前的形状 ---
        #print(f"[DEBUG] Before CBAM: {out.shape}")
        # 这里的 Channels 维度必须等于 256！
        out = self.cbam(out)
        #print(f"[DEBUG] After CBAM: {out.shape}")
        r_out = self.rnn(out)
        out = self.dense_layers(r_out)
        return out
