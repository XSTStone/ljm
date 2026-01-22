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

#创建CTA模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        # 对应论文 Eq (7)
        # 针对 1D 时间序列: [Batch, Channel, Time]
        # Global Avg & Max Pooling 压缩 Time 维度
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # 共享 MLP (Shared Weight Multilayer Perceptron)
        # 论文提到: neuron = C -> C/r -> C
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [Batch, Channel, Time]
        # avg_out: [Batch, Channel, 1]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))

        # 对应论文 Eq (7) & (8): Wc(X) * X
        out = avg_out + max_out
        return x * self.sigmoid(out)


class TimeAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(TimeAttention, self).__init__()
        # 对应论文 Eq (9) & (10)
        # 论文 Table II 中提到 kernel size = 3x3 (针对2D输入)
        # 针对 1D 时间序列，我们使用 kernel size = 3 的 Conv1d
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2

        # 输入通道为2 (AvgPool + MaxPool 的结果拼接)
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [Batch, Channel, Time]

        # 沿着通道维度进行池化 -> [Batch, 1, Time]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接 -> [Batch, 2, Time]
        x_cat = torch.cat([avg_out, max_out], dim=1)

        # 卷积 + Sigmoid -> 生成时间权重
        scale = self.sigmoid(self.conv1(x_cat))

        # 对应论文 Eq (10): Ws(X) * X
        return x * scale


class CTA(nn.Module):
    '''
    Channel-Time Attention Module
    串行结构：先 Channel Attention 后 Time Attention
    '''

    def __init__(self, planes, ratio=2, kernel_size=3):
        super(CTA, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.ta = TimeAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.ta(x)
        return x
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

        # 1. AFFB 模块
        self.affb = AFFB(channels=num_channels, T=T, num_filters=3)

        self.F = [num_channels * 2] + [num_channels * 4]
        self.K = 10
        self.S = 2

        # 2. 卷积特征提取层
        net = []
        net.append(self.spatial_block(num_channels, self.dropout_level))
        net.append(self.enhanced_block(self.F[0], self.F[1], self.dropout_level, self.K, self.S))
        self.conv_layers = nn.Sequential(*net)

        # 3. 计算卷积层输出维度
        # calculateOutSize 返回的是 (Channels, Height, Time)，例如 (256, 1, 60)
        self.fcSize = self.calculateOutSize(self.conv_layers, num_channels, T)

        # 4. CTA 模块
        self.cta = CTA(planes=self.F[1], ratio=2, kernel_size=3)

        # 5. LSTM
        self.rnn = LSTM(input_size=self.F[1], hidden_size=self.F[1])

        # 6. 全连接分类层 (关键修改点！)
        # 修正维度计算：Time * Channel * 2
        # self.fcSize[2] 是时间维度(约60)，self.fcSize[0] 是通道维度(256)
        self.fcUnit = self.fcSize[2] * self.fcSize[0] * 2

        # 打印一下维度，确保正确 (调试用)
        print(f"Calculated fcUnit: {self.fcUnit} (Should be around 30720)")

        self.D1 = self.fcUnit // 10
        self.D2 = self.D1 // 5

        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fcUnit, self.D1),
            nn.PReLU(),
            nn.Linear(self.D1, self.D2),
            nn.PReLU(),
            nn.Dropout(self.dropout_level),
            nn.Linear(self.D2, num_classes)
        )

    def forward(self, x):
        #修改：第一步进行自适应滤波
        x = self.affb(x)

        out = self.conv_layers(x)
        out = out.squeeze(2)
        #增加：se
        #out = self.se(out)
        out = self.cta(out)
        r_out = self.rnn(out)
        out = self.dense_layers(r_out)
        return out
