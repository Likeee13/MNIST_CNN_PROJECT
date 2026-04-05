import torch.nn as nn
import torch.nn.functional as F
from config import cfg

class Net(nn.Module):
    def __init__(self):

        super().__init__()
        """
        初始化父类。nn.Module 是所有神经网络模块的基类。调用它的初始化方法，是为了在 Net 内部建立一套及其复杂的追踪机制。
        它会在后台创建几个字典 (比如 _parameters, _modules等)。当我们写下 self.conv1 = nn.Conv2d(...) 时，因为我们继承并初始化了父类，
        PyTorch 会在后台自动把 conv1 里的权重 (weights) 和偏置 (biases) 注册到这些字典里。
        如果没有这一步，网络就没有"记忆"，后续优化器 (Optimizer) 就找不到需要更新的参数，反向传播也就无法进行了。

        """

        # 第一个卷积层
        self.conv1 = nn.Conv2d(1, 10, kernel_size = cfg.kernel_size)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(10, 20, kernel_size = cfg.kernel_size)

        # Dropout 随机失活层
        self.conv2_drop = nn.Dropout2d()

        if cfg.kernel_size == 5:
            self.flatten_dim = 320
        elif cfg.kernel_size == 3:
            self.flatten_dim = 500
        else:
            raise ValueError("为了不把数学算太复杂，目前仅支持 kernel_size = 3 或 5")

        # 第一个全连接层
        self.fc1 = nn.Linear(self.flatten_dim, 50)

        # 第二个全连接层
        self.fc2 = nn.Linear(50, 10)



    # 神经网络架构
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # 展平
        x = x.view(-1, self.flatten_dim)

        x = F.relu(self.fc1(x))

        # 随机把神经元中的某些值变成0，防止过拟合
        x = F.dropout(x, training=self.training)

        x = self.fc2(x)

        # 将预测的可能性数值归一化
        return F.log_softmax(x, dim=1)
