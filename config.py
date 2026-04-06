import torch
from dataclasses import dataclass

@dataclass
class Config:
    # 训练相关的超参数
    n_epochs: int = 3              # 训练轮数
    batch_size_train: int = 64     # 训练批大小
    batch_size_test: int = 1000    # 测试批大小
    learning_rate: float = 0.01    # 学习率
    momentum: float = 0.5          # 动量
    kernel_size: int = 3           # 卷积核大小

    # 系统配置
    log_interval: int = 10         # 打印日志的间隔
    random_seed: int = 1           # 随机种子，保证实验可复现

    # 路径配置
    data_dir: str = './data/'      # 数据集存放路径
    model_save_path: str = './model.pth'   # 模型保存路径


cfg = Config()

# 固定随机种子
torch.manual_seed(cfg.random_seed)
 