import torch
import torch.nn.functional as F
from config import cfg

def train_one_epoch(network, optimizer, train_loader, epoch):
    # 把模型设置为训练模式，这会激活模型中的 Dropout 层
    network.train()

    # 记录损失，方便后续画图
    train_losses = []

    for batch_idx, (images, labels) in enumerate(train_loader):

        # 步骤 1：清空历史梯度 (PyTorch默认会累加梯度，每次循环必须清零)
        optimizer.zero_grad()

        # 步骤 2：前向传播，得到模型的输出结果
        output = network(images)

        # 步骤 3：计算损失
        loss = F.nll_loss(output, labels)

        # 步骤 4：反向传播，计算网络中每个权重的梯度
        loss.backward()

        # 步骤 5：更新权重。优化器根据刚刚算出来的梯度，微调网络中的参数
        optimizer.step()

        # 打印日志
        if batch_idx % cfg.log_interval == 0:
            print(f"Train Epoch:{epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)}" 
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss:{loss.item():.6f}")
            
            train_losses.append(loss.item())

    return train_losses