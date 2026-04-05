import torch
import torch.nn.functional as F

def evaluate(network, test_loader):

    # 把模型设置为测试模式，会立即关闭 Dropout 层，使用全部的神经元预测结果
    network.eval()

    test_loss = 0
    correct = 0

    # torch.no_grad() 是一个上下文管理器，关闭梯度记录，关闭反向传播，节省显存和计算时间
    with torch.no_grad():
        for images, labels in test_loader:
            output = network(images)

            # 计算这一个 batch 的总损失
            test_loss += F.nll_loss(output, labels, reduction='sum').item()

            # 获取预测结果
            pred = output.max(1, keepdim=True)[1]

            # 对比预测结果和正确答案
            correct += pred.eq(labels.view_as(pred)).sum().item()

    # 计算平均损失
    test_loss /= len(test_loader.dataset)

    # 计算准确度
    accuracy = 100. * correct / len(test_loader.dataset)

    # 打印最终的测试结果
    print(f"\n Test set: Avg. loss: {test_loss:.4f}, "
          f"Accuracy: {correct} / {len(test_loader.dataset)}, "
          f"({accuracy:.2f}%)\n")
    
    return test_loss, accuracy