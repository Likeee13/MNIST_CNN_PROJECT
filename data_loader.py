import torch
import torchvision
from config import cfg
import matplotlib.pyplot as plt

# 定义数据预处理流程 (Transforms)
# 1. ToTensor(): 把灰度图(0-255)变成 PyTorch 能看懂的浮点数张量(0.0-1.0), 并且把维度调整为 [C, H, W]
# 2. Normalize(): 数据标准化。减去均值 0.1307, 除以标准差 0.3081
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,),(0.3081,))
])

# 准备训练集数据加载器
def get_train_loader():
    train_dataset = torchvision.datasets.MNIST(
        root = cfg.data_dir,
        train = True,
        download = True,
        transform = transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = cfg.batch_size_train,
        shuffle = True                             # 训练时打乱顺序，防止模型死记硬背
    )
    return train_loader

# 准备测试集数据加载器
def get_test_loader():
    test_dataset = torchvision.datasets.MNIST(
        root = cfg.data_dir,
        train = False,
        download = True,
        transform = transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = cfg.batch_size_test,
        shuffle = True              
    )
    return test_loader

if __name__ == "__main__":
    train_loader = get_train_loader()
    test_loader = get_test_loader()

    print(f"训练集包含{len(train_loader.dataset)}个样本")
    print(f"训练集包含{len(test_loader.dataset)}个样本\n")

    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"训练批次中图像张量的维度是 {images.shape}")
        print(f"训练批次中标签张量的维度是 {labels.shape}\n")

        fig = plt.figure()            # 创建图像窗口

        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(images[i][0], cmap="gray", interpolation="none")
            plt.title(f"Label:{labels[i]}")
            plt.xticks([])
            plt.yticks([])

        plt.show()

        break

    for batch_idx, (images, labels) in enumerate(test_loader):
        print(f"测试批次中图像张量的维度是 {images.shape}")
        print(f"测试批次中标签张量的维度是 {labels.shape}")
        break