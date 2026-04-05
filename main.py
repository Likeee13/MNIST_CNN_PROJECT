import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time

from config import cfg
from model import Net
from data_loader import get_train_loader, get_test_loader
from train import train_one_epoch
from evaluate import evaluate


def main():
    print("正在准备数据...")
    train_loader = get_train_loader()
    test_loader = get_test_loader()

    print("正在初始化网络和优化器...")
    network = Net()

    # 实例化优化器(SGD 随机梯度下降)
    # 把网络里的所有参数 network.parameters() 交给优化器保管
    optimizer = optim.SGD(network.parameters(), lr = cfg.learning_rate, momentum = cfg.momentum)

    # 准备画图用的数据收集器
    train_losses = []
    train_counter = []  # 记录蓝线对应的横坐标
    test_losses = []
    test_counter = []   # 记录红点对应的横坐标

    # 拿到训练集的总图片数（MNIST 是 60000）
    train_dataset_size = len(train_loader.dataset)

    # 在正式训练之前，先看看神经网络盲猜能有多少准确率
    print("【初始状态评估】")
    init_test_loss, init_test_acc = evaluate(network, test_loader)
    test_losses.append(init_test_loss)
    test_counter.append(0)

    start_time = time.time()

    print("开始正式训练...")
    for epoch in range(1, cfg.n_epochs + 1):
        epoch_losses = train_one_epoch(network, optimizer, train_loader, epoch)
        train_losses.extend(epoch_losses)

        # 计算蓝线的横坐标
        for i in range(len(epoch_losses)):
            images_seen_before = (epoch - 1) * train_dataset_size
            images_seen_now = i * cfg.log_interval * cfg.batch_size_train
            train_counter.append(images_seen_before + images_seen_now)

        final_test_loss, final_test_acc = evaluate(network, test_loader)
        test_losses.append(final_test_loss)
        test_counter.append(epoch * train_dataset_size)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"总训练时间: {total_time:.4f} 秒")

    # 训练结束后，保存模型参数
    torch.save(network.state_dict(), cfg.model_save_path)
    print(f"模型已保存至: {cfg.model_save_path}")

    # 实验记录与画图
    result_dir = "./results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 写入文本日志
    log_path = os.path.join(result_dir, "experiment_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        log_text = (f"Epochs: {cfg.n_epochs} | LR: {cfg.learning_rate} | Kernel: {cfg.kernel_size} "
                    f"==> Test Loss: {final_test_loss:.4f}, Accuracy: {final_test_acc:.2f}%, Total_trainging_time:{total_time:.4f}\n")
        f.write(log_text)
    print("✅ 实验数据已成功追加写入 experiment_log.txt")


    # 画出本次实验的 loss 曲线图
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_counter, train_losses, color='blue', label='Train Loss')
    plt.scatter(test_counter, test_losses, color='red', label='Test Loss', zorder=5)
    
    plt.title(f"Training and Test Loss Curve\n"
              f"(Epochs: {cfg.n_epochs}, LR: {cfg.learning_rate}, Kernel: {cfg.kernel_size})", 
              fontsize=14)
              
    plt.xlabel("number of training examples seen")
    plt.ylabel("negative log likelihood loss")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plot_filename = f"loss_E{cfg.n_epochs}_LR{cfg.learning_rate}_K{cfg.kernel_size}.png"
    plot_path = os.path.join(result_dir, plot_filename)
    plt.savefig(plot_path)
    print(f"✅ Loss 曲线图已成功保存为 {plot_filename}")

    # 画出6张预测图
    examples = enumerate(test_loader)
    batch_idx, (example_images, example_labels) = next(examples)

    with torch.no_grad():
        output = network(example_images)

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_images[i][0], cmap="gray", interpolation="none")
        plt.title(f"Prediction:{output.max(1, keepdim=True)[1][i].item()}")
        plt.xticks()
        plt.yticks()

    plt.show()

if __name__ == "__main__":
    main()