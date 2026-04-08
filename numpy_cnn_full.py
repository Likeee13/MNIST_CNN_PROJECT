import numpy as np
import struct
import os

# ================= 1. 数据集加载与处理 =================
def load_mnist_images(filepath):
    with open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, 1, rows, cols)
        return images.astype(np.float32) / 255.0

def load_mnist_labels(filepath):
    with open(filepath, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

def to_one_hot(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


# ================= 2. 激活函数与损失 =================
class ReLU:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x > 0)
        return np.maximum(0, x)
    def backward(self, dout):
        return dout * self.mask

class SoftmaxCrossEntropy:
    def __init__(self):
        self.y_pred, self.y_true = None, None
    def forward(self, x, y_true):
        self.y_true = y_true
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.y_pred = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return -np.sum(y_true * np.log(self.y_pred + 1e-7)) / x.shape[0]
    def backward(self):
        return (self.y_pred - self.y_true) / self.y_true.shape[0]


# ================= 3. 神经网络骨架 =================
class Linear:
    def __init__(self, in_features, out_features):
        # Kaiming 初始化，防止梯度消失
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout, lr=0.01):
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T)
        self.W -= lr * dW
        self.b -= lr * db
        return dx

class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size, self.stride = pool_size, stride
        self.x, self.max_indices = None, None

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        out_H, out_W = H // self.stride, W // self.stride
        out = np.zeros((N, C, out_H, out_W))
        self.max_indices = np.zeros_like(x)
        
        for h in range(out_H):
            for w in range(out_W):
                h_start, w_start = h * self.stride, w * self.stride
                h_end, w_end = h_start + self.pool_size, w_start + self.pool_size
                window = x[:, :, h_start:h_end, w_start:w_end]
                # 提取窗口最大值
                out[:, :, h, w] = np.max(window, axis=(2, 3))
                
                # 记录最大值位置用于反向传播
                for n in range(N):
                    for c in range(C):
                        win = window[n, c]
                        r, col = np.unravel_index(np.argmax(win), win.shape)
                        self.max_indices[n, c, h_start+r, w_start+col] = 1
        return out

    def backward(self, dout, lr=None):
        dx = np.zeros_like(self.x)
        N, C, out_H, out_W = dout.shape
        for h in range(out_H):
            for w in range(out_W):
                h_start, w_start = h * self.stride, w * self.stride
                h_end, w_end = h_start + self.pool_size, w_start + self.pool_size
                # 只有最大值的位置才有梯度回传
                for n in range(N):
                    for c in range(C):
                        dx[n, c, h_start:h_end, w_start:w_end] += \
                            dout[n, c, h, w] * self.max_indices[n, c, h_start:h_end, w_start:w_end]
        return dx

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = kernel_size
        # Kaiming 初始化
        self.W = np.random.randn(out_channels, in_channels, self.k, self.k) * np.sqrt(2.0 / (in_channels * self.k * self.k))
        self.b = np.zeros(out_channels)
        self.x = None

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        out_H, out_W = H - self.k + 1, W - self.k + 1
        out = np.zeros((N, self.out_channels, out_H, out_W))
        
        # 使用张量收缩 (Tensor Dot) 代替内层极慢的 for 循环
        for h in range(out_H):
            for w in range(out_W):
                x_slice = x[:, :, h:h+self.k, w:w+self.k] # shape: (N, C_in, K, K)
                # 矩阵相乘累加：将输入切片与权重在 C_in, K, K 三个维度上进行点积
                out[:, :, h, w] = np.tensordot(x_slice, self.W, axes=([1, 2, 3], [1, 2, 3])) + self.b
        return out

    def backward(self, dout, lr=0.01):
        N, C, H, W = self.x.shape
        _, _, out_H, out_W = dout.shape
        dW = np.zeros_like(self.W)
        db = np.sum(dout, axis=(0, 2, 3))
        dx = np.zeros_like(self.x)
        
        for h in range(out_H):
            for w in range(out_W):
                x_slice = self.x[:, :, h:h+self.k, w:w+self.k] # (N, C_in, K, K)
                dout_slice = dout[:, :, h, w]                  # (N, C_out)
                
                # 反向传播核心微积分推导的张量化实现
                dW += np.tensordot(dout_slice, x_slice, axes=([0], [0]))
                dx[:, :, h:h+self.k, w:w+self.k] += np.tensordot(dout_slice, self.W, axes=([1], [0]))
                
        self.W -= lr * dW
        self.b -= lr * db
        return dx

class Flatten:
    def __init__(self):
        self.x_shape = None
    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(x.shape[0], -1)
    def backward(self, dout, lr=None):
        return dout.reshape(self.x_shape)


# ================= 4. 组装与训练流水线 =================
def main():
    print("1. [NumPy底层引擎] 正在加载 MNIST 数据集...")
    train_images = load_mnist_images('./data/MNIST/raw/train-images-idx3-ubyte')
    train_labels = load_mnist_labels('./data/MNIST/raw/train-labels-idx1-ubyte')
    test_images = load_mnist_images('./data/MNIST/raw/t10k-images-idx3-ubyte')
    test_labels = load_mnist_labels('./data/MNIST/raw/t10k-labels-idx1-ubyte')

    print("2. [NumPy底层引擎] 截取 1000 张图像进行小样本训练验证...")
    X_train, Y_train = train_images[:1000], train_labels[:1000]
    X_test, Y_test = test_images[:200], test_labels[:200]
    Y_train_onehot = to_one_hot(Y_train)

    print("3. [NumPy底层引擎] 正在组装双层卷积网络...")
    print("   架构: Conv(1,10,5) -> MaxPool -> Conv(10,20,5) -> MaxPool -> Flatten -> FC(320,50) -> FC(50,10)")


    conv1 = Conv2D(in_channels=1, out_channels=10, kernel_size=5)
    relu1 = ReLU()
    pool1 = MaxPool2D(pool_size=2, stride=2)
    
    conv2 = Conv2D(in_channels=10, out_channels=20, kernel_size=5)
    relu2 = ReLU()
    pool2 = MaxPool2D(pool_size=2, stride=2)
    
    flatten = Flatten()
    fc1 = Linear(320, 50)
    relu3 = ReLU()
    fc2 = Linear(50, 10)
    
    loss_fn = SoftmaxCrossEntropy()

    epochs = 10
    batch_size = 32
    learning_rate = 0.05
    num_batches = len(X_train) // batch_size

    print("4. [NumPy底层引擎] 启动微积分反向传播链...")
    for epoch in range(epochs):
        epoch_loss = 0
        correct_train = 0
        
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = start_idx + batch_size
            
            x_batch = X_train[start_idx:end_idx]
            y_batch_onehot = Y_train_onehot[start_idx:end_idx]
            y_batch_real = Y_train[start_idx:end_idx]

            # ---- 前向传播工程 ----
            x = conv1.forward(x_batch)
            x = relu1.forward(x)
            x = pool1.forward(x)
            x = conv2.forward(x)
            x = relu2.forward(x)
            x = pool2.forward(x)
            x = flatten.forward(x)
            x = fc1.forward(x)
            x = relu3.forward(x)
            logits = fc2.forward(x)
            
            # 计算 Loss 与准确率
            loss = loss_fn.forward(logits, y_batch_onehot)
            epoch_loss += loss
            predictions = np.argmax(logits, axis=1)
            correct_train += np.sum(predictions == y_batch_real)

            # ---- 链式反向传播 ----
            dout = loss_fn.backward()
            dout = fc2.backward(dout, lr=learning_rate)
            dout = relu3.backward(dout)
            dout = fc1.backward(dout, lr=learning_rate)
            dout = flatten.backward(dout)
            dout = pool2.backward(dout)
            dout = relu2.backward(dout)
            dout = conv2.backward(dout, lr=learning_rate)
            dout = pool1.backward(dout)
            dout = relu1.backward(dout)
            dout = conv1.backward(dout, lr=learning_rate)
            
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {epoch_loss/num_batches:.4f} | Train Acc: {correct_train/len(X_train)*100:.2f}%")

    print("\n5. [NumPy底层引擎] 开始测试集评估...")
    # 测试集前向推理
    x = conv1.forward(X_test)
    x = relu1.forward(x)
    x = pool1.forward(x)
    x = conv2.forward(x)
    x = relu2.forward(x)
    x = pool2.forward(x)
    x = flatten.forward(x)
    x = fc1.forward(x)
    x = relu3.forward(x)
    test_logits = fc2.forward(x)
    
    test_preds = np.argmax(test_logits, axis=1)
    test_acc = np.sum(test_preds == Y_test) / len(Y_test) * 100
    print(f"✅ 手工 CNN 测试集准确率: {test_acc:.2f}%")

if __name__ == '__main__':
    main()