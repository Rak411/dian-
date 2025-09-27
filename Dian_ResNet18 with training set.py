import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
from torchsummary import summary



# 设置设备（自动检测GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 如果有GPU，显示GPU信息
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)                                                
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,stride=stride,kernel_size=1,bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self,input):
        identity = input
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):  # 修改为10类，适应CIFAR-10
        super(ResNet18,self).__init__()
        # 针对CIFAR-10的小尺寸图像调整初始层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 移除maxpool，因为CIFAR-10图像太小
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layer = []
        layer.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layer.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layer)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 移除maxpool
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# 数据加载和预处理
def get_cifar10_dataloaders(batch_size=128):
    """获取CIFAR-10数据加载器"""
    # 数据增强和归一化
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 加载数据集
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    # 创建数据加载器
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

# 训练函数
def train_model(model, trainloader, testloader, epochs=100, learning_rate=0.1):
    """训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                         momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
    
    # 记录训练过程
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    learning_rates = []
    
    print("开始训练...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # 计算训练准确率
        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total
        
        # 测试阶段
        test_acc = test_model(model, testloader, device)
        
        # 记录数据
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f'Epoch [{epoch+1:3d}/{epochs}] | '
                  f'Loss: {train_loss:.4f} | '
                  f'Train Acc: {train_acc:.2f}% | '
                  f'Test Acc: {test_acc:.2f}% | '
                  f'LR: {learning_rates[-1]:.6f}')
        
        scheduler.step()
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f'训练完成! 总耗时: {training_time/60:.2f}分钟')
    
    return train_losses, train_accuracies, test_accuracies, learning_rates

def test_model(model, testloader, device):
    """测试模型准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100. * correct / total

# 可视化函数
def plot_training_results(train_losses, train_accuracies, test_accuracies, learning_rates):
    """绘制训练结果"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    ax1.plot(train_losses, 'b-', linewidth=2)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(train_accuracies, 'g-', label='Train Accuracy', linewidth=2)
    ax2.plot(test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_title('Training and Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 学习率变化
    ax3.plot(learning_rates, 'purple', linewidth=2)
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 最终准确率对比
    epochs = range(1, len(train_accuracies) + 1)
    ax4.bar(['Train', 'Test'], 
            [train_accuracies[-1], test_accuracies[-1]], 
            color=['green', 'red'], alpha=0.7)
    ax4.set_title('Final Accuracy Comparison')
    ax4.set_ylabel('Accuracy (%)')
    for i, v in enumerate([train_accuracies[-1], test_accuracies[-1]]):
        ax4.text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('resnet18_cifar10_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_model_summary(model):
    """打印模型摘要"""
    print("=" * 60)
    print("ResNet-18 模型结构")
    print("=" * 60)
    summary(model, (3, 32, 32))  # CIFAR-10图像尺寸为32x32
    print("=" * 60)

# 主函数
def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建模型
    model = ResNet18(num_classes=10)
    
    # 打印模型信息
    print_model_summary(model)
    
    # 获取数据
    trainloader, testloader, classes = get_cifar10_dataloaders(batch_size=128)
    
    print(f"训练集样本数: {len(trainloader.dataset)}")
    print(f"测试集样本数: {len(testloader.dataset)}")
    print(f"类别数: {len(classes)}")
    print(f"类别名称: {classes}")
    
    # 训练模型
    train_losses, train_accuracies, test_accuracies, learning_rates = train_model(
        model, trainloader, testloader, epochs=100, learning_rate=0.1)
    
    # 最终测试准确率
    final_test_acc = test_accuracies[-1]
    final_train_acc = train_accuracies[-1]
    
    print("\n" + "=" * 60)
    print("训练结果总结")
    print("=" * 60)
    print(f"最终训练准确率: {final_train_acc:.2f}%")
    print(f"最终测试准确率: {final_test_acc:.2f}%")
    print(f"过拟合程度: {final_train_acc - final_test_acc:.2f}%")
    
    # 可视化结果
    plot_training_results(train_losses, train_accuracies, test_accuracies, learning_rates)
    
    # 保存模型
    torch.save(model.state_dict(), 'resnet18_cifar10_final.pth')
    print("模型已保存为 'resnet18_cifar10_final.pth'")

if __name__ == "__main__":
    main()