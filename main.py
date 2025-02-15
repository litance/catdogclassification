import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import os
from tqdm import tqdm
# 1️⃣ 指定数据集路径
print("Torch Version:", torch.__version__)
print("Cuda Available:", torch.cuda.is_available())
print("Cuda:", torch.version.cuda)
DATASET_PATH = "" #数据集路径
print("🔍 数据集目录:", DATASET_PATH)
# 2️⃣ 检查数据集结构
dataset_structure = os.listdir(DATASET_PATH)
print("📂 目录内容:", dataset_structure)
# 判断数据集是否包含 'cats' 和 'dogs' 文件夹
if "cats" in dataset_structure and "dogs" in dataset_structure:
    root_dir = DATASET_PATH
else:
    raise FileNotFoundError("❌ 未找到 'cats' 和 'dogs' 文件夹，请检查数据集！")
print("✅ 使用数据集目录:", root_dir)
# 3️⃣ 数据预处理（增强 + 归一化）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 统一图片大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转换为 PyTorch Tensor
    transforms.Normalize([0.5], [0.5])  # 归一化
])
# 4️⃣ 自定义数据集类
class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_labels = []
        for label, cls in enumerate(["cats", "dogs"]):  # 类别映射：cats -> 0, dogs -> 1
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                raise FileNotFoundError(f"❌ 目录 {cls_dir} 不存在，请检查数据集！")
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # 过滤非图片文件
                    self.img_labels.append((img_path, label))
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
# 5️⃣ 加载数据集并划分训练/验证集
full_dataset = CatDogDataset(root_dir=root_dir, transform=transform)
train_size = int(0.8 * len(full_dataset))  # 80% 训练集
val_size = len(full_dataset) - train_size  # 20% 验证集
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
batch_size = 512  # 可调整 batch_size
num_workers = 0  # Windows 设置 num_workers=0 以避免 multiprocessing 问题
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print(f"📊 训练样本: {train_size}, 验证样本: {val_size}")
# 6️⃣ 设备选择（自动检测 GPU）
device = torch.device("cuda:0")  # 强制使用 GPU 0
print(f"✅ 使用设备: {device}")
# 7️⃣ 定义 ResNet 模型（迁移学习）
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 2)  # 修改最后一层，输出 2 个类别（猫 vs 狗）
model = model.to(device)
# 8️⃣ 训练参数
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # 每 5 个 epoch 学习率减小
# 9️⃣ 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        print(f"📌 Epoch [{epoch + 1}/{epochs}] 开始训练...")
        # 🔄 训练过程
        for images, labels in tqdm(train_loader, desc=f"🚀 训练中 Epoch {epoch + 1}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_acc = 100 * correct / total
        scheduler.step()  # 更新学习率
        print(f"📌 Epoch [{epoch + 1}/{epochs}], Loss: {running_loss:.4f}, Accuracy: {train_acc:.2f}%")
        # 🔎 评估模型
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100 * val_correct / val_total
        print(f"✅ 验证集 Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%\n")
# **🚀 运行训练**
if __name__ == '__main__':
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10)
    # 🔟 保存模型
    torch.save(model.state_dict(), "cat_dog_classifier.pth")
    print("✅ 模型已保存为 'cat_dog_classifier.pth'")
