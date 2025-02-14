# catdogclassification
## Cat and Dog Image Recognition Based on Pytorch Deep Learning (ResNet Model (Transfer Learning))<br/>基于Pytorch深度学习的猫狗图像识别（ResNet模型（迁移学习））

Trained model：https://drive.google.com/file/d/1TGSMWplYMm_VNHXzAGRQZtSCy-UYIH6l/view?usp=sharing

### 所需要第三方库 Third-party libraries needed to run this program
- torch
- torchvision
- torchaudio
- tqdm
- pillow
- numpy

### 数据增强 Data Augmentation
- Resize
- RandomHorizontalFlip
- ToTensor
- Normalize

## 1.首先, 安装第三方库 Install Third-party libraries
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tqdm pillow numpy
```

## 2.配置好代码 Configure the code(main.py)
DATASET_PATH(line:15)
```
DATASET_PATH = ""
```

Optimizer(line:93)
```
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

Scheduler(line:94)
```
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
```

Epoch(line:98)
```
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100):
```

## 3.运行 Run this program
