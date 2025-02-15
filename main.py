import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QLineEdit
from PyQt6.QtGui import QPixmap
from PIL import Image

# **1️⃣ 设备选择**
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# **2️⃣ 加载训练好的模型**
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 2)  # 修改最后一层
model.load_state_dict(torch.load("cat_dog_classifier.pth", map_location=device))
model.to(device)
model.eval()

# **3️⃣ 图像预处理**
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# **4️⃣ GUI 界面**
class CatDogClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("猫狗分类器")
        self.setGeometry(100, 100, 400, 400)

        # **界面元素**
        self.label = QLabel("请选择图片", self)
        self.label.setStyleSheet("font-size: 16px;")
        self.img_label = QLabel(self)
        self.img_label.setFixedSize(224, 224)

        self.upload_button = QPushButton("上传图片", self)
        self.upload_button.clicked.connect(self.upload_image)

        self.path_input = QLineEdit(self)
        self.path_input.setPlaceholderText("或输入图片路径")

        self.predict_button = QPushButton("分类", self)
        self.predict_button.clicked.connect(self.predict)

        # **布局**
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.img_label)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.path_input)
        layout.addWidget(self.predict_button)
        self.setLayout(layout)

        self.image_path = ""

    def upload_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path).scaled(224, 224)
            self.img_label.setPixmap(pixmap)
            self.path_input.setText(file_path)

    def predict(self):
        file_path = self.path_input.text()
        if not file_path:
            self.label.setText("❌ 请选择或输入图片路径！")
            return

        image = Image.open(file_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            label = "🐱 猫" if predicted.item() == 0 else "🐶 狗"
            self.label.setText(f"✅ 预测结果: {label}")


# **运行应用**
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CatDogClassifierApp()
    window.show()
    sys.exit(app.exec())
