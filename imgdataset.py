import os
import glob
import cv2
import torch
from torch import nn
from PIL import Image, ImageOps
import models
import numpy as np
import image_features
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm                                           # 进度条
import matplotlib.pyplot as plt                                 # 可视化
plt.rcParams["font.family"] = ["SimHei"]                        # 中文字体
plt.rcParams['axes.unicode_minus'] = False                      # 解决负号显示问题

# 生成训练集测试集，目前也测试训练模型
np.random.seed(42)  
torch.manual_seed(42)

class ImageDataset(Dataset):
    def __init__(self, imgs, metrics, labels):
        self.imgs = imgs
        self.metrics = metrics
        self.labels = labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx].astype(np.float32) / 255.0              # 归一化
        img = torch.from_numpy(img).permute(2, 0, 1)
        metric = torch.from_numpy(self.metrics[idx])
        metric = (metric - metric.mean()) / (metric.std() + 1e-8)   # 标准化
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, metric, label



imgs = []
metrics = []
labels = []

valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

def _process_folder(folder_path, label):
    # =================读入对照组=================
    for fname in os.listdir(folder_path):
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_exts:
            full_path = os.path.join(folder_path, fname)
            img = Image.open(full_path).convert("RGB")
            img = ImageOps.exif_transpose(img)  # 处理可能的EXIF旋转
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            H, W = img.shape[:2]

            mask = Image.open("./mask/" + os.path.splitext(fname)[0] + ".png")
            mask = np.array(ImageOps.exif_transpose(mask))

            mask = cv2.resize(mask, (W, H),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
            img[~mask] = [0, 0, 0]  # 将 Mask 外的像素设为黑色

            
            coords = np.argwhere(mask > 0)

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            img_crop = img[y_min:y_max+1, x_min:x_max+1]
            img_crop = cv2.resize(img_crop, (224, 224), interpolation=cv2.INTER_AREA)
            
            imgs.append(img_crop)
            labels.append(label)
            metrics.append(image_features.convert_metrics_to_array(image_features.calculate_image_metrics(img_crop)))

            # ====== 数据增强：水平翻转 ======
            img_flip = cv2.flip(img_crop, 1)
            imgs.append(img_flip)
            labels.append(label)
            metrics.append(image_features.convert_metrics_to_array(image_features.calculate_image_metrics(img_flip)))

            # ====== 数据增强：亮度扰动 ======
            img_bright = cv2.convertScaleAbs(img_crop, alpha=1.0, beta=np.random.randint(-30, 30))
            imgs.append(img_bright)
            labels.append(label)
            metrics.append(image_features.convert_metrics_to_array(image_features.calculate_image_metrics(img_bright)))

            # angle = np.random.uniform(-5, 5)  # 随机旋转角度， 但发现这个data 96%argument会导致最后验证集准确率下降，原因未知
            # h, w = img_crop.shape[:2]
            # center = (w // 2, h // 2)
            # M = cv2.getRotationMatrix2D(center, angle, 1.0)
            # img_rotate = cv2.warpAffine(img_crop, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            # imgs.append(img_rotate)
            # labels.append(label)
            # metrics.append(image_features.convert_metrics_to_array(image_features.calculate_image_metrics(img_rotate)))


            # ====== 可视化调试（保留原逻辑） ======
            if np.random.randint(0, 300) < 1:
                imgshow = cv2.cvtColor(img_bright, cv2.COLOR_BGR2RGB)
                imgshow = Image.fromarray(imgshow)
                imgshow.show()

_process_folder("./对照组舌苔/对照组舌苔图像102例", label=0)
_process_folder("./舌苔/1.MH健康对照组（有转录）", label=0)
_process_folder("./舌苔/2.MYH健康对照（无转录）", label=0)

_process_folder("./舌苔/3.MY抑郁组/type-1 0W", label=1)  # 抑郁组标签为1
_process_folder("./舌苔/3.MY抑郁组/type-2 2W", label=1)  # 抑郁组标签为1
_process_folder("./舌苔/3.MY抑郁组/type-3 4W", label=1)  # 抑郁组标签为1
            

# Resnet_18训练测试（注释了“原版”的是仅输入图片的版本）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 60    # 测试用，可调整
LR = 1e-4      # 学习率
WEIGHT_DECAY = 1e-4  # 权重衰减
SAVE_PATH = "./resnet_test.pth"

handcraft_dim = np.array(metrics).shape[1]
dataset = ImageDataset(imgs, metrics, labels)
# model = models.ResNet(input_channels=3).to(DEVICE)  # 原版
model = models.ResNet_attn(
    input_channels=3, handcraft_dim=handcraft_dim).to(DEVICE)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
criterion = nn.CrossEntropyLoss()  # 二分类用CrossEntropyLoss
optimizer = torch.optim.Adam(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

from torch.optim.lr_scheduler import StepLR
lr_scheduler = StepLR(optimizer, step_size=20, gamma=0.6)


def train_model():
    best_acc = 0.0
    # 初始化存储训练过程数据的列表
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    # 创建可视化画布
    plt.ion()  # 开启交互模式
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('模型训练过程可视化', fontsize=14)

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, metrics, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Train"):
            imgs, metrics, labels = imgs.to(
                DEVICE), metrics.to(DEVICE), labels.to(DEVICE)

            # 前向传播
            outputs = model(imgs, metrics)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, metrics, labels in val_loader:
                imgs, metrics, labels = imgs.to(
                    DEVICE), metrics.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs, metrics) 
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        # 计算指标
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        # 打印结果
        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 更新可视化曲线
        ax1.clear()
        ax1.plot(train_losses, label='训练损失', color='blue')
        ax1.plot(val_losses, label='验证损失', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('训练/验证损失曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.clear()
        ax2.plot(train_accs, label='训练准确率', color='blue')
        ax2.plot(val_accs, label='验证准确率', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('训练/验证准确率曲线')
        ax2.set_ylim(0, 1.05)  # 准确率范围固定在0-1.05
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.draw()
        plt.pause(0.1)  # 暂停0.1秒让图像更新

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"保存最佳模型，验证准确率: {best_acc:.4f}")

        lr_scheduler.step()
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

    print(f"\n训练完成！最佳验证准确率: {best_acc:.4f}")
    return model


if __name__ == "__main__":
    trained_model = train_model()
    print("模型训练完成")
    input("按任意键关闭窗口并结束程序...")
