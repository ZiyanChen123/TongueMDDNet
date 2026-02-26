import models
from imgdataset import *
from torch.utils.data import DataLoader, random_split
from torch import nn
from tqdm import tqdm 
imgs = []
metrics = []
labels = []
dataset = ImageDataset(imgs, metrics, labels)
# dataset.process_folder("./对照组舌苔/对照组舌苔图像102例", label=0)
dataset.process_folder("./舌苔/1.MH健康对照组（有转录）", label=0)
dataset.process_folder("./舌苔/2.MYH健康对照（无转录）", label=0)
dataset.process_folder("./舌苔/3.MY抑郁组/type-1 0W", label=1)  # 抑郁组标签为1
dataset.process_folder("./舌苔/3.MY抑郁组/type-2 2W", label=1)  # 抑郁组标签为1
dataset.process_folder("./舌苔/3.MY抑郁组/type-3 4W", label=1)  # 抑郁组标签为1
            
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{DEVICE}")
EPOCHS = 10    # 测试用，可调整
LR = 1e-4      # 学习率
WEIGHT_DECAY = 1e-4  # 权重衰减
SAVE_PATH = "./test.pth"

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


model = models.CrossModalBinaryClassifier().to(DEVICE)
optimizer = torch.optim.Adam(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

from torch.optim.lr_scheduler import StepLR
lr_scheduler = StepLR(optimizer, step_size=20, gamma=0.6)


def train_model(model, train_loader, val_loader, optimizer, lr_scheduler):
    """
    适配CrossModalBinaryClassifier的训练函数
    Args:
        model: 初始化后的CrossModalBinaryClassifier模型
        train_loader: 训练数据加载器，返回(imgs, metrics, labels)
        val_loader: 验证数据加载器，返回(imgs, metrics, labels)
        optimizer: 优化器（如Adam）
        lr_scheduler: 学习率调度器
        criterion: 兼容参数（实际使用模型内部的损失计算，此处保留以兼容）
    """
    best_acc = 0.0
    # 初始化存储训练过程数据的列表
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, metrics, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Train"):
            imgs, metrics, labels = imgs.to(DEVICE), metrics.to(DEVICE), labels.to(DEVICE)
            
            labels = labels.unsqueeze(1)
            # 前向传播：传入labels，模型返回包含损失和预测的字典
            outputs = model(imgs, metrics, labels)
            
            # 反向传播
            optimizer.zero_grad()
            outputs["total_loss"].backward()
            optimizer.step()
            
            # 统计训练损失（使用模型返回的total_loss）
            train_loss += outputs["total_loss"].item()
            
            # 计算二分类准确率：概率>=0.5为正例，否则为负例
            preds = (outputs["pred_prob"] >= 0.5).float()
            train_correct += (preds == labels.float()).sum().item()
            train_total += labels.size(0)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, metrics, labels in val_loader:
                imgs, metrics, labels = imgs.to(DEVICE), metrics.to(DEVICE), labels.to(DEVICE)
                labels = labels.unsqueeze(1)
                # 验证阶段1：先获取预测概率（无labels）
                pred_prob = model(imgs, metrics)
                
                # 验证阶段2：计算验证损失（需要传入labels让模型计算）
                val_outputs = model(imgs, metrics, labels)
                val_loss += val_outputs["total_loss"].item()
                
                # 计算二分类准确率
                preds = (pred_prob >= 0.5).float()
                val_correct += (preds == labels.float()).sum().item()
                val_total += labels.size(0)

        # 计算本轮平均指标
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        # 记录指标
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        # 打印本轮结果
        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 打印对比损失（如果使用contrastive融合方式）
        if model.fusion_type == "contrastive" and epoch == 0:
            print("提示：当前使用对比损失融合，对比损失权重为0.1")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, SAVE_PATH)
            print(f"保存最佳模型，验证准确率: {best_acc:.4f}")

        # 更新学习率
        lr_scheduler.step()
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

    # 返回训练过程的指标，方便后续可视化
    return {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "best_acc": best_acc
    }


if __name__ == "__main__":
    trained_model = train_model(model,train_loader,val_loader,optimizer,lr_scheduler)
    print("模型训练完成")
    input("按任意键关闭窗口并结束程序...")