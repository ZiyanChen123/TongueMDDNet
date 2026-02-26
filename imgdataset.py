import os
import cv2
import torch
import random
from PIL import Image, ImageOps
import numpy as np
import image_features
from torch.utils.data import Dataset                                         # 进度条
import matplotlib.pyplot as plt                                 # 可视化
plt.rcParams["font.family"] = ["SimHei"]                        # 中文字体
plt.rcParams['axes.unicode_minus'] = False                      # 解决负号显示问题

# np.random.seed(42)  
# torch.manual_seed(42)

class ImageDataset(Dataset):
    def __init__(self, imgs=[], metrics=[], labels=[]):
        self.imgs = imgs
        self.metrics = metrics
        self.labels = labels
        self.valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx].astype(np.float32) / 255.0             # 归一化
        img = torch.from_numpy(img).permute(2, 0, 1)
        metric = torch.from_numpy(self.metrics[idx])
        metric = (metric - metric.mean()) / (metric.std() + 1e-8)   # 标准化
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, metric, label

    def AUG(self,img):
        """图像增强函数,生成12张处理后的图像"""
        augmented_imgs = []
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

        # 生成基础翻转图像（包含原图+3种翻转）
        base_imgs = [
            img,                  # 原图
            cv2.flip(img, 1),     # 水平翻转
            cv2.flip(img, 0),     # 垂直翻转
            cv2.flip(img, -1)     # 水平+垂直翻转
        ]

        # 裁剪和色彩增强
        for base_img in base_imgs:
            # 随机裁剪：面积为10%~100%的区域，裁剪后resize回224x224
            area_ratio = random.uniform(0.1, 1.0)
            crop_w = int(w * np.sqrt(area_ratio))
            crop_h = int(h * np.sqrt(area_ratio))
            x_start = random.randint(0, w - crop_w)
            y_start = random.randint(0, h - crop_h)
            img_crop = base_img[y_start:y_start+crop_h, x_start:x_start+crop_w]
            img_crop_resize = cv2.resize(img_crop, (224, 224), interpolation=cv2.INTER_AREA)
            augmented_imgs.append(img_crop_resize)

            # 随机改变亮度、对比度、饱和度、色调（基于HSV空间处理）
            img_hsv = cv2.cvtColor(base_img, cv2.COLOR_BGR2HSV)
            h_delta = random.randint(-10, 10)
            img_hsv[:, :, 0] = (img_hsv[:, :, 0] + h_delta) % 180
            s_scale = random.uniform(0.7, 1.3)
            img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * s_scale, 0, 255)
            v_scale = random.uniform(0.7, 1.3)
            img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * v_scale, 0, 255)
            img_color = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
            augmented_imgs.append(img_color)

        augmented_imgs = [aug for aug in augmented_imgs if aug.shape == (224,224,3)]
        augmented_imgs.extend(base_imgs)
        
        return augmented_imgs

    def process_folder(self,folder_path, label):
        # 读入数据集
        print("开始加载：{}".format(folder_path))
        for fname in os.listdir(folder_path):
            ext = os.path.splitext(fname)[1].lower()
            if ext in self.valid_exts:
                full_path = os.path.join(folder_path, fname)
                img = Image.open(full_path).convert("RGB")
                img = ImageOps.exif_transpose(img)  # 处理EXIF旋转
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                H, W = img.shape[:2]

                # 读取并处理mask
                mask_path = os.path.join("./mask", os.path.splitext(fname)[0] + ".png")
                if not os.path.exists(mask_path):
                    print(f"警告 未找到mask文件 {mask_path}，跳过该图像")
                    continue
                mask = Image.open(mask_path)
                mask = np.array(ImageOps.exif_transpose(mask))
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                img[~mask] = [0, 0, 0]  # Mask外设为黑色

                # 裁剪到mask有效区域
                coords = np.argwhere(mask > 0)
                if len(coords) == 0:
                    print(f"警告：{fname} 的mask无有效区域")
                    continue
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                img_crop = img[y_min:y_max+1, x_min:x_max+1]
                img_crop = cv2.resize(img_crop, (224, 224), interpolation=cv2.INTER_AREA)

                # 1. 添加原图
                self.imgs.append(img_crop)
                self.labels.append(label)
                self.metrics.append(image_features.convert_metrics_to_array(
                    image_features.calculate_image_metrics(img_crop)
                ))

                # # 2. 调用增强函数，添加增强后的图像
                # augmented_imgs = self.AUG(img_crop)
                # for aug_img in augmented_imgs:
                #     self.imgs.append(aug_img)
                #     self.labels.append(label)
                #     self.metrics.append(image_features.convert_metrics_to_array(
                #         image_features.calculate_image_metrics(aug_img)
                #     ))

