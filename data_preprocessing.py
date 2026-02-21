# @Author: Liyuan Tuo , Zixuan Gu
# @Description: 数据处理脚本，负责从coating_mask中提取舌苔、舌体 两者的特征并保存为 tongue_features_dataset.xlsx 这个 Excel 文件

import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from tqdm import tqdm
from PIL import Image, ImageOps

def load_image(image_path, filename):
    img_resized = np.array(ImageOps.exif_transpose(Image.open(image_path)))[:,:,0:3]
    H, W, _ = img_resized.shape
    try:                                 # mask 文件夹中有对应的掩膜图像，命名规则为原图文件名去掉扩展名后加上 .png
        # 确保一定是单通道 (H, W)，防止 Mask 是 RGB 格式导致后面报错
        img_masked = np.array(ImageOps.exif_transpose(Image.open(f"./coating_mask/{os.path.splitext(filename)[0]}.png")).resize((W, H)))

    except Exception as e:
        print(f"无法找到掩膜文件: {os.path.splitext(filename)[0]}.png, 错误: {e}")
        return img_resized, None
    
    return img_resized, img_masked

def calculate_ltp(img_gray, mask, tau=5):
    """计算 LTP (Local Ternary Pattern) 特征"""
    # 转为 int32 以处理负值
    img_int = img_gray.astype(np.int32)
    rows, cols = img_gray.shape
    
    # 定义8邻域偏移量
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]
    
    # 遍历像素 (由于 Python 循环慢，这里只提取 mask 区域的边界框内进行计算优化)
    # 或者使用切片操作来加速：
    
    # 中心像素
    center = img_int[1:-1, 1:-1]
    
    upper_code = np.zeros_like(center, dtype=np.uint8)
    lower_code = np.zeros_like(center, dtype=np.uint8)
    
    for i, (dy, dx) in enumerate(offsets):
        # 邻域像素
        # 注意：切片范围需根据 dy, dx 调整，保证与 center 对应
        neighbor = img_int[1+dy:rows-1+dy, 1+dx:cols-1+dx]
        diff = neighbor - center
        
        # 计算 upper pattern (diff > tau -> 1)
        bit_u = (diff > tau).astype(np.uint8)
        upper_code |= (bit_u << i)
        
        # 计算 lower pattern (diff < -tau -> 1)
        bit_l = (diff < -tau).astype(np.uint8)
        lower_code |= (bit_l << i)

    # 将计算结果填回全图尺寸 (边缘留黑)
    upper_full = np.zeros_like(img_gray, dtype=np.uint8)
    lower_full = np.zeros_like(img_gray, dtype=np.uint8)
    upper_full[1:-1, 1:-1] = upper_code
    lower_full[1:-1, 1:-1] = lower_code
    
    
    u_vals = upper_full[mask]
    l_vals = lower_full[mask] # 只取 Mask 区域的 LTP 值的均值和标准差作为特征，其实也可以把整个mask区域的值都当作特征输入模型，但维度会很大，这里先统计一下均值和标准差作为简化的特征。后续可以考虑增加更多统计量或者直接使用全局的 LTP 直方图等更丰富的特征表示。
    
    return np.mean(u_vals), np.std(u_vals), np.mean(l_vals), np.std(l_vals)


def calculate_hog(img_gray, mask):
    """计算 HOG (Histogram of Oriented Gradients) 特征"""
    # 1. 裁剪出 mask 的最小外接矩形，避免全图计算引入大量背景噪声
    coords = np.argwhere(mask)
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # 裁剪
    roi_gray = img_gray[y_min:y_max+1, x_min:x_max+1]
    
    # 缩放到固定尺寸 (如 64x64/128x128) 以保证特征维度一致性或更稳定的梯度统计
    # HOG 对分辨率敏感，这里标准化到 128x128
    roi_resized = cv2.resize(roi_gray, (128, 128))
    
    try:
        # 计算 HOG
        fd = hog(roi_resized, orientations=8, pixels_per_cell=(16, 16),
                 cells_per_block=(1, 1), visualize=False)
        return np.mean(fd), np.std(fd)
    except Exception as e:
        print(f"HOG error: {e}")
        return 0, 0


def extract_features(img_rgb, mask):
    
    features = {}

    # === 辅助函数：提取特定区域特征 (包含 LBP, SIFT, LTP, HOG) ===
    def get_roi_features(valid_pixels, suffix):
        roi_feats = {}
        
        # 转灰度
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # --- 1. 颜色特征 ---
        # RGB
        for i, c in enumerate(['R', 'G', 'B']):
            pixels = img_rgb[:,:,i][valid_pixels]
            roi_feats[f'{c}_Mean{suffix}'] = np.mean(pixels)
            roi_feats[f'{c}_Std{suffix}'] = np.std(pixels)
        
        # HSV
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        for i, c in enumerate(['H', 'S', 'V']):
            pixels = img_hsv[:,:,i][valid_pixels]
            roi_feats[f'{c}_Mean{suffix}'] = np.mean(pixels)
            roi_feats[f'{c}_Std{suffix}'] = np.std(pixels)
        # Lab
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        pixels_l = img_lab[:,:,0][valid_pixels]
        pixels_a = img_lab[:,:,1][valid_pixels]
        pixels_b_lab = img_lab[:,:,2][valid_pixels]
        roi_feats[f'L_Mean{suffix}'] = np.mean(pixels_l)
        roi_feats[f'a_Mean{suffix}'] = np.mean(pixels_a)
        roi_feats[f'b_Mean{suffix}_Lab'] = np.mean(pixels_b_lab)
        
        # --- 2. 传统纹理特征 (GLCM) ---
        roi_feats[f'Mean{suffix}'] = np.mean(img_gray[valid_pixels])

        # GLCM (修复暗像素丢失问题)
        n_levels = 128
        # 量化并+1，使得有效范围变成 1-64，0 留给背景
        img_gray_quant = (img_gray // (256 // n_levels)).astype(np.uint8) + 1
        img_gray_quant[~valid_pixels] = 0  # 背景设为 0
        
        # levels=65 (0..64)
        g = graycomatrix(img_gray_quant, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=n_levels+1, symmetric=True, normed=False)
        
        # 剔除第0行和第0列
        g = g[1:, 1:, :, :] 
        
        g_sum = np.sum(g)
        if g_sum > 0:
            g_norm = g.astype(np.float64) / g_sum
            roi_feats[f'Contrast{suffix}'] = np.mean(graycoprops(g_norm, 'contrast'))
            roi_feats[f'ASM{suffix}'] = np.mean(graycoprops(g_norm, 'ASM'))
            p_nz = g_norm[g_norm > 0]
            roi_feats[f'Entropy{suffix}'] = -np.sum(p_nz * np.log2(p_nz))
        else:
            roi_feats[f'Contrast{suffix}'] = 0
            roi_feats[f'ASM{suffix}'] = 0
            roi_feats[f'Entropy{suffix}'] = 0

        # --- 3. LBP 特征 (Local Binary Patterns) ---
        # radius=1, n_points=8, method='uniform' 是最常用的配置
        # 为了只计算 Mask 区域，我们先计算全图 LBP，然后取 Valid 区域
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
        lbp_valid = lbp[valid_pixels]
        
        if len(lbp_valid) > 0:
            roi_feats[f'LBP_Mean{suffix}'] = np.mean(lbp_valid)
            roi_feats[f'LBP_Std{suffix}'] = np.std(lbp_valid)
            # 也可以计算 LBP 的熵
            hist, _ = np.histogram(lbp_valid, bins=np.arange(0, n_points + 3), density=True)
            hist = hist[hist > 0]
            roi_feats[f'LBP_Entropy{suffix}'] = -np.sum(hist * np.log2(hist))
        else:
            roi_feats[f'LBP_Mean{suffix}'] = 0
            roi_feats[f'LBP_Std{suffix}'] = 0
            roi_feats[f'LBP_Entropy{suffix}'] = 0
            
        # --- 3.1 LTP 特征 (Local Ternary Patterns) ---
        # 计算 LTP
        ltp_u_mean, ltp_u_std, ltp_l_mean, ltp_l_std = calculate_ltp(img_gray, valid_pixels)
        roi_feats[f'LTP_Upper_Mean{suffix}'] = ltp_u_mean
        roi_feats[f'LTP_Upper_Std{suffix}'] = ltp_u_std
        roi_feats[f'LTP_Lower_Mean{suffix}'] = ltp_l_mean
        roi_feats[f'LTP_Lower_Std{suffix}'] = ltp_l_std
        
        # --- 3.2 HOG 特征 ---
        hog_mean, hog_std = calculate_hog(img_gray, valid_pixels)
        roi_feats[f'HOG_Mean{suffix}'] = hog_mean
        roi_feats[f'HOG_Std{suffix}'] = hog_std

        # --- 4. SIFT 特征 ---
        # SIFT 查找关键点，我们统计在 Mask 区域内的关键点数量
        try:
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(img_gray, None)
            
            sift_count = 0
            sift_response_sum = 0
            
            if keypoints:
                for kp in keypoints:
                    # kp.pt 是 (x, y)，我们需要检查这个坐标是否在 Mask 内
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    if 0 <= y < valid_pixels.shape[0] and 0 <= x < valid_pixels.shape[1]:
                        if valid_pixels[y, x]: # 如果该点在 Mask 内
                            sift_count += 1
                            sift_response_sum += kp.response
            
            roi_feats[f'SIFT_Count{suffix}'] = sift_count
            roi_feats[f'SIFT_MeanResponse{suffix}'] = (sift_response_sum / sift_count) if sift_count > 0 else 0
            
        except Exception as e:
            # print(f"SIFT error: {e}") 
            roi_feats[f'SIFT_Count{suffix}'] = 0
            roi_feats[f'SIFT_MeanResponse{suffix}'] = 0

        # --- 5. perAll ---
        s_values = img_hsv[:,:,1][valid_pixels]
        if len(s_values) > 0:
            coating_threshold = np.mean(s_values) 
            roi_feats[f'perAll{suffix}'] = np.sum(s_values < coating_threshold) / len(s_values)
        else:
            roi_feats[f'perAll{suffix}'] = 0
            
        return roi_feats

    # 提取两部分特征
    features.update(get_roi_features(mask == 128, "1"))
    features.update(get_roi_features(mask == 255, "2"))

    return features

base_dir = "./"

# 两个数据来源文件夹
control_dir = os.path.join(base_dir, "对照组舌苔", "对照组舌苔图像102例")
depression_dir_root = os.path.join(base_dir, "舌苔")

# 存储所有图片信息的列表   总计674张 由于有一张mp4 所以提取出673张图片进行处理
data_records = []
valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']


# =================读入对照组=================
for fname in os.listdir(control_dir):
    # print(f"正在处理对照组图片: {fname}")
    ext = os.path.splitext(fname)[1].lower()
    if ext in valid_exts:
        full_path = os.path.join(control_dir, fname)
        data_records.append({
            "filepath": full_path,
            "filename": fname,
        })

# =================读入抑郁组=================
for root, dirs, files in os.walk(depression_dir_root):
    # print(f"正在处理抑郁组文件夹: {root}, 包含 {len(files)} 张图片 , dir 数量: {len(dirs)}")
    for fname in files:
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_exts:
            full_path = os.path.join(root, fname)
            folder_name = os.path.basename(root) # e.g., type-1 0W
            data_records.append({
                "filepath": full_path,
                "filename": fname,
            })


# 转换为 DataFrame
df_data = pd.DataFrame(data_records)


print(f"共发现图片: {len(df_data)} 张")

all_features = []

for idx, row in tqdm(df_data.iterrows(), total=len(df_data)):

    img_resized, img_masked = load_image(row['filepath'], row['filename'])
        # --- 保存处理后的图片 ---
        # 保持文件名，加上前缀或放入对应文件夹，避免重名
        # save_name = f"{row['label']}_{idx}_{row['filename']}" 
    
    
    img_copy = img_resized.copy()
    img_copy[img_masked == 255] = [0, 0, 0]  # 将mask为255的区域设为黑色

    
    features = extract_features(img_resized, img_masked)
    
    if features:
        # 合并基本信息
        all_features.append(features)

# 转为 DataFrame
all_features = pd.DataFrame(all_features)

# 保存
output_file = "./tongue_features_dataset.xlsx"

# 保存 Excel
all_features.to_csv(output_file, index=False)
print(f"特征数据已保存至: {output_file}")
    


