import cv2
import numpy as np
from skimage import feature

def get_square_crop_and_resize(img, mask=None, target_size=(224, 224)):
    """
    从遮罩图提取有效区域的外接正方形，裁剪原图并缩放到指定尺寸
    参数：
        img: 原图
        mask: 黑白遮罩图
        target_size: 目标缩放尺寸,需为正方形
    返回：
        square_img: 裁剪并缩放后的正方形图片
    """
    # 1. 读取图片和遮罩
    # 统一遮罩尺寸和原图一致
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    # 2. 二值化遮罩
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 3. 找到遮罩中白色区域的所有坐标
    coords = np.argwhere(mask_bin > 0)
    # 处理无有效区域的情况
    if len(coords) == 0 or mask==None:
        print("WARNING：未给定有效遮罩，直接将原图缩放到指定尺寸")
        # 直接将原图缩放到target_size（不再裁剪中心正方形）
        square_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        return square_img
    else:
        img = cv2.bitwise_and(img, img, mask=mask)
        # 4. 计算有效区域的最小外接矩形
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # 5. 将矩形转为正方形（取最大边长，居中对齐）
        rect_h = y_max - y_min + 1
        rect_w = x_max - x_min + 1
        side_len = max(rect_h, rect_w)  # 正方形边长
        
        # 计算正方形的起始坐标（居中对齐）
        y_center = (y_min + y_max) // 2
        x_center = (x_min + x_max) // 2
        y_start = y_center - side_len // 2
        x_start = x_center - side_len // 2
        
        # 6. 处理边界越界问题（确保坐标在图片范围内）
        y_start = max(0, y_start)
        x_start = max(0, x_start)
        # 确保正方形不超出图片下/右边界
        if y_start + side_len > img.shape[0]:
            y_start = img.shape[0] - side_len
        if x_start + side_len > img.shape[1]:
            x_start = img.shape[1] - side_len
    
        # 7. 裁剪正方形区域
        y_end = y_start + side_len
        x_end = x_start + side_len
        square_crop = img[y_start:y_end, x_start:x_end]
        
        # 8. 缩放到指定尺寸
        square_img = cv2.resize(square_crop, target_size, interpolation=cv2.INTER_AREA)
        
        return square_img


def calculate_image_metrics(img_masked):
    """
    计算已遮罩图片的所有指标：
    输入img(3通道BGR图)
    输出计算得到的手工特征
    """
    # 获取有效区域的像素
    mask = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY) > 0
    valid_pixels = img_masked[mask]

    # 处理空mask情况（无有效像素）
    if len(valid_pixels) == 0:
        base_metrics = {
            'R_Mean': 0, 'G_Mean': 0, 'B_Mean': 0,
            'R_Std': 0, 'G_Std': 0, 'B_Std': 0,
            'H_Mean': 0, 'S_Mean': 0, 'V_Mean': 0,
            'L_Mean': 0, 'a_Mean': 0, 'b_Mean_Lab': 0,
            'Mean': 0, 'Contrast': 0, 'ASM': 0, 'Entropy': 0, 'perAll': 0
        }
        # 空mask时的特征默认值
        new_features = {
            'LBP_Hist': np.zeros(256), 'LBP_Mean': 0,
            'LTP_Hist': np.zeros(512), 'LTP_Mean': 0,
            'SIFT_Keypoint_Count': 0, 'SIFT_Descriptor': np.array([]),
            'SURF_Keypoint_Count': 0, 'SURF_Descriptor': np.array([]),
            'HOG_Feature': np.array([]), 'HOG_Mean': 0
        }
        base_metrics.update(new_features)
        return base_metrics

    # ========== RGB/HSV/Lab/GLCM/熵等 ==========
    # OpenCV读取的是BGR，需转为RGB
    r_pixels = valid_pixels[:, 2]
    g_pixels = valid_pixels[:, 1]
    b_pixels = valid_pixels[:, 0]
    
    r_mean = np.mean(r_pixels)
    g_mean = np.mean(g_pixels)
    b_mean = np.mean(b_pixels)
    r_std = np.std(r_pixels)
    g_std = np.std(g_pixels)
    b_std = np.std(b_pixels)
    
    # HSV颜色统计
    img_hsv = cv2.cvtColor(img_masked, cv2.COLOR_BGR2HSV)
    img_hsv = img_hsv[mask]
    h_mean = np.mean(img_hsv[:, 0])
    s_mean = np.mean(img_hsv[:, 1])
    v_mean = np.mean(img_hsv[:, 2])
    
    # Lab颜色统计
    img_lab = cv2.cvtColor(img_masked, cv2.COLOR_BGR2Lab)
    img_lab = img_lab[mask]
    l_mean = np.mean(img_lab[:, 0])
    a_mean = np.mean(img_lab[:, 1])
    b_mean_lab = np.mean(img_lab[:, 2])
    
    # 灰度纹理特征
    img_gray = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
    gray_mean = np.mean(img_gray[mask])
    
    # GLCM特征
    glcm = feature.graycomatrix(
        img_gray, 
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )
    contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
    asm = feature.graycoprops(glcm, 'ASM')[0, 0]

    # 熵计算
    gray_masked_pixels = img_gray[mask]
    hist, _ = np.histogram(gray_masked_pixels, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    
    # 有效像素占比
    total_pixels = img_masked.shape[0] * img_masked.shape[1]
    valid_pixel_count = len(valid_pixels)
    per_all = (valid_pixel_count / total_pixels)
    
    # ========== LBP（局部二值模式） ==========
    # 计算LBP（8邻域，半径1），仅在mask区域有效
    lbp = feature.local_binary_pattern(img_gray, P=8, R=1, method='uniform')
    # 只取mask覆盖区域的LBP值
    lbp_valid = lbp[mask]
    # 计算LBP直方图（特征核心）
    lbp_hist, _ = np.histogram(lbp_valid, bins=256, range=(0, 256), density=True)
    lbp_mean = np.mean(lbp_valid)
    
    # ========== LTP（局部三元模式） ==========
    # 基于LBP改进，引入阈值tau（经验值5）
    tau = 5
    ltp = np.zeros_like(img_gray, dtype=np.int32)
    img_gray_int = img_gray.astype(np.int32)
    # 8邻域坐标
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)]
    h, w = img_gray.shape
    for i in range(1, h-1):
        for j in range(1, w-1):
            if mask[i, j] == 0:
                continue
            center = img_gray_int[i, j]
            code = 0
            for k, (dx, dy) in enumerate(neighbors):
                neighbor_val = img_gray_int[i+dx, j+dy]
                diff = neighbor_val - center
                if diff > tau:
                    code |= (1 << (2*k))
                elif diff < -tau:
                    code |= (1 << (2*k + 1))
            ltp[i, j] = code
    # 提取mask区域的LTP值
    ltp_valid = ltp[mask]
    ltp_hist, _ = np.histogram(ltp_valid, bins=512, range=(0, 512), density=True)
    ltp_mean = np.mean(ltp_valid)
    
    # ========== SIFT（尺度不变特征变换） ==========
    # 初始化SIFT（OpenCV 4.x以上需用cv2.SIFT_create()）
    sift = cv2.SIFT_create()
    sift_mask = (mask * 255).astype(np.uint8)
    sift_kp, sift_desc = sift.detectAndCompute(img_gray, mask=sift_mask)
    sift_kp_count = len(sift_kp)
    # 若无关键点，descriptor设为空数组
    if sift_desc is None:
        sift_desc = np.array([])
    
    # ========== HOG（方向梯度直方图） ==========

    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    # 裁剪有效区域
    gray_crop = img_gray[y_min:y_max+1, x_min:x_max+1]
    # 关键：缩放到固定尺寸
    target_hog_size = (64, 64)
    gray_crop_resized = cv2.resize(gray_crop, target_hog_size, interpolation=cv2.INTER_AREA)
    # 计算HOG特征
    hog_features, _ = feature.hog(
        gray_crop_resized,
        orientations=8,        # 8个方向
        pixels_per_cell=(8, 8),# 单元格大小：8×8
        cells_per_block=(2, 2),# 块大小：2x2
        block_norm='L2-Hys',   # 归一化
        visualize=True,
        feature_vector=True,   # 输出一维特征向量
        transform_sqrt=True    # 增强对比度
    )
    hog_features = hog_features.astype(np.float32)
    hog_mean = np.mean(hog_features) if len(hog_features) > 0 else 0.0
    
    # ========== 整合所有指标 ==========
    all_metrics = {
        'R_Mean': r_mean, 'G_Mean': g_mean, 'B_Mean': b_mean,
        'R_Std': r_std, 'G_Std': g_std, 'B_Std': b_std,
        'H_Mean': h_mean, 'S_Mean': s_mean, 'V_Mean': v_mean,
        'L_Mean': l_mean, 'a_Mean': a_mean, 'b_Mean_Lab': b_mean_lab,
        'Mean': gray_mean, 'Contrast': contrast, 'ASM': asm, 
        'Entropy': entropy, 'perAll': per_all,
        'LBP_Hist': lbp_hist, 'LBP_Mean': lbp_mean,
        'LTP_Hist': ltp_hist, 'LTP_Mean': ltp_mean,
        'SIFT_Keypoint_Count': sift_kp_count, 'SIFT_Descriptor': sift_desc,
        'HOG_Feature': hog_features, 'HOG_Mean': hog_mean
    }
    
    return all_metrics

def convert_metrics_to_array(metrics):
    """
    将手工特征字典转换为一维数组，适配模型融合
    参数：
        metrics: calculate_image_metrics返回的特征字典
        device: 张量设备（'cuda'）
    返回：
        handcrafted_tensor: 手工特征一维数组 (1, D)，D为总特征维度
    """
    scalar_keys = [
        'R_Mean', 'G_Mean', 'B_Mean',
        'R_Std', 'G_Std', 'B_Std',
        'H_Mean', 'S_Mean', 'V_Mean',
        'L_Mean', 'a_Mean', 'b_Mean_Lab',
        'Mean', 'Contrast', 'ASM', 'Entropy', 'perAll',
        'LBP_Mean', 'LTP_Mean', 'SIFT_Keypoint_Count', 'HOG_Mean'
    ]
    # 提取标量值并转为numpy数组
    scalar_features = np.array([metrics[k] for k in scalar_keys], dtype=np.float32)

    # LBP_Hist：固定256维
    lbp_hist = metrics['LBP_Hist'].astype(np.float32)
    # LTP_Hist：固定512维
    ltp_hist = metrics['LTP_Hist'].astype(np.float32)
    # HOG_Feature：
    hog_feature = metrics['HOG_Feature'].astype(np.float32)
    
    # ---------------- 3. 处理不规则的SIFT特征----------------
    sift_desc = metrics['SIFT_Descriptor']
    if len(sift_desc) == 0:
        # 无SIFT关键点时，返回全零向量（128维，SIFT描述子默认维度）
        sift_feature = np.zeros(128, dtype=np.float32)
    else:
        # 有关键点时，对所有描述子做均值池化（聚合为128维）
        sift_feature = np.mean(sift_desc, axis=0).astype(np.float32)
    
    # ---------------- 4. 拼接所有特征为一维数组 ----------------
    all_handcrafted = np.concatenate([
        scalar_features,
        lbp_hist,
        ltp_hist,
        hog_feature,
        sift_feature
    ])
    return all_handcrafted