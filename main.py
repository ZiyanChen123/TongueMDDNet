# @Author: Liyuan Tuo, Zixuan Gu
# @Description: 负责从二值 mask 文件夹中， 尝试区分舌苔、舌体两类区域 并保存为 coating_mask 文件夹中的 png 图片 以及 compair_kmean 文件夹中对比图，对比图左侧是去除阴影后的原图，右侧是 K-means 分割结果（黑色=苔, 灰色=体, 白色=背景）

from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans  # 引入 MiniBatchKMeans 作为备选
import cv2
import numpy as np
from PIL import Image, ImageOps
import os

# 必须在导入 numpy/sklearn 之前设置线程数，防止 OpenBLAS/MKL 内存泄漏
# 设置为 1 可能会慢一点，但最稳定，解决了 Windows 下 KMeans 的内存泄漏问题
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # 限制 joblib 的最大进程数


def shadow_remove(img, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]

    # 当提供mask时，为了避免背景的黑色(0)影响边缘的光照估计，
    # 我们将背景填充为mask内亮度的平均值（这一步相当于简单的边界填充/外推）
    # 这样高斯模糊在边缘处就不会骤降
    V_for_blur = V.copy()
    mean_v = np.mean(V[mask])
    V_for_blur[~mask] = int(mean_v)
    illumination = cv2.GaussianBlur(V_for_blur, (21, 21), sigmaX=40, sigmaY=40)

    # 光照归一化
    # 避免除以0，添加一个小常数
    illumination = np.maximum(illumination, 1)
    V_corrected = np.clip((V.astype(np.float32) / illumination) * 255, 0, 255)

    hsv[:, :, 2] = V_corrected.astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img


def dual_channel_kmeans_segmentation_with_mask(img_Lab, mask, n_clusters=2,
                                               use_spatial_features=False, random_state=42,
                                               spatial_weight=0.0000001, background_label=255):
    """
    在指定掩膜范围内进行双通道K-means聚类图像分割

    参数:
    ----------
    img_Lab : numpy.ndarray
        LAB色彩空间图像，形状为 (H, W, 3)
    mask : numpy.ndarray
        掩膜图像，形状为 (H, W)，布尔类型或0/1值，True/1表示需要处理的区域
    n_clusters : int, 可选
        聚类数量，默认为2
    random_state : int, 可选
        随机种子，用于可重复性
    use_spatial_features : bool, 可选
        是否添加空间位置特征
    spatial_weight : float, 可选
        空间特征的权重（0-1之间）
    background_label : int, 可选
        掩膜外区域的标签值，默认为-1
    normalize_channels : bool, 可选
        是否对通道进行归一化，默认为True
        - 如果为True，将G通道归一化到[0,1]，b*通道归一化到[-1,1]
        - 如果为False，则假设输入已经适当归一化

    返回:
    ----------
    mask_inner : numpy.ndarray
        完整分割掩码，形状为 (H, W)，掩膜内为0到n_clusters-1，掩膜外为background_label = -1
    cluster_centers : numpy.ndarray
        聚类中心，形状为 (n_clusters, 2)
    labels_in_mask : numpy.ndarray
        掩膜内像素的标签，一维数组
    """

    H, W = img_Lab.shape[:2]

    # 提取掩膜内的像素值
    channel1_masked = img_Lab[mask].astype(np.float32)
    # channel2_masked = channel2[mask].astype(np.float32)

    # 将双通道数据组合成特征向量
    # 每个像素点是一个2D向量 [channel1_value, channel2_value]
    features = channel1_masked

    # 如果只针对掩膜内的像素
    if use_spatial_features:
        # 获取掩膜内像素的坐标
        y_coords, x_coords = np.where(mask)

        # 添加空间特征
        spatial_features = np.stack(
            [x_coords.astype(np.float32), y_coords.astype(np.float32)], axis=1)

        # 将空间特征与通道特征结合，并加权， 现在的特征变成了 (G * （1 - ）, b * （1 - ）, weight*x, weight*y)
        features = np.hstack([
            features,
            spatial_features
        ])

    # 标准化特征（K-means对尺度敏感）
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    if use_spatial_features:
        # features_scaled 的前两列是颜色 (G, b*)，后两列是位置 (x, y)
        # 我们按照权重比例对它们进行缩放
        features_scaled[:, :2] *= (1.0 - spatial_weight)  # 颜色权重
        features_scaled[:, 2:] *= spatial_weight           # 空间位置权重

    # 应用K-means聚类（只在掩膜内的像素上进行）
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=random_state, n_init=10)
    labels_in_mask = kmeans.fit_predict(features_scaled)

    # 获取聚类中心（反标准化后）
    cluster_centers_scaled = kmeans.cluster_centers_
    cluster_centers = scaler.inverse_transform(cluster_centers_scaled)

    # 如果使用了空间特征，聚类中心可能包含空间坐标
    # 我们只返回原始通道的聚类中心（前两列）
    if use_spatial_features:
        # 从加权特征中提取原始通道的聚类中心
        # 注意：这里需要反向计算权重
        n_original_features = 2  # channel1和channel2
        cluster_centers_original = cluster_centers[:,
                                                   :n_original_features] / (1 - spatial_weight)
    else:
        cluster_centers_original = cluster_centers

    # 创建完整的掩膜（包括背景区域）
    mask_inner = np.full((H, W), background_label, dtype=np.uint8)
    mask_inner[mask] = labels_in_mask

    # 测试是否成功分类
    # print(np.unique(labels_in_mask))

    return mask_inner, cluster_centers_original


def tongue_Gb_segmentation(img, mask):

    # 提取R分量                          为什么是R分量
    # 提取b*分量
    img_Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # 进行k聚类分割
    mask_inner, _ = dual_channel_kmeans_segmentation_with_mask(img_Lab, mask)
    # print(mask_inner)
    # cv2.imshow("img", mask_inner)
    # cv2.waitKey(0)
    # there -1 will be converted to 255 in astype uint8, which is what we want for the background
    return mask_inner


# 存储所有图片信息的列表   总计674张 由于有一张mp4 所以提取出673张图片进行处理
data_records = []
valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']


# =================读入对照组=================
for fname in os.listdir("./对照组舌苔/对照组舌苔图像102例"):
    ext = os.path.splitext(fname)[1].lower()
    if ext in valid_exts:
        full_path = os.path.join("./对照组舌苔/对照组舌苔图像102例", fname)
        data_records.append({
            "filename": fname,
            "filepath": full_path
        })

for fname in os.listdir("./舌苔/1.MH健康对照组（有转录）"):
    ext = os.path.splitext(fname)[1].lower()
    if ext in valid_exts:
        full_path = os.path.join("./舌苔/1.MH健康对照组（有转录）", fname)
        data_records.append({
            "filename": fname,
            "filepath": full_path
        })

for fname in os.listdir("./舌苔/2.MYH健康对照（无转录）"):
    ext = os.path.splitext(fname)[1].lower()
    if ext in valid_exts:
        full_path = os.path.join("./舌苔/2.MYH健康对照（无转录）", fname)
        data_records.append({
            "filename": fname,
            "filepath": full_path
        })

# =================读入抑郁组=================
for root, dirs, files in os.walk("./舌苔/3.MY抑郁组"):
    # print(f"正在处理抑郁组文件夹: {root}, 包含 {len(files)} 张图片 , dir 数量: {len(dirs)}")
    for fname in files:
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_exts:
            full_path = os.path.join(root, fname)
            folder_name = os.path.basename(root)  # e.g., type-1 0W
            data_records.append({
                "filename": fname,
                "filepath": full_path
            })


# 转换为 DataFrame
df_data = pd.DataFrame(data_records)

# 显式排序，确保每次运行的处理顺序完全一致
df_data = df_data.sort_values(by=['filename']).reset_index(drop=True)


print(f"共发现图片: {len(df_data)} 张")

all_features = []


for idx, row in tqdm(df_data.iterrows(), total=len(df_data)):
    # mask读取
    PILmask = Image.open(
        "./mask/" + os.path.splitext(row['filename'])[0] + ".png")
    PILmask = ImageOps.exif_transpose(PILmask)  # 处理可能的EXIF旋转
    mask = np.array(PILmask)

    print(f"正在处理图片: {row['filename']}")
    PILimg = Image.open(row['filepath']).convert("RGB")
    PILimg = ImageOps.exif_transpose(PILimg)  # 处理可能的EXIF旋转
    img_initial = cv2.cvtColor(np.array(PILimg), cv2.COLOR_RGB2BGR)

    # 使用原尺寸
    # MAX_DIM = 1280  # 设置最大长/宽限制
    H, W = img_initial.shape[:2]
    # if max(H, W) > MAX_DIM:
    #     scale = MAX_DIM / max(H, W)
    #     new_W, new_H = int(W * scale), int(H * scale)
    #     img_initial = cv2.resize(
    #         img_initial, (new_W, new_H), interpolation=cv2.INTER_AREA)
    #     # print(f"  -> 图像过大 ({W}x{H})，已缩放至 ({new_W}x{new_H})")
    #     H, W = new_H, new_W  # 更新当前的H和W
    # ==================================

    # 这里的mask都是8位的，值仅为0，255
    mask = cv2.resize(mask, (W, H),
                      interpolation=cv2.INTER_NEAREST).astype(bool)

    # cv2.imshow("img", img_initial)
    # display_size = (640, int(640 * H / W)) # 统一缩小到宽度 640 进行预览
    # cv2.imshow("img_initial", cv2.resize(img_initial, display_size))

    # 亮度均衡化
    img_unshadowed = shadow_remove(img_initial, mask)
    # cv2.imshow("img_unshadowed", cv2.resize(img_unshadowed, display_size))

    # cv2.waitKey(0)

    # 图像分割
    mask_inner = tongue_Gb_segmentation(
        img_unshadowed, mask)  # mask的类型是 bool

    # 0(第一类) -> 黑色(0)
    # 1(第二类) -> 灰色(128)
    # -1(背景)  -> 白色(255)
    mask_inner[mask_inner == 1] = 128

    mask_inner = Image.fromarray(mask_inner)
    mask_inner.save(f"./coating_mask/" +
                    os.path.splitext(row['filename'])[0] + ".png")

    # --- 替换 matplotlib 绘图保存逻辑，改用 cv2 直接拼接保存，防止内存泄漏 ---

    # 1. 准备左图：Unshadowed (BGR)
    # img_unshadowed 已经是 BGR，且非 mask 区域已经处理过（可选：确保背景是白色）

    img_unshadowed[~mask] = [255, 255, 255]

    # 2. 准备右图：K-means Result (Grayscale -> BGR)
    # mask_inner 是 (H, W) 的单通道，0=苔, 128=体, 255=背景
    # 为了拼接，需要转成 3 通道 BGR
    img_right = cv2.cvtColor(np.array(mask_inner), cv2.COLOR_GRAY2BGR)

    # 3. 拼接 (水平拼接)
    # 确保两张图的高度一致（通常是一致的，因为都来自原图尺寸）
    # 中间可以加一条黑线或白线分隔
    sep_line = np.zeros((H, 10, 3), dtype=np.uint8)  # 10像素宽的黑线
    img_combined = np.hstack([img_unshadowed, sep_line, img_right])

    # 4. 添加文字说明 (可选)
    # cv2.putText(img_combined, "Unshadowed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 5. 保存
    save_path = f"./compair_kmean/{os.path.splitext(row['filename'])[0]}.png"
    cv2.imwrite(save_path, img_combined)
