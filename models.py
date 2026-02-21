import torch
from torch import nn
from torch.nn import functional as F
########原版Resnet18########
class Residual(nn.Module):
    def __init__(self,input_channels,num_channels,
                 use1x1conv=False,strides=1):
        super().__init__()
        self.conv1 =nn.Conv2d(input_channels,num_channels,kernel_size=3,padding=1,stride=strides)
        self.conv2 =nn.Conv2d(num_channels,num_channels,kernel_size=3,padding=1)
        if use1x1conv:
            self.conv3 =nn.Conv2d(input_channels,num_channels,kernel_size=1,stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
    def forward(self,X):
        Y =F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y+=X
        return F.relu(Y)

def resnet_block(input_channels,num_channels,num_residuals,
                 first_block =False):
    blk=[]
    for i in range(num_residuals):
        if i==0 and not first_block:
            blk.append(Residual(input_channels,num_channels,
                 use1x1conv=True,strides=2))
        else:
             blk.append(Residual(num_channels,num_channels))
    return blk

class ResNet(nn.Module):
    """封装后的ResNet模型类"""
    def __init__(self, input_channels=3, num_classes=2):
        """
        参数说明：
        - input_channels: 输入图像的通道数（1=灰度图，3=彩色图）
        - num_classes: 分类数（默认2分类）
        """
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # 构建ResNet的网络结构
        self._build_layers()

    def _build_layers(self):
        """构建网络各层（内部方法）"""
        # 第一个阶段：7x7卷积 + 批归一化 + ReLU + 最大池化
        self.b1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 第二到第五阶段：残差块组成
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))
        
        # 分类头：自适应平均池化 + 展平 + 全连接层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, X):
        X = self.b1(X)
        X = self.b2(X)
        X = self.b3(X)
        X = self.b4(X)
        X = self.b5(X)
        
        output = self.classifier(X)
        return output
############################

#########融合手工特征########
class ResNet_attn(nn.Module):
    """"通过注意力机制融合手工特征的resnet18"""
    def __init__(self, input_channels=3, num_classes=2, handcraft_dim=None):
        """
        参数说明：
        - input_channels: 输入图像的通道数（默认3=彩色图，修复原默认值1的问题）
        - num_classes: 分类数（默认2分类）
        """
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.handcraft_dim =handcraft_dim
        # 构建ResNet的基础网络结构
        self._build_layers()

    def _build_layers(self):
        """构建网络各层（内部方法）"""
        self.b1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))

        # 线性层：将手工特征对齐到512维（和图像深度特征维度一致）

        self.handcraft_proj = nn.Linear(self.handcraft_dim, 512)
        
        # 多头注意力（自注意力+交叉注意力）
        self.self_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        
        # 融合后的特征投影+层归一化
        self.fusion_proj = nn.Sequential(
            nn.Linear(512 * 2, 512),  # 拼接自注意力+交叉注意力特征后降维
            nn.LayerNorm(512),
            nn.ReLU()
        )

        # 先池化展平图像特征，再对接融合层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(512, self.num_classes)

    def forward(self, X, Y):
        """
        前向传播：
        - X: 图像输入 (batch, 3, H, W)
        - Y: 手工特征输入 (batch, handcraft_dim)
        """
        # ========== 第一步：提取图像深度特征 ==========
        X = self.b1(X)
        X = self.b2(X)
        X = self.b3(X)
        X = self.b4(X)
        X = self.b5(X)  # (batch, 512, H', W')
        
        # 池化+展平：(batch, 512, 1, 1) → (batch, 512)
        img_feat = self.avgpool(X)
        img_feat = self.flatten(img_feat)  # (batch, 512)
        
        # ========== 第二步：处理手工特征 ==========
        # 维度对齐：(batch, handcraft_dim) → (batch, 512)
        hand_feat = self.handcraft_proj(Y)  # (batch, 512)
        
        # ========== 第三步：注意力融合 ==========
        # 增加序列维度（MultiheadAttention要求3维输入：(batch, seq_len, embed_dim)）
        img_feat_seq = img_feat.unsqueeze(1)  # (batch, 1, 512)
        hand_feat_seq = hand_feat.unsqueeze(1)  # (batch, 1, 512)
        
        # 3.1 图像特征自注意力（增强自身表征）
        img_self_attn, _ = self.self_attn(
            query=img_feat_seq, 
            key=img_feat_seq, 
            value=img_feat_seq
        )  # (batch, 1, 512)
        
        # 3.2 图像-手工特征交叉注意力（融合双方信息）
        cross_attn, _ = self.cross_attn(
            query=img_feat_seq,       # 以图像特征为查询
            key=hand_feat_seq,        # 以手工特征为键
            value=hand_feat_seq       # 以手工特征为值
        )  # (batch, 1, 512)
        
        # 3.3 特征拼接+投影（融合自注意力和交叉注意力结果）
        img_self_attn = img_self_attn.squeeze(1)  # (batch, 512)
        cross_attn = cross_attn.squeeze(1)        # (batch, 512)
        fused_feat = torch.cat([img_self_attn, cross_attn], dim=1)  # (batch, 1024)
        fused_feat = self.fusion_proj(fused_feat)  # (batch, 512)
        
        # ========== 第四步：分类输出 ==========
        output = self.classifier(fused_feat)  # (batch, num_classes)
        return output