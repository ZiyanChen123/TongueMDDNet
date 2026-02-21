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

###跨模态对齐方法###
#占位，各种特征的编码器
class ImageEmbedding(nn.Module):
    def __init__(self, input_dim,hidden_dim):
        super().__init__()

    def forward(self,images):
        return images
    
class HandcraftEmbedding(nn.Module):
    def __init__(self, input_dim,hidden_dim):
        super().__init__()

    def forward(self,handcraft_features):
        return handcraft_features
    
class AudioEmbedding(nn.Module):
    def __init__(self, input_dim,hidden_dim):
        super().__init__()

    def forward(self,audios):
        return audios
    
#1. 通过嵌入对齐和优化损失函数融合
#基于余弦相似度的对比损失(训练时用这个损失函数)
class ContrastiveLoss(nn.Module):
    def __init__(self, margin = 1.0):
        super().__init__()
        self.margin =margin

    def forward(self, image_embeddings,handcraft_embeddings,labels):
        cosine_sim = torch.cosine_similarity(image_embeddings,handcraft_embeddings)
        positive_loss = (1-labels)*(1-cosine_sim)
        negative_loss = labels*torch.clamp(cosine_sim-self.margin,min=0)
        return positive_loss.mean()+negative_loss.mean()

    
#2. 基于交叉注意力机制的跨模态对齐(以手工特征为锚点，图像特征主动对齐手工特征)
class CrossModalAlignment(nn.Module):
    def __init__(self,img_input_dim,handcraft_input_dim,embed_size,num_heads):
        super().__init__()
        self.image_proj = nn.Linear(img_input_dim,embed_size)
        self.handcraft_proj = nn.Linear(handcraft_input_dim,embed_size)
        self.attention = nn.MultiheadAttention(embed_dim=embed_size,
                                               num_heads=num_heads,
                                               batch_first=True)
        self.fc = nn.Linear(embed_size,embed_size)

    def forward(self,image_embeddings,handcraft_embeddings):
        image_embeddings = self.image_proj(image_embeddings.unsqueeze(1))
        handcraft_embeddings = self.handcraft_proj(handcraft_embeddings.unsqueeze(1))
        combined,_ = self.attention(image_embeddings,handcraft_embeddings,handcraft_embeddings)
        return self.fc(combined).squeeze(1)
    
#3. 数据级融合，即简单拼接两个模态，无法学习模态间关系
class Datafusion(nn.Module):
    def __init__(self, img_input_dim,handcraft_input_dim,hidden_dim):
        super().__init__()
        self.image_proj = nn.Linear(img_input_dim,hidden_dim)
        self.handcraft_proj = nn.Linear(handcraft_input_dim,hidden_dim)
        self.fusion_layer = nn.Linear(hidden_dim*2,hidden_dim)
        self.relu = nn.ReLU()

    def forward(self,image_embeddings,handcraft_embeddings):
        image_embeddings = self.relu(self.image_proj(image_embeddings))
        handcraft_embeddings = self.relu(self.handcraft_proj(handcraft_embeddings))
        fused_features = torch.cat((image_embeddings,handcraft_embeddings),dim=1)
        fused_output = self.relu(self.fusion_layer(fused_features))
        return(fused_output)


#4. 特征级融合（基于注意力机制）
class AttentionFusion(nn.Module):
    def __init__(self, img_input_dim,handcraft_input_dim,hidden_dim):
        super().__init__()
        self.image_proj = nn.Linear(img_input_dim,hidden_dim)
        self.handcraft_proj = nn.Linear(handcraft_input_dim,hidden_dim)
        self.image_attention = nn.Linear(hidden_dim,1)
        self.handcraft_attention = nn.Linear(hidden_dim,1)
        self.fusion_layer = nn.Linear(hidden_dim*2,hidden_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,image_embeddings,handcraft_embeddings):
        image_embeddings = self.relu(self.image_proj(image_embeddings))
        handcraft_embeddings = self.relu(self.handcraft_proj(handcraft_embeddings))
        #两个模态的权重
        image_weights = self.softmax(self.image_attention(image_embeddings))
        handcraft_weights = self.softmax(self.handcraft_attention(handcraft_embeddings))
        #加权拼接
        weighted_image = image_embeddings*image_weights
        weighted_handcraft = handcraft_embeddings*handcraft_weights
        fused_features = torch.cat((weighted_image,weighted_handcraft),dim=1)
        fused_output = self.relu(self.fusion_layer(fused_features))
        return fused_output
        
# 自注意力模块
class SingleModalSelfAttention(nn.Module):
    def __init__(self,input_size,embed_size, num_heads):
        super().__init__()
        self.proj = nn.Linear(input_size,embed_size)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            batch_first=True
        )
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, modal_embeddings):
        modal_seq = self.proj(modal_embeddings.unsqueeze(1))
        attn_out, _ = self.self_attention(
            query=modal_seq,
            key=modal_seq,
            value=modal_seq
        )
        enhanced_emb = self.fc(attn_out).squeeze(1)
        return enhanced_emb
