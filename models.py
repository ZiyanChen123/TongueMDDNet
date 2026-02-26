import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
from torchvision.models import (
    DenseNet169_Weights, MobileNet_V3_Small_Weights,
    SqueezeNet1_1_Weights, VGG19_BN_Weights
)
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


#基于预训练模型的编码器
class ImageEmbedding(nn.Module):
    def __init__(self, model_name="resnet50", hidden_dim=512):
        super().__init__()
        # 根据model_name选择预训练模型
        if model_name == "densenet169":
            self.backbone_model = models.densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(self.backbone_model.children())[:-1])  # 移除分类层
            self.backbone_out_dim = 1664  # DenseNet169输出维度
        elif model_name == "mobilenet_v3_small":
            self.backbone_model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(self.backbone_model.children())[:-1])  # 移除分类层
            self.backbone_out_dim = 576  # MobileNetV3Small输出维度
        elif model_name == "squeezenet1_1":
            self.backbone_model = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(self.backbone_model.children())[:-1])  # 移除分类层
            self.backbone_out_dim = 512  # SqueezeNet输出维度
        elif model_name == "vgg19_bn":
            self.backbone_model = models.vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(self.backbone_model.children())[:-1])  # 移除分类层
            self.backbone_out_dim = 512  # VGG19_bn输出维度
        else:  # 默认ResNet50
            self.backbone_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(self.backbone_model.children())[:-2])
            self.backbone_out_dim = 2048

        # 固定预训练权重
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 全局池化（适配所有模型）
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 特征压缩到hidden_dim（512）
        self.fc = nn.Linear(self.backbone_out_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, images):
        features = self.backbone(images)
        pooled_features = self.avg_pool(features)
        flattened = pooled_features.view(pooled_features.size(0), -1)
        embedding = self.dropout(self.relu(self.fc(flattened)))
        return embedding

# 图像手工特征编码器
class HandcraftEmbedding(nn.Module):
    def __init__(self, input_dim = 2485 , hidden_dim=512):
        super().__init__()
        self.handcraft_dim = input_dim
        self.handcraft_proj = nn.Linear(self.handcraft_dim, hidden_dim)
        # 手工特征权重
        self.feature_units = [
            # 标量特征：21个，每个特征1个权重（共21个）
            (0, 21, 21),
            # LBP直方图：256维，1个权重
            (21, 277, 1),
            # LTP直方图：512维，1个权重
            (277, 789, 1),
            # HOG特征：1568维，1个权重
            (789, 2357, 1),
            # SIFT特征：128维，1个权重
            (2357, 2485, 1)
        ]
        total_weight_num = sum([unit[2] for unit in self.feature_units])
        self.weights = nn.Parameter(torch.ones(total_weight_num, dtype=torch.float32))
        self.relu = nn.ReLU()

    def forward(self, handcraft_features):
        weighted_parts = []
        weight_idx = 0  # 权重索引指针
        
        for start, end, weight_num in self.feature_units:
            # 提取当前单元特征
            feat_part = handcraft_features[:, start:end]  # (batch, unit_dim)
            # 提取当前单元对应的权重
            w = self.weights[weight_idx:weight_idx+weight_num]  # (weight_num,)
            weight_idx += weight_num
            
            # 加权：标量逐特征加权，高维组广播加权
            if weight_num > 1:
                # 标量单元：逐特征加权
                weighted_part = feat_part * w.unsqueeze(0)
            else:
                # 高维组：1个权重广播到整个组
                weighted_part = feat_part * w
            
            weighted_parts.append(weighted_part)
        
        # 拼接所有加权特征 → 恢复2485维
        weighted_feat = torch.cat(weighted_parts, dim=1)
        return self.relu(self.handcraft_proj(weighted_feat))


# 二分类头
class BinaryClassificationHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, dropout_rate=0.1):
        """
        通用二分类头

        """
        super().__init__()
        self.classifier = nn.Sequential(
            # 第一层特征压缩
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 层归一化，提升训练稳定性
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # 第二层特征压缩
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # 输出层：二分类（logits）
            nn.Linear(hidden_dim // 2, 1)
        )
        self.sigmoid = nn.Sigmoid()  # 将logits转为0-1概率

    def forward(self, fused_features, return_prob=True):
        logits = self.classifier(fused_features)
        if return_prob:
            return self.sigmoid(logits)
        return logits


###跨模态对齐方法###
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
        

#5. 门控融合机制
class GatedFusionNetwork(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size):
        super(GatedFusionNetwork, self).__init__()
        self.pathway1 = nn.Linear(input_size1, hidden_size)
        self.pathway2 = nn.Linear(input_size2, hidden_size)
        self.gating = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        self.fusion_layer = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, input1, input2):
        out1 = self.relu(self.pathway1(input1))
        out2 = self.relu(self.pathway2(input2))
        fused = torch.cat((out1, out2), dim=1)
        gate = self.gating(fused)
        gated_output = gate * out1 + (1 - gate) * out2
        fused_output = self.relu(self.fusion_layer(gated_output))
        return fused_output

class CrossModalBinaryClassifier(nn.Module):
    def __init__(
        self,
        img_input_dim=3,
        handcraft_input_dim=2485,
        embed_size=512,
        num_heads=8,
        fusion_type="attention_fusion",
        dropout_rate=0.1
    ):
        """
        整合图像编码器、手工特征编码器、融合模块、分类头的完整模型
        Args:
            fusion_type: 融合方式，可选值：
                - contrastive: 对比损失对齐（需配合ContrastiveLoss训练）
                - cross_attention: 交叉注意力对齐
                - datafusion: 简单拼接
                - attention_fusion: 注意力加权融合
                - gated: 门控融合
        """
        super().__init__()
        
        # 1. 基础编码器
        self.image_encoder = ImageEmbedding(model_name="squeezenet1_1", hidden_dim=512)
        self.handcraft_encoder = HandcraftEmbedding(input_dim=handcraft_input_dim, hidden_dim=embed_size)
        
        # 2. 选择融合模块
        self.fusion_type = fusion_type
        if fusion_type == "cross_attention":
            self.fusion_module = CrossModalAlignment(
                img_input_dim=embed_size,
                handcraft_input_dim=embed_size,
                embed_size=embed_size,
                num_heads=num_heads
            )
        elif fusion_type == "datafusion":
            self.fusion_module = Datafusion(
                img_input_dim=embed_size,
                handcraft_input_dim=embed_size,
                hidden_dim=embed_size
            )
        elif fusion_type == "attention_fusion":
            self.fusion_module = AttentionFusion(
                img_input_dim=embed_size,
                handcraft_input_dim=embed_size,
                hidden_dim=embed_size
            )
        elif fusion_type == "gated":
            self.fusion_module = GatedFusionNetwork(
                input_size1=embed_size,
                input_size2=embed_size,
                hidden_size=embed_size
            )
        elif fusion_type == "contrastive":
            # 对比损失仅用于对齐，融合用简单拼接
            self.fusion_module = Datafusion(
                img_input_dim=embed_size,
                handcraft_input_dim=embed_size,
                hidden_dim=embed_size
            )
            self.contrastive_loss = ContrastiveLoss(margin=1.0)
        
        # 3. 通用二分类头
        self.classification_head = BinaryClassificationHead(
            input_dim=embed_size,
            hidden_dim=embed_size // 2,
            dropout_rate=dropout_rate
        )

    def forward(self, images, handcraft_features, labels=None):
        """
        前向传播
        Args:
            images: 图像张量 [batch_size, 3, 224, 224]
            handcraft_features: 手工特征张量 [batch_size, handcraft_input_dim]
            labels: 标签（仅训练时需要）[batch_size, 1]
        Returns:
            如果labels=None: 二分类概率 [batch_size, 1]
            如果labels≠None: 字典，包含概率、分类损失（、对比损失）
        """
        # 1. 编码特征
        img_emb = self.image_encoder(images)
        hand_emb = self.handcraft_encoder(handcraft_features)
        
        # 2. 特征融合
        fused_feat = self.fusion_module(img_emb, hand_emb)
        
        # 3. 分类预测
        pred_prob = self.classification_head(fused_feat, return_prob=True)
        
        # 4. 计算损失（训练阶段）
        if labels is not None:
            # 分类损失（用logits计算，避免sigmoid梯度消失）
            pred_logits = self.classification_head(fused_feat, return_prob=False)
            bce_loss = nn.functional.binary_cross_entropy_with_logits(pred_logits, labels.float())
            
            # 如果是对比损失融合，额外计算对比损失
            total_loss = bce_loss
            if self.fusion_type == "contrastive":
                contr_loss = self.contrastive_loss(img_emb, hand_emb, labels)
                total_loss = bce_loss + 0.1 * contr_loss  # 对比损失权重可调整
                
            return {
                "pred_prob": pred_prob,
                "classification_loss": bce_loss,
                "total_loss": total_loss,
                "contrastive_loss": contr_loss if self.fusion_type == "contrastive" else None
            }
        return pred_prob

# ------------------- 测试代码 -------------------
if __name__ == "__main__":
    # 1. 初始化模型（测试门控融合+二分类头）
    model = CrossModalBinaryClassifier(
        handcraft_input_dim=2485,  # 假设手工特征是100维
        fusion_type="gated"
    )
    
    # 2. 构造测试数据
    batch_size = 4
    test_images = torch.randn(batch_size, 3, 224, 224)  # 图像输入
    test_handcraft = torch.randn(batch_size, 2485)       # 手工特征输入
    test_labels = torch.randint(0, 2, (batch_size, 1))  # 二分类标签
    
    # 3. 训练模式（带损失计算）
    model.train()
    output = model(test_images, test_handcraft, test_labels)
    print("训练模式输出：")
    print(f"预测概率形状: {output['pred_prob'].shape}")
    print(f"分类损失: {output['classification_loss'].item():.4f}")
    print(f"总损失: {output['total_loss'].item():.4f}")
    
    # 4. 推理模式（仅预测）
    model.eval()
    with torch.no_grad():
        pred = model(test_images, test_handcraft)
    print("\n推理模式输出：")
    print(f"预测概率形状: {pred.shape}")
    
    # 修复：格式化numpy数组的正确方式
    pred_np = pred.squeeze().numpy()
    # 方式1：遍历元素格式化
    pred_formatted = [f"{p:.4f}" for p in pred_np]
    print(f"预测概率示例: {', '.join(pred_formatted)}")