import torch
import torch.nn as nn
#from einops import rearrange

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import VisionTransformer
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
# 多头自注意力机制
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        qkv = self.qkv_proj(x)
        batch_size, seq_length, _ = x.size()
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        output = self.out_proj(attn_output)
        return output

# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.relu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Transformer 编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Vision Transformer 模型
class VisionTransformer(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=10, embed_dim=192, num_heads=3, num_layers=12,
                 hidden_dim=768, dropout=0.001):
        super(VisionTransformer, self).__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size."
        num_patches = (image_size // patch_size) ** 2

        # 添加重塑层
        self.reshape_layer = nn.Unflatten(1, (3, image_size, image_size))

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # 重塑输入
        x = self.reshape_layer(x)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)
        cls_output = x[:, 0]
        output = self.fc(cls_output)
        return output

###########################################################
# 加载本地 CIFAR-10 数据集
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'data'], dict[b'labels']

# 假设 CIFAR-10 数据集存储在 'cifar-10-batches-py' 文件夹中
data_dir = 'cifar-10-batches-py/cifar-10-batches-py'
batch_files = [f'{data_dir}/data_batch_{i}' for i in range(1, 6)]
test_file = f'{data_dir}/test_batch'

# 加载所有批次的数据
data, labels = [], []
for file in batch_files:
    batch_data, batch_labels = load_cifar10_batch(file)
    data.append(batch_data)
    labels.extend(batch_labels)

# # 加载测试数据
test_data, test_labels = load_cifar10_batch(test_file)
data.append(test_data)
labels.extend(test_labels)

# 计算每个类别的样本数量
classes, counts = np.unique(labels, return_counts=True)
class_labels = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# 合并所有数据
data = np.concatenate(data)
labels = np.array(labels)
# 分割数据集为训练集和测试集
train_data = data[:50000]
train_labels = labels[:50000]
test_data = data[50000:]
test_labels = labels[50000:]

# 归一化
train_data = train_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0

# # 调整数据形状
# train_data = train_data.reshape(-1, 3, 32, 32)
# test_data = test_data.reshape(-1, 3, 32, 32)


# 转换为 PyTorch 张量
train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)

# 创建数据集和数据加载器
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
##############################################################################

# 创建 ViT 模型实例
model = VisionTransformer(
            image_size=32,
            patch_size=4,
            num_classes=10,
            num_heads=8,
            num_layers=12,
            hidden_dim=128
).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test accuracy: {100 * correct / total}%')

# 预测功能
def predict(model, data):
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
    return predicted



################################################################

vit_result_gpu=predict(model,test_data)
print(device)
vit_result = vit_result_gpu.cpu().numpy()
test_labels=test_labels.cpu().numpy()
#
vit_accuracy = accuracy_score(test_labels, vit_result)
print(f"vit 模型的准确率: {vit_accuracy * 100:.2f}%")
