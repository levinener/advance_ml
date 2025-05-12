import torch
import torch.nn as nn
#from einops import rearrange
from sklearn.metrics import accuracy_score
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import VisionTransformer
import numpy as np
import pickle
from torch.cuda.amp import autocast, GradScaler
import time
scaler = GradScaler(init_scale=2**16)
class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=10, embed_dim=768, 
                 num_layers=12,hidden_dim=128, num_heads=12, dropout=0.01):
        super().__init__()
        # 添加重塑层
        self.reshape_layer = nn.Unflatten(1, (3, image_size, image_size))

        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        
        x = self.reshape_layer(x)
        x = self.patch_embed(x)  # [B, C, H, W] -> [B, C, H/p, W/p]
        #x = rearrange(x, 'b c h w -> b (h w) c')
        x = x.flatten(2).transpose(1, 2)
        
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embed
        
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.norm(x)
        cls_output = x[:, 0]
        return self.fc(cls_output)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.01):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = MLP(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):###没使用dropout
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        # 使用自定义算子
        #attn = vit_ops.qk_matmul(q.half(), k.half()).float() * self.scale
        attn = (q @ k.transpose(-2, -1)) * self.scale
        #attn = vit_ops.optimized_softmax(attn.half())
        attn = torch.nn.functional.softmax(attn, dim=-1)
  
        x = torch.matmul(attn,v)  # 可继续优化
        x = x.transpose(1,2).reshape(B, N, C).float()
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


###########################################################
# 加载本地 CIFAR-10 数据集
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'data'], dict[b'labels']

# 假设 CIFAR-10 数据集存储在 'cifar-10-batches-py' 文件夹中
data_dir = 'cifar-10-batches-py/'
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
model = ViT(
            image_size=32,
            patch_size=4,
            num_classes=10,
            num_heads=8,
            num_layers=12,
            hidden_dim=128
).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 20

time_list=[]
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    torch.cuda.synchronize()
    start_time = time.time() 
    for inputs, labels in train_loader:
        optimizer.zero_grad()
         
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        # 使用scaler缩放梯度并反向传播
        scaler.scale(loss).backward()
        #梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # scaler执行参数更新
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    torch.cuda.synchronize()
    end_time = time.time()
    time_list.append(end_time - start_time) 
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
print(f"vit 模型所用时间: {sum(time_list):.2f}s")