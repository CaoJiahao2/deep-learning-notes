# 深度学习完整训练流程

> 从数据准备到模型部署的端到端实践指南。

---

## 目录

1. [总体流程概览](#1)
2. [数据准备](#2)
3. [数据预处理与增强](#3)
4. [模型构建](#4)
5. [损失函数与评价指标](#5)
6. [优化器与学习率调度](#6)
7. [训练与验证循环](#7)
8. [模型测试与性能评估](#8)
9. [模型保存与加载](#9)
10. [模型推理与部署](#10)
11. [实践建议与常见技巧](#11)

---

## 1. 总体流程概览

```mermaid
graph LR
    A[数据准备] --> B[数据预处理]
    B --> C[数据增强]
    C --> D[模型构建]
    D --> E[训练循环]
    E --> F[验证评估]
    F --> E
    F --> G[模型测试]
    G --> H[模型保存]
    H --> I[推理部署]
```

一个完整的深度学习项目包含以下关键阶段：

1. **数据准备**：收集、清洗、标注数据
2. **数据预处理**：归一化、Tokenization、格式转换
3. **数据增强**：扩充训练数据，提高泛化能力
4. **模型构建**：设计或选择网络架构
5. **训练循环**：前向传播、损失计算、反向传播、参数更新
6. **验证评估**：监控训练过程，调整超参数
7. **测试评估**：在独立测试集上评估最终性能
8. **模型保存**：保存检查点，记录训练配置
9. **推理部署**：将模型投入生产环境

---

## 2. 数据准备

### 2.1 数据集划分

| 集合 | 比例 | 用途 |
|------|------|------|
| 训练集 (Train) | 60-80% | 模型参数学习 |
| 验证集 (Validation) | 10-20% | 超参数调优、早停 |
| 测试集 (Test) | 10-20% | 最终性能评估 |

**原则**：测试集在训练过程中**绝对不能**被模型看到。

### 2.2 数据加载最佳实践

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

# 高效 DataLoader 配置
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,           # 训练集打乱
    num_workers=4,           # 多进程加载
    pin_memory=True,         # 加速 GPU 传输
    prefetch_factor=2,       # 预取数据
    drop_last=True           # 丢弃不完整 batch
)
```

### 2.3 不平衡数据处理

| 方法 | 说明 |
|------|------|
| 重采样 | 过采样少数类 / 欠采样多数类 |
| 类别权重 | 给少数类更高损失权重 |
| Focal Loss | 降低易分类样本的权重 |
| 数据增强 | 对少数类进行增强 |

---

## 3. 数据预处理与增强

### 3.1 图像预处理

```python
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet 统计量
        std=[0.229, 0.224, 0.225]
    )
])
```

### 3.2 文本预处理

- **Tokenization**：BPE, WordPiece, SentencePiece
- **Padding/Truncation**：统一序列长度
- **特殊标记**：[CLS], [SEP], [PAD], [MASK]

### 3.3 数据增强

详见 [正则化技术文档](./regularization.md) 第 5 节。

---

## 4. 模型构建

### 4.1 推荐的项目结构

```
project/
├── configs/               # 配置文件
│   └── default.yaml
├── data/                  # 数据处理
│   └── dataset.py
├── models/                # 模型定义
│   ├── backbone.py
│   ├── head.py
│   └── model.py
├── losses/                # 损失函数
│   └── loss.py
├── utils/                 # 工具函数
│   ├── metrics.py
│   └── logger.py
├── train.py               # 训练入口
├── eval.py                # 评估脚本
└── inference.py           # 推理脚本
```

### 4.2 模型初始化

**Xavier/Glorot 初始化**（适用于 Tanh/Sigmoid）：

$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$$

**Kaiming/He 初始化**（适用于 ReLU）：

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

```python
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
```

---

## 5. 损失函数与评价指标

### 5.1 常用损失函数

| 任务 | 损失函数 | 公式 |
|------|---------|------|
| 二分类 | BCE Loss | $-\frac{1}{N}\sum[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$ |
| 多分类 | Cross Entropy | $-\frac{1}{N}\sum y_i \log \hat{y}_i$ |
| 回归 | MSE Loss | $\frac{1}{N}\sum (y_i - \hat{y}_i)^2$ |
| 回归 | Huber Loss | 结合 MSE 和 MAE 的优点 |
| 不平衡分类 | Focal Loss | $-\alpha_t(1-p_t)^\gamma \log p_t$ |

### 5.2 常用评价指标

| 任务 | 指标 | 说明 |
|------|------|------|
| 分类 | Accuracy | 整体准确率 |
| 分类 | Precision/Recall/F1 | 不平衡数据更可靠 |
| 分类 | AUC-ROC | 二分类排序能力 |
| 回归 | MAE/MSE/RMSE | 误差大小 |
| 回归 | R² Score | 拟合优度 |
| 生成 | FID/IS | 图像生成质量 |
| NLP | BLEU/ROUGE | 文本生成质量 |

---

## 6. 优化器与学习率调度

详见 [优化器详解](./optimization.md)。

**推荐配置**：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_steps - warmup_steps
)
```

---

## 7. 训练与验证循环

### 7.1 训练循环模板

```python
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    for batch in train_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
    
    # 验证阶段
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            output = model(x)
            val_loss += criterion(output, y).item()
    
    val_loss /= len(val_loader)
    print(f"Epoch {epoch}: val_loss = {val_loss:.4f}")
```

### 7.2 混合精度训练

```python
scaler = torch.amp.GradScaler('cuda')

for batch in train_loader:
    optimizer.zero_grad()
    
    with torch.amp.autocast('cuda'):
        output = model(x)
        loss = criterion(output, y)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 7.3 梯度累积

当 GPU 显存不足以支持大 batch size 时：

```python
accumulation_steps = 4

for i, batch in enumerate(train_loader):
    loss = criterion(model(x), y) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 8. 模型测试与性能评估

### 8.1 测试集评估

```python
def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            total_loss += criterion(output, y).item()
            
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    return total_loss / len(test_loader), accuracy
```

### 8.2 混淆矩阵与分类报告

```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(all_labels, all_preds))
print(confusion_matrix(all_labels, all_preds))
```

---

## 9. 模型保存与加载

### 9.1 检查点保存

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_loss': best_val_loss,
    'config': config_dict
}
torch.save(checkpoint, f'checkpoint_epoch{epoch}.pt')
```

### 9.2 模型加载

```python
checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### 9.3 仅保存模型权重（推荐用于部署）

```python
# 保存
torch.save(model.state_dict(), 'model_weights.pt')

# 加载
model = MyModel()
model.load_state_dict(torch.load('model_weights.pt'))
model.eval()
```

---

## 10. 模型推理与部署

### 10.1 推理模板

```python
@torch.no_grad()
def inference(model, input_data):
    model.eval()
    input_tensor = preprocess(input_data).to(device)
    output = model(input_tensor)
    return postprocess(output)
```

### 10.2 模型导出

- **TorchScript**：`torch.jit.trace(model, example_input)`
- **ONNX**：`torch.onnx.export(model, example_input, 'model.onnx')`
- **TensorRT**：NVIDIA 推理加速引擎

详见 [模型部署文档](../deployment/model-serving.md)。

---

## 11. 实践建议与常见技巧

### 11.1 调试与诊断

| 技巧 | 说明 |
|------|------|
| 过拟合小 batch | 先在少量数据上过拟合，验证模型和学习 pipeline 正确 |
| 梯度检查 | 验证反向传播实现的正确性 |
| 学习率范围测试 | 通过 LR Range Test 找到最佳学习率 |
| TensorBoard | 可视化 loss、梯度、权重分布 |
| 打印统计信息 | 定期打印输入/输出的均值、方差 |

### 11.2 训练加速

| 技巧 | 加速效果 |
|------|---------|
| 混合精度训练 (AMP) | 1.5-2× |
| 多 GPU 数据并行 (DDP) | 近线性加速 |
| 增大 batch size | 配合 LR scaling |
| 数据预取 (prefetch) | 消除 I/O 瓶颈 |
| torch.compile() | PyTorch 2.0+, 10-30% 提升 |

### 11.3 常见错误

1. **忘记 model.eval() / model.train()**：影响 BN、Dropout 行为
2. **忘记 zero_grad()**：梯度累积错误
3. **数据泄露**：在训练前对整个数据集做了归一化
4. **学习率与 batch size 不匹配**：batch size 加倍时，学习率也应适当调整
5. **未设置随机种子**：导致实验不可复现
