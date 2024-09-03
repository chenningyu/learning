import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# GPT 配置类，用于存储模型的超参数
class GPTConfig:
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd):
        self.vocab_size = vocab_size  # 词汇表大小
        self.block_size = block_size  # 输入序列的最大长度
        self.n_layer = n_layer        # Transformer Block 的层数
        self.n_head = n_head          # 自注意力层中的头数
        self.n_embd = n_embd          # 嵌入维度

# GPT 模型类
class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        
        # 词嵌入层，将输入的标记转换为高维向量表示
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # 位置嵌入，用于为每个标记的位置信息编码
        self.position_embedding = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        # 包含多个 Transformer Block 的层列表
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # 层归一化，用于标准化最后的输出
        self.ln_f = nn.LayerNorm(config.n_embd)
        # 最后的线性层，用于生成预测
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        # 初始化模型权重
        self.apply(self._init_weights)
    
    # 初始化权重的函数，确保权重遵循正态分布
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    # 定义前向传播的逻辑
    def forward(self, idx, targets=None):
        b, t = idx.size()  # 获取批量大小和序列长度
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # 获取标记嵌入和位置嵌入，并将它们相加
        token_embeddings = self.token_embedding(idx)
        position_embeddings = self.position_embedding[:, :t, :]
        x = token_embeddings + position_embeddings
        
        # 依次通过每个 Transformer Block
        for layer in self.layers:
            x = layer(x)
        
        # 层归一化后通过线性层生成 logits
        x = self.ln_f(x)
        logits = self.head(x)
        
        # 如果提供了目标，则计算交叉熵损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

# Transformer Block 类
class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        # 层归一化和自注意力层
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        # 层归一化和前馈神经网络
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    # 定义前向传播的逻辑
    def forward(self, x):
        # 残差连接，输入通过自注意力机制后再加回原输入
        x = x + self.attn(self.ln1(x))
        # 前馈网络的残差连接
        x = x + self.mlp(self.ln2(x))
        return x

# 自注意力机制类
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0

        # 定义用于计算注意力机制的 query, key, value 线性层
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        # 定义用于防止过拟合的 dropout 层
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

        # 用于将注意力结果投影回原始维度的线性层
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    # 定义前向传播的逻辑
    def forward(self, x):
        B, T, C = x.size()  # 获取批量大小、序列长度和嵌入维度
        # 计算 query, key, value， 并变换形状以适应多头注意力
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 计算注意力权重矩阵，并应用因果遮掩，防止模型“看到”未来的信息
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(
            torch.tril(torch.ones(T, T)).to(x.device) == 0, float('-inf')
        )
        # 对注意力权重进行 softmax 归一化和 dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        # 加权求和得到注意力输出
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 通过线性层和 dropout 层，并返回最终的输出
        y = self.resid_dropout(self.proj(y))
        return y

# 使用示例
config = GPTConfig(vocab_size=50257, block_size=128, n_layer=12, n_head=8, n_embd=512)
model = GPT(config)

# 生成一些随机输入数据
input_ids = torch.randint(0, config.vocab_size, (1, config.block_size))
logits, loss = model(input_ids, targets=input_ids)
print(logits, loss)  # 打印输出 logits 和损失