import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class DebuggableLinear(nn.Linear):
    """带调试信息的线性层"""

    def forward(self, input):
        if input is None:
            raise ValueError("Linear层接收到None输入")

        if torch.is_tensor(input) and input.numel() == 0:
            raise ValueError("Linear层接收到空张量")

        # 检查输入是否包含NaN或inf
        if torch.is_tensor(input) and (torch.isnan(input).any() or torch.isinf(input).any()):
            print(f"警告: 线性层输入包含NaN或inf")

        return F.linear(input, self.weight, self.bias)


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        if x is None:
            raise ValueError("位置编码输入不能为None")
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """多头自注意力机制 - 带调试版本"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 使用可调试的线性层
        self.W_q = DebuggableLinear(d_model, d_model, bias=False)
        self.W_k = DebuggableLinear(d_model, d_model, bias=False)
        self.W_v = DebuggableLinear(d_model, d_model, bias=False)
        self.W_o = DebuggableLinear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        # 输入验证
        for name, tensor in [("query", query), ("key", key), ("value", value)]:
            if tensor is None:
                raise ValueError(f"{name} cannot be None")
            if not torch.is_tensor(tensor):
                raise ValueError(f"{name} must be a tensor, got {type(tensor)}")

        batch_size, seq_len = query.size(0), query.size(1)

        # 线性变换并分头 (batch_size, seq_len, n_heads, d_k)
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数 (batch_size, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用mask（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重到values
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, n_heads, seq_len, d_k)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # 输出投影
        output = self.W_o(attn_output)

        return output, attn_weights


class PositionwiseFFN(nn.Module):
    """逐位前馈网络 - 带调试版本"""

    def __init__(self, d_model, d_ff, dropout=0.1, activation="relu"):
        super().__init__()
        self.linear1 = DebuggableLinear(d_model, d_ff)
        self.linear2 = DebuggableLinear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        if x is None:
            raise ValueError("FFN输入不能为None")

        # 打印调试信息（仅在调试模式下）
        if self.training and torch.isnan(x).any():
            print("FFN输入包含NaN")

        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


class SublayerConnection(nn.Module):
    """残差连接 + LayerNorm"""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """x -> norm -> sublayer -> dropout -> + x"""
        if x is None:
            raise ValueError("SublayerConnection输入不能为None")

        return x + self.dropout(sublayer(self.norm(x)))


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层 - 带调试版本"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFFN(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        if x is None:
            raise ValueError(f"Decoder层 {self.layer_idx} 输入为None")

        # 自注意力子层
        def self_attn_sublayer(x):
            attn_out, weights = self.self_attn(x, x, x, tgt_mask)
            return attn_out

        x = self.sublayer1(x, self_attn_sublayer)

        # 前馈网络子层
        x = self.sublayer2(x, self.feed_forward)

        return x


class TransformerDecoder(nn.Module):
    """Transformer解码器 - 带调试版本"""

    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # 创建带索引的解码器层
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, i)
            for i in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        # 嵌入 + 位置编码
        if x is None:
            raise ValueError("Decoder输入不能为None")

        x_embed = self.token_embedding(x)
        if x_embed is None:
            raise ValueError("词嵌入输出为None")

        x_embed = self.pos_encoding(x_embed)
        x = self.dropout(x_embed)

        # 通过所有解码器层
        for i, layer in enumerate(self.layers):
            if x is None:
                raise ValueError(f"第 {i} 层后输出为None")

            x = layer(x, memory, src_mask, tgt_mask)

        return self.norm(x)


class TransformerLM(nn.Module):
    """仅用于语言建模的Transformer（只有Decoder）- 带调试版本"""

    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.debug_mode = True  # 调试模式

        self.decoder = TransformerDecoder(vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len, dropout)
        self.output_proj = DebuggableLinear(d_model, vocab_size)

        # 参数初始化
        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x, mask=None):
        if x is None:
            raise ValueError("模型输入不能为None")

        if not torch.is_tensor(x):
            raise ValueError(f"输入必须是张量，得到 {type(x)}")

        # 检查输入范围
        if x.dtype != torch.long:
            print(f"警告: 输入数据类型为 {x.dtype}，期望 torch.long")

        # 创建因果mask
        batch_size, seq_len = x.size(0), x.size(1)
        causal_mask = self._generate_causal_mask(seq_len).to(x.device)

        # 通过解码器
        decoder_output = self.decoder(x, None, None, causal_mask)

        if decoder_output is None:
            raise ValueError("解码器输出为None")

        # 输出投影
        output = self.output_proj(decoder_output)

        return output

    def _generate_causal_mask(self, seq_len):
        """生成因果mask（防止看到未来信息）"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return ~mask

    def generate(self, start_tokens, max_len, temperature=1.0):
        """生成文本"""
        self.eval()
        with torch.no_grad():
            generated = start_tokens.clone()

            for _ in range(max_len - len(start_tokens)):
                logits = self.forward(generated.unsqueeze(0))[0, -1, :]
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_token])

            return generated