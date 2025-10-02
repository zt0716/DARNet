import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
# (64, 14, 820)-->(64, 820, 14)
class NativeSparseAttention(nn.Module):
    def __init__(self, embed_dim=800, num_heads=16, comp_block=25, sel_block=25, win_size=200):      # [820, 20, 7, 7, 7]
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 820/20=41
        self.comp_block = comp_block
        self.sel_block = sel_block
        self.win_size = win_size

        self.q_proj = nn.Linear(embed_dim, embed_dim)   # (800, 800)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.comp_mlp = nn.Sequential(
            nn.Linear(comp_block*self.head_dim, self.head_dim//2),      # (7*41,41/2)=(1250, 25)
            nn.ReLU(),
            nn.Linear(self.head_dim//2, self.head_dim)      # (41/2, 41)
        )

        self.gate = nn.Sequential(
            nn.Linear(self.head_dim, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        B, L, E = x.shape  # Batch size, sequence length, embedding dimension
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, E/H)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, E/H)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, E/H)

        # Compressed attention
        k_compressed = self._compress(k, self.comp_block)  # Compress keys
        v_compressed = self._compress(v, self.comp_block)  # Compress values
        attn_compressed = self._scaled_dot_product_attention(q, k_compressed, v_compressed)
        comp_attn = self._comp_attn(q, k_compressed)

        # Selected attention
        k_selected, v_selected = self._select(k, v, comp_attn)
        attn_selected = self._scaled_dot_product_attention(q, k_selected, v_selected)

        # Sliding window attention
        k_window = k[:, :, -self.win_size:]  # Take the last win_size tokens
        v_window = v[:, :, -self.win_size:]
        attn_window = self._scaled_dot_product_attention(q, k_window, v_window)

        # Combine attention outputs using gating mechanism
        gate_weights = self.gate(q)
        attn_output = gate_weights[:, :, :, 0].unsqueeze(-1) * attn_compressed + \
                      gate_weights[:, :, :, 1].unsqueeze(-1) * attn_selected + \
                      gate_weights[:, :, :, 2].unsqueeze(-1) * attn_window

        attn_output = attn_output.transpose(1, 2).reshape(B, L, self.num_heads*self.head_dim)

        return attn_output

    def _comp_attn(self, q, k):
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        return attn_weights

    def _scaled_dot_product_attention(self, q, k, v):
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output

    def _compress(self, x, comp_block):
        # Compress keys/values by aggregating blocks
        B, H, L, E = x.shape
        x_blocks = x.view(B, H, -1, comp_block, E)
        x_compressed = self.comp_mlp(x_blocks.flatten(3))
        return x_compressed

    def _select(self, k, v, comp_attn):
        # Select important blocks based on attention scores
        B, H, L, E = k.shape

        k_blocks = k.view(B, H, -1, self.sel_block, E)
        v_blocks = v.view(B, H, -1, self.sel_block, E)

        block_scores = comp_attn.sum(dim=2)  # B, H, L_compressed
        topk = min(16, block_scores.shape[-1])  # 动态调整 k
        topk_scores, topk_indices = torch.topk(
            block_scores,
            k=topk,  # 16
            dim=-1
        )

        k_selected = torch.gather(k_blocks, dim=2, index=topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, k_blocks.shape[-2], k_blocks.shape[-1])).view(B, H, -1, E)
        v_selected = torch.gather(v_blocks, dim=2, index=topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, v_blocks.shape[-2], v_blocks.shape[-1])).view(B, H, -1, E)

        return k_selected, v_selected

class GatedMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: int = 4,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish'
    ):
        super().__init__()

        self.hidden_size = hidden_size
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)  # 256对齐

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        if hidden_act != 'swish':
            raise ValueError(f'Unsupported hidden_act: {hidden_act}')

        # 线性层
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算门控值和上采样值
        gate = self.gate_proj(x)
        y = self.up_proj(x)

        # Swish-Gated Linear Unit: Swish(x) * y
        gate = gate * torch.sigmoid(gate)

        # 线性变换回原始维度
        return self.down_proj(gate * y)


class NSABlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, comp_block=32, sel_block=32, win_size=512,
                 eps=1e-6,
                 hidden_ratio=4, intermediate_size=None, hidden_act="swish"
                 ):
        super().__init__()
        #注意力的参数
        self.embed_dim = embed_dim              # 820
        self.num_heads = num_heads              # 20
        self.head_dim = embed_dim // num_heads      # 820/20=41
        self.comp_block = comp_block            # 7
        self.sel_block = sel_block              # 7
        self.win_size = win_size                # 7
        #归一化的参数
        self.eps =eps
        #多层感知机的参数
        self.hidden_ratio= hidden_ratio
        self.intermediate_size= intermediate_size
        self.hidden_act= hidden_act

        self.attn_norm = nn.RMSNorm(self.embed_dim, self.eps)
        self.attn = NativeSparseAttention(self.embed_dim, self.num_heads, self.comp_block, self.sel_block, self.win_size)
        self.mlp_norm = nn.RMSNorm(self.embed_dim, self.eps)
        self.mlp = GatedMLP(self.embed_dim, self.hidden_ratio, self.intermediate_size, self.hidden_act)   # [embed_dim=820, hidden_ratio=4, intermediate_size=None, hidden_act="swish"]


    def forward(self, x):
        #采用残差连接
        residual = x

        x = self.attn_norm(x)
        x = self.attn(x)

        x = residual + x
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)

        outputs = residual + x
        return outputs


if __name__ == '__main__':
    batch_size = 64
    seq_length = 14
    embedding_dim = 820
    num_heads = 20

    input_tensor = torch.randn(batch_size, seq_length, embedding_dim)   # (64, 14, 820)
    print(f"Input shape: {input_tensor.shape}")
    nsa_module = NSABlock(embed_dim=embedding_dim, num_heads=num_heads, comp_block=7, sel_block=7, win_size=7)
    output = nsa_module(input_tensor)
    print(f"Output shape: {output.shape}")
