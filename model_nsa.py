# -* coding:utf-8 -*

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import ml_collections
import copy
import logging
import math
from os.path import join as pjoin
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from torch.autograd import Variable
from typing import Optional
from thop import profile

def get_config():
    """Custom configuration for Att_channel model"""
    config = ml_collections.ConfigDict()

    config.hidden_size = 820
    config.classifier = 'token'
    config.representation_size = None

    config.transformer = ml_collections.ConfigDict()
    config.transformer.num_heads = 41
    config.transformer.num_layers = 1
    config.transformer.mlp_dim = 3072
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.transformer.comp_block = 7
    config.transformer.sel_block = 2
    config.transformer.win_size = 5

    return config


config = get_config()

class Complex_dense(nn.Module):
    def __init__(self, n_sym=14, K=82, m_iq=2, n_sc=82):
        super(Complex_dense, self).__init__()
        self.m_iq = m_iq
        self.n_sym = n_sym
        self.K = K
        self.n_sc = n_sc

        self.antenna_norm = nn.BatchNorm2d(num_features=2)


        self.remove_cp = nn.Conv1d(
            in_channels=2,
            out_channels=2,
            kernel_size=16 + 1,
            stride=1,
            padding=0,
            bias=False
        )

        self.final_norm = nn.LayerNorm([n_sym, K, 1, 2])

    def forward(self, x):
        batch_size = x.shape[0]

        x = x.permute(0, 3, 1, 2)
        x = self.antenna_norm(x)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, 2, 1164)

        x = self.remove_cp(x)

        x = x.reshape(batch_size, 2, self.n_sym, self.K, 2)
        x = x.permute(0, 2, 3, 1, 4)
        x = x.sum(dim=3)
        x = x.unsqueeze(3)

        return self.final_norm(x)


class NativeSparseAttention(nn.Module):
    def __init__(self, embed_dim=820, num_heads=20, comp_block=7, sel_block=7, win_size=7):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.comp_block = comp_block
        self.sel_block = sel_block
        self.win_size = win_size

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.comp_mlp = nn.Sequential(
            nn.Linear(comp_block * self.head_dim, self.head_dim // 2),
            nn.ReLU(),
            nn.Linear(self.head_dim // 2, self.head_dim)
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

        attn_output = attn_output.transpose(1, 2).reshape(B, L, self.num_heads * self.head_dim)

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
        topk = min(16, block_scores.shape[-1])
        topk_scores, topk_indices = torch.topk(
            block_scores,
            k=topk,
            dim=-1
        )
        k_selected = torch.gather(k_blocks, dim=2, index=topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, k_blocks.shape[-2], k_blocks.shape[-1])).view(B, H,-1, E)
        v_selected = torch.gather(v_blocks, dim=2, index=topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, v_blocks.shape[-2], v_blocks.shape[-1])).view(B, H,-1, E)

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
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        if hidden_act != 'swish':
            raise ValueError(f'Unsupported hidden_act: {hidden_act}')

        # 线性层
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        y = self.up_proj(x)

        gate = gate * torch.sigmoid(gate)

        return self.down_proj(gate * y)


class NSABlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, comp_block=32, sel_block=32, win_size=512,
                 eps=1e-6,
                 hidden_ratio=4, intermediate_size=None, hidden_act="swish"
                 ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.comp_block = comp_block
        self.sel_block = sel_block
        self.win_size = win_size

        self.eps = eps

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.attn_norm = nn.RMSNorm(self.embed_dim, self.eps)
        self.attn = NativeSparseAttention(self.embed_dim, self.num_heads, self.comp_block, self.sel_block,
                                          self.win_size)
        self.mlp_norm = nn.RMSNorm(self.embed_dim, self.eps)
        self.mlp = GatedMLP(self.embed_dim, self.hidden_ratio, self.intermediate_size, self.hidden_act)

    def forward(self, x):
        residual = x

        x = self.attn_norm(x)
        x = self.attn(x)

        x = residual + x
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)

        outputs = residual + x

        return outputs, None

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = NSABlock(
            embed_dim=config.hidden_size,
            num_heads=config.transformer.num_heads,
            comp_block=config.transformer.get('comp_block',7),
            sel_block=config.transformer.get('sel_block',7),
            win_size=config.transformer.get('win_size',7))
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.transformer["mlp_dim"]),
            nn.ReLU(),
            nn.Dropout(config.transformer["dropout_rate"]),    # 0.1
            nn.Linear(config.transformer["mlp_dim"], config.hidden_size),
            nn.Dropout(config.transformer["dropout_rate"])
        )
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
    def forward(self, x):

        attn_output, weights = self.attn(x)
        x = self.attention_norm(x + attn_output)

        return x, weights


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Att_channel(nn.Module):
    def __init__(self, config, vis):
        super(Att_channel, self).__init__()
        self.filters = 1
        self.K = 82
        self.m_iq = 2
        self.n_sym = 14
        self.drop_cp = Complex_dense(n_sym=14, K=82, m_iq=2, n_sc=82)

        self.conv2D = nn.Sequential(
            nn.Conv2d(self.K * self.m_iq, 320, kernel_size=3, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(),
            nn.Conv2d(320, 640, kernel_size=3, padding=1),
            nn.BatchNorm2d(640),
            nn.Conv2d(640, config.hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.hidden_size),
            nn.ReLU()
        )

        self.ln_before_encoder = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.encoder = Encoder(config, vis)
        self.ln_after_encoder = nn.LayerNorm(config.hidden_size, eps=1e-6)

        self.complex_2D = nn.Conv3d(
            in_channels=5,
            out_channels=2,
            kernel_size=(1, 81, 1),
            stride=1,
            padding=(0, 40, 0)
        )
        self.bn3d = nn.BatchNorm3d(2)
        self.relu = nn.ReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.drop_cp(x)

        x = torch.reshape(x, [-1, self.K*self.m_iq, self.n_sym, 1])
        x = self.conv2D(x)

        x = x.permute(0, 2, 1, 3).squeeze(-1)

        x = self.ln_before_encoder(x)

        encoded, attn_weights = self.encoder(x)

        encoded = self.ln_after_encoder(encoded)

        encoded = encoded.reshape(-1, 5, self.n_sym, self.K, self.m_iq)
        x = self.complex_2D(encoded)

        x = self.bn3d(x)
        x = self.relu(x)

        conv = x
        shapes_conv = list(Variable(conv).size())
        conv = torch.reshape(conv, [-1, self.filters, shapes_conv[2], shapes_conv[3], shapes_conv[4] * 2])

        conv_re = conv[:, :, :, :, 0] - conv[:, :, :, :, 3]
        conv_im = conv[:, :, :, :, 1] - conv[:, :, :, :, 2]
        conv_re = torch.reshape(conv_re, [-1, self.filters, shapes_conv[2], shapes_conv[3], 1])
        conv_im = torch.reshape(conv_im, [-1, self.filters, shapes_conv[2], shapes_conv[3], 1])
        output = torch.cat([conv_re, conv_im], dim=4)
        output = torch.complex(output[:, :, :, :, 0], output[:, :, :, :, 1])
        output = output.permute(0, 2, 3, 1)

        return output


class Transition(nn.Module):

    def __init__(self, filters=82, kernal_size=(82, 1, 1), K=82):
        super(Transition, self).__init__()
        self.filters = filters
        self.kernal_size = kernal_size
        self.K = K
        # self.channel_estimatiion = Att_channel(config, vis=True)
        self.channel_estimation = Channel_estimation(filters=1, n_sym=14, K=82, m_iq=2, pilot_size=32) # according block to redefine
        self.layers_conv2D_complex = nn.Conv3d(in_channels=1,
                                               out_channels=filters*2,
                                               kernel_size=kernal_size,
                                               stride=(1, 1, 1),
                                               padding=0,
                                               )
        self.BatchNorm3D = nn.BatchNorm3d(164)
        self.relu = nn.ReLU()  # 12.8_________
        # 添加权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x, input_complex = self.channel_estimation(x)
        input_complex = torch.reshape(input_complex, (-1, 14, self.K, 1))
        equalized_complex, chest = transition_block(x, input_complex)
        equalized_complex = preparation(equalized_complex)
        equalized_complex = equalized_complex.permute(0, 4, 2, 3, 1)
        x = self.layers_conv2D_complex(equalized_complex)
        x = self.BatchNorm3D(x)
        x = self.relu(x)
        x = x.permute(0, 4, 2, 3, 1)
        conv = x
        shapes_conv = list(Variable(conv).size())
        conv = torch.reshape(conv, [-1, shapes_conv[1], shapes_conv[2], shapes_conv[3] * 2, self.filters])  #
        conv_re = conv[:, :, :, 0, :] - conv[:, :, :, 3, :]
        conv_im = conv[:, :, :, 1, :] - conv[:, :, :, 2, :]
        conv_re = torch.reshape(conv_re, [-1, shapes_conv[1], shapes_conv[2], 1, self.filters])  #
        conv_im = torch.reshape(conv_im, [-1, shapes_conv[1], shapes_conv[2], 1, self.filters])
        output = torch.cat([conv_re, conv_im], dim=3)
        output = torch.transpose(output, dim0=4, dim1=3)
        output = torch.complex(output[:, :, :, :, 0], output[:, :, :, :, 1])
        equalized_complex = torch.transpose(output, dim0=3, dim1=2)
        return equalized_complex, chest

class Att_cat_channel(nn.Module):
    def __init__(self, filters=1):
        super(Att_cat_channel, self).__init__()
        self.filters = filters
        self.att_channel = Att_channel(config, vis=True)
        self.transition = Transition(filters=82, kernal_size=(82, 1, 1), K=82)
        self.LayerNorm = nn.LayerNorm([14, 82, 2])


    def forward(self, x):
        x1 = self.att_channel(x)
        x2, _ = self.transition(x)
        x_comb = torch.mul(x1, x2)
        x = torch.cat([torch.real(x_comb), torch.imag(x_comb)], dim=-1)
        x = self.LayerNorm(x)

        return x


class Extraction(nn.Module):

    def __init__(self, n_sym=14, n_sc=82, m_iq=2, data_ofdm=832):
        super(Extraction, self).__init__()
        self.n_sym = n_sym
        self.n_sc = n_sc
        self.m_iq = m_iq
        self.data_ofdm = data_ofdm
        self.att = Att_cat_channel(filters=1)
        self.linear1 = nn.Linear(in_features=1148*m_iq, out_features=data_ofdm*m_iq)

        self.dropout = nn.Dropout(p=0.8)
        self.LayerNorm = nn.LayerNorm([1, 832, 2])

    def forward(self, x):
        x = self.att(x)
        x = torch.reshape(x, (-1, self.n_sym*self.n_sc*self.m_iq))
        x = self.linear1(x)
        x = self.dropout(x)
        x = torch.reshape(x, (-1, 1, self.data_ofdm, self.m_iq))
        x = self.LayerNorm(x)
        return x


class Con_cat(nn.Module):
    def __init__(self, m_iq=2, nbits=4):
        super(Con_cat, self).__init__()
        self.m_iq = 2
        self.nbits = 4
        self.extraction = Extraction(n_sym=14, n_sc=82, m_iq=2, data_ofdm=832)
        self.conv1 = nn.Conv2d(in_channels=m_iq, out_channels=2**nbits, kernel_size=3, stride=1, padding=1, bias=True)
        self.BatchNorm = nn.BatchNorm2d(2**nbits)
        self.Relu = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.extraction(x)
        out_iq = x
        x = x.permute(0, 3, 2, 1)
        x = self.conv1(x)
        x = self.BatchNorm(x)
        x = self.Relu(x)
        x = x.permute(0, 3, 2, 1)
        x = torch.cat([x, out_iq], dim=-1)
        return x

class DARNet(nn.Module):
    def __init__(self, data_carrier=832, nbits=4):
        super(DARNet, self).__init__()
        self.data_carrier = data_carrier
        self.nbits = nbits
        self.concat = Con_cat()

        self.demapper = nn.Sequential(
            nn.Conv2d(in_channels=data_carrier, out_channels=2*data_carrier, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(2*data_carrier),
            nn.ReLU(),
            nn.Conv2d(in_channels=2*data_carrier, out_channels=4*data_carrier, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(4 * data_carrier),
            nn.ReLU() )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=4*data_carrier, out_channels=2 * data_carrier, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(2 * data_carrier),
            nn.ReLU() )

        self.linear = nn.Sequential(
            nn.Linear(in_features=2**nbits+2, out_features=9, bias=True),
            # nn.Dropout(p=0.4),
            nn.Linear(in_features=9, out_features=2, bias=True),
            # nn.Dropout(p=0.4),
            # nn.ReLU()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.concat(x)
        x = x.permute(0, 2, 1, 3)
        x = self.demapper(x)
        x = self.decoder(x)
        x = torch.reshape(x, (-1, 2**self.nbits+2))
        x = self.linear(x)
        x = torch.reshape(x, (-1, self.data_carrier*2, 1, 2))
        return x


# FLOPs and Parameters calculation
model = DARNet()
input = torch.randn(64, 2, 1164, 2)  # Example input tensor
flops, params = profile(model, inputs=(input,))
print(f"FLOPs: {flops / 1e9:.2f} G")
print(f"Params: {params / 1e6:.2f} M")
