# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional

import torch
from torch import nn as nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import transformer

from timm.models.vision_transformer import VisionTransformer, PatchEmbed

import copy
def clones(_to_clone_module, _clone_times, _is_deep=True):
    """Produce N identical layers."""
    copy_method = copy.deepcopy if _is_deep else copy.copy
    return nn.ModuleList([copy_method(_to_clone_module) for _ in range(_clone_times if _is_deep else 1)])

class MultiHeadAttention(torch.jit.ScriptModule):
    def __init__(self, _multi_attention_heads, _dimensions, _dropout=0.1):
        """

        :param _multi_attention_heads: number of self attention head
        :param _dimensions: dimension of model
        :param _dropout:
        """
        super(MultiHeadAttention, self).__init__()

        assert _dimensions % _multi_attention_heads == 0
        # requires d_v = d_k, d_q = d_k = d_v = d_m / h
        self.d_k = int(_dimensions / _multi_attention_heads)
        self.h = _multi_attention_heads
        self.linears = clones(nn.Linear(_dimensions, _dimensions), 4)  # (q, k, v, last output layer)
        self.attention = None
        self.dropout = nn.Dropout(p=_dropout)

    @torch.jit.script_method
    def dot_product_attention(self, _query, _key, _value, _mask):
        """
        Compute 'Scaled Dot Product Attention

        :param _query: (N, h, seq_len, d_q), h is multi-head
        :param _key: (N, h, seq_len, d_k)
        :param _value: (N, h, seq_len, d_v)
        :param _mask: None or (N, 1, seq_len, seq_len), 0 will be replaced with -1e9
        :return:
        """

        d_k = _value.size(-1)
        score = torch.matmul(_query, _key.transpose(-2, -1)) / math.sqrt(d_k)  # (N, h, seq_len, seq_len)
        if _mask is not None:
            score = score.masked_fill(_mask == 0, -1e9)  # score (N, h, seq_len, seq_len)
        p_attn = F.softmax(score, dim=-1)
        # (N, h, seq_len, d_v), (N, h, seq_len, seq_len)
        return torch.matmul(p_attn, _value), p_attn

    @torch.jit.script_method
    def forward(self, _query, _key, _value, _mask):
        batch_size = _query.size(0)

        # do all the linear projections in batch from d_model => h x d_k
        # (N, seq_len, d_m) -> (N, seq_len, h, d_k) -> (N, h, seq_len, d_k)
        _query, _key, _value = \
            [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (_query, _key, _value))]

        # apply attention on all the projected vectors in batch.
        # (N, h, seq_len, d_v), (N, h, seq_len, seq_len)
        product_and_attention = self.dot_product_attention(_query, _key, _value, _mask=_mask)
        x = product_and_attention[0]
        # self.attention = self.dropout(product_and_attention[1])

        # "Concat" using a view and apply a final linear.
        # (N, seq_len, d_m)
        x = x.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.h * self.d_k)

        # (N, seq_len, d_m)
        return self.linears[-1](x)


class DecoderLayer(nn.Module):
    """A Transformer decoder layer supporting two-stream attention (XLNet)
       This implements a pre-LN decoder, as opposed to the post-LN default in PyTorch."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu',
                 layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super().__setstate__(state)

    def forward_stream(self, tgt: Tensor, tgt_norm: Tensor, tgt_kv: Tensor, memory: Tensor, tgt_mask: Optional[Tensor],
                       tgt_key_padding_mask: Optional[Tensor]):
        """Forward pass for a single stream (i.e. content or query)
        tgt_norm is just a LayerNorm'd tgt. Added as a separate parameter for efficiency.
        Both tgt_kv and memory are expected to be LayerNorm'd too.
        memory is LayerNorm'd by ViT.  和论文一致
        """
        tgt2, sa_weights = self.self_attn(tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask,
                                          key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)

        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(tgt)))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, sa_weights, ca_weights

    def forward(self, query, content, memory, query_mask: Optional[Tensor] = None, content_mask: Optional[Tensor] = None,
                content_key_padding_mask: Optional[Tensor] = None, update_content: bool = True):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(query, query_norm, content_norm, memory, query_mask, content_key_padding_mask)[0]
        if update_content:
            content = self.forward_stream(content, content_norm, content_norm, memory, content_mask,
                                          content_key_padding_mask)[0]
        return query, content


#总的来说，这段代码定义了一个解码器类，用于将查询和内容输入经过多个解码器层进行处理，以生成模型的输出。
# 解码器层通常包含了注意力机制和其他处理，用于生成目标序列或执行其他序列处理任务。这是序列到序列模型中解码器的一般结构。
# 这段代码是一个自定义的TransformerDecoder类的forward方法，它的作用是根据输入的query, content, memory和相应的掩码，返回一个经过多层解码器层和层归一化处理后的query向量。

# 参数说明：

# query(Tensor)：解码器的输入序列，形状为[batch_size, tgt_len, d_model]，其中batch_size是批次大小，tgt_len是目标序列长度，d_model是模型维度。
# content(Tensor)：解码器自身的输出序列，形状为[batch_size, tgt_len, d_model]，与query相同。
# memory(Tensor)：编码器的输出序列，形状为[batch_size, src_len, d_model]，其中src_len是源序列长度。
# query_mask(Optional[Tensor])：用于屏蔽query序列中未来信息的掩码，形状为[tgt_len, tgt_len]，其中tgt_len是目标序列长度。如果该位置应该被屏蔽，则为True，否则为False。
# content_mask(Optional[Tensor])：用于屏蔽content序列中未来信息的掩码，形状与query_mask相同。
# content_key_padding_mask(Optional[Tensor])：用于屏蔽content序列中无效值（如填充符）的掩码，形状为[batch_size, tgt_len]，其中batch_size是批次大小，tgt_len是目标序列长度。如果该位置是填充符，则为True，否则为False。
# 方法说明：

# self.layers：一个包含多个TransformerDecoderLayer对象的模块列表，每个对象都实现了一个解码器层的功能。
# self.norm：一个层归一化（LayerNorm）对象，用于对最后一层解码器层的输出进行归一化处理。
# for i, mod in enumerate(self.layers)：遍历每个解码器层对象，并记录其索引i。
# last = i == len(self.layers) - 1：判断是否是最后一层解码器层。
# query, content = mod(query, content, memory, query_mask, content_mask, content_key_padding_mask, update_content=not last)：调用当前解码器层对象的forward方法，传入query, content, memory和相应的掩码，并根据update_content参数决定是否更新content。返回更新后的query和content。
# query = self.norm(query)：对最后一层解码器层的输出query进行归一化处理。
# return query：返回最终的query向量。
class Decoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, content, memory, query_mask: Optional[Tensor] = None, content_mask: Optional[Tensor] = None,
                content_key_padding_mask: Optional[Tensor] = None):
        for i, mod in enumerate(self.layers):
            last = i == len(self.layers) - 1 #True
            query, content = mod(query, content, memory, query_mask, content_mask, content_key_padding_mask,
                                 update_content=not last)
        query = self.norm(query)
        return query


class Encoder(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed):
        super().__init__(img_size, patch_size, in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         num_classes=0, global_pool='', class_token=False)  # these disable the classifier head

    def forward(self, x):
        # Return all tokens
        return self.forward_features(x)


class TokenEmbedding(nn.Module):

    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]  192*26*512
        output : contextual feature [batch_size x T x output_size]
        """
        # 为了提高内存的利用率和效率，调用flatten_parameters让parameter的数据存放成contiguous chunk(连续的块)
        self.rnn.flatten_parameters()
        
        #只需要隐藏层的状态即可
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output  # 192*26*256
    
#动态融合机制
class DynamicalFusionLayer(nn.Module):

    def __init__(self, dim_model, dim=-1):
        super(DynamicalFusionLayer, self).__init__()

        self.dim_model = dim_model
        self.dim = dim

        self.linear_layer = nn.Linear(dim_model * 2, dim_model * 2)
        self.glu_layer = nn.GLU(dim=dim) #Gated Linear Unit 线性门控单元

    # def forward(self, x0, x1, x2):
    #     assert x0.size() == x1.size()
	#     # assert x0.size() == x2.size()
    #     fusion_input = torch.cat([x0, x1], self.dim)
    #     fusion_input = torch.cat([fusion_input, x2], self.dim)
    #     output = self.linear_layer(fusion_input)
    #     output = self.glu_layer(output)

    #     return output

    def forward(self, x0, x1):
        assert x0.size() == x1.size()
	    # assert x0.size() == x2.size()
        fusion_input = torch.cat([x0, x1], self.dim)
        # fusion_input = torch.cat([fusion_input, x2], self.dim)
        output = self.linear_layer(fusion_input)
        output = self.glu_layer(output)

        return output