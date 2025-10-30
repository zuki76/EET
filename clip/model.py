from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _default_init_func(_m):
    if isinstance(_m, nn.Linear):
        nn.init.trunc_normal_(_m.weight, std=.02)
        if _m.bias is not None:
            nn.init.constant_(_m.bias, 0)
    elif isinstance(_m, nn.LayerNorm):
        nn.init.constant_(_m.bias, 0)
        nn.init.constant_(_m.weight, 1.0)
    elif isinstance(_m, nn.Parameter):
        nn.init.trunc_normal_(_m, std=.02)


class EET_transformer_TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.):
        super().__init__()
        
        # Attention for Layer Summarization
        self.layer_summarizer_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.layer_summary_token = nn.Parameter(torch.zeros(1, 1, d_model, dtype=torch.float16), requires_grad=True)
        # self.layer_summary_token = nn.Parameter(torch.zeros(1, 12, d_model, dtype=torch.float16), requires_grad=True)
        self.norm_summarizer = LayerNorm(d_model)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = F.gelu

    def forward(self, query, value):
        """
        Args:
            query (torch.Tensor): (B, L_q, D)
            value (torch.Tensor): (B, L_v, N, D) 
        """
        B, L_v, N, D = value.shape
        
        # (B, L_v, N, D) -> (B * L_v, N, D)
        value_reshaped = value.reshape(B * L_v, N, D)
        # value_reshaped = value.reshape(B, N* L_v, D)
        
        # [SUMMARY] token: (1, 1, D) -> (B * L_v, 1, D)
        summary_token = self.layer_summary_token.expand(B * L_v, -1, -1)
        # summary_token = self.layer_summary_token.expand(B * L_v // 12, 12, 1, -1)  # (12,1,D) -> (K,12,1,D)
        # summary_token = summary_token.reshape(B * L_v, 1, -1)  # (K*12,1,D) = (B*L_v,1,D)
        
        value_with_summary = torch.cat([summary_token, value_reshaped], dim=1)
        
        summarized_value = self.layer_summarizer_attn(
            query=self.norm_summarizer(value_with_summary[:, 0:1, :]), # summary_token
            key=self.norm_summarizer(value_with_summary),
            value=self.norm_summarizer(value_with_summary),
            need_weights=False
        )[0]
        
        summarized_value = summarized_value.reshape(B, L_v*1, D)

        # # ----------------- Ablation ------------------
        # summarized_value = value.mean(dim=2)

        q_shortcut_2 = query
        query_norm_2 = self.norm2(query)
        cross_attn_out = self.multihead_attn(
            query=query_norm_2,
            key=summarized_value,
            value=summarized_value,
            need_weights=False
        )[0]
        query = q_shortcut_2 + self.dropout2(cross_attn_out)
        
        q_shortcut_3 = query
        query_norm_3 = self.norm3(query)
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(query_norm_3))))
        query = q_shortcut_3 + self.dropout3(ffn_out)
        
        return query


class EET_resnet50_TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.):
        super().__init__()
        # Attention for Layer Summarization
        self.layer_summarizer_attn_1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.layer_summarizer_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.layer_summarizer_attn_3 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.layer_summarizer_attn_4 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.layer_summary_token_1 = nn.Parameter(torch.zeros(1, 1, d_model, dtype=torch.float16), requires_grad=True)
        self.layer_summary_token_2 = nn.Parameter(torch.zeros(1, 1, d_model, dtype=torch.float16), requires_grad=True)
        self.layer_summary_token_3 = nn.Parameter(torch.zeros(1, 1, d_model, dtype=torch.float16), requires_grad=True)
        self.layer_summary_token_4 = nn.Parameter(torch.zeros(1, 1, d_model, dtype=torch.float16), requires_grad=True)

        self.norm_summarizer = LayerNorm(d_model)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = F.gelu

    def forward(self, query, value):
        """
        Args:
            query (torch.Tensor): (B, L_q, D)
            value (torch.Tensor): (B, L_v, N, D)
        """
        B, N, D = value.shape

        value_1 = value[:, :3136, :]
        value_2 = value[:, 3136:3136+784, :]
        value_3 = value[:, 3136+784:3136+784+196, :]
        value_4 = value[:, 3136+784+196:, :]

        summary_token_1 = self.layer_summary_token_1.expand(B, -1, -1)
        value_with_summary_1 = torch.cat([summary_token_1, value_1], dim=1)
        summarized_value_1 = self.layer_summarizer_attn_1(
            query=self.norm_summarizer(value_with_summary_1[:, 0:1, :]),
            key=self.norm_summarizer(value_with_summary_1),
            value=self.norm_summarizer(value_with_summary_1),
            need_weights=False
        )[0]
        summary_token_2 = self.layer_summary_token_2.expand(B, -1, -1)
        value_with_summary_2 = torch.cat([summary_token_2, value_2], dim=1)
        summarized_value_2 = self.layer_summarizer_attn_2(
            query=self.norm_summarizer(value_with_summary_2[:, 0:1, :]),
            key=self.norm_summarizer(value_with_summary_2),
            value=self.norm_summarizer(value_with_summary_2),
            need_weights=False
        )[0]
        summary_token_3 = self.layer_summary_token_3.expand(B, -1, -1)
        value_with_summary_3 = torch.cat([summary_token_3, value_3], dim=1)
        summarized_value_3 = self.layer_summarizer_attn_3(
            query=self.norm_summarizer(value_with_summary_3[:, 0:1, :]),
            key=self.norm_summarizer(value_with_summary_3),
            value=self.norm_summarizer(value_with_summary_3),
            need_weights=False
        )[0]
        summary_token_4 = self.layer_summary_token_4.expand(B, -1, -1)
        value_with_summary_4 = torch.cat([summary_token_4, value_4], dim=1)
        summarized_value_4 = self.layer_summarizer_attn_4(
            query=self.norm_summarizer(value_with_summary_4[:, 0:1, :]),
            key=self.norm_summarizer(value_with_summary_4),
            value=self.norm_summarizer(value_with_summary_4),
            need_weights=False
        )[0]

        summarized_value = torch.cat([summarized_value_1, summarized_value_2, summarized_value_3, summarized_value_4], dim=1)

        q_shortcut_2 = query
        query_norm_2 = self.norm2(query)
        cross_attn_out = self.multihead_attn(
            query=query_norm_2,
            key=summarized_value,
            value=summarized_value,
            need_weights=False
        )[0]
        query = q_shortcut_2 + self.dropout2(cross_attn_out)
        
        q_shortcut_3 = query
        query_norm_3 = self.norm3(query)
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(query_norm_3))))
        query = q_shortcut_3 + self.dropout3(ffn_out)
        
        return query


class EET_transformer_WeightsGenerater(nn.Module):
    def __init__(self, dim, depth, \
                 weights_num, lora_rank, \
                 mlp_ratio=4, num_heads=16):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, weights_num, dim, dtype=torch.float16), requires_grad=True)
        # self.queries = nn.Parameter(torch.zeros(1, weights_num, dim, dtype=torch.float16), requires_grad=True)
        self.decoders = nn.ModuleList([EET_transformer_TransformerDecoderLayer(dim, num_heads, int(dim*mlp_ratio)) for i in range(depth)])
        self.norm = LayerNorm(dim)
        self.weights_head = nn.Linear(dim, dim*lora_rank)

    def forward(self, img_feature):
        # img_feature: B,196,dim
        query = self.queries.expand(img_feature.shape[0], -1, -1) # B,12,dim
        for layer in self.decoders:
            query = layer(query, img_feature)
        query = self.weights_head(self.norm(query))
        return query  # [batch_size, weights_num, dim*lora_rank]


class EET_resnet50_WeightsGenerater(nn.Module):
    def __init__(self, dim, depth, \
                 weights_num, lora_rank, \
                 mlp_ratio=4, num_heads=16):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, weights_num, dim, dtype=torch.float16), requires_grad=True)
        # self.queries = nn.Parameter(torch.zeros(1, weights_num, dim, dtype=torch.float16), requires_grad=True)
        self.decoders = nn.ModuleList([EET_resnet50_TransformerDecoderLayer(dim, num_heads, int(dim*mlp_ratio)) for i in range(depth)])
        self.norm = LayerNorm(dim)
        self.weights_head = nn.Linear(dim, dim*lora_rank)
        
    def forward(self, img_feature):
        # img_feature: B,196,dim
        query = self.queries.expand(img_feature.shape[0], -1, -1) # B,12,dim
        for layer in self.decoders:
            query = layer(query, img_feature)
        query = self.weights_head(self.norm(query))
        return query  # [batch_size, weights_num, dim*lora_rank]


class EET_transformer_LoRAGenerater(nn.Module):
    def __init__(self, dim, N, \
                 llm_depth=12, length=9, lora_rank=64, \
                 visual_dim=768, pos_num=50, \
                 mlp_ratio=4, num_heads=16, skip_layers=1, vlora_alpha=1.0):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, pos_num, dim, dtype=torch.float16), requires_grad=True)
        self.skip_layers = skip_layers
        self.gen = EET_transformer_WeightsGenerater(dim, N, llm_depth//skip_layers, lora_rank, mlp_ratio, num_heads)
        # initialization before init self.Bs
        self.apply(_default_init_func)
        # self.Bs = nn.Parameter(0.02*torch.randn(1, llm_depth//skip_layers, dim*lora_rank, dtype=torch.float16), requires_grad=True)
        self.Bs = nn.Parameter(torch.zeros(1, llm_depth//skip_layers, dim*lora_rank, dtype=torch.float16), requires_grad=True)
        self.vlora_alpha = vlora_alpha if vlora_alpha is not None else lora_rank
        # self.vlora_alpha = nn.Parameter(torch.tensor(vlora_alpha if vlora_alpha is not None else lora_rank, dtype=torch.float16), requires_grad=True)
        self.llm_depth = llm_depth
        self.dim = dim
        self.lora_rank = lora_rank
        self.length = length
    
    def forward(self, img_feature):
        """
        img_features: output by vision encoder, shape of [batch_size, 196, 768]
        output: [(A, B),...] A's shape: [batch_size, dim, r]
                                   B's shape: [1, dim, r]
        """
        img_feature = self.visual_proj(img_feature) + self.pos_embed
        weights = self.gen(img_feature)  # [batch_size, llm_depth, dim*lora_rank]
        weights = weights.reshape(weights.shape[0], self.llm_depth//self.skip_layers, self.dim, self.lora_rank)

        Bs = self.Bs.reshape(-1, self.llm_depth//self.skip_layers, self.dim, self.lora_rank)
        Bs = self.vlora_alpha / self.lora_rank * Bs

        lora_weights_list = []
        for depth in range(self.llm_depth):
            if((depth+1) % self.skip_layers == 0 and depth < self.length):
                A = weights[:, depth//self.skip_layers]
                B = Bs[:, depth//self.skip_layers]
                lora_weights = (A, B)
            else:
                lora_weights = (None, None)
            lora_weights_list.append(lora_weights)
        return lora_weights_list


class ResNetFeatureAdapter(nn.Module):
    def __init__(self, resnet_channels, d_model):
        """
        Args:
            resnet_channels (list[int]): [256, 512, 1024, 2048]
            d_model (int): Transformer hidden dimension
        """
        super().__init__()
        self.d_model = d_model
        
        # 1x1 conv for unifying
        self.proj_layer1 = nn.Conv2d(resnet_channels[0], d_model, kernel_size=1)
        self.proj_layer2 = nn.Conv2d(resnet_channels[1], d_model, kernel_size=1)
        self.proj_layer3 = nn.Conv2d(resnet_channels[2], d_model, kernel_size=1)
        self.proj_layer4 = nn.Conv2d(resnet_channels[3], d_model, kernel_size=1)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.norm4 = LayerNorm(d_model)

    def forward(self, features_list):
        """
        Args:
            features_list (list[torch.Tensor]):
                - features_list[0]: (B, 256, 56, 56)
                - features_list[1]: (B, 512, 28, 28)
                - features_list[2]: (B, 1024, 14, 14)
                - features_list[3]: (B, 2048, 7, 7)

        Returns:
            torch.Tensor: (B, N_total, D)
                          N_total = 56*56 + 28*28 + 14*14 + 7*7 = 4165
        """
        # (B, 256, 56, 56) -> (B, D, 56, 56)
        p1 = self.proj_layer1(features_list[0])
        B, D, H, W = p1.shape
        # (B, D, 56*56) -> (B, 56*56, D)
        p1 = p1.flatten(2).permute(0, 2, 1)
        p1 = self.norm1(p1)

        # (B, 512, 28, 28) -> (B, D, 28, 28) -> (B, 28*28, D)
        p2 = self.proj_layer2(features_list[1]).flatten(2).permute(0, 2, 1)
        p2 = self.norm2(p2)

        # (B, 1024, 14, 14) -> (B, D, 14, 14) -> (B, 14*14, D)
        p3 = self.proj_layer3(features_list[2]).flatten(2).permute(0, 2, 1)
        p3 = self.norm3(p3)

        # (B, 2048, 7, 7) -> (B, D, 7, 7) -> (B, 7*7, D)
        p4 = self.proj_layer4(features_list[3]).flatten(2).permute(0, 2, 1)
        p4 = self.norm4(p4)

        # (B, 3136, D), (B, 784, D), (B, 196, D), (B, 49, D) -> (B, 4165, D)
        all_tokens = torch.cat([p1, p2, p3, p4], dim=1)
        
        return all_tokens


class EET_resnet50_LoRAGenerater(nn.Module):
    def __init__(self, dim, N, \
                 llm_depth=12, length=9, lora_rank=64, \
                 visual_dim=768, pos_num=4165, \
                 mlp_ratio=4, num_heads=16, skip_layers=1, vlora_alpha=1.0):
        super().__init__()
        # self.visual_proj = nn.Linear(visual_dim, dim)
        self.feature_adapter = ResNetFeatureAdapter(
            resnet_channels=[256, 512, 1024, 2048], 
            d_model=dim
        )
        self.pos_embed = nn.Parameter(torch.randn(1, pos_num, dim, dtype=torch.float16), requires_grad=True)
        self.skip_layers = skip_layers
        self.gen = EET_resnet50_WeightsGenerater(dim, N, llm_depth//skip_layers, lora_rank, mlp_ratio, num_heads)
        self.apply(_default_init_func)
        self.Bs = nn.Parameter(torch.zeros(1, llm_depth//skip_layers, dim*lora_rank, dtype=torch.float16), requires_grad=True)
        self.vlora_alpha = vlora_alpha if vlora_alpha is not None else lora_rank
        # self.vlora_alpha = nn.Parameter(torch.tensor(vlora_alpha if vlora_alpha is not None else lora_rank, dtype=torch.float16), requires_grad=True)
        self.llm_depth = llm_depth
        self.dim = dim
        self.lora_rank = lora_rank
        self.length = length
    
    def forward(self, img_feature):
        """
        img_features: output by vision encoder, shape of [batch_size, 196, 768]
        output: [(A, B),...] A's shape: [batch_size, dim, r]
                                   B's shape: [1, dim, r]
        """
        # img_feature = self.visual_proj(img_feature) + self.pos_embed
        img_feature = self.feature_adapter(img_feature) + self.pos_embed
        
        weights = self.gen(img_feature)  # [batch_size, llm_depth, dim*lora_rank]
        weights = weights.reshape(weights.shape[0], self.llm_depth//self.skip_layers, self.dim, self.lora_rank)

        Bs = self.Bs.reshape(-1, self.llm_depth//self.skip_layers, self.dim, self.lora_rank)
        Bs = self.vlora_alpha / self.lora_rank * Bs

        lora_weights_list = []
        for depth in range(self.llm_depth):
            if((depth+1) % self.skip_layers == 0 and depth < self.length):
                A = weights[:, depth//self.skip_layers]
                B = Bs[:, depth//self.skip_layers]
                lora_weights = (A, B)
            else:
                lora_weights = (None, None)
            lora_weights_list.append(lora_weights)
        return lora_weights_list


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock_IVLP(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False,
                 text_layer=False, i=0, design_details=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # Only add learnable tokens if flag is set True
        # For the first iteration i, we should not add the learnable parameters
        # as it is already been taken care of in the very start, for both text
        # and the visual branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        if i != 0:
            self.add_prompt = add_prompt
            if self.add_prompt:
                if self.text_layer:
                    self.n_ctx_text = design_details["language_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_text, d_model)
                else:
                    self.n_ctx_visual = design_details["vision_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_visual, d_model)
                # Code snippet for per layer visual prompts
                nn.init.normal_(ctx_vectors, std=0.02)
                self.VPT_shallow = nn.Parameter(ctx_vectors)
        else:
            self.add_prompt = False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # Will need to append the learnable tokens for this layer here
        # Check if flag was set for this layer or not
        if self.add_prompt:
            # Also see if this is textual transformer layer or not
            if not self.text_layer:
                # Remove the outputs produced by learnable tokens of previous layer
                prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
                # Create/configure learnable tokens of this layer
                visual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, by replacing the previous
                # layer learnable tokens
                x = torch.cat([prefix, visual_context], dim=0)
            else:
                # Appending the learnable tokens in different way
                # x -> [77, NCLS, DIM]
                # First remove the learnable tokens from previous layer
                prefix = x[:1, :, :]
                suffix = x[1 + self.n_ctx_text:, :, :]
                # Create/configure learnable tokens of this layer
                textual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, replaced by previous
                # layer learnable tokens
                x = torch.cat([prefix, textual_context, suffix], dim=0)

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock_MaPLe(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, design_details=None,
                 text_layer=False, i=0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # For the first iteration i, we do not need to add the learnable parameters here
        # as it will be added in the beginning, for both text and the vision branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        # This must be consistent with the config file prompt
        self.compound_prompt_nctx = design_details['maple_length']
        if i == 0:
            self.first_layer = True
        else:
            self.first_layer = False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, inputs):
        # For the first layer, we do not need to add any duplicate, as it is already added
        # as the shallow version
        x = inputs[0]
        compound_prompts_deeper = inputs[1]
        counter = inputs[2]
        if not self.first_layer:
            if len(compound_prompts_deeper) > 0:
                # This means that deeper compound prompts are turned on
                # Here it behaves differently for text and visual side
                # Forward function is same for both

                if not self.text_layer:
                    # First check if the ith layer needs compound prompts or not
                    if not (counter > len(compound_prompts_deeper) - 1):
                        # Remove the outputs produced by learnable tokens of previous layer
                        prefix = x[0:x.shape[0] - self.compound_prompt_nctx, :, :]
                        # Create/configure learnable tokens of this layer
                        visual_context = compound_prompts_deeper[counter]  # extract the correct index
                        visual_context = visual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        # Add the learnable tokens of this layer with the input, by replacing previous
                        # layer learnable tokens
                        x = torch.cat([prefix, visual_context], dim=0)

                        # Once done, update the counter, so that the next time, it does not use same learnable tokens
                        counter += 1
                else:
                    # First check if the ith layer needs compound prompts or not
                    if not (counter > len(compound_prompts_deeper) - 1):
                        # Appending the learnable tokens in different way
                        # x -> [77, NCLS, DIM]
                        # First remove the learnable tokens from previous layer
                        prefix = x[:1, :, :]
                        suffix = x[1 + self.compound_prompt_nctx:, :, :]
                        # Create/configure learnable tokens of this layer
                        textual_context = compound_prompts_deeper[counter]
                        textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        # Add the learnable tokens of this layer with the input, replaced by previous
                        # layer learnable tokens
                        x = torch.cat([prefix, textual_context, suffix], dim=0)
                        # Once done, update the counter, so that the next time, it does not use same learnable tokens
                        counter += 1
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return [x, compound_prompts_deeper, counter]  # return again as a list, so that nn.seq can work


class ResidualAttentionBlock_EET_transformer(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, design_details=None,
                 text_layer=False, i=0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # For the first iteration i, we do not need to add the learnable parameters here
        # as it will be added in the beginning, for both text and the vision branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        self.d_model = d_model
        self.n_head = n_head

        self.register_buffer('noise_buffer', None)
        self.noise_shape = None
        # This must be consistent with the config file prompt
        self.compound_prompt_nctx = design_details['maple_length']
        if i == 0:
            self.first_layer = True
        else:
            self.first_layer = False
        self.i = i

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, inputs_weights):
        # For the first layer, we do not need to add any duplicate, as it is already added
        # as the shallow version
        inputs = inputs_weights[0]
        lora_weights_list = inputs_weights[1]
        lora_weight_A, lora_weight_B = lora_weights_list[self.i] # b*dim*r, 1*dim*r

        x = inputs[0]
        compound_prompts_deeper = inputs[1]
        counter = inputs[2]
        if not self.first_layer:
            if len(compound_prompts_deeper) > 0:
                # This means that deeper compound prompts are turned on
                # Here it behaves differently for text and visual side
                # Forward function is same for both

                if not self.text_layer:
                    # First check if the ith layer needs compound prompts or not
                    if not (counter > len(compound_prompts_deeper) - 1):
                        # Remove the outputs produced by learnable tokens of previous layer
                        prefix = x[0:x.shape[0] - self.compound_prompt_nctx, :, :]
                        # Create/configure learnable tokens of this layer
                        visual_context = compound_prompts_deeper[counter]  # extract the correct index
                        visual_context = visual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        # Add the learnable tokens of this layer with the input, by replacing previous
                        # layer learnable tokens
                        x = torch.cat([prefix, visual_context], dim=0)

                        # Once done, update the counter, so that the next time, it does not use same learnable tokens
                        counter += 1
                else:
                    # First check if the ith layer needs compound prompts or not
                    if not (counter > len(compound_prompts_deeper) - 1):
                        # Appending the learnable tokens in different way
                        # x -> [77, NCLS, DIM]
                        # First remove the learnable tokens from previous layer
                        prefix = x[:1, :, :]
                        suffix = x[1 + self.compound_prompt_nctx:, :, :]
                        # Create/configure learnable tokens of this layer
                        textual_context = compound_prompts_deeper[counter]
                        textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        # Add the learnable tokens of this layer with the input, replaced by previous
                        # layer learnable tokens
                        x = torch.cat([prefix, textual_context, suffix], dim=0)
                        # Once done, update the counter, so that the next time, it does not use same learnable tokens
                        counter += 1
        lnx = self.ln_1(x)
        att_x = self.attention(lnx)
        
        if lora_weight_A is None or lora_weight_B is None:
            x = x + att_x
        else:
            pw_mean = torch.mean(att_x)
            pw_mean = pw_mean * (1 + 0.1 * torch.rand(1, device=pw_mean.device, dtype=torch.float16))
            # pw_mean = torch.mean(x)
            B_x, N_x, C_x = x.shape
            # pw_noise = pw_mean * torch.randn(B_x, N_x, C_x, dtype=torch.float16).to(pw_mean.device)
            if self.noise_buffer is None or self.noise_shape != (B_x, N_x, C_x):
                # regenerate with shape change
                self.noise_buffer = torch.randn(B_x, N_x, C_x,
                                                dtype=torch.float16,
                                                device=pw_mean.device)
                self.noise_shape = (B_x, N_x, C_x)
            pw_noise = self.noise_buffer * pw_mean

            if not self.text_layer:
                lora_weight_B = lora_weight_B.permute(0, 2, 1).expand(lora_weight_A.shape[0], -1, -1)
                
                delta_x = lnx.permute(1, 0, 2)
                delta_x = torch.einsum('bld,bdr->blr', delta_x, lora_weight_A)
                delta_x = torch.einsum('blr,brd->bld', delta_x, lora_weight_B)
                delta_x = delta_x.permute(1, 0, 2)
            else:
                lora_weight_A = lora_weight_A.mean(dim=0)
                lora_weight_B = lora_weight_B.permute(0, 2, 1).squeeze()
                delta_x = lnx @ lora_weight_A @ lora_weight_B
        
            x = x + att_x + 0.1 * delta_x + pw_noise

        x = x + self.mlp(self.ln_2(x))
        outputs = [x, compound_prompts_deeper, counter]
        return [outputs, lora_weights_list]  # return again as a list, so that nn.seq can work


class ResidualAttentionBlock_EET_resnet50(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, design_details=None,
                 text_layer=False, i=0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # For the first iteration i, we do not need to add the learnable parameters here
        # as it will be added in the beginning, for both text and the vision branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        self.d_model = d_model
        self.n_head = n_head

        self.register_buffer('noise_buffer', None)
        self.noise_shape = None
        # This must be consistent with the config file prompt
        self.compound_prompt_nctx = design_details['maple_length']
        if i == 0:
            self.first_layer = True
        else:
            self.first_layer = False
        self.i = i

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, inputs_weights):
        # For the first layer, we do not need to add any duplicate, as it is already added
        # as the shallow version
        inputs = inputs_weights[0]
        lora_weights_list = inputs_weights[1]
        lora_weight_A, lora_weight_B = lora_weights_list[self.i] # b*dim*r, 1*dim*r

        x = inputs[0]
        compound_prompts_deeper = inputs[1]
        counter = inputs[2]
        if not self.first_layer:
            if len(compound_prompts_deeper) > 0:
                # This means that deeper compound prompts are turned on
                # Here it behaves differently for text and visual side
                # Forward function is same for both

                if not self.text_layer:
                    # First check if the ith layer needs compound prompts or not
                    if not (counter > len(compound_prompts_deeper) - 1):
                        # Remove the outputs produced by learnable tokens of previous layer
                        prefix = x[0:x.shape[0] - self.compound_prompt_nctx, :, :]
                        # Create/configure learnable tokens of this layer
                        visual_context = compound_prompts_deeper[counter]  # extract the correct index
                        visual_context = visual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        # Add the learnable tokens of this layer with the input, by replacing previous
                        # layer learnable tokens
                        x = torch.cat([prefix, visual_context], dim=0)

                        # Once done, update the counter, so that the next time, it does not use same learnable tokens
                        counter += 1
                else:
                    # First check if the ith layer needs compound prompts or not
                    if not (counter > len(compound_prompts_deeper) - 1):
                        # Appending the learnable tokens in different way
                        # x -> [77, NCLS, DIM]
                        # First remove the learnable tokens from previous layer
                        prefix = x[:1, :, :]
                        suffix = x[1 + self.compound_prompt_nctx:, :, :]
                        # Create/configure learnable tokens of this layer
                        textual_context = compound_prompts_deeper[counter]
                        textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        # Add the learnable tokens of this layer with the input, replaced by previous
                        # layer learnable tokens
                        x = torch.cat([prefix, textual_context, suffix], dim=0)
                        # Once done, update the counter, so that the next time, it does not use same learnable tokens
                        counter += 1
        lnx = self.ln_1(x)
        att_x = self.attention(lnx)
        if lora_weight_A is None or lora_weight_B is None:
            x = x + att_x
        else:
            pw_mean = torch.mean(att_x)
            pw_mean = pw_mean * (1 + 0.1 * torch.rand(1, device=pw_mean.device, dtype=torch.float16))
            # pw_mean = torch.mean(x)
            B_x, N_x, C_x = x.shape
            # pw_noise = pw_mean * torch.randn(B_x, N_x, C_x, dtype=torch.float16).to(pw_mean.device)
            if self.noise_buffer is None or self.noise_shape != (B_x, N_x, C_x):
                # regenerate with shape change
                self.noise_buffer = torch.randn(B_x, N_x, C_x,
                                                dtype=torch.float16,
                                                device=pw_mean.device)
                self.noise_shape = (B_x, N_x, C_x)
            pw_noise = self.noise_buffer * pw_mean

            if not self.text_layer:
                lora_weight_B = lora_weight_B.permute(0, 2, 1).expand(lora_weight_A.shape[0], -1, -1)
    
                delta_x = lnx.permute(1, 0, 2)
                delta_x = torch.einsum('bld,bdr->blr', delta_x, lora_weight_A)
                delta_x = torch.einsum('blr,brd->bld', delta_x, lora_weight_B)
                delta_x = delta_x.permute(1, 0, 2)
            else:
                lora_weight_A = lora_weight_A.mean(dim=0)
                lora_weight_B = lora_weight_B.permute(0, 2, 1).squeeze()
                delta_x = lnx @ lora_weight_A @ lora_weight_B
        
            x = x + att_x + 0.1 * delta_x + pw_noise

        x = x + self.mlp(self.ln_2(x))
        outputs = [x, compound_prompts_deeper, counter]
        return [outputs, lora_weights_list]  # return again as a list, so that nn.seq can work


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=0,
                 text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        # Implements respective encoder blocks for a given design choice
        current_trainer = design_details['trainer']
        if current_trainer == 'IVLP' or current_trainer == 'VPT':
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP(width, heads, attn_mask, True,
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_IVLP(width, heads, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])
        elif current_trainer == 'MaPLe' or current_trainer == 'DMPP_v2' or current_trainer == 'DMPW_v2' or current_trainer == 'DMPW_v8':
            self.resblocks = nn.Sequential(
                *[ResidualAttentionBlock_MaPLe(width, heads, attn_mask, design_details, text_layer, i)
                  for i in range(layers)])
        else:
            # Corresponds to default CoOp or CoCoOp
            assert current_trainer == 'CoOp' or current_trainer == 'CoCoOp' or current_trainer == 'DMPP' or current_trainer == 'DMPW'
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class Transformer_EET_transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=0,
                 text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        # Implements respective encoder blocks for a given design choice
        current_trainer = design_details['trainer']

        assert current_trainer == "EET_RS_ViT" or current_trainer == "EET_RS_ViTAE" or current_trainer == "EET_Med_ViT"
        self.lora_rank = 16

        if text_layer:
            self.LG_dim = 512
            self.DMPW_LoRAGenerater_1 = EET_transformer_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1) # 12*512*512
            self.DMPW_LoRAGenerater_2 = EET_transformer_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_3 = EET_transformer_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_4 = EET_transformer_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_5 = EET_transformer_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_6 = EET_transformer_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_7 = EET_transformer_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_8 = EET_transformer_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_9 = EET_transformer_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_10 = EET_transformer_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_11 = EET_transformer_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_12 = EET_transformer_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
        else:
            self.LG_dim = 768
            self.DMPW_LoRAGenerater_1 = EET_transformer_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1) # 12*768*768
            self.DMPW_LoRAGenerater_2 = EET_transformer_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_3 = EET_transformer_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_4 = EET_transformer_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_5 = EET_transformer_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_6 = EET_transformer_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_7 = EET_transformer_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_8 = EET_transformer_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_9 = EET_transformer_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_10 = EET_transformer_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_11 = EET_transformer_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_12 = EET_transformer_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
        
        self.Bs_no_grad = nn.Parameter(0.02*torch.randn(1, 12//1, self.LG_dim*self.lora_rank, dtype=torch.float16), requires_grad=False)
        
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock_EET_transformer(width, heads, attn_mask, design_details, text_layer, i) for i in range(layers)])

    def forward(self, x: torch.Tensor, domain_out):
        lora_weights_1 = self.DMPW_LoRAGenerater_1(domain_out)
        lora_weights_2 = self.DMPW_LoRAGenerater_2(domain_out)
        lora_weights_3 = self.DMPW_LoRAGenerater_3(domain_out)
        lora_weights_4 = self.DMPW_LoRAGenerater_4(domain_out)
        lora_weights_5 = self.DMPW_LoRAGenerater_5(domain_out)
        lora_weights_6 = self.DMPW_LoRAGenerater_6(domain_out)
        lora_weights_7 = self.DMPW_LoRAGenerater_7(domain_out)
        lora_weights_8 = self.DMPW_LoRAGenerater_8(domain_out)
        lora_weights_9 = self.DMPW_LoRAGenerater_9(domain_out)
        lora_weights_10 = self.DMPW_LoRAGenerater_10(domain_out)
        lora_weights_11 = self.DMPW_LoRAGenerater_11(domain_out)
        lora_weights_12 = self.DMPW_LoRAGenerater_12(domain_out)
        lora_weights_list = lora_weights_1 + lora_weights_2 + lora_weights_3 + lora_weights_4 + lora_weights_5 + lora_weights_6 + lora_weights_7 + lora_weights_8 + lora_weights_9 + lora_weights_10 + lora_weights_11 + lora_weights_12

        return self.resblocks([x, lora_weights_list])[0]


class Transformer_EET_resnet50(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=0,
                 text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        # Implements respective encoder blocks for a given design choice
        current_trainer = design_details['trainer']

        assert current_trainer == "EET_RS_ResNet50" or current_trainer == "EET_Med_ResNet50"
        self.lora_rank = 16
        
        if text_layer:
            self.LG_dim = 512
            self.DMPW_LoRAGenerater_1 = EET_resnet50_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1) # 12*512*512
            self.DMPW_LoRAGenerater_2 = EET_resnet50_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_3 = EET_resnet50_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_4 = EET_resnet50_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_5 = EET_resnet50_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_6 = EET_resnet50_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_7 = EET_resnet50_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_8 = EET_resnet50_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_9 = EET_resnet50_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_10 = EET_resnet50_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_11 = EET_resnet50_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_12 = EET_resnet50_LoRAGenerater(dim=512, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
        else:
            self.LG_dim = 768
            self.DMPW_LoRAGenerater_1 = EET_resnet50_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1) # 12*768*768
            self.DMPW_LoRAGenerater_2 = EET_resnet50_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_3 = EET_resnet50_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_4 = EET_resnet50_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_5 = EET_resnet50_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_6 = EET_resnet50_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_7 = EET_resnet50_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_8 = EET_resnet50_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_9 = EET_resnet50_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_10 = EET_resnet50_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_11 = EET_resnet50_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
            self.DMPW_LoRAGenerater_12 = EET_resnet50_LoRAGenerater(dim=768, N=1, skip_layers=1, lora_rank=self.lora_rank, length=1, llm_depth=1)
        
        self.Bs_no_grad = nn.Parameter(0.02*torch.randn(1, 12//1, self.LG_dim*self.lora_rank, dtype=torch.float16), requires_grad=False)
        
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock_EET_resnet50(width, heads, attn_mask, design_details, text_layer, i) for i in range(layers)])

    def forward(self, x: torch.Tensor, domain_out):
        lora_weights_1 = self.DMPW_LoRAGenerater_1(domain_out)
        lora_weights_2 = self.DMPW_LoRAGenerater_2(domain_out)
        lora_weights_3 = self.DMPW_LoRAGenerater_3(domain_out)
        lora_weights_4 = self.DMPW_LoRAGenerater_4(domain_out)
        lora_weights_5 = self.DMPW_LoRAGenerater_5(domain_out)
        lora_weights_6 = self.DMPW_LoRAGenerater_6(domain_out)
        lora_weights_7 = self.DMPW_LoRAGenerater_7(domain_out)
        lora_weights_8 = self.DMPW_LoRAGenerater_8(domain_out)
        lora_weights_9 = self.DMPW_LoRAGenerater_9(domain_out)
        lora_weights_10 = self.DMPW_LoRAGenerater_10(domain_out)
        lora_weights_11 = self.DMPW_LoRAGenerater_11(domain_out)
        lora_weights_12 = self.DMPW_LoRAGenerater_12(domain_out)
        lora_weights_list = lora_weights_1 + lora_weights_2 + lora_weights_3 + lora_weights_4 + lora_weights_5 + lora_weights_6 + lora_weights_7 + lora_weights_8 + lora_weights_9 + lora_weights_10 + lora_weights_11 + lora_weights_12

        return self.resblocks([x, lora_weights_list])[0]


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int, design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        if design_details["vision_depth"] == 0:
            self.VPT_shallow = False
        else:
            self.VPT_shallow = True
        if self.VPT_shallow:
            # Add visual prompt tokens here
            n_ctx = design_details["vision_ctx"]  # hyperparameter
            ctx_vectors = torch.empty(n_ctx, width)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT = nn.Parameter(ctx_vectors)
            # self.VPT.half()
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = design_details["vision_depth"]
        self.transformer = Transformer(width, layers, heads, prompts_needed=self.prompt_till_layer_visual,
                                       design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = self.VPT.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class VisionTransformer_MaPLe(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.VPT_shallow = True
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = 0
        self.transformer = Transformer(width, layers, heads, design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, shared_ctx, compound_deeper_prompts):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = shared_ctx.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # Again combine the inputs, so nn.sequential can work
        outputs = self.transformer([x, compound_deeper_prompts, 0])  # third argument is counter
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class VisionTransformer_EET_transformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.VPT_shallow = True
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = 0
        
        self.transformer = Transformer_EET_transformer(width, layers, heads, design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, shared_ctx, compound_deeper_prompts, domain_out):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = shared_ctx.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # Again combine the inputs, so nn.sequential can work
        outputs = self.transformer([x, compound_deeper_prompts, 0], domain_out)  # third argument is counter
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class VisionTransformer_EET_resnet50(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.VPT_shallow = True
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = 0
        
        self.transformer = Transformer_EET_resnet50(width, layers, heads, design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, shared_ctx, compound_deeper_prompts, domain_out):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = shared_ctx.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # Again combine the inputs, so nn.sequential can work
        outputs = self.transformer([x, compound_deeper_prompts, 0], domain_out)  # third argument is counter
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 design_details
                 ):
        super().__init__()

        self.context_length = context_length
        trainer = design_details['trainer']

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            if trainer == "MaPLe":
                self.visual = VisionTransformer_MaPLe(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                )
            elif trainer == "EET_RS_ViT" or trainer == "EET_RS_ViTAE" or trainer == "EET_Med_ViT":
                self.visual = VisionTransformer_EET_transformer(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                )
            elif trainer == "EET_RS_ResNet50" or trainer == "EET_Med_ResNet50":
                self.visual = VisionTransformer_EET_resnet50(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                )
            else:
                self.visual = VisionTransformer(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                )
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        prompt_till_layer_text = design_details['language_depth']
        if trainer == "EET_RS_ViT" or trainer == "EET_RS_ViTAE" or trainer == "EET_Med_ViT":
            self.transformer = Transformer_EET_transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask(),
                prompts_needed=prompt_till_layer_text,
                text_layer=True,
                design_details=design_details
            )
        elif trainer == "EET_RS_ResNet50" or trainer == "EET_Med_ResNet50":
            self.transformer = Transformer_EET_resnet50(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask(),
                prompts_needed=prompt_till_layer_text,
                text_layer=True,
                design_details=design_details
            )
        else:
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask(),
                prompts_needed=prompt_till_layer_text,
                text_layer=True,
                design_details=design_details
            )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, design_details):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        # vision_layers = len(
        #     [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj.weight")])
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, design_details
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    try:
        model.load_state_dict(state_dict)
    except:
        missing_keys, _ = model.load_state_dict(state_dict, strict=False)
        print('Weights not found for some missing keys: ', missing_keys)
    return model.eval()
