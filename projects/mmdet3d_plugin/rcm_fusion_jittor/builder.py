def build_jittor_module(cfg):
    if cfg is None:
        return None
    t = cfg['type']
    kwargs = {k: v for k, v in cfg.items() if k != 'type'}
    
    if t == 'RadarGuidedBEVEncoder':
        from .modules.radar_guided_bev_encoder import RadarGuidedBEVEncoder
        return RadarGuidedBEVEncoder(**kwargs)
    elif t == 'RadarGuidedBEVEncoderLayer':
        from .modules.radar_guided_bev_encoder import RadarGuidedBEVEncoderLayer
        return RadarGuidedBEVEncoderLayer(**kwargs)
    elif t == 'RadarGuidedBEVAttention':
        from .modules.radar_guided_bev_attention import RadarGuidedBEVAttention
        return RadarGuidedBEVAttention(**kwargs)
    elif t == 'SpatialCrossAttention':
        from .modules.spatial_cross_attention import SpatialCrossAttention
        return SpatialCrossAttention(**kwargs)
    elif t == 'MSDeformableAttention3D':
        from .modules.spatial_cross_attention import MSDeformableAttention3D
        return MSDeformableAttention3D(**kwargs)
    elif t == 'DetectionTransformerDecoder':
        from .modules.decoder import DetectionTransformerDecoder
        return DetectionTransformerDecoder(**kwargs)
    elif t == 'DetrTransformerDecoderLayer':
        from .modules.custom_base_transformer_layer import DetrTransformerDecoderLayer
        return DetrTransformerDecoderLayer(**kwargs)
    elif t == 'MultiheadAttention':
        import jittor as jt
        import jittor.nn as nn
        import math
        class _AttnParams(nn.Module):
            """Sub-module to match PyTorch's `attn.in_proj_weight` parameter naming."""
            def __init__(self, embed_dims):
                super().__init__()
                self.in_proj_weight = nn.Parameter(jt.zeros((3 * embed_dims, embed_dims)))
                self.in_proj_bias = nn.Parameter(jt.zeros((3 * embed_dims,)))
                self.out_proj = nn.Linear(embed_dims, embed_dims)
        class JittorMultiheadAttention(nn.Module):
            def __init__(self, embed_dims, num_heads=8, dropout=0.0, batch_first=True, **kw):
                super().__init__()
                self.embed_dims = embed_dims
                self.num_heads = num_heads
                self.dropout = dropout
                self.batch_first = batch_first
                head_dim = embed_dims // num_heads
                assert head_dim * num_heads == embed_dims
                self.head_dim = head_dim
                self.attn = _AttnParams(embed_dims)
                self.attn_drop = nn.Dropout(dropout)

            def _linear(self, x, weight, bias):
                x = jt.matmul(x, weight.transpose(0, 1))
                if bias is not None:
                    x = x + bias
                return x

            def execute(self, query, key, value, identity=None, query_pos=None, key_pos=None, attn_mask=None, key_padding_mask=None, **kw):
                if identity is None:
                    identity = query
                if query_pos is not None:
                    query = query + query_pos
                if key_pos is not None:
                    key = key + key_pos
                if not self.batch_first:
                    query = query.permute(1, 0, 2)
                    key = key.permute(1, 0, 2)
                    value = value.permute(1, 0, 2)
                w_q, w_k, w_v = jt.split(self.attn.in_proj_weight, self.embed_dims, dim=0)
                b_q, b_k, b_v = jt.split(self.attn.in_proj_bias, self.embed_dims, dim=0)
                q = self._linear(query, w_q, b_q)
                k = self._linear(key, w_k, b_k)
                v = self._linear(value, w_v, b_v)
                bs, tgt_len, _ = q.shape
                src_len = k.shape[1]
                q = q.view(bs, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
                k = k.view(bs, src_len, self.num_heads, self.head_dim).transpose(1, 2)
                v = v.view(bs, src_len, self.num_heads, self.head_dim).transpose(1, 2)
                attn = jt.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
                if attn_mask is not None:
                    if attn_mask.ndim == 2:
                        attn = attn + attn_mask[None, None, :, :]
                    elif attn_mask.ndim == 3:
                        attn = attn + attn_mask[:, None, :, :]
                if key_padding_mask is not None:
                    mask = key_padding_mask[:, None, None, :].astype(attn.dtype)
                    attn = attn + mask * (-10000.0)
                attn = jt.nn.softmax(attn, dim=-1)
                attn = self.attn_drop(attn)
                out = jt.matmul(attn, v)
                out = out.transpose(1, 2).reshape(bs, tgt_len, self.embed_dims)
                out = self.attn.out_proj(out)
                if not self.batch_first:
                    out = out.permute(1, 0, 2)
                return out + identity
        return JittorMultiheadAttention(**kwargs)
    elif t == 'CustomMSDeformableAttention':
        from .modules.decoder import CustomMSDeformableAttention
        return CustomMSDeformableAttention(**kwargs)
    elif t == 'FFN':
        import jittor.nn as nn
        class JittorFFN(nn.Module):
            def __init__(self, embed_dims=256, feedforward_channels=1024, num_fcs=2, act_cfg=dict(type='ReLU', inplace=True), ffn_drop=0., add_identity=True, dropout_layer=None, **kw):
                super().__init__()
                # Match mmcv FFN structure: layers[0..num_fcs-2] = Sequential(Linear, act, Dropout)
                # layers[num_fcs-1] = Linear, layers[num_fcs] = Dropout
                layers_list = []
                in_channels = embed_dims
                for i in range(num_fcs - 1):
                    layers_list.append(nn.Sequential(
                        nn.Linear(in_channels, feedforward_channels),
                        nn.ReLU(),
                        nn.Dropout(ffn_drop),
                    ))
                    in_channels = feedforward_channels
                layers_list.append(nn.Linear(feedforward_channels, embed_dims))
                layers_list.append(nn.Dropout(ffn_drop))
                self.layers = nn.Sequential(*layers_list)
                self.add_identity = add_identity
            def execute(self, x, identity=None):
                out = self.layers(x)
                if not self.add_identity:
                    return out
                if identity is None:
                    identity = x
                return identity + out
        return JittorFFN(**kwargs)
    else:
        raise ValueError(f"Unknown Jittor module type: {t}")
