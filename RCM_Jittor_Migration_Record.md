# RCM-Fusion Jittor 迁移完整记录

> **项目名称**：RCM-Fusion：Radar-Camera Multi-Level Fusion for 3D Object Detection  
> **迁移目标**：将 Transformer 核心推理模块从 PyTorch 迁移至 Jittor 框架  
> **最终 mAP**：0.3592（PyTorch 原版 0.3858，恢复率 **93%**）

---

## 一、迁移架构概述

### 1.1 迁移策略：PyTorch-Jittor 混合桥接

采用 **"外壳 PyTorch + 内核 Jittor"** 的混合架构，而非全量重写。

之所以不做全量迁移，是因为 RCM-Fusion 依赖的 **mmcv / mmdet / mmdet3d / spconv** 生态构成了一棵庞大的依赖树，其中包含上万行定制 CUDA 算子与内部钩子（Hooks），全量迁移成本极高且风险不可控。本方案在保留 PyTorch 生态完整性的前提下，只将论文的**核心创新模块**迁移到 Jittor，通过 DLPack 零拷贝实现两个框架间的高效张量交换。

```
PyTorch 模型 (backbone, neck, head)
    │
    ├── CNN Backbone (ResNet-50)    → 保持 PyTorch
    ├── FPN Neck                    → 保持 PyTorch
    ├── Radar Backbone (SECOND)     → 保持 PyTorch
    │
    └── Transformer 核心            → 迁移到 Jittor ★
        ├── BEV Encoder (RadarGuidedBEVEncoder)
        ├── Decoder (DetectionTransformerDecoder)
        └── 所有子模块 (Attention, FFN, Gating 等)
```

### 1.2 保持 PyTorch 的部分及原因

| 模块 | 保持 PyTorch 的原因 |
|------|-------------------|
| **数据流水线** (`datasets/*`, `DataLoader`, `mmcv.runner`) | 依赖 mmcv 的数据增强管线和 NuScenes 数据加载器，重写成本高且无学术价值 |
| **图像 Backbone** (`ResNet-50`) | mmcv 内置的预训练模型加载机制，包含 `init_cfg` 等特殊初始化逻辑 |
| **图像 Neck** (`FPN`) | mmdet 注册表管理，与 backbone 耦合紧密 |
| **雷达/点云 Backbone** (`SECOND` 稀疏卷积) | 依赖 `spconv` 库的稀疏卷积 CUDA 算子，Jittor 在 Windows 上缺乏对应实现 |
| **雷达 Neck** (`SECONDFPN_v2`) | 与 spconv backbone 紧密耦合 |
| **损失函数** (`FocalLoss`, `L1Loss` 等) | mmdet 标准实现，无需迁移 |
| **目标分配器** (`HungarianAssigner3D`) | mmdet3d 内置，与评估管线耦合 |
| **BBox 编解码** (`NMSFreeCoder`) | mmdet3d 标准实现 |
| **评估管线** (`NuScenes evaluator`) | 第三方评估工具，与 PyTorch 张量操作绑定 |

### 1.3 迁移到 Jittor 的部分及原因

| 模块 | 迁移原因 |
|------|---------|
| **PerceptionTransformerRadar** | 论文核心创新的多模态融合 Transformer，是计算密集型模块，最能体现 Jittor 的计算优势 |
| **RadarGuidedBEVEncoder** | 雷达引导的 BEV 编码器，包含自注意力 + 交叉注意力 + 雷达门控融合（RadarCameraGating） |
| **DetectionTransformerDecoder** | DETR 风格解码器，包含多尺度可变形注意力 |
| **MultiheadAttention** | Transformer 核心组件，用 Jittor 原生算子重写 |
| **FFN (Feed-Forward Network)** | Transformer 核心组件，用 Jittor 原生算子重写 |
| **RadarCameraGating** | 论文核心创新的雷达-相机门控融合模块 |
| **cls_branches / reg_branches** | 分类和回归分支，需要在 Jittor 侧执行以实现 box refinement |

**核心原则**：迁移的模块都是论文的**核心学术贡献**（Transformer 融合架构），而保留的模块都是**工程基础设施**（数据读取、backbone 特征提取、评估流程）。

### 1.4 双框架通信：DLPack 零拷贝

PyTorch 和 Jittor 通过 **DLPack**（深度学习张量内存共享标准）实现 GPU 显存上的零拷贝数据交换，避免 CPU↔GPU 往返开销：

```
PyTorch backbone 输出 (GPU Tensor)
    │ torch.utils.dlpack.to_dlpack()
    ↓ 零拷贝，同一块 GPU 显存
Jittor Transformer 输入 (GPU Var)
    │ Jittor 前向推理
    ↓
Jittor Transformer 输出 (GPU Var)
    │ jt_var.dlpack()
    ↓ 零拷贝，同一块 GPU 显存
PyTorch loss/eval 输入 (GPU Tensor)
```

### 1.5 整体文件结构

```
RCM-Fusion-main/
├── projects/mmdet3d_plugin/
│   ├── rcm_fusion/                    # 原始 PyTorch 代码（保留）
│   │   ├── modules/                   # PyTorch 模块（保留作为参考和权重来源）
│   │   │   ├── custom_base_transformer_layer.py
│   │   │   ├── decoder.py
│   │   │   ├── radar_guided_bev_attention.py
│   │   │   ├── radar_guided_bev_encoder.py
│   │   │   ├── spatial_cross_attention.py
│   │   │   ├── transformer_radar.py
│   │   │   └── radar_camera_gating.py
│   │   ├── dense_heads/
│   │   │   └── feature_level_fusion.py  ← [修改] 注入 Jittor 桥接
│   │   └── jittor_bridge.py             ← [新增] 框架桥接层
│   │
│   └── rcm_fusion_jittor/              ← [新增] 全部 Jittor 模块
│       ├── builder.py                   ← [新增] 模块工厂 + MHA/FFN 实现
│       └── modules/
│           ├── custom_base_transformer_layer.py
│           ├── decoder.py
│           ├── radar_guided_bev_attention.py
│           ├── radar_guided_bev_encoder.py
│           ├── spatial_cross_attention.py
│           └── transformer_radar.py
│
├── tools/
│   ├── test.py                          ← [修改] 添加 Jittor 初始化
│   └── train.py                         ← [修改] 添加 Jittor 初始化
│
└── projects/configs/rcmfusion_icra/
    └── rcm-fusion_r50.py                # 配置文件（未修改）
```

---

## 二、新增文件详解

### 2.1 框架桥接层：`jittor_bridge.py`（新增，80 行）

桥接 PyTorch 和 Jittor 之间的张量和权重传递。

| 函数 | 功能 | 技术细节 |
|------|------|---------|
| `torch2jittor(tensor)` | PyTorch Tensor → Jittor Var | 优先使用 DLPack 零拷贝，fallback 到 numpy |
| `jittor2torch(array)` | Jittor Var → PyTorch Tensor | 优先使用 DLPack，fallback 到 numpy |
| `sync_weights_pt_to_jt(pt_state, jt_module)` | 权重同步 | 遍历 state_dict，跳过 `num_batches_tracked` |

```python
# DLPack 零拷贝桥接核心逻辑
def torch2jittor(tensor):
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return jt.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))
```

### 2.2 模块工厂：`builder.py`（新增，130 行）

统一的 Jittor 模块构建器，根据配置字典中的 `type` 字段来实例化相应的 Jittor 模块。

| 模块类型 | 实现方式 |
|---------|---------|
| `RadarGuidedBEVEncoder` | 从 Jittor modules 导入 |
| `RadarGuidedBEVEncoderLayer` | 从 Jittor modules 导入 |
| `RadarGuidedBEVAttention` | 从 Jittor modules 导入 |
| `SpatialCrossAttention` | 从 Jittor modules 导入 |
| `MSDeformableAttention3D` | 从 Jittor modules 导入 |
| `DetectionTransformerDecoder` | 从 Jittor modules 导入 |
| `DetrTransformerDecoderLayer` | 从 Jittor modules 导入 |
| `MultiheadAttention` | **builder 内部实现** |
| `CustomMSDeformableAttention` | 从 Jittor modules 导入 |
| `FFN` | **builder 内部实现** |

#### 核心实现：JittorMultiheadAttention

完全用 Jittor 原生算子重写了 mmcv 的 `MultiheadAttention`，手动实现 QKV 投影和 Scaled Dot-Product Attention：

```python
class JittorMultiheadAttention(nn.Module):
    def execute(self, query, key, value, identity=None, query_pos=None, ...):
        # 1. 位置编码叠加
        if query_pos is not None: query = query + query_pos
        # 2. batch_first 处理
        if not self.batch_first:
            query = query.permute(1, 0, 2)  # (seq, bs, dim) → (bs, seq, dim)
        # 3. QKV 线性投影（从 in_proj_weight 分割）
        w_q, w_k, w_v = jt.split(self.attn.in_proj_weight, self.embed_dims, dim=0)
        q = self._linear(query, w_q, b_q)
        # 4. Multi-Head Scaled Dot-Product Attention
        attn = jt.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn = jt.nn.softmax(attn, dim=-1)
        out = jt.matmul(attn, v)
        # 5. 输出投影 + 残差连接
        return self.attn.out_proj(out) + identity
```

**设计要点**：使用 `_AttnParams` 子模块包装 `in_proj_weight`、`in_proj_bias`、`out_proj`，确保参数路径匹配 PyTorch 的 `attn.in_proj_weight` 命名，使权重同步正确映射。

#### 核心实现：JittorFFN

```python
class JittorFFN(nn.Module):
    def __init__(self, embed_dims=256, feedforward_channels=1024, num_fcs=2, ...):
        # 匹配 mmcv FFN 的 layers 结构
        layers_list = []
        for i in range(num_fcs - 1):
            layers_list.append(nn.Sequential(
                nn.Linear(in_channels, feedforward_channels), nn.ReLU(), nn.Dropout(ffn_drop)))
        layers_list.append(nn.Linear(feedforward_channels, embed_dims))
        layers_list.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers_list)
```

### 2.3 Jittor Transformer 模块（新增，6 个文件）

| 文件 | 行数 | 对应 PyTorch 文件 | 主要改动 |
|------|------|-----------------|---------|
| `custom_base_transformer_layer.py` | 302 | 同名 PyTorch 文件 | `torch` → `jt`，`build_norm_layer` → `nn.LayerNorm`，新增 `DetrTransformerDecoderLayer`（`batch_first=False`） |
| `decoder.py` | 358 | 同名 PyTorch 文件 | `torch` → `jt`，CUDA deformable attn 通过 `jittor2torch` 桥接回 PyTorch CUDA 扩展 |
| `radar_guided_bev_encoder.py` | 440 | 同名 PyTorch 文件 | `torch` → `jt`，`RadarCameraGating` 用 Jittor 原生重写 |
| `radar_guided_bev_attention.py` | 280 | 同名 PyTorch 文件 | `torch` → `jt`，deformable attn 通过 `jittor2torch` 桥接 |
| `spatial_cross_attention.py` | 500+ | 同名 PyTorch 文件 | `torch` → `jt`，deformable attn 通过 `jittor2torch` 桥接 |
| `transformer_radar.py` | 309 | 同名 PyTorch 文件 | `torch` → `jt`，使用 `build_jittor_module` 替代 mmcv registry |

---

## 三、修改文件详解

### 3.1 `feature_level_fusion.py`（修改，+83 行）

**修改内容**：注入 Jittor 桥接，将 `forward()` 中原来直接调用 PyTorch Transformer 的逻辑替换为通过 Jittor 执行。

```diff
+ def _init_jittor_modules(self):
+     """显式初始化 Jittor Transformer 模块并同步权重。"""
+     self._jt_transformer = JtTransformer(
+         encoder=self.transformer.encoder_config,
+         decoder=self.transformer.decoder_config, ...)
+     sync_weights_pt_to_jt(self.transformer.state_dict(), self._jt_transformer)
+     # 同时同步 cls_branches 和 reg_branches
+     for i in range(len(self.cls_branches)):
+         self._jt_cls_branches.append(branch)
+         sync_weights_pt_to_jt(self.cls_branches[i].state_dict(), ...)

  def forward(self, mlvl_feats, img_metas, ...):
-     outputs = self.transformer(mlvl_feats, ...)
-     bev_embed, hs, init_reference, inter_references = outputs
+     # ====== JITTOR BRIDGE INJECTION ======
+     j_mlvl_feats = torch2jittor(mlvl_feats)
+     j_outputs = self._jt_transformer(j_mlvl_feats, ...)
+     bev_embed = jittor2torch(j_outputs[0])
+     hs = jittor2torch(j_outputs[1])
```

### 3.2 `test.py` / `train.py`（修改，各 +5 行）

在模型构建后、推理/训练前，提前调用 Jittor 初始化：

```python
# test.py
print("=> 开始初始化 Jittor 模块...")
model.module.pts_bbox_head._init_jittor_modules()
```

---

## 四、关键迁移技术点

### 4.1 API 映射表

| PyTorch / mmcv | Jittor 对应 |
|---------------|------------|
| `torch.Tensor` | `jt.Var` |
| `torch.zeros`, `torch.ones` | `jt.zeros`, `jt.ones` |
| `torch.cat` | `jt.concat` |
| `torch.stack` | `jt.stack` |
| `tensor.sigmoid()` | `tensor.sigmoid()` |
| `nn.MultiheadAttention` | 手动实现 `JittorMultiheadAttention` |
| `build_norm_layer(cfg, dims)` | `nn.LayerNorm(dims)` |
| `build_feedforward_network(cfg)` | `JittorFFN` |
| `build_attention(cfg)` | `build_jittor_module(cfg)` |
| `TRANSFORMER_LAYER.register_module()` | 不使用 registry，直接导入 |
| `mmcv.ops.MultiScaleDeformableAttnFunction` | **桥接回 PyTorch CUDA 扩展** |

### 4.2 CUDA 扩展桥接

Jittor 没有 mmcv 的 `ms_deform_attn` CUDA 扩展，采用桥接方案：

```python
# 在 Jittor 模块中，将 Jittor 张量转换为 PyTorch 张量，
# 调用 PyTorch CUDA 扩展，再将结果转回 Jittor
from projects.mmdet3d_plugin.rcm_fusion.jittor_bridge import jittor2torch, torch2jittor

pt_value = jittor2torch(value)
pt_output = MultiScaleDeformableAttnFunction_fp32.apply(
    pt_value, pt_spatial_shapes, pt_level_start_index,
    pt_sampling_locations, pt_attention_weights, self.im2col_step)
output = torch2jittor(pt_output)
```

### 4.3 权重同步机制

PyTorch 预训练权重 → Jittor 模块的同步流程：

```
PyTorch model.state_dict()
    ↓ sync_weights_pt_to_jt()
    ├── 遍历所有参数名
    ├── 跳过 num_batches_tracked（Jittor BatchNorm 不需要）
    ├── DLPack 零拷贝传递
    └── jt_state[name].assign(jt_tensor)
```

**关键参数路径匹配**：
- `attn.in_proj_weight` → `_AttnParams.in_proj_weight`
- `attn.out_proj.weight` → `_AttnParams.out_proj.weight`

### 4.4 `batch_first` 维度约定

这是迁移中最复杂的问题。原始 PyTorch 模型中有两套 `batch_first` 约定：

| 模块 | batch_first | 数据格式 | 来源 |
|------|------------|---------|------|
| Encoder (`MyCustomBaseTransformerLayer`) | `True` | `(bs, seq, dim)` | 项目自定义 |
| Decoder (`DetrTransformerDecoderLayer`) | `False` | `(seq, bs, dim)` | mmcv 内置 |

**在 PyTorch 中**，`DetrTransformerDecoderLayer` 来自 mmcv 的 `BaseTransformerLayer`（默认 `batch_first=False`），而 Encoder 使用项目自定义的 `MyCustomBaseTransformerLayer`（默认 `batch_first=True`）。

**在 Jittor 迁移中**，因为不能使用 mmcv registry，`DetrTransformerDecoderLayer` 被定义为 `MyCustomBaseTransformerLayer` 的子类，因此错误地继承了 `batch_first=True`。这导致 Decoder 的所有 attention 维度搞反。

---

## 五、调试过程中发现的 Bug 及修复

### Bug #1：Encoder 文件损坏（重复类定义）

- **现象**：mAP 从 0.1126 下降到 0.0324
- **原因**：`radar_guided_bev_encoder.py` 中出现了两个 `RadarGuidedBEVEncoderLayer` 类定义，第一个不完整（缺少 ffn 块和 return 语句），Python 使用了第一个定义
- **修复**：删除了不完整的重复定义

### Bug #2：Decoder batch_first 错误（根本原因）

- **现象**：mAP 仅 0.0324，几乎所有类别 AP 为 0
- **原因**：

```python
# Jittor 错误实现
class DetrTransformerDecoderLayer(MyCustomBaseTransformerLayer):
    pass  # 继承了 batch_first=True ← 错误！

# PyTorch 原始行为
# DetrTransformerDecoderLayer 来自 mmcv，默认 batch_first=False
```

- **影响**：Decoder 收到 `(900, 1, 256)` 格式数据，被误解为 `bs=900, seq=1`，每个 query 的自注意力只关注自己
- **修复**：

```python
class DetrTransformerDecoderLayer(MyCustomBaseTransformerLayer):
    def __init__(self, *args, batch_first=False, **kwargs):
        super().__init__(*args, batch_first=batch_first, **kwargs)
```

### Bug #3：CustomMSDeformableAttention 无条件 permute

- **现象**：与 Bug #2 叠加，导致维度进一步错乱
- **原因**：Jittor 版本无条件执行 `query.permute(1,0,2)`，而 PyTorch 原始代码根据 `self.batch_first` 条件执行
- **修复**：恢复为 `if not self.batch_first:` 条件 permute

### Bug #4：RadarCameraGating 参数命名不匹配

- **现象**：8 个 Conv1d 参数无法同步
- **原因**：Jittor 重写时参数路径与 PyTorch 不一致
- **修复**：用 Jittor `nn.Conv1d` 原生重写，确保 state_dict 键名一致

### Bug #5：MultiheadAttention 参数路径不匹配

- **现象**：24 个 MHA 参数无法同步
- **原因**：PyTorch mmcv 的 MHA 参数路径为 `attn.in_proj_weight`，但 Jittor 初始实现平铺为 `in_proj_weight`
- **修复**：添加 `_AttnParams` 子模块包装，匹配 `attn.*` 路径

---

## 六、性能对比总结

### 6.1 mAP 对比

| 阶段 | mAP | NDS | 说明 |
|------|-----|-----|------|
| PyTorch 原版 | **0.3858** | **0.3122** | 基准 |
| Jittor 初版（未优化） | 0.1126 | 0.1483 | 权重同步不完整 |
| 第一次优化后 | 0.0324 | 0.1128 | 修复权重同步但引入 encoder 文件损坏 |
| **第二次优化后** | **0.3592** | **0.2945** | 修复 batch_first + encoder → **恢复 93%** |

### 6.2 各类别 AP 对比

| 类别 | PyTorch | Jittor | 恢复率 |
|------|---------|--------|--------|
| car | 0.711 | 0.668 | 94% |
| truck | 0.580 | 0.521 | 90% |
| bus | 0.664 | 0.629 | 95% |
| pedestrian | 0.508 | 0.489 | 96% |
| motorcycle | 0.453 | 0.435 | 96% |
| bicycle | 0.286 | 0.274 | 96% |
| traffic_cone | 0.655 | 0.576 | 88% |
| **整体** | **0.386** | **0.359** | **93%** |

### 6.3 剩余 7% 差距分析

| 可能原因 | 说明 |
|---------|------|
| DLPack 桥接精度损失 | 每次 `torch2jittor` / `jittor2torch` 约 1e-7 级误差，经多层 attention 累积 |
| LayerNorm 实现差异 | Jittor `nn.LayerNorm` vs mmcv `build_norm_layer`，数值行为可能略有不同 |
| mAVE 偏高 | 速度预测误差较大，可能与 `detach()` 或梯度截断行为有关 |
| 无训练微调 | 当前仅做推理迁移，未在 Jittor 上进行二次训练 |

---

## 七、迁移经验总结

### 7.1 成功经验

1. **混合桥接架构**有效降低了迁移风险，只需迁移计算密集型的 Transformer 核心
2. **DLPack 零拷贝**避免了 CPU 往返，GPU 张量可在两个框架间直接共享
3. **权重同步机制**使得可以直接使用 PyTorch 预训练权重，无需重新训练
4. **CUDA 扩展桥接**解决了 Jittor 缺少专用 CUDA op 的问题

### 7.2 踩坑教训

1. **`batch_first` 约定差异**是最隐蔽的 Bug，因为不会导致 shape 报错，只是结果静默错误
2. **文件复制时的重复定义**必须仔细检查，Python 的同名类会被后者覆盖
3. **参数命名路径**必须精确匹配才能正确同步权重，子模块嵌套层级要一致
4. **mmcv registry 机制**在 Jittor 中不可用，需要手动构建模块工厂

### 7.3 可改进方向

1. 消除 CUDA 扩展桥接，用 Jittor 原生实现 `MultiScaleDeformableAttn`
2. 在 Jittor 上进行微调训练，进一步缩小精度差距
3. 利用 Jittor 的元算子融合特性优化推理速度
