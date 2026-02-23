<div align="center">

# RCM-Fusion: Radar-Camera Multi-Level Fusion for 3D Object Detection

**Jittor Mixed Migration Version**

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2307.10249-red)](https://arxiv.org/abs/2307.10249)
[![Conference](https://img.shields.io/badge/Conference-ICRA%202024-blue)](https://2024.ieee-icra.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%2B%20Jittor-orange)](https://cg.cs.tsinghua.edu.cn/jittor/)

**[中文](README.md) | [English](README_EN.md)**

</div>

---

## 📖 Overview

This repository is the **Jittor mixed migration version** of **RCM-Fusion** (Radar-Camera Multi-Level Fusion for 3D Object Detection, ICRA 2024).

**Original Paper**: [RCM-Fusion: Radar-Camera Multi-Level Fusion for 3D Object Detection](https://arxiv.org/abs/2307.10249)  
**Original Authors**: Jisong Kim\*, Minjae Seong\*, Geonho Bang, Dongsuk Kum, Jun Won Choi (KAIST)

### Abstract

Existing radar-camera fusion methods fail to fully exploit the potential of radar information. This paper proposes **RCM-Fusion**, which performs multi-modal fusion at both feature and instance levels:

- **Feature-level fusion**: A Radar Guided BEV Encoder is proposed, which transforms camera features into precise BEV representations guided by radar BEV features, and fuses both radar and camera BEV features.
- **Instance-level fusion**: A Radar Grid Point Refinement module is proposed to reduce localization errors by leveraging the characteristics of radar point clouds.

On the public nuScenes benchmark, RCM-Fusion achieves **state-of-the-art (SOTA) performance** among single-frame radar-camera fusion methods.

---

## 🏗 Migration Architecture

### Hybrid Bridging Strategy: "PyTorch Shell + Jittor Core"

This project adopts a **PyTorch-Jittor hybrid migration** strategy rather than a full rewrite. This is because RCM-Fusion depends on a large ecosystem of mmcv / mmdet / mmdet3d / spconv, making a full migration extremely costly. Instead, this solution migrates only the **core innovative Transformer modules** to Jittor while preserving the PyTorch ecosystem, using **DLPack zero-copy** for efficient tensor exchange between the two frameworks.

```
PyTorch Model (backbone, neck, head)
    │
    ├── CNN Backbone (ResNet-50)      → Keep in PyTorch
    ├── FPN Neck                      → Keep in PyTorch
    ├── Radar Backbone (SECOND)       → Keep in PyTorch
    │
    └── Transformer Core              → Migrated to Jittor ★
        ├── BEV Encoder (RadarGuidedBEVEncoder)
        ├── Decoder (DetectionTransformerDecoder)
        └── All sub-modules (Attention, FFN, Gating, etc.)
```

### DLPack Zero-Copy Dual-Framework Communication

```
PyTorch backbone output (GPU Tensor)
    │ torch.utils.dlpack.to_dlpack()
    ↓ Zero-copy, same GPU memory
Jittor Transformer input (GPU Var)
    │ Jittor forward inference
    ↓
Jittor Transformer output (GPU Var)
    │ jt_var.dlpack()
    ↓ Zero-copy, same GPU memory
PyTorch loss/eval input (GPU Tensor)
```

---

## 📊 Performance Comparison

### mAP Recovery Rate

| Version | mAP | NDS | Notes |
|:---:|:---:|:---:|:---:|
| PyTorch Original (Baseline) | 0.3858 | 0.3122 | — |
| Jittor Initial Version | 0.1126 | 0.1483 | Incomplete weight synchronization |
| After 1st Optimization | 0.0324 | 0.1128 | Introduced `batch_first`-related Bug |
| **Jittor Mixed Version (Final)** | **0.3592** | **0.2945** | **93% Recovery** ✅ |

### Per-Class AP Comparison

| Class | PyTorch | Jittor | Recovery |
|:---:|:---:|:---:|:---:|
| Car | 0.711 | 0.668 | 94% |
| Truck | 0.580 | 0.521 | 90% |
| Bus | 0.664 | 0.629 | 95% |
| Pedestrian | 0.508 | 0.489 | 96% |
| Motorcycle | 0.453 | 0.435 | 96% |
| Bicycle | 0.286 | 0.274 | 96% |
| Traffic Cone | 0.655 | 0.576 | 88% |
| **Overall** | **0.386** | **0.359** | **93%** |

### Model Zoo

| Backbone | Method | Lr Schd | NDS | mAP | Config | Download |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| R50 | RCM-Fusion-R50 | 24ep | 53.5 | 45.2 | [config](projects/configs/rcmfusion_icra/rcm-fusion_r50.py) | [model](https://arxiv.org/abs/2307.10249) |
| R101 | RCM-Fusion-R101 | 24ep | 58.7 | 50.6 | [config](projects/configs/rcmfusion_icra/rcm-fusion_r101.py) | [model](https://arxiv.org/abs/2307.10249) |

---

## 🧩 Model Architecture

![Model Architecture](rcm-fusion-overall.png)

<div align="center">
  <img src="figs/arch.png" alt="Detailed Network Architecture" width="90%"/>
</div>

---

## 🛠 Environment Setup

### Requirements

- Python == 3.10
- CUDA == 11.6
- PyTorch == 1.13.1+cu116
- [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) == 1.3.8.1
- mmcv-full == 1.7.0
- mmdet == 2.28.2
- mmsegmentation == 0.30.0
- mmdet3d == 1.0.0rc6
- spconv-cu116 == 2.3.6
- nuscenes-devkit == 1.1.11

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/RCM-Jittor-MixedMigration.git
cd RCM-Jittor-MixedMigration
```

**2. Create conda environment**

```bash
conda create -n rcm_jittor python=3.10 -y
conda activate rcm_jittor
```

**3. Install PyTorch (CUDA 11.6)**

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

**4. Install Jittor**

```bash
pip install jittor==1.3.8.1
```

**5. Install mmcv / mmdet / mmdet3d**

```bash
pip install openmim==0.3.9
mim install mmcv-full==1.7.0
mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0
# mmdet3d will be installed from source (see step 8)
```

**6. Install remaining dependencies**

```bash
pip install -r requirements.txt
```

**7. Install spconv and other 3D tools**

```bash
pip install spconv-cu116==2.3.6 -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install nuscenes-devkit==1.1.11
```

**8. Install local mmdetection3d extension (mmdet3d==1.0.0rc6)**

```bash
cd mmdetection3d
pip install -e .
cd ..
```

---

## 📁 Data Preparation

### Download nuScenes Dataset

Please visit the [nuScenes official website](https://www.nuscenes.org/download) to download the **v1.0-trainval** full dataset and the **CAN bus expansion pack**.

**Extract CAN bus data**

```bash
unzip can_bus.zip
# Move the can_bus folder to the data directory
```

**Generate nuScenes annotation files**

```bash
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0 \
    --canbus ./data
```

### Directory Structure

```
RCM-Jittor-MixedMigration/
├── projects/
├── tools/
├── mmdetection3d/
├── ckpts/
│   ├── rcm-fusion-r50-icra-final.pth
│   └── rcm-fusion-r101-icra-final.pth
└── data/
    ├── can_bus/
    └── nuscenes/
        ├── maps/
        ├── samples/
        ├── sweeps/
        ├── v1.0-trainval/
        ├── nuscenes_infos_train_rcmfusion.pkl
        └── nuscenes_infos_val_rcmfusion.pkl
```

---

## 🚀 Getting Started

### Testing (Inference & Evaluation)

Single GPU evaluation with R50 backbone:

```bash
python tools/test.py \
    projects/configs/rcmfusion_icra/rcm-fusion_r50.py \
    ckpts/rcm-fusion-r50-icra-final.pth \
    --eval bbox
```

Distributed evaluation with 4 GPUs:

```bash
./tools/dist_test.sh \
    projects/configs/rcmfusion_icra/rcm-fusion_r101.py \
    ckpts/rcm-fusion-r101-icra-final.pth \
    4 --eval bbox
```

### Training

Distributed training with 4 GPUs:

```bash
./tools/dist_train.sh \
    projects/configs/rcmfusion_icra/rcm-fusion_r101.py \
    4 --work-dir ./workdirs/rcm_fusion_r101
```

---

## 📂 Project Structure

```
RCM-Jittor-MixedMigration/
├── projects/mmdet3d_plugin/
│   ├── rcm_fusion/                      # Original PyTorch modules (preserved)
│   │   ├── modules/                     # PyTorch Transformer sub-modules
│   │   ├── dense_heads/
│   │   │   └── feature_level_fusion.py  # [Modified] Inject Jittor bridge
│   │   └── jittor_bridge.py             # [New] Framework bridge layer (DLPack)
│   │
│   └── rcm_fusion_jittor/               # [New] All Jittor modules
│       ├── builder.py                   # [New] Module factory + MHA/FFN impl
│       └── modules/
│           ├── custom_base_transformer_layer.py
│           ├── decoder.py
│           ├── radar_guided_bev_attention.py
│           ├── radar_guided_bev_encoder.py
│           ├── spatial_cross_attention.py
│           └── transformer_radar.py
│
├── tools/
│   ├── train.py                         # [Modified] Add Jittor initialization
│   └── test.py                          # [Modified] Add Jittor initialization
│
├── docs/
│   ├── install.md
│   ├── prepare_dataset.md
│   └── getting_started.md
│
├── figs/
│   ├── arch.png
│   └── sota_results.png
│
├── RCM_Jittor_Migration_Record.md       # Full migration record document
└── requirements.txt
```

---

## 🔧 Key Technical Details

### PyTorch ↔ Jittor API Mapping

| PyTorch / mmcv | Jittor Equivalent |
|:---|:---|
| `torch.Tensor` | `jt.Var` |
| `torch.zeros` / `torch.ones` | `jt.zeros` / `jt.ones` |
| `torch.cat` | `jt.concat` |
| `torch.stack` | `jt.stack` |
| `nn.MultiheadAttention` | Manually implemented `JittorMultiheadAttention` |
| `build_norm_layer(cfg, dims)` | `nn.LayerNorm(dims)` |
| `build_feedforward_network(cfg)` | `JittorFFN` |
| `build_attention(cfg)` | `build_jittor_module(cfg)` |
| `mmcv.ops.MultiScaleDeformableAttnFunction` | **Bridged back to PyTorch CUDA extension** |

### Modules Migrated to Jittor

| Module | Reason for Migration |
|:---|:---|
| `PerceptionTransformerRadar` | Core multi-modal fusion Transformer of the paper; compute-intensive |
| `RadarGuidedBEVEncoder` | Contains self-attention + cross-attention + radar-camera gating |
| `DetectionTransformerDecoder` | DETR-style decoder with multi-scale deformable attention |
| `MultiheadAttention` | Manually reimplemented with native Jittor operators |
| `FFN` | Manually reimplemented with native Jittor operators |
| `RadarCameraGating` | Core innovation: radar-camera gated fusion module |

---

## 🐛 Bug Fixes During Migration

During this migration, **5 critical bugs** were identified and fixed, lifting mAP from 0.1126 (initial) to 0.3592 (93% recovery).

| Bug | Symptom | Root Cause | Impact |
|:---|:---|:---|:---:|
| Encoder file corruption (duplicate class definition) | mAP dropped to 0.0324 | Two incomplete `RadarGuidedBEVEncoderLayer` definitions in one file | Major recovery |
| Decoder `batch_first` error | Nearly all class APs ≈ 0 | `DetrTransformerDecoderLayer` incorrectly inherited `batch_first=True` | Core fix |
| `CustomMSDeformableAttention` unconditional permute | Dimension mismatch | Missing `if not self.batch_first:` guard before permute | Auxiliary fix |
| `RadarCameraGating` parameter name mismatch | 8 Conv1d params failed to sync | `state_dict` key names inconsistent after Jittor rewrite | Weight sync |
| `MultiheadAttention` parameter path mismatch | 24 MHA params failed to sync | Missing `_AttnParams` sub-module wrapper | Weight sync |

> 📋 For the complete migration walkthrough, code analysis, and detailed bug reports, see [RCM_Jittor_Migration_Record.md](RCM_Jittor_Migration_Record.md).

---

## 📈 SOTA Comparison

<div align="center">
  <img src="figs/sota_results.png" alt="SOTA Performance Comparison" width="85%"/>
</div>

---

## 📝 Citation

If this work is helpful for your research, please consider citing the original paper:

```bibtex
@article{icra2024RCMFusion,
  title={RCM-Fusion: Radar-Camera Multi-Level Fusion for 3D Object Detection},
  author={Kim, Jisong and Seong, Minjae and Bang, Geonho and Kum, Dongsuk and Choi, Jun Won},
  journal={arXiv preprint arXiv:2307.10249},
  year={2024}
}
```

---

## 🙏 Acknowledgements

Many thanks to these excellent open-source projects:

- [RCM-Fusion (Original PyTorch Version)](https://github.com/mjseong0414/RCM-Fusion)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [detr3d](https://github.com/WangYueFt/detr3d)
- [Jittor](https://github.com/Jittor/jittor)

---

<div align="center">
  <sub>This repository is a Jittor mixed migration of RCM-Fusion, with the core Transformer modules running in the Jittor framework.</sub>
</div>
