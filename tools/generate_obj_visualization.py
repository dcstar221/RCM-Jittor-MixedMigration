"""
generate_obj_visualization.py
==============================
从 NuScenes 数据集生成 .obj 文件，用于在 MeshLab 等 3D 软件中可视化：
  1. ***_points.obj  —— LiDAR 原始点云
  2. ***_gt.obj      —— Ground Truth 3D 边界框（用 trimesh 导出为网格）
  3. ***_pred.obj    —— 预测 3D 边界框
  4. ***_seg.obj     —— 按类别伪色分割的点云（XYZRGB 格式 .obj）

用法:
  conda activate rcm_jittor
  cd d:\T03-jittor\RCM-Fusion-main
  python tools/generate_obj_visualization.py \
      --dataroot  data/RCM_Data/v1.0-mini  \
      --version   v1.0-mini                \
      --results   test/rcm-fusion_r50/Mon_Feb_23_03_00_21_2026/pts_bbox/results_nusc.json \
      --out-dir   output_obj               \
      --num-samples 5
"""

import argparse
import os
import sys
import json
import pickle
import math
import numpy as np
from pathlib import Path

# ─── 颜色表：nuScenes 10 类检测类别 ────────────────────────────────────────
CLASS_COLORS = {
    'car':                  (0.8, 0.1, 0.1),    # 红
    'truck':                (0.9, 0.5, 0.1),    # 橙
    'bus':                  (1.0, 0.8, 0.0),    # 黄
    'trailer':              (0.4, 0.8, 0.2),    # 浅绿
    'construction_vehicle': (0.2, 0.6, 0.9),    # 蓝
    'motorcycle':           (0.6, 0.2, 0.8),    # 紫
    'bicycle':              (0.8, 0.5, 0.9),    # 淡紫
    'pedestrian':           (0.1, 0.9, 0.7),    # 青绿
    'traffic_cone':         (1.0, 0.6, 0.2),    # 橙黄
    'barrier':              (0.5, 0.5, 0.5),    # 灰
    'unknown':              (1.0, 1.0, 1.0),    # 白
}

# nuScenes 类别名 → 检测类别名（部分映射）
NUS_CATEGORY_TO_DET = {
    'vehicle.car':                      'car',
    'vehicle.truck':                    'truck',
    'vehicle.bus.bendy':                'bus',
    'vehicle.bus.rigid':                'bus',
    'vehicle.trailer':                  'trailer',
    'vehicle.construction':             'construction_vehicle',
    'vehicle.motorcycle':               'motorcycle',
    'vehicle.bicycle':                  'bicycle',
    'human.pedestrian.adult':           'pedestrian',
    'human.pedestrian.child':           'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer':  'pedestrian',
    'movable_object.trafficcone':       'traffic_cone',
    'movable_object.barrier':           'barrier',
}

# ─── .obj / mesh 写入工具函数 ─────────────────────────────────────────────

def write_points_obj(points: np.ndarray, filepath: str):
    """将点云写入 .obj（纯顶点格式）。
    points: (N, 3) 或 (N, 4+)，如果恰好是 (N, 6) 则前3列xyz 后3列rgb[0-255]。
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w') as f:
        f.write("# OBJ point cloud\n")
        for row in points:
            if points.shape[1] >= 6:
                r, g, b = int(row[3]), int(row[4]), int(row[5])
                f.write(f"v {row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {r} {g} {b}\n")
            else:
                f.write(f"v {row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
    print(f"  [✓] 写入点云 .obj : {filepath}  ({len(points)} 点)")


def _rotation_z(yaw: float) -> np.ndarray:
    """绕 Z 轴旋转矩阵。"""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


def _box_to_corners(cx, cy, cz, dx, dy, dz, yaw) -> np.ndarray:
    """返回 3D 框 8 个角点 (8, 3)，cx/cy/cz 为中心，dx/dy/dz 为长宽高。"""
    # 局部坐标下 8 角
    half = np.array([dx, dy, dz]) / 2.0
    corners_local = np.array([
        [ 1,  1,  1],
        [ 1, -1,  1],
        [-1, -1,  1],
        [-1,  1,  1],
        [ 1,  1, -1],
        [ 1, -1, -1],
        [-1, -1, -1],
        [-1,  1, -1],
    ], dtype=float) * half  # (8, 3)

    R = _rotation_z(yaw)
    corners_world = (R @ corners_local.T).T + np.array([cx, cy, cz])
    return corners_world


def write_boxes_obj(boxes: list, filepath: str, color=(0.2, 0.8, 0.2)):
    """将 3D 边界框写入 .obj（线框网格），每个框导出 12 条棱。
    boxes: list of dict, 每个 dict 含 'translation'(x,y,z), 'size'(w,l,h),
           'rotation'(四元数 w,x,y,z), 'detection_name'(str), 'detection_score'(float)
    """
    if len(boxes) == 0:
        print(f"  [!] 无框可写: {filepath}")
        return

    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

    # OBJ 不支持内置颜色，需要 MTL；这里把颜色写入注释并 MTL
    mtl_path = filepath.replace('.obj', '.mtl')
    used_materials = {}  # mat_name -> (r,g,b)

    lines_v = []    # 顶点行
    lines_f = []    # 面/线行（用 l 表示线段）
    v_offset = 1    # OBJ 顶点从 1 开始

    for box in boxes:
        # ── 解析位置 / 尺寸 / 旋转 ──
        tx, ty, tz = box['translation']
        # nuScenes: size = [width, length, height]  ← w=y, l=x, h=z
        w, l, h = box['size']
        # nuScenes 四元数 (qw, qx, qy, qz)
        q = box.get('rotation', [1, 0, 0, 0])
        qw, qx, qy, qz = q
        # 提取绕 Z 轴 yaw
        yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

        name = box.get('detection_name', 'unknown')
        score = box.get('detection_score', -1)
        mat_name = name.replace(' ', '_')
        clr = CLASS_COLORS.get(name, CLASS_COLORS['unknown'])
        used_materials[mat_name] = clr

        corners = _box_to_corners(tx, ty, tz, l, w, h, yaw)  # (8,3)

        # 写顶点
        for c in corners:
            lines_v.append(f"v {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")

        # 12 条棱 (OBJ 用 l 定义线段)
        edges = [
            (1,2),(2,3),(3,4),(4,1),   # 顶面
            (5,6),(6,7),(7,8),(8,5),   # 底面
            (1,5),(2,6),(3,7),(4,8),   # 竖棱
        ]
        lines_f.append(f"usemtl {mat_name}\n")
        for e in edges:
            a, b = e[0] + v_offset - 1, e[1] + v_offset - 1
            lines_f.append(f"l {a} {b}\n")

        v_offset += 8

    # 写 .mtl
    with open(mtl_path, 'w') as f:
        f.write("# MTL material file\n")
        for mat, (r, g, b) in used_materials.items():
            f.write(f"\nnewmtl {mat}\n")
            f.write(f"Kd {r:.3f} {g:.3f} {b:.3f}\n")
            f.write(f"Ka 0.1 0.1 0.1\n")
            f.write(f"Ks 0.0 0.0 0.0\n")
            f.write(f"illum 1\n")

    # 写 .obj
    with open(filepath, 'w') as f:
        f.write(f"# OBJ bbox file  ({len(boxes)} boxes)\n")
        f.write(f"mtllib {os.path.basename(mtl_path)}\n\n")
        f.writelines(lines_v)
        f.write("\n")
        f.writelines(lines_f)

    print(f"  [✓] 写入边界框 .obj : {filepath}  ({len(boxes)} 框)")


def write_seg_obj(points: np.ndarray, labels: np.ndarray,
                  palette: np.ndarray, filepath: str, ignore_index: int = None):
    """将带分割标签的点云写成带颜色的 .obj。
    points: (N, 3+)    labels: (N,)    palette: (C, 3) uint8
    """
    if ignore_index is not None:
        mask = labels != ignore_index
        points = points[mask]
        labels = labels[mask]

    colors = palette[labels % len(palette)]  # (N, 3) uint8
    xyzrgb = np.concatenate([points[:, :3], colors.astype(float)], axis=1)  # (N,6)
    write_points_obj(xyzrgb, filepath)
    print(f"  [✓] 写入分割掩码 .obj : {filepath}")


# ─── NuScenes 简易读取（不依赖完整 mmdet3d，仅需 nuscenes-devkit）──────────

def load_lidar_points_from_nuscenes(nusc, sample_token: str) -> np.ndarray:
    """从 NuScenes 加载 LiDAR 点云，返回 (N, 4) float32: x,y,z,intensity"""
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_filepath = os.path.join(nusc.dataroot, lidar_data['filename'])

    pts = np.fromfile(lidar_filepath, dtype=np.float32).reshape(-1, 5)
    # nuScenes LiDAR: x, y, z, intensity, ring_index
    return pts[:, :4]  # 只取 xyz + intensity


def load_gt_boxes_from_nuscenes(nusc, sample_token: str) -> list:
    """从 NuScenes 加载该帧 GT 3D 框（转换到全局坐标系，可与点云对齐）"""
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion

    sample = nusc.get('sample', sample_token)
    ann_tokens = sample['anns']
    boxes = []
    for ann_token in ann_tokens:
        ann = nusc.get('sample_annotation', ann_token)
        cat = ann['category_name']
        det_name = NUS_CATEGORY_TO_DET.get(cat, None)
        if det_name is None:
            continue

        q = ann['rotation']  # [qw, qx, qy, qz]
        boxes.append({
            'translation': ann['translation'],
            'size': ann['size'],         # [width, length, height]
            'rotation': q,
            'detection_name': det_name,
            'detection_score': 1.0,
        })
    return boxes


# ─── 主逻辑 ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='生成 .obj 3D 可视化文件')
    parser.add_argument('--dataroot', default='data/RCM_Data/v1.0-mini',
                        help='NuScenes 数据集根目录')
    parser.add_argument('--version',  default='v1.0-mini',
                        help='NuScenes 版本，如 v1.0-mini / v1.0-trainval')
    parser.add_argument('--results',
                        default='test/rcm-fusion_r50/Mon_Feb_23_03_00_21_2026/pts_bbox/results_nusc.json',
                        help='模型预测结果 JSON (results_nusc.json)')
    parser.add_argument('--out-dir',  default='output_obj',
                        help='输出目录')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='最多处理几个 sample（0 = 全部）')
    parser.add_argument('--score-thr', type=float, default=0.2,
                        help='预测框置信度阈值')
    parser.add_argument('--no-lidar', action='store_true',
                        help='跳过 LiDAR 点云（数据较大）')
    return parser.parse_args()


def main():
    args = parse_args()

    # 路径处理（相对路径 → 绝对路径，相对于脚本的上两级目录）
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..'))
    def abspath(p):
        return p if os.path.isabs(p) else os.path.join(project_root, p)

    dataroot  = abspath(args.dataroot)
    results_f = abspath(args.results)
    out_dir   = abspath(args.out_dir)

    print("=" * 60)
    print("  RCM-Fusion .OBJ 可视化生成器")
    print("=" * 60)
    print(f"  NuScenes dataroot : {dataroot}")
    print(f"  预测结果 JSON     : {results_f}")
    print(f"  输出目录          : {out_dir}")
    print(f"  置信度阈值        : {args.score_thr}")
    print()

    # ── 检查依赖 ──
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError:
        print("[ERROR] 请安装 nuscenes-devkit: pip install nuscenes-devkit")
        sys.exit(1)

    # ── 加载 NuScenes ──
    print("[INFO] 加载 NuScenes 数据库...")
    nusc = NuScenes(version=args.version, dataroot=dataroot, verbose=False)
    print(f"[INFO] 数据库加载完成，共 {len(nusc.sample)} 条 sample")

    # ── 加载预测结果 ──
    if not os.path.exists(results_f):
        print(f"[ERROR] 未找到预测结果文件: {results_f}")
        sys.exit(1)
    print(f"[INFO] 加载预测结果: {results_f}")
    with open(results_f, 'r') as f:
        pred_data = json.load(f)
    pred_results = pred_data.get('results', {})
    print(f"[INFO] 预测结果共 {len(pred_results)} 条 sample")

    # ── 取交集 token ──
    nusc_tokens = {s['token'] for s in nusc.sample}
    valid_tokens = [t for t in pred_results.keys() if t in nusc_tokens]
    print(f"[INFO] 数据集与预测结果的交集: {len(valid_tokens)} 条")

    if len(valid_tokens) == 0:
        print("[ERROR] 没有匹配的 sample token，请检查版本是否对应！")
        sys.exit(1)

    n = len(valid_tokens) if args.num_samples == 0 else min(args.num_samples, len(valid_tokens))
    tokens_to_process = valid_tokens[:n]
    print(f"[INFO] 将处理 {n} 个 sample\n")

    os.makedirs(out_dir, exist_ok=True)

    # ── 逐 sample 生成 .obj ──
    for idx, token in enumerate(tokens_to_process):
        sample_dir = os.path.join(out_dir, f"sample_{idx:03d}_{token[:8]}")
        os.makedirs(sample_dir, exist_ok=True)

        print(f"[{idx+1}/{n}] token: {token}")
        print(f"  输出目录: {sample_dir}")

        # 1. 点云 .obj
        if not args.no_lidar:
            try:
                pts = load_lidar_points_from_nuscenes(nusc, token)
                pts_file = os.path.join(sample_dir, f"{token[:8]}_points.obj")
                write_points_obj(pts[:, :3], pts_file)
            except Exception as e:
                print(f"  [!] 点云加载失败: {e}")

        # 2. GT 框 .obj
        try:
            gt_boxes = load_gt_boxes_from_nuscenes(nusc, token)
            gt_file = os.path.join(sample_dir, f"{token[:8]}_gt.obj")
            write_boxes_obj(gt_boxes, gt_file)
        except Exception as e:
            print(f"  [!] GT 框加载失败: {e}")

        # 3. 预测框 .obj
        try:
            pred_boxes = pred_results.get(token, [])
            # 过滤置信度
            pred_boxes = [b for b in pred_boxes if b.get('detection_score', 0) >= args.score_thr]
            pred_file = os.path.join(sample_dir, f"{token[:8]}_pred.obj")
            write_boxes_obj(pred_boxes, pred_file)
        except Exception as e:
            print(f"  [!] 预测框处理失败: {e}")

        # 4. 分割掩码 .obj（按类别伪色，基于 GT 标注生成）
        try:
            if not args.no_lidar:
                pts_full = load_lidar_points_from_nuscenes(nusc, token)
                gt_boxes_raw = load_gt_boxes_from_nuscenes(nusc, token)
                # 简易点云分割：判断每个点是否在某个 GT 框内并分配标签
                labels = np.zeros(len(pts_full), dtype=np.int32)  # 0 = background
                class_list = list(CLASS_COLORS.keys())
                for box in gt_boxes_raw:
                    tx, ty, tz = box['translation']
                    w, l, h = box['size']
                    q = box['rotation']
                    qw, qx, qy, qz = q
                    yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
                    # 反旋转点云到框局部坐标系
                    R = _rotation_z(-yaw)
                    local = (R @ (pts_full[:, :3] - np.array([tx, ty, tz])).T).T
                    in_box = (
                        (np.abs(local[:, 0]) <= l/2) &
                        (np.abs(local[:, 1]) <= w/2) &
                        (np.abs(local[:, 2]) <= h/2)
                    )
                    det_name = box['detection_name']
                    cls_idx = class_list.index(det_name) + 1 if det_name in class_list else 0
                    labels[in_box] = cls_idx

                # 构建调色盘
                palette = np.array([
                    [200, 200, 200],   # 0: background (灰)
                ] + [
                    [int(c[0]*255), int(c[1]*255), int(c[2]*255)]
                    for c in list(CLASS_COLORS.values())
                ], dtype=np.uint8)

                seg_file = os.path.join(sample_dir, f"{token[:8]}_seg.obj")
                write_seg_obj(pts_full, labels, palette, seg_file, ignore_index=None)
        except Exception as e:
            print(f"  [!] 分割掩码生成失败: {e}")

        print()

    print("=" * 60)
    print(f"[完成] 所有 .obj 文件已保存至: {out_dir}")
    print()
    print("  文件说明:")
    print("    *_points.obj  —— LiDAR 原始点云（顶点）")
    print("    *_gt.obj      —— Ground Truth 3D 边界框（线框，含 .mtl 颜色文件）")
    print("    *_pred.obj    —— 模型预测 3D 边界框（线框，含 .mtl 颜色文件）")
    print("    *_seg.obj     —— 按目标类别着色的点云分割掩码")
    print()
    print("  推荐查看方式:")
    print("    MeshLab: 拖入 .obj 文件即可查看（File → Import Mesh）")
    print("    CloudCompare: 支持带颜色点云，直接打开 *_points.obj")
    print("=" * 60)


if __name__ == '__main__':
    main()
