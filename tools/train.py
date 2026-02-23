from __future__ import division

# import sys
# sys.path.append('/home/spalab/RCM_fusion/mmdetection3d')
# sys.path.append('/home/spalab/RCM_fusion/mmdetection3d/RCM-Fusion')

import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
warnings.filterwarnings("ignore", message=".*grid_sample.*align_corners.*")
warnings.filterwarnings("ignore", message=".*affine_grid.*align_corners.*")
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
#from mmdet3d.apis import train_model

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

from mmcv.utils import TORCH_VERSION, digit_version


def _get_jittor_info():
    """安全获取 Jittor 信息"""
    try:
        import jittor as jt
        version = jt.__version__
        return version, True
    except Exception:
        return "N/A", False


def _run_nvidia_smi():
    """尝试多种方式运行 nvidia-smi（兼容 Windows 路径）"""
    import subprocess
    candidates = [
        'nvidia-smi',
        os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32', 'nvidia-smi.exe'),
        r'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe',
    ]
    for cmd in candidates:
        try:
            result = subprocess.run(
                [cmd, '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            continue
    return None


def _get_cuda_info():
    """安全获取 CUDA 信息（兼容 Jittor 和 PyTorch 环境）"""
    info = {'available': False, 'device_name': 'N/A', 'total_mem_gb': 'N/A'}

    # 1) 尝试通过 Jittor 检测 CUDA
    try:
        import jittor as jt
        if getattr(jt.flags, 'use_cuda', 0) or getattr(jt.compiler, 'has_cuda', False):
            info['available'] = True
    except Exception:
        pass

    # 2) 尝试通过 PyTorch 检测 CUDA
    if not info['available']:
        try:
            if torch.cuda.is_available():
                info['available'] = True
        except Exception:
            pass

    # 3) 获取 GPU 名称和显存
    if info['available']:
        try:
            if torch.cuda.is_available():
                info['device_name'] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                info['total_mem_gb'] = f"{props.total_mem / (1024**3):.1f} GB"
        except Exception:
            pass

        if info['device_name'] == 'N/A':
            smi_output = _run_nvidia_smi()
            if smi_output:
                try:
                    line = smi_output.split('\n')[0]
                    parts = line.split(',')
                    info['device_name'] = parts[0].strip()
                    mem_mb = float(parts[1].strip())
                    info['total_mem_gb'] = f"{mem_mb / 1024:.1f} GB"
                except Exception:
                    pass

        if info['total_mem_gb'] == 'N/A':
            try:
                import jittor as jt
                mem_info = jt.get_mem_info()
                if mem_info and 'total' in str(mem_info):
                    total_bytes = mem_info.get('total', 0) if isinstance(mem_info, dict) else 0
                    if total_bytes > 0:
                        info['total_mem_gb'] = f"{total_bytes / (1024**3):.1f} GB"
            except Exception:
                pass

    # 4) 即使上面检测 CUDA 不可用，也尝试 nvidia-smi
    if not info['available']:
        smi_output = _run_nvidia_smi()
        if smi_output:
            try:
                line = smi_output.split('\n')[0]
                parts = line.split(',')
                info['available'] = True
                info['device_name'] = parts[0].strip()
                mem_mb = float(parts[1].strip())
                info['total_mem_gb'] = f"{mem_mb / 1024:.1f} GB"
            except Exception:
                pass

    return info


def _print_info_panel(title, rows):
    """打印带边框的信息面板"""
    max_label = max(len(r[0]) for r in rows)
    max_value = max(len(str(r[1])) for r in rows)
    inner_width = max(max_label + max_value + 7, len(title) + 4)

    border = "=" * (inner_width + 4)
    print(f"\n{border}")
    pad = (inner_width + 2 - len(title)) // 2
    print(f"  {' ' * pad}{title}")
    print(border)
    for label, value in rows:
        padding = " " * (max_label - len(label))
        print(f"  {label}{padding}  : {value}")
    print(f"{border}\n")


def print_train_info(args, cfg, datasets):
    """打印训练运行信息面板"""
    jt_ver, jt_enabled = _get_jittor_info()
    cuda_info = _get_cuda_info()

    jt_status = f"{jt_ver} (已启用)" if jt_enabled else "未启用"
    cuda_status = (f"✅ 可用 - {cuda_info['device_name']}"
                   if cuda_info['available'] else "❌ 不可用")

    try:
        max_epochs = str(cfg.runner.max_epochs)
    except Exception:
        max_epochs = "N/A"
    try:
        lr = str(cfg.optimizer.lr)
    except Exception:
        lr = "N/A"
    try:
        optim_type = str(cfg.optimizer.type)
    except Exception:
        optim_type = "N/A"
    try:
        batch_size = str(cfg.data.samples_per_gpu)
    except Exception:
        batch_size = "N/A"
    try:
        resume = str(cfg.resume_from) if cfg.resume_from else (str(cfg.load_from) if cfg.get('load_from') else "无")
    except Exception:
        resume = "无"

    rows = [
        ("Jittor 状态", jt_status),
        ("CUDA 状态", cuda_status),
        ("GPU 显存", cuda_info['total_mem_gb']),
        ("总周期(Epochs)", max_epochs),
        ("训练样本总数", str(len(datasets[0]))),
        ("Batch Size", batch_size),
        ("学习率(LR)", lr),
        ("优化器", optim_type),
        ("Work Dir", str(cfg.work_dir)),
        ("Resume/Load", osp.basename(resume) if resume != "无" else "无"),
        ("配置文件", osp.basename(args.config)),
    ]

    _print_info_panel("🚀 TRAIN 运行信息", rows)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=None,
        help='override the maximum number of training epochs'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='override the maximum number of training samples'
    )
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    if args.max_epochs is not None:
        cfg.runner.max_epochs = args.max_epochs
        cfg.total_epochs = args.max_epochs
        
    if args.max_samples is not None:
        if 'dataset' in cfg.data.train and isinstance(cfg.data.train.dataset, dict):
            cfg.data.train.dataset.max_samples = args.max_samples
            # Bypass CBGSDataset since oversampling might fail on a tiny subset
            if cfg.data.train.type == 'CBGSDataset':
                cfg.data.train = cfg.data.train.dataset
        else:
            cfg.data.train.max_samples = args.max_samples
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

            from projects.mmdet3d_plugin.rcm_fusion.apis.train import custom_train_model
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # if args.resume_from is not None:
    if args.resume_from is not None and osp.isfile(args.resume_from):
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    if digit_version(TORCH_VERSION) == digit_version('1.8.1') and cfg.optimizer['type'] == 'AdamW':
        cfg.optimizer['type'] = 'AdamW2' # fix bug in Adamw
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    try:
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    except Exception as e:
        env_info = f"Failed to collect env info: {e}"
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.debug(f'Config:\n{cfg.pretty_text}')
    logger.info('Config loaded successfully. (use DEBUG log level to see full config)')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # ====== 提前初始化 Jittor 模块（体现 Jittor 迁移）======
    logger.info('Initializing Jittor modules...')
    model.pts_bbox_head._init_jittor_modules()
    logger.info('Jittor modules initialized successfully.')
    
    logger.debug(f'Model:\n{model}')
    logger.info('Model built successfully. (use DEBUG log level to see full model architecture)')
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # ====== 打印训练运行信息面板 ======
    print_train_info(args, cfg, datasets)

    custom_train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)

if __name__ == '__main__':
    main()
