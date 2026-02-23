# import sys
# sys.path.append('/home/spalab/RCM_fusion/mmdetection3d')
# sys.path.append('/home/spalab/RCM_fusion/mmdetection3d/RCM-Fusion')

import argparse
import mmcv
import os
import torch
import warnings
warnings.filterwarnings("ignore", message=".*grid_sample.*align_corners.*")
warnings.filterwarnings("ignore", message=".*affine_grid.*align_corners.*")
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.rcm_fusion.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp


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
        # 先尝试 PyTorch API
        try:
            if torch.cuda.is_available():
                info['device_name'] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                info['total_mem_gb'] = f"{props.total_mem / (1024**3):.1f} GB"
        except Exception:
            pass

        # 若 PyTorch 未获取到，用 nvidia-smi
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

        # 若仍未获取显存，尝试 Jittor API
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

    # 4) 即使上面检测 CUDA 不可用，也尝试 nvidia-smi 获取信息（可能只是 API 不通）
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
    # 计算最大宽度
    max_label = max(len(r[0]) for r in rows)
    max_value = max(len(str(r[1])) for r in rows)
    inner_width = max(max_label + max_value + 7, len(title) + 4)

    border = "=" * (inner_width + 4)
    print(f"\n{border}")
    # 居中标题
    pad = (inner_width + 2 - len(title)) // 2
    print(f"  {' ' * pad}{title}")
    print(border)
    for label, value in rows:
        padding = " " * (max_label - len(label))
        print(f"  {label}{padding}  : {value}")
    print(f"{border}\n")


def print_test_info(args, dataset, samples_per_gpu):
    """打印测试运行信息面板"""
    jt_ver, jt_enabled = _get_jittor_info()
    cuda_info = _get_cuda_info()

    jt_status = f"{jt_ver} (已启用)" if jt_enabled else "未启用"
    cuda_status = (f"✅ 可用 - {cuda_info['device_name']}"
                   if cuda_info['available'] else "❌ 不可用")

    rows = [
        ("Jittor 状态", jt_status),
        ("CUDA 状态", cuda_status),
        ("GPU 显存", cuda_info['total_mem_gb']),
        ("样本总数", str(len(dataset))),
        ("模型文件", osp.basename(args.checkpoint)),
        ("配置文件", osp.basename(args.config)),
        ("Samples/GPU", str(samples_per_gpu)),
    ]

    _print_info_panel("🔍 TEST 运行信息", rows)


def print_train_info(args, cfg, datasets):
    """打印训练运行信息面板"""
    jt_ver, jt_enabled = _get_jittor_info()
    cuda_info = _get_cuda_info()

    jt_status = f"{jt_ver} (已启用)" if jt_enabled else "未启用"
    cuda_status = (f"✅ 可用 - {cuda_info['device_name']}"
                   if cuda_info['available'] else "❌ 不可用")

    # 安全获取各项配置
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
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
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
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
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

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    # ====== 提前初始化 Jittor 模块（体现 Jittor 迁移）======
    print("=" * 60)
    print("=> 开始初始化 Jittor 模块...")
    model.pts_bbox_head._init_jittor_modules()
    print("=> Jittor 模块初始化完成。")
    print("=" * 60)
    
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        model.eval()
        outputs = []
        dataset = data_loader.dataset

        # ====== 打印测试运行信息面板 ======
        print_test_info(args, dataset, samples_per_gpu)

        prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            outputs.extend(result)
            batch_size = len(result)
            for _ in range(batch_size):
                prog_bar.update()
        print() # Adding a newline after progress bar

    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir,
                                        args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            assert False
            #mmcv.dump(outputs['bbox_results'], args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        if args.format_only:
            dataset.format_results(outputs, **kwargs)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

            # print(dataset.evaluate(_, **eval_kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
