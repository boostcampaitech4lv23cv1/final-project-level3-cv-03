import copy
import os.path as osp
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         rfnext_init_model, setup_multi_processes)
import numpy as np
import random
import wandb

def set_seed(seed):
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def main():
    wandb.login()
    runs=wandb.Api().runs(path="aitech4_cv3/Final\ Project", order="created_at")
    try:
        this_run_num=f"{int(runs[0].name[1:4])+1:03d}"
    except:
        this_run_num="000"
    wandb.init(
        entity="aitech4_cv3",
        project="Final Project",
        config="/opt/ml/final-project-level3-cv-03/sweeps/config.yaml"
        )
    this_run_name=f"[{this_run_num}]-cascade-swin-large"
    wandb.run.name=this_run_name
    wandb.run.save()

    cfg = Config.fromfile(wandb.config.model_config)
    cfg.gpu_ids = [0]
    cfg.work_dir = osp.join("/opt/ml/final-project-level3-cv-03/mmdetection/work_dirs",
                            wandb.run.id)
    
    # create work_dir
    mmcv.mkdir_or_exist(cfg.work_dir)
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(wandb.config.model_config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # set multi-process settings
    setup_multi_processes(cfg)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Config:\n{cfg.pretty_text}')

    cfg.device = get_device()
    cfg.seed = wandb.config.seed
    set_seed(wandb.config.seed)
    meta['exp_name'] = osp.basename(wandb.config.model_config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    rfnext_init_model(model, cfg=cfg)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.get(
            'pipeline', cfg.data.train.dataset.get('pipeline'))
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=False,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()