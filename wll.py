# -*- coding:utf-8 -*-
# import warnings
# warnings.filterwarnings('ignore') #屏蔽警告信息：在一些情况下，比如对于已知或不可避免的警告信息，很好的拯救了：tqdm
from logging import Logger
import logging
import os
import yaml
import os.path as osp
from pathlib import Path
import torch
import torch.distributed as dist
import sys
# 我添加的
import types
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import glob
# 
import os.path as osp
import shutil
import tempfile
from importlib import import_module
from addict import Dict
# 
from wllEngine import Trainer

######################### from yolov6.utils.envs import get_envs, set_random_seed ###############
def get_envs():    
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    rank = int(os.getenv('RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', 1)) #1
    return local_rank, rank, world_size

def set_random_seed(seed, deterministic=False):   
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True

######################### from yolov6.utils.events import LOGGER ###############

def set_logging(name=None):
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)

LOGGER = set_logging(__name__)
NCOLS = min(100, shutil.get_terminal_size().columns)

##########################################################

# 检查与初始化 - 使用更多的参数
def check_and_init(args):    
    
    # check files        
    args.save_dir = str(osp.join(args.output_dir, args.name)) #保存目录: ./runs/train/*
    if not os.path.exists(args.save_dir):        
        os.makedirs(args.save_dir)    

    from munch import DefaultMunch as dictToObj #字典转对象 pip install munch -i https://pypi.tuna.tsinghua.edu.cn/simple
    cfg={
    'use_checkpoint': False,
    'model': {
        'type': 'GoldYOLO-s',
        'pretrained': None,
        'depth_multiple': 0.33,
        'width_multiple': 0.5,
        'backbone': {'type': 'EfficientRep', 'num_repeats': [1, 6, 12, 18, 6], 'out_channels': [64, 128, 256, 512, 1024], 'fuse_P2': True, 'cspsppf': True},
        'neck': {'type': 'RepGDNeck', 'num_repeats': [12, 12, 12, 12], 'out_channels': [256, 128, 128, 256, 256, 512],            
            'extra_cfg': {
                'norm_cfg': {'type': 'SyncBN', 'requires_grad': True},
                'depths': 2, 'fusion_in': 960, 'ppa_in': 704, 'fusion_act': {'type': 'ReLU6'}, 'fuse_block_num': 3,
                'embed_dim_p': 128, 'embed_dim_n': 704, 'key_dim': 8, 'num_heads': 4, 'mlp_ratios': 1, 'attn_ratios': 2, 'c2t_stride': 2,
                'drop_path_rate': 0.1, 'trans_channels': [128, 64, 128, 256], 
                'pool_mode': 'torch'
            }
        },
        'head': {'type': 'EffiDeHead', 'in_channels': [128, 256, 512], 'num_layers': 3, 'begin_indices': 24, 'anchors': 3, 'anchors_init': [[10, 13, 19, 19, 33, 23], [30, 61, 59, 59, 59, 119], [116, 90, 185, 185, 373, 326]], 'out_indices': [17, 20, 23], 'strides': [8, 16, 32], 'atss_warmup_epoch': 0, 'iou_type': 'giou', 'use_dfl': True, 'reg_max': 16, 'distill_weight': {'class': 1.0, 'dfl': 1.0}}},
        'solver': {'optim': 'SGD', 'lr_scheduler': 'Cosine', 'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1},
        'data_aug': {'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0},
        'training_mode': 'repvgg', #附加的
    }
    cfg = dictToObj.fromDict(cfg) #字典转对象   
      

    # 获取 cpu 或 gpu 驱动
    device = args.device #select_device

    # set random seed
    set_random_seed(1+args.rank, deterministic=(args.rank == -1)) #1,False

    return cfg, device, args #新的cfg配置信息,cpu或gpu设备,旧的传进来的参数（可能有修改）



##########################################################

# 训练
args = types.SimpleNamespace()
args.batch_size=2    
args.bs_per_gpu=32
args.calib=False
args.check_images=False
args.check_labels=False
args.conf_file='configs/gold_yolo-s.py'
args.data_path='data/coco.yaml'
args.device='cpu'
args.dist_url='env://'

# 教师模型
args.distill=False
args.distill_feat=False

args.epochs=2
args.eval_final_only=False
args.eval_interval=20
args.fuse_ab=False
args.gpu_count=0
args.heavy_eval_range=50
args.img_size=640
args.local_rank=-1
args.name='gold_yolo-s'
args.output_dir='./runs/train'
args.quant=False #量化模型
args.resume=False #继续接着训练-训练到半的模型权重文件
args.save_ckpt_on_last_n_epoch=-1
args.stop_aug_last_n_epoch=15
args.teacher_model_path=None
args.temperature=20
args.use_syncbn=False
args.workers=0
args.write_trainbatch_tb=False        


# 使用更多的参数
args.local_rank, args.rank, args.world_size = get_envs()
cfg, device, args = check_and_init(args) #检查与初始化 - 使用更多的参数

# 从新设置这几个值 - 重载环境，因为在check_and_init（args）中更改了args
args.local_rank, args.rank, args.world_size = get_envs()

print(f'training args are: {args}\n')


# Start - 训练类
trainer = Trainer(args, cfg, device)

# PTQ
if args.quant and args.calib:
    trainer.calibrate(cfg)
    print('返回空')
    
trainer.train()

# End
if args.world_size > 1 and args.rank == 0:
    LOGGER.info('Destroying process group... ')
    dist.destroy_process_group()