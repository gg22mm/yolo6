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
from yolov6.core.engine import Trainer

######################### from yolov6.utils.envs import get_envs, select_device, set_random_seed ###############
def get_envs():    
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    rank = int(os.getenv('RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', 1))
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

def select_device(device):   
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        LOGGER.info('Using CPU for training... ')
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available()
        nd = len(device.strip().split(','))
        LOGGER.info(f'Using {nd} GPU for training... ')
    cuda = device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    return device


######################### from yolov6.utils.config import Config ###############
class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError("'{}' object has no attribute '{}'".format(
                self.__class__.__name__, name))
        except Exception as e:
            ex = e
        else:
            return value
        raise ex

class Config(object):
    @staticmethod
    def _file2dict(filename):
        filename = str(filename)
        if filename.endswith('.py'):
            with tempfile.TemporaryDirectory() as temp_config_dir:
                shutil.copyfile(filename,
                                osp.join(temp_config_dir, '_tempconfig.py'))
                sys.path.insert(0, temp_config_dir)
                mod = import_module('_tempconfig')
                sys.path.pop(0)
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith('__')
                }
                # delete imported module
                del sys.modules['_tempconfig']
        else:
            raise IOError('Only .py type are supported now!')
        cfg_text = filename + '\n'
        with open(filename, 'r') as f:
            cfg_text += f.read()

        return cfg_dict, cfg_text

    @staticmethod
    def fromfile(filename):
        cfg_dict, cfg_text = Config._file2dict(filename)
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but got {}'.format(
                type(cfg_dict)))

        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super(Config, self).__setattr__('_filename', filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, 'r') as f:
                text = f.read()
        else:
            text = ''
        super(Config, self).__setattr__('_text', text)

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    def __repr__(self):
        return 'Config (path: {}): {}'.format(self.filename,
                                              self._cfg_dict.__repr__())

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

######################### from yolov6.utils.events import LOGGER, save_yaml ###############

def set_logging(name=None):
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)

LOGGER = set_logging(__name__)
NCOLS = min(100, shutil.get_terminal_size().columns)

def load_yaml(file_path):    
    if isinstance(file_path, str):
        with open(file_path, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
    return data_dict

def save_yaml(data_dict, save_path):    
    with open(save_path, 'w') as f:
        yaml.safe_dump(data_dict, f, sort_keys=False)

def write_tblog(tblogger, epoch, results, losses):    
    tblogger.add_scalar("val/mAP@0.5", results[0], epoch + 1)
    tblogger.add_scalar("val/mAP@0.50:0.95", results[1], epoch + 1)

    tblogger.add_scalar("train/iou_loss", losses[0], epoch + 1)
    tblogger.add_scalar("train/dist_focalloss", losses[1], epoch + 1)
    tblogger.add_scalar("train/cls_loss", losses[2], epoch + 1)

    tblogger.add_scalar("x/lr0", results[2], epoch + 1)
    tblogger.add_scalar("x/lr1", results[3], epoch + 1)
    tblogger.add_scalar("x/lr2", results[4], epoch + 1)

def write_tbimg(tblogger, imgs, step, type='train'):    
    if type == 'train':
        tblogger.add_image(f'train_batch', imgs, step + 1, dataformats='HWC')
    elif type == 'val':
        for idx, img in enumerate(imgs):
            tblogger.add_image(f'val_img_{idx + 1}', img, step + 1, dataformats='HWC')
    else:
        LOGGER.warning('WARNING: Unknown image type to visualize.\n')


######################### from yolov6.utils.general import increment_name, find_latest_checkpoint ###############

def increment_name(path):   
    path = Path(path)
    sep = ''
    if path.exists():
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(1, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not os.path.exists(p):
                break
        path = Path(p)
    return path

def find_latest_checkpoint(search_dir='.'):    
    checkpoint_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(checkpoint_list, key=os.path.getctime) if checkpoint_list else ''

def dist2bbox(distance, anchor_points, box_format='xyxy'):    
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if box_format == 'xyxy':
        bbox = torch.cat([x1y1, x2y2], -1)
    elif box_format == 'xywh':
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        bbox = torch.cat([c_xy, wh], -1)
    return bbox

def bbox2dist(anchor_points, bbox, reg_max):    
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    dist = torch.cat([lt, rb], -1).clip(0, reg_max - 0.01)
    return dist

def xywh2xyxy(bboxes):    
    bboxes[..., 0] = bboxes[..., 0] - bboxes[..., 2] * 0.5
    bboxes[..., 1] = bboxes[..., 1] - bboxes[..., 3] * 0.5
    bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2]
    bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3]
    return bboxes

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


##########################################################

# 检查与初始化 - 使用更多的参数
def check_and_init(args):    
    
    # check files
    # master_process = args.rank == 0 if args.world_size > 1 else args.rank == -1
    master_process = True     #是否要穿建目录 和 是否要生成新的： args.yaml
   
    if args.resume: #False - 继续接着训练
        
        checkpoint_path = args.resume # if isinstance(args.resume, str) else find_latest_checkpoint() #默认使用传进来的，不传使用最近的
        
        assert os.path.isfile(checkpoint_path), f'找不到继续训练的权重文件: {checkpoint_path}'   

        # 读取之前训练的配置： \runs\train\gold_yolo-s2\args.yaml
        resume_opt_file_path = Path(checkpoint_path).parent.parent / 'args.yaml'
        if osp.exists(resume_opt_file_path):
            with open(resume_opt_file_path) as f:
                args = argparse.Namespace(**yaml.safe_load(f))  # load args value from args.yaml
        else:            
            args.save_dir = str(Path(checkpoint_path).parent.parent) #保存目录，保存在之前的目录

        # args.resume = checkpoint_path  # 这如何不自动获取最后一个权重，就不用设置了

    else: #从新训练 - 默认就是走这里     
        args.save_dir = str(increment_name(osp.join(args.output_dir, args.name))) #保存目录: ./runs/train/*
        if master_process:
            os.makedirs(args.save_dir)


    #读取配置信息
    cfg = Config.fromfile(args.conf_file) 

    if not hasattr(cfg, 'training_mode'):
        setattr(cfg, 'training_mode', 'repvgg')

    # 获取 cpu 或 gpu 驱动
    device = select_device(args.device)

    # set random seed
    set_random_seed(1+args.rank, deterministic=(args.rank == -1))

    # 生成新的： args.yaml
    if master_process:
        save_yaml(vars(args), osp.join(args.save_dir, 'args.yaml'))

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
args.quant=False
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