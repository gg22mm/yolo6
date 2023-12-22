# -*- coding:utf-8 -*-
from ast import Pass
import os
import time
from copy import deepcopy
import os.path as osp
from tqdm import tqdm
import cv2
import numpy as np
import math
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
# 我添加的
import types
import random
import torch.backends.cudnn as cudnn
import glob
from logging import Logger
import logging
import yaml
from pathlib import Path
import torch.distributed as dist
import sys
import tempfile
# 
import os.path as osp
import shutil
import tempfile
from importlib import import_module
from addict import Dict
#yolo6
from wllYolo6 import build_model 
from wllLoss import ComputeLoss as ComputeLoss # loss
from wll_data_load import create_dataloader # 数据集，这个自己定义

# from yolov6.utils.ema import ModelEMA, de_parallel
# from yolov6.solver.build import build_optimizer, build_lr_scheduler  # 优化器
# from yolov6.models.losses.loss import ComputeLoss as ComputeLoss # loss
# from yolov6.utils.nms import xywh2xyxy
# from yolov6.utils.RepOptimizer import extract_scales #没什么用
# from yolov6.utils.checkpoint import load_state_dict, save_checkpoint, strip_optimizer
# from yolov6.utils.events import LOGGER, NCOLS, write_tblog, write_tbimg
# from yolov6.data.data_load import create_dataloader # 数据集，这个自己定义
import tools.eval as eval

def load_yaml(file_path):    
    if isinstance(file_path, str):
        with open(file_path, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
    return data_dict

# 获取数据集合 - 后观会替换掉
def get_data_loader(args, cfg, data_dict):#旧的传进来的参数（可能有修改）,新的cfg配置信息,数据集data coco.yaml参数
    train_path, val_path = data_dict['train'], data_dict['val']
    # check data
    nc = int(data_dict['nc']) #分类数
    class_names = data_dict['names'] #数据label
    assert len(class_names) == nc, f'coco.yaml中设置的分类数与数据label不匹配'
    grid_size = max(int(max(cfg.model.head.strides)), 32) #32
    
    # create train dataloader
    train_loader = create_dataloader(train_path, args.img_size, args.batch_size // args.world_size, grid_size, #args.world_size=1
                                     hyp=dict(cfg.data_aug), augment=True, rect=False, rank=args.local_rank,
                                     workers=args.workers, shuffle=True, check_images=args.check_images,
                                     check_labels=args.check_labels, data_dict=data_dict, task='train')[0]
    # create val dataloader
    val_loader = None
    if args.rank in [-1, 0]:
        val_loader = create_dataloader(val_path, args.img_size, args.batch_size // args.world_size * 2, grid_size,
                                       hyp=dict(cfg.data_aug), rect=True, rank=-1, pad=0.5,
                                       workers=args.workers, check_images=args.check_images,
                                       check_labels=args.check_labels, data_dict=data_dict, task='val')[0]
    
    return train_loader, val_loader

############################## from yolov6.utils.nms import xywh2xyxy ##############################

def xywh2xyxy(x):
    '''Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right.'''
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

############################## from yolov6.utils.checkpoint import load_state_dict, save_checkpoint, strip_optimizer ##############################

def load_state_dict(weights, model, map_location=None):    
    ckpt = torch.load(weights, map_location=map_location)
    state_dict = ckpt['model'].float().state_dict()
    model_state_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
    model.load_state_dict(state_dict, strict=False)
    del ckpt, state_dict, model_state_dict
    return model

def save_checkpoint(ckpt, is_best, save_dir, model_name=""):    
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    filename = osp.join(save_dir, model_name + '.pt')
    torch.save(ckpt, filename)
    if is_best:
        best_filename = osp.join(save_dir, 'best_ckpt.pt')
        shutil.copyfile(filename, best_filename)

def strip_optimizer(ckpt_dir, epoch):    
    for s in ['best', 'last']:
        ckpt_path = osp.join(ckpt_dir, '{}_ckpt.pt'.format(s))
        if not osp.exists(ckpt_path):
            continue
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        if ckpt.get('ema'):
            ckpt['model'] = ckpt['ema']  # replace model with ema
        for k in ['optimizer', 'ema', 'updates']:  # keys
            ckpt[k] = None
        ckpt['epoch'] = epoch
        ckpt['model'].half()  # to FP16
        for p in ckpt['model'].parameters():
            p.requires_grad = False
        torch.save(ckpt, ckpt_path)

############################## from yolov6.utils.events import LOGGER, NCOLS, write_tblog, write_tbimg ##############################

def set_logging(name=None):
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)

LOGGER = set_logging(__name__)
NCOLS = min(100, shutil.get_terminal_size().columns)

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

############################## from yolov6.utils.ema import ModelEMA, de_parallel yolo特有 ##############################

class ModelEMA:
    def __init__(self, model, decay=0.9999, updates=0):
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            decay = self.decay(self.updates)

            state_dict = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, item in self.ema.state_dict().items():
                if item.dtype.is_floating_point:
                    item *= decay
                    item += (1 - decay) * state_dict[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        copy_attr(self.ema, model, include, exclude)

def copy_attr(a, b, include=(), exclude=()):    
    for k, item in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, item)

def is_parallel(model):    
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):    
    return model.module if is_parallel(model) else model

############################## from yolov6.solver.build import build_optimizer, build_lr_scheduler  ##############################

# 优化器
def build_optimizer(cfg, model):    
    g_bnw, g_w, g_b = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g_b.append(v.bias)
        if isinstance(v, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            g_bnw.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g_w.append(v.weight)
    
    assert cfg.solver.optim == 'SGD' or 'Adam', 'ERROR: unknown optimizer, use SGD defaulted'
    if cfg.solver.optim == 'SGD':
        optimizer = torch.optim.SGD(g_bnw, lr=cfg.solver.lr0, momentum=cfg.solver.momentum, nesterov=True)
    elif cfg.solver.optim == 'Adam':
        optimizer = torch.optim.Adam(g_bnw, lr=cfg.solver.lr0, betas=(cfg.solver.momentum, 0.999))
    
    optimizer.add_param_group({'params': g_w, 'weight_decay': cfg.solver.weight_decay})
    optimizer.add_param_group({'params': g_b})
    
    del g_bnw, g_w, g_b
    return optimizer

# 步长
def build_lr_scheduler(cfg, optimizer, epochs):    
    if cfg.solver.lr_scheduler == 'Cosine':
        lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (cfg.solver.lrf - 1) + 1
    elif cfg.solver.lr_scheduler == 'Constant':
        lf = lambda x: 1.0
    else:
        LOGGER.error('unknown lr scheduler, use Cosine defaulted')
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return scheduler, lf


##########################################################################################


class Trainer:
    def __init__(self, args, cfg, device):
        self.args = args
        self.cfg = cfg
        self.device = device
        
        if args.resume: #如果是继续训练
            self.ckpt = torch.load(args.resume, map_location='cpu')
        
        self.rank = args.rank
        self.local_rank = args.local_rank
        self.world_size = args.world_size
        self.main_process = self.rank in [-1, 0]
        self.save_dir = args.save_dir
        
        # 获取数据集合 - 后观会替换掉
        self.data_dict = load_yaml(args.data_path)
        self.num_classes = self.data_dict['nc']
        self.train_loader, self.val_loader = get_data_loader(args, cfg, self.data_dict) #旧的传进来的参数（可能有修改）,新的cfg配置信息,数据集data/coco.yaml参数
        

        # get model and optimizer
        self.distill_ns = True if self.args.distill and self.cfg.model.type in ['YOLOv6n', 'YOLOv6s', 'GoldYOLO-n', 'GoldYOLO-s'] else False #False        

        # 创建模型
        model = self.get_model(args, cfg, self.num_classes, device)

        # ema 模型 yolo特有
        self.ema = ModelEMA(model) if self.main_process else None

        # print(model)
        # exit()

        # 优化器
        self.optimizer = self.get_optimizer(args, cfg, model)

        # 步长
        self.scheduler, self.lf = self.get_lr_scheduler(args, cfg, self.optimizer)
           
        # tensorboard
        self.tblogger = SummaryWriter(self.save_dir) if self.main_process else None
        self.start_epoch = 0

        # resume - 如果是继续训练 - 一般不执行
        if hasattr(self, "ckpt"): #检查是否有继续训练的属性
            resume_state_dict = self.ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            model.load_state_dict(resume_state_dict, strict=True)  # load
            self.start_epoch = self.ckpt['epoch'] + 1
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
            if self.main_process:
                self.ema.ema.load_state_dict(self.ckpt['ema'].float().state_dict())
                self.ema.updates = self.ckpt['updates']

        self.model = model # self.parallel_model(args, model, device)
        self.model.nc, self.model.names = self.data_dict['nc'], self.data_dict['names']
        
        self.max_epoch = args.epochs
        self.max_stepnum = len(self.train_loader)
        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.vis_imgs_list = []
        self.write_trainbatch_tb = args.write_trainbatch_tb
        
        # 随机设置 label的标签颜色
        self.color = [tuple(np.random.choice(range(256), size=3)) for _ in range(self.model.nc)]
        
        # loss 字段信息
        self.loss_num = 3
        self.loss_info = ['Epoch', 'iou_loss', 'dfl_loss', 'cls_loss'] #DFL 损失在训练神经网络时“考虑”了类别不平衡的问题。当一个类出现过于频繁而另一类出现较少时，就会出现类不平衡 : https://qa.1r1g.com/sf/ask/5316519841/        
        # if self.args.distill:
        #     self.loss_num += 1
        #     self.loss_info += ['cwd_loss']
    
    # 创建模型
    def get_model(self, args, cfg, nc, device):
        model = build_model(cfg, nc, device, fuse_ab=self.args.fuse_ab, distill_ns=self.distill_ns)
        weights = cfg.model.pretrained #None

        # 不执行
        if weights:  # None 如果设置了预训练的模型，则进行微调
            LOGGER.info(f'Loading state_dict from {weights} for fine-tuning...')
            model = load_state_dict(weights, model, map_location=device)
        
        # 不执行
        if args.use_syncbn and self.rank != -1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)        
        
        return model

    # 优化器
    def get_optimizer(self, args, cfg, model):
        accumulate = max(1, round(64 / args.batch_size))
        cfg.solver.weight_decay *= args.batch_size * accumulate / 64
        cfg.solver.lr0 *= args.batch_size / (self.world_size * args.bs_per_gpu)  # rescale lr0 related to batchsize
        optimizer = build_optimizer(cfg, model)
        return optimizer

    # 步长
    @staticmethod
    def get_lr_scheduler(args, cfg, optimizer):
        epochs = args.epochs
        lr_scheduler, lf = build_lr_scheduler(cfg, optimizer, epochs)
        return lr_scheduler, lf   

    # 训练入口
    def train(self):
        try:
            self.train_before_loop()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.train_in_loop(self.epoch)
            self.strip_model()
        
        except Exception as _:
            LOGGER.error('ERROR in training loop or eval/save model.')
            raise
        finally:
            self.train_after_loop()
    
    # 训练之前准备
    def train_before_loop(self):
        LOGGER.info('Training start...')
        self.start_time = time.time()
        self.warmup_stepnum = max(round(self.cfg.solver.warmup_epochs * self.max_stepnum),
                                  1000) if self.args.quant is False else 0
        self.scheduler.last_epoch = self.start_epoch - 1
        self.last_opt_step = -1
        self.scaler = amp.GradScaler(enabled=self.device != 'cpu')
        
        self.best_ap, self.ap = 0.0, 0.0
        self.best_stop_strong_aug_ap = 0.0
        self.evaluate_results = (0, 0)  # AP50, AP50_95
        
        self.compute_loss = ComputeLoss(num_classes=self.data_dict['nc'],
                                        ori_img_size=self.img_size,
                                        warmup_epoch=self.cfg.model.head.atss_warmup_epoch,
                                        use_dfl=self.cfg.model.head.use_dfl,
                                        reg_max=self.cfg.model.head.reg_max,
                                        iou_type=self.cfg.model.head.iou_type,
                                        fpn_strides=self.cfg.model.head.strides)        

    ################ 训练中 ########################

    # 训练中 - 循环
    def train_in_loop(self, epoch_num):
        try:
            self.prepare_for_steps()
            for self.step, self.batch_data in self.pbar:
                try:
                    self.train_in_steps(epoch_num, self.step)
                except Exception as e:
                    LOGGER.error(f'ERROR in training steps: {e}')
                self.print_details()
        except Exception as _:
            LOGGER.error('ERROR in training steps.')
            raise
        try:
            self.eval_and_save()
        except Exception as _:
            LOGGER.error('ERROR in evaluate and save model.')
            raise
    
    def prepare_for_steps(self):
        if self.epoch > self.start_epoch:
            self.scheduler.step()
        # stop strong aug like mosaic and mixup from last n epoch by recreate dataloader
        if self.epoch == self.max_epoch - self.args.stop_aug_last_n_epoch:
            self.cfg.data_aug.mosaic = 0.0
            self.cfg.data_aug.mixup = 0.0
            self.train_loader, self.val_loader = self.get_data_loader(self.args, self.cfg, self.data_dict)
        self.model.train()
        if self.rank != -1:
            self.train_loader.sampler.set_epoch(self.epoch)
        self.mean_loss = torch.zeros(self.loss_num, device=self.device)
        self.optimizer.zero_grad()
        
        LOGGER.info(('\n' + '%10s' * (self.loss_num + 1)) % (*self.loss_info,))
        self.pbar = enumerate(self.train_loader)
        if self.main_process:
            self.pbar = tqdm(self.pbar, total=self.max_stepnum, ncols=NCOLS,
                             bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    #-------1-------

    # Training loop for batchdata
    def train_in_steps(self, epoch_num, step_num):
        images, targets = self.prepro_data(self.batch_data, self.device)
        # plot train_batch and save to tensorboard once an epoch
        if self.write_trainbatch_tb and self.main_process and self.step == 0:
            self.plot_train_batch(images, targets)
            write_tbimg(self.tblogger, self.vis_train_batch, self.step + self.max_stepnum * self.epoch, type='train')
        
        # forward
        with amp.autocast(enabled=self.device != 'cpu'):
            preds, s_featmaps = self.model(images)
            total_loss, loss_items = self.compute_loss(preds, targets, epoch_num, step_num)  # YOLOv6_af

            if self.rank != -1:
                total_loss *= self.world_size

        # backward
        self.scaler.scale(total_loss).backward()
        self.loss_items = loss_items
        self.update_optimizer()

    def plot_train_batch(self, images, targets, max_size=1920, max_subplots=16):
        # Plot train_batch with labels
        if isinstance(images, torch.Tensor):
            images = images.cpu().float().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if np.max(images[0]) <= 1:
            images *= 255  # de-normalise (optional)
        bs, _, h, w = images.shape  # batch size, _, height, width
        bs = min(bs, max_subplots)  # limit plot images
        ns = np.ceil(bs ** 0.5)  # number of subplots (square)
        paths = self.batch_data[2]  # image paths
        # Build Image
        mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
        for i, im in enumerate(images):
            if i == max_subplots:  # if last batch has fewer images than we expect
                break
            x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
            im = im.transpose(1, 2, 0)
            mosaic[y:y + h, x:x + w, :] = im
        # Resize (optional)
        scale = max_size / ns / max(h, w)
        if scale < 1:
            h = math.ceil(scale * h)
            w = math.ceil(scale * w)
            mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))
        for i in range(bs):
            x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
            cv2.rectangle(mosaic, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)  # borders
            cv2.putText(mosaic, f"{os.path.basename(paths[i])[:40]}", (x + 5, y + 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, color=(220, 220, 220), thickness=1)  # filename
            if len(targets) > 0:
                ti = targets[targets[:, 0] == i]  # image targets
                boxes = xywh2xyxy(ti[:, 2:6]).T
                classes = ti[:, 1].astype('int')
                labels = ti.shape[1] == 6  # labels if no conf column
                if boxes.shape[1]:
                    if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                        boxes[[0, 2]] *= w  # scale to pixels
                        boxes[[1, 3]] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        boxes *= scale
                boxes[[0, 2]] += x
                boxes[[1, 3]] += y
                for j, box in enumerate(boxes.T.tolist()):
                    box = [int(k) for k in box]
                    cls = classes[j]
                    color = tuple([int(x) for x in self.color[cls]])
                    cls = self.data_dict['names'][cls] if self.data_dict['names'] else cls
                    if labels:
                        label = f'{cls}'
                        cv2.rectangle(mosaic, (box[0], box[1]), (box[2], box[3]), color, thickness=1)
                        cv2.putText(mosaic, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, color,
                                    thickness=1)
        self.vis_train_batch = mosaic.copy()

    @staticmethod
    def prepro_data(batch_data, device):
        images = batch_data[0].to(device, non_blocking=True).float() / 255
        targets = batch_data[1].to(device)
        return images, targets

    def update_optimizer(self):
        curr_step = self.step + self.max_stepnum * self.epoch
        self.accumulate = max(1, round(64 / self.batch_size))
        if curr_step <= self.warmup_stepnum:
            self.accumulate = max(1, np.interp(curr_step, [0, self.warmup_stepnum], [1, 64 / self.batch_size]).round())
            for k, param in enumerate(self.optimizer.param_groups):
                warmup_bias_lr = self.cfg.solver.warmup_bias_lr if k == 2 else 0.0
                param['lr'] = np.interp(curr_step, [0, self.warmup_stepnum],
                                        [warmup_bias_lr, param['initial_lr'] * self.lf(self.epoch)])
                if 'momentum' in param:
                    param['momentum'] = np.interp(curr_step, [0, self.warmup_stepnum],
                                                  [self.cfg.solver.warmup_momentum, self.cfg.solver.momentum])
        if curr_step - self.last_opt_step >= self.accumulate:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)
            self.last_opt_step = curr_step
    #--------1------


    # Print loss after each steps
    def print_details(self):
        if self.main_process:
            self.mean_loss = (self.mean_loss * self.step + self.loss_items) / (self.step + 1)
            self.pbar.set_description(('%10s' + '%10.4g' * self.loss_num) % (f'{self.epoch}/{self.max_epoch - 1}', \
                                                                           *(self.mean_loss)))   


    #---------2-----
    def eval_and_save(self):
        remaining_epochs = self.max_epoch - self.epoch
        eval_interval = self.args.eval_interval if remaining_epochs > self.args.heavy_eval_range else 3
        is_val_epoch = (not self.args.eval_final_only or (remaining_epochs == 1)) and (self.epoch % eval_interval == 0)
        if self.main_process:
            self.ema.update_attr(self.model, include=['nc', 'names', 'stride'])  # update attributes for ema model
            if is_val_epoch:
                self.eval_model()
                self.ap = self.evaluate_results[1]
                self.best_ap = max(self.ap, self.best_ap)
            # save ckpt
            ckpt = {
                    'model': deepcopy(de_parallel(self.model)).half(),
                    'ema': deepcopy(self.ema.ema).half(),
                    'updates': self.ema.updates,
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': self.epoch,
            }
            
            save_ckpt_dir = osp.join(self.save_dir, 'weights')
            save_checkpoint(ckpt, (is_val_epoch) and (self.ap == self.best_ap), save_ckpt_dir, model_name='last_ckpt')
            if self.epoch >= self.max_epoch - self.args.save_ckpt_on_last_n_epoch:
                save_checkpoint(ckpt, False, save_ckpt_dir, model_name=f'{self.epoch}_ckpt')
            
            # default save best ap ckpt in stop strong aug epochs
            if self.epoch >= self.max_epoch - self.args.stop_aug_last_n_epoch:
                if self.best_stop_strong_aug_ap < self.ap:
                    self.best_stop_strong_aug_ap = max(self.ap, self.best_stop_strong_aug_ap)
                    save_checkpoint(ckpt, False, save_ckpt_dir, model_name='best_stop_aug_ckpt')
            
            del ckpt
            # log for learning rate
            lr = [x['lr'] for x in self.optimizer.param_groups]
            self.evaluate_results = list(self.evaluate_results) + lr
            
            # log for tensorboard
            write_tblog(self.tblogger, self.epoch, self.evaluate_results, self.mean_loss)
            # save validation predictions to tensorboard
            write_tbimg(self.tblogger, self.vis_imgs_list, self.epoch, type='val')

    def eval_model(self):
        if not hasattr(self.cfg, "eval_params"):
            results, vis_outputs, vis_paths = eval.run(self.data_dict,
                                                       batch_size=self.batch_size // self.world_size * 2,
                                                       img_size=self.img_size,
                                                       model=self.ema.ema if self.args.calib is False else self.model,
                                                       conf_thres=0.03,
                                                       dataloader=self.val_loader,
                                                       save_dir=self.save_dir,
                                                       task='train')
        else:
            def get_cfg_value(cfg_dict, value_str, default_value):
                if value_str in cfg_dict:
                    if isinstance(cfg_dict[value_str], list):
                        return cfg_dict[value_str][0] if cfg_dict[value_str][0] is not None else default_value
                    else:
                        return cfg_dict[value_str] if cfg_dict[value_str] is not None else default_value
                else:
                    return default_value
            
            eval_img_size = get_cfg_value(self.cfg.eval_params, "img_size", self.img_size)
            results, vis_outputs, vis_paths = eval.run(self.data_dict,
                                                       batch_size=get_cfg_value(self.cfg.eval_params, "batch_size",
                                                                                self.batch_size // self.world_size * 2),
                                                       img_size=eval_img_size,
                                                       model=self.ema.ema if self.args.calib is False else self.model,
                                                       conf_thres=get_cfg_value(self.cfg.eval_params, "conf_thres",
                                                                                0.03),
                                                       dataloader=self.val_loader,
                                                       save_dir=self.save_dir,
                                                       task='train',
                                                       test_load_size=get_cfg_value(self.cfg.eval_params,
                                                                                    "test_load_size", eval_img_size),
                                                       letterbox_return_int=get_cfg_value(self.cfg.eval_params,
                                                                                          "letterbox_return_int",
                                                                                          False),
                                                       force_no_pad=get_cfg_value(self.cfg.eval_params, "force_no_pad",
                                                                                  False),
                                                       not_infer_on_rect=get_cfg_value(self.cfg.eval_params,
                                                                                       "not_infer_on_rect", False),
                                                       scale_exact=get_cfg_value(self.cfg.eval_params, "scale_exact",
                                                                                 False),
                                                       verbose=get_cfg_value(self.cfg.eval_params, "verbose", False),
                                                       do_coco_metric=get_cfg_value(self.cfg.eval_params,
                                                                                    "do_coco_metric", True),
                                                       do_pr_metric=get_cfg_value(self.cfg.eval_params, "do_pr_metric",
                                                                                  False),
                                                       plot_curve=get_cfg_value(self.cfg.eval_params, "plot_curve",
                                                                                False),
                                                       plot_confusion_matrix=get_cfg_value(self.cfg.eval_params,
                                                                                           "plot_confusion_matrix",
                                                                                           False),
                                                       )
        
        LOGGER.info(f"Epoch: {self.epoch} | mAP@0.5: {results[0]} | mAP@0.50:0.95: {results[1]}")
        self.evaluate_results = results[:2]
        # plot validation predictions
        self.plot_val_pred(vis_outputs, vis_paths)

    def plot_val_pred(self, vis_outputs, vis_paths, vis_conf=0.3, vis_max_box_num=5):
        # plot validation predictions
        self.vis_imgs_list = []
        for (vis_output, vis_path) in zip(vis_outputs, vis_paths):
            vis_output_array = vis_output.cpu().numpy()  # xyxy
            ori_img = cv2.imread(vis_path)
            for bbox_idx, vis_bbox in enumerate(vis_output_array):
                x_tl = int(vis_bbox[0])
                y_tl = int(vis_bbox[1])
                x_br = int(vis_bbox[2])
                y_br = int(vis_bbox[3])
                box_score = vis_bbox[4]
                cls_id = int(vis_bbox[5])
                # draw top n bbox
                if box_score < vis_conf or bbox_idx > vis_max_box_num:
                    break
                cv2.rectangle(ori_img, (x_tl, y_tl), (x_br, y_br), tuple([int(x) for x in self.color[cls_id]]),
                              thickness=1)
                cv2.putText(ori_img, f"{self.data_dict['names'][cls_id]}: {box_score:.2f}", (x_tl, y_tl - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple([int(x) for x in self.color[cls_id]]), thickness=1)
            self.vis_imgs_list.append(torch.from_numpy(ori_img[:, :, ::-1].copy()))

    #----------2----

    ################### 训练中 #####################

    def strip_model(self):
        if self.main_process:
            LOGGER.info(f'\nTraining completed in {(time.time() - self.start_time) / 3600:.3f} hours.')
            save_ckpt_dir = osp.join(self.save_dir, 'weights')
            strip_optimizer(save_ckpt_dir, self.epoch)  # strip optimizers for saved pt model
    
    # 训练之结束之后 - Empty cache if training finished
    def train_after_loop(self):
        if self.device != 'cpu':
            torch.cuda.empty_cache()

    
    
   