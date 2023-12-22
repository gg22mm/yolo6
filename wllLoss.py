# 2023.09.18-Changed for loss implementation of Gold-YOLO
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import glob
from pathlib import Path
import math

# from yolov6.assigners.anchor_generator import generate_anchors
# from yolov6.utils.general import dist2bbox, bbox2dist, xywh2xyxy, box_iou
# from yolov6.utils.figure_iou import IOUloss
# from yolov6.assigners.atss_assigner import ATSSAssigner
# from yolov6.assigners.tal_assigner import TaskAlignedAssigner

################################# from yolov6.assigners.atss_assigner import ATSSAssigner #################################

def cast_tensor_type(x, scale=1., dtype=None):
    if dtype == 'fp16':        
        x = (x / scale).half()
    return x

def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:        
        return x.float().clamp(min, max).half()
    return x.clamp(min, max)

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

def iou2d_calculator(bboxes1, bboxes2, mode='iou', is_aligned=False, scale=1., dtype=None):    
    assert bboxes1.size(-1) in [0, 4, 5]
    assert bboxes2.size(-1) in [0, 4, 5]
    if bboxes2.size(-1) == 5:
        bboxes2 = bboxes2[..., :4]
    if bboxes1.size(-1) == 5:
        bboxes1 = bboxes1[..., :4]

    if dtype == 'fp16':        
        bboxes1 = cast_tensor_type(bboxes1, scale, dtype)
        bboxes2 = cast_tensor_type(bboxes2, scale, dtype)
        overlaps = bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
        if not overlaps.is_cuda and overlaps.dtype == torch.float16:
        # resume cpu float32
            overlaps = overlaps.float()
        return overlaps

    return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

######### from yolov6.assigners.assigner_utils import dist_calculator 

def dist_calculator(gt_bboxes, anchor_bboxes):   
    gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
    gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
    gt_points = torch.stack([gt_cx, gt_cy], dim=1)
    ac_cx = (anchor_bboxes[:, 0] + anchor_bboxes[:, 2]) / 2.0
    ac_cy = (anchor_bboxes[:, 1] + anchor_bboxes[:, 3]) / 2.0
    ac_points = torch.stack([ac_cx, ac_cy], dim=1)

    distances = (gt_points[:, None, :] - ac_points[None, :, :]).pow(2).sum(-1).sqrt()

    return distances, ac_points

def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):   
    n_anchors = xy_centers.size(0)
    bs, n_max_boxes, _ = gt_bboxes.size()
    _gt_bboxes = gt_bboxes.reshape([-1, 4])
    xy_centers = xy_centers.unsqueeze(0).repeat(bs * n_max_boxes, 1, 1)
    gt_bboxes_lt = _gt_bboxes[:, 0:2].unsqueeze(1).repeat(1, n_anchors, 1)
    gt_bboxes_rb = _gt_bboxes[:, 2:4].unsqueeze(1).repeat(1, n_anchors, 1)
    b_lt = xy_centers - gt_bboxes_lt
    b_rb = gt_bboxes_rb - xy_centers
    bbox_deltas = torch.cat([b_lt, b_rb], dim=-1)
    bbox_deltas = bbox_deltas.reshape([bs, n_max_boxes, n_anchors, -1])
    return (bbox_deltas.min(axis=-1)[0] > eps).to(gt_bboxes.dtype)

def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):    
    fg_mask = mask_pos.sum(axis=-2)
    if fg_mask.max() > 1:
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])
        max_overlaps_idx = overlaps.argmax(axis=1)
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
        fg_mask = mask_pos.sum(axis=-2)
    target_gt_idx = mask_pos.argmax(axis=-2)
    return target_gt_idx, fg_mask , mask_pos

def iou_calculator(box1, box2, eps=1e-9):    
    box1 = box1.unsqueeze(2)  # [N, M1, 4] -> [N, M1, 1, 4]
    box2 = box2.unsqueeze(1)  # [N, M2, 4] -> [N, 1, M2, 4]
    px1y1, px2y2 = box1[:, :, :, 0:2], box1[:, :, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, :, 0:2], box2[:, :, :, 2:4]
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps

    return overlap / union

class ATSSAssigner(nn.Module):    
    def __init__(self,topk=9,num_classes=80):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes

    @torch.no_grad()
    def forward(self,anc_bboxes,n_level_bboxes,gt_labels,gt_bboxes,mask_gt,pd_bboxes):
        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py

        Args:
            anc_bboxes (Tensor): shape(num_total_anchors, 4)
            n_level_bboxes (List):len(3)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
            pd_bboxes (Tensor): shape(bs, n_max_boxes, 4)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        self.n_anchors = anc_bboxes.size(0)
        self.bs = gt_bboxes.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return torch.full( [self.bs, self.n_anchors], self.bg_idx).to(device), \
                   torch.zeros([self.bs, self.n_anchors, 4]).to(device), \
                   torch.zeros([self.bs, self.n_anchors, self.num_classes]).to(device), \
                   torch.zeros([self.bs, self.n_anchors]).to(device)


        overlaps = iou2d_calculator(gt_bboxes.reshape([-1, 4]), anc_bboxes)
        overlaps = overlaps.reshape([self.bs, -1, self.n_anchors])

        distances, ac_points = dist_calculator(gt_bboxes.reshape([-1, 4]), anc_bboxes)
        distances = distances.reshape([self.bs, -1, self.n_anchors])

        is_in_candidate, candidate_idxs = self.select_topk_candidates(
            distances, n_level_bboxes, mask_gt)

        overlaps_thr_per_gt, iou_candidates = self.thres_calculator(
            is_in_candidate, candidate_idxs, overlaps)

        # select candidates iou >= threshold as positive
        is_pos = torch.where(
            iou_candidates > overlaps_thr_per_gt.repeat([1, 1, self.n_anchors]),
            is_in_candidate, torch.zeros_like(is_in_candidate))

        is_in_gts = select_candidates_in_gts(ac_points, gt_bboxes)
        mask_pos = is_pos * is_in_gts * mask_gt

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes)

        # assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # soft label with iou
        if pd_bboxes is not None:
            ious = iou_calculator(gt_bboxes, pd_bboxes) * mask_pos
            ious = ious.max(axis=-2)[0].unsqueeze(-1)
            target_scores *= ious

        return target_labels.long(), target_bboxes, target_scores, fg_mask.bool()

    def select_topk_candidates(self,distances,n_level_bboxes,mask_gt):
        mask_gt = mask_gt.repeat(1, 1, self.topk).bool()
        level_distances = torch.split(distances, n_level_bboxes, dim=-1)
        is_in_candidate_list = []
        candidate_idxs = []
        start_idx = 0
        for per_level_distances, per_level_boxes in zip(level_distances, n_level_bboxes):

            end_idx = start_idx + per_level_boxes
            selected_k = min(self.topk, per_level_boxes)
            _, per_level_topk_idxs = per_level_distances.topk(selected_k, dim=-1, largest=False)
            candidate_idxs.append(per_level_topk_idxs + start_idx)
            per_level_topk_idxs = torch.where(mask_gt,
                per_level_topk_idxs, torch.zeros_like(per_level_topk_idxs))
            is_in_candidate = F.one_hot(per_level_topk_idxs, per_level_boxes).sum(dim=-2)
            is_in_candidate = torch.where(is_in_candidate > 1,
                torch.zeros_like(is_in_candidate), is_in_candidate)
            is_in_candidate_list.append(is_in_candidate.to(distances.dtype))
            start_idx = end_idx

        is_in_candidate_list = torch.cat(is_in_candidate_list, dim=-1)
        candidate_idxs = torch.cat(candidate_idxs, dim=-1)

        return is_in_candidate_list, candidate_idxs

    def thres_calculator(self,is_in_candidate,candidate_idxs,overlaps):
        n_bs_max_boxes = self.bs * self.n_max_boxes
        _candidate_overlaps = torch.where(is_in_candidate > 0,
            overlaps, torch.zeros_like(overlaps))
        candidate_idxs = candidate_idxs.reshape([n_bs_max_boxes, -1])
        assist_idxs = self.n_anchors * torch.arange(n_bs_max_boxes, device=candidate_idxs.device)
        assist_idxs = assist_idxs[:,None]
        faltten_idxs = candidate_idxs + assist_idxs
        candidate_overlaps = _candidate_overlaps.reshape(-1)[faltten_idxs]
        candidate_overlaps = candidate_overlaps.reshape([self.bs, self.n_max_boxes, -1])

        overlaps_mean_per_gt = candidate_overlaps.mean(axis=-1, keepdim=True)
        overlaps_std_per_gt = candidate_overlaps.std(axis=-1, keepdim=True)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        return overlaps_thr_per_gt, _candidate_overlaps

    def get_targets(self,gt_labels,gt_bboxes,target_gt_idx,fg_mask):
        # assigned target labels
        batch_idx = torch.arange(self.bs, dtype=gt_labels.dtype, device=gt_labels.device)
        batch_idx = batch_idx[...,None]
        target_gt_idx = (target_gt_idx + batch_idx * self.n_max_boxes).long()
        target_labels = gt_labels.flatten()[target_gt_idx.flatten()]
        target_labels = target_labels.reshape([self.bs, self.n_anchors])
        target_labels = torch.where(fg_mask > 0,
            target_labels, torch.full_like(target_labels, self.bg_idx))

        # assigned target boxes
        target_bboxes = gt_bboxes.reshape([-1, 4])[target_gt_idx.flatten()]
        target_bboxes = target_bboxes.reshape([self.bs, self.n_anchors, 4])

        # assigned target scores
        target_scores = F.one_hot(target_labels.long(), self.num_classes + 1).float()
        target_scores = target_scores[:, :, :self.num_classes]

        return target_labels, target_bboxes, target_scores

################################# from yolov6.assigners.tal_assigner import TaskAlignedAssigner #################################

class TaskAlignedAssigner(nn.Module):
    def __init__(self,topk=13,num_classes=80,alpha=1.0,beta=6.0,eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
    
    @torch.no_grad()
    def forward(self,pd_scores,pd_bboxes,anc_points,gt_labels,gt_bboxes,mask_gt):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)
        
        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), \
                torch.zeros_like(pd_bboxes).to(device), \
                torch.zeros_like(pd_scores).to(device), \
                torch.zeros_like(pd_scores[..., 0]).to(device)
        
        cycle, step, self.bs = (1, self.bs, self.bs) if self.n_max_boxes <= 100 else (self.bs, 1, 1)
        target_labels_lst, target_bboxes_lst, target_scores_lst, fg_mask_lst = [], [], [], []
        # loop batch dim in case of numerous object box
        for i in range(cycle):
            start, end = i * step, (i + 1) * step
            pd_scores_ = pd_scores[start:end, ...]
            pd_bboxes_ = pd_bboxes[start:end, ...]
            gt_labels_ = gt_labels[start:end, ...]
            gt_bboxes_ = gt_bboxes[start:end, ...]
            mask_gt_ = mask_gt[start:end, ...]
            
            mask_pos, align_metric, overlaps = self.get_pos_mask(
                    pd_scores_, pd_bboxes_, gt_labels_, gt_bboxes_, anc_points, mask_gt_)
            
            target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(
                    mask_pos, overlaps, self.n_max_boxes)
            
            # assigned target
            target_labels, target_bboxes, target_scores = self.get_targets(
                    gt_labels_, gt_bboxes_, target_gt_idx, fg_mask)
            
            # normalize
            align_metric *= mask_pos
            pos_align_metrics = align_metric.max(axis=-1, keepdim=True)[0]
            pos_overlaps = (overlaps * mask_pos).max(axis=-1, keepdim=True)[0]
            norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).max(-2)[0].unsqueeze(-1)
            target_scores = target_scores * norm_align_metric
            
            # append
            target_labels_lst.append(target_labels)
            target_bboxes_lst.append(target_bboxes)
            target_scores_lst.append(target_scores)
            fg_mask_lst.append(fg_mask)
        
        # concat
        target_labels = torch.cat(target_labels_lst, 0)
        target_bboxes = torch.cat(target_bboxes_lst, 0)
        target_scores = torch.cat(target_scores_lst, 0)
        fg_mask = torch.cat(fg_mask_lst, 0)
        
        return target_labels, target_bboxes, target_scores, fg_mask.bool()
    
    def get_pos_mask(self,pd_scores,pd_bboxes,gt_labels,gt_bboxes,anc_points,mask_gt):        
        # get anchor_align metric
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        # get in_gts mask
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # get topk_metric mask
        mask_topk = self.select_topk_candidates(align_metric * mask_in_gts, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        # merge all mask to a final mask
        mask_pos = mask_topk * mask_in_gts * mask_gt
        
        return mask_pos, align_metric, overlaps
    
    def get_box_metrics(self,pd_scores,pd_bboxes,gt_labels,gt_bboxes):        
        pd_scores = pd_scores.permute(0, 2, 1)
        gt_labels = gt_labels.to(torch.long)
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores = pd_scores[ind[0], ind[1]]
        
        overlaps = iou_calculator(gt_bboxes, pd_bboxes)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        
        return align_metric, overlaps
    
    def select_topk_candidates(self,metrics,largest=True,topk_mask=None):        
        num_anchors = metrics.shape[-1]
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, axis=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(axis=-1, keepdim=True) > self.eps).tile( [1, 1, self.topk] )
        topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
        is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)
        is_in_topk = torch.where(is_in_topk > 1, torch.zeros_like(is_in_topk), is_in_topk)
        return is_in_topk.to(metrics.dtype)
    
    def get_targets(self,gt_labels,gt_bboxes,target_gt_idx,fg_mask):        
        # assigned target labels
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]
        
        # assigned target boxes
        target_bboxes = gt_bboxes.reshape([-1, 4])[target_gt_idx]
        
        # assigned target scores
        target_labels[target_labels < 0] = 0
        target_scores = F.one_hot(target_labels, self.num_classes)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, torch.full_like(target_scores, 0))
        
        return target_labels, target_bboxes, target_scores

################################# from yolov6.utils.figure_iou import IOUloss #################################

class IOUloss:    
    def __init__(self, box_format='xywh', iou_type='ciou', reduction='none', eps=1e-7):        
        self.box_format = box_format
        self.iou_type = iou_type.lower()
        self.reduction = reduction
        self.eps = eps

    def __call__(self, box1, box2):        
        if box1.shape[0] != box2.shape[0]:
            box2 = box2.T
            if self.box_format == 'xyxy':
                b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
                b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
            elif self.box_format == 'xywh':
                b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
                b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
                b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
                b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        else:
            if self.box_format == 'xyxy':
                b1_x1, b1_y1, b1_x2, b1_y2 = torch.split(box1, 1, dim=-1)
                b2_x1, b2_y1, b2_x2, b2_y2 = torch.split(box2, 1, dim=-1)

            elif self.box_format == 'xywh':
                b1_x1, b1_y1, b1_w, b1_h = torch.split(box1, 1, dim=-1)
                b2_x1, b2_y1, b2_w, b2_h = torch.split(box2, 1, dim=-1)
                b1_x1, b1_x2 = b1_x1 - b1_w / 2, b1_x1 + b1_w / 2
                b1_y1, b1_y2 = b1_y1 - b1_h / 2, b1_y1 + b1_h / 2
                b2_x1, b2_x2 = b2_x1 - b2_w / 2, b2_x1 + b2_w / 2
                b2_y1, b2_y2 = b2_y1 - b2_h / 2, b2_y1 + b2_h / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps
        union = w1 * h1 + w2 * h2 - inter + self.eps
        iou = inter / union

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if self.iou_type == 'giou':
            c_area = cw * ch + self.eps  # convex area
            iou = iou - (c_area - union) / c_area
        elif self.iou_type in ['diou', 'ciou']:
            c2 = cw ** 2 + ch ** 2 + self.eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if self.iou_type == 'diou':
                iou = iou - rho2 / c2
            elif self.iou_type == 'ciou':
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + self.eps))
                iou = iou - (rho2 / c2 + v * alpha)
        elif self.iou_type == 'siou':
            # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + self.eps
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + self.eps
            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            rho_x = (s_cw / cw) ** 2
            rho_y = (s_ch / ch) ** 2
            gamma = angle_cost - 2
            distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
            iou = iou - 0.5 * (distance_cost + shape_cost)
        loss = 1.0 - iou

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss

################################# from yolov6.assigners.anchor_generator import generate_anchors #################################

def generate_anchors(feats, fpn_strides, grid_cell_size=5.0, grid_cell_offset=0.5,  device='cpu', is_eval=False, mode='af'):    
    anchors = []
    anchor_points = []
    stride_tensor = []
    num_anchors_list = []
    assert feats is not None
    if is_eval:#不运行这里        
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            shift_x = torch.arange(end=w, device=device) + grid_cell_offset
            shift_y = torch.arange(end=h, device=device) + grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack(
                    [shift_x, shift_y], axis=-1).to(torch.float)
            if mode == 'af': # anchor-free
                anchor_points.append(anchor_point.reshape([-1, 2]))
                stride_tensor.append(
                torch.full(
                    (h * w, 1), stride, dtype=torch.float, device=device))
            elif mode == 'ab': # anchor-based
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3,1))
                stride_tensor.append(
                    torch.full(
                        (h * w, 1), stride, dtype=torch.float, device=device).repeat(3,1))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor
    else: # 只运行这里
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            cell_half_size = grid_cell_size * stride * 0.5
            shift_x = (torch.arange(end=w, device=device) + grid_cell_offset) * stride
            shift_y = (torch.arange(end=h, device=device) + grid_cell_offset) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor = torch.stack(
                [
                    shift_x - cell_half_size, shift_y - cell_half_size,
                    shift_x + cell_half_size, shift_y + cell_half_size
                ],
                axis=-1).clone().to(feats[0].dtype)
            anchor_point = torch.stack(
                [shift_x, shift_y], axis=-1).clone().to(feats[0].dtype)

            if mode == 'af': # anchor-free
                anchors.append(anchor.reshape([-1, 4]))
                anchor_points.append(anchor_point.reshape([-1, 2]))
            elif mode == 'ab': # anchor-based
                anchors.append(anchor.reshape([-1, 4]).repeat(3,1))
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3,1))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor.append(
                torch.full(
                    [num_anchors_list[-1], 1], stride, dtype=feats[0].dtype)) 
        anchors = torch.cat(anchors)
        anchor_points = torch.cat(anchor_points).to(device)
        stride_tensor = torch.cat(stride_tensor).to(device)
        return anchors, anchor_points, num_anchors_list, stride_tensor

################################# from yolov6.utils.general import dist2bbox, bbox2dist, xywh2xyxy, box_iou #################################

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
    '''Transform distance(ltrb) to box(xywh or xyxy).'''
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
    '''Transform bbox(xyxy) to dist(ltrb).'''
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    dist = torch.cat([lt, rb], -1).clip(0, reg_max - 0.01)
    return dist

def xywh2xyxy(bboxes):
    '''Transform bbox(xywh) to box(xyxy).'''
    bboxes[..., 0] = bboxes[..., 0] - bboxes[..., 2] * 0.5
    bboxes[..., 1] = bboxes[..., 1] - bboxes[..., 3] * 0.5
    bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2]
    bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3]
    return bboxes

def box_iou(box1, box2):
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

########################################################################################

class ComputeLoss:    
    def __init__(self,
                 fpn_strides=[8, 16, 32],
                 grid_cell_size=5.0,
                 grid_cell_offset=0.5,
                 num_classes=80,
                 ori_img_size=640,
                 warmup_epoch=4,
                 use_dfl=True,
                 reg_max=16,
                 iou_type='giou',
                 loss_weight={
                         'class': 1.0,
                         'iou': 2.5,
                         'dfl': 0.5}
                 ):
        
        self.fpn_strides = fpn_strides
        self.grid_cell_size = grid_cell_size
        self.grid_cell_offset = grid_cell_offset
        self.num_classes = num_classes
        self.ori_img_size = ori_img_size
        
        self.warmup_epoch = warmup_epoch
        self.warmup_assigner = ATSSAssigner(9, num_classes=self.num_classes) #ATSSAssigner 标签分配算法步骤: https://www.yii666.com/blog/651623.html
        self.formal_assigner = TaskAlignedAssigner(topk=13, num_classes=self.num_classes, alpha=1.0, beta=6.0)
        
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.iou_type = iou_type
        self.varifocal_loss = VarifocalLoss().cuda()
        self.bbox_loss = BboxLoss(self.num_classes, self.reg_max, self.use_dfl, self.iou_type).cuda()
        self.loss_weight = loss_weight
    
    def __call__(self,outputs,targets,epoch_num,step_num):
        
        feats, pred_scores, pred_distri = outputs
        anchors, anchor_points, n_anchors_list, stride_tensor = \
            generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset,
                             device=feats[0].device)
        
        assert pred_scores.type() == pred_distri.type()
        gt_bboxes_scale = torch.full((1, 4), self.ori_img_size).type_as(pred_scores)
        batch_size = pred_scores.shape[0]
        
        # targets
        targets = self.preprocess(targets, batch_size, gt_bboxes_scale)
        gt_labels = targets[:, :, :1]
        gt_bboxes = targets[:, :, 1:]  # xyxy
        mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()
        
        # pboxes
        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri)  # xyxy
        
        try:
            if epoch_num < self.warmup_epoch:
                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.warmup_assigner(
                            anchors,
                            n_anchors_list,
                            gt_labels,
                            gt_bboxes,
                            mask_gt,
                            pred_bboxes.detach() * stride_tensor)
            else:
                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.formal_assigner(
                            pred_scores.detach(),
                            pred_bboxes.detach() * stride_tensor,
                            anchor_points,
                            gt_labels,
                            gt_bboxes,
                            mask_gt)
        
        except RuntimeError:
            print(
                    "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                        CPU mode is applied in this batch. If you want to avoid this issue, \
                        try to reduce the batch size or image size."
            )
            torch.cuda.empty_cache()
            print("------------CPU Mode for This Batch-------------")
            if epoch_num < self.warmup_epoch:
                _anchors = anchors.cpu().float()
                _n_anchors_list = n_anchors_list
                _gt_labels = gt_labels.cpu().float()
                _gt_bboxes = gt_bboxes.cpu().float()
                _mask_gt = mask_gt.cpu().float()
                _pred_bboxes = pred_bboxes.detach().cpu().float()
                _stride_tensor = stride_tensor.cpu().float()
                
                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.warmup_assigner(
                            _anchors,
                            _n_anchors_list,
                            _gt_labels,
                            _gt_bboxes,
                            _mask_gt,
                            _pred_bboxes * _stride_tensor)
            
            else:
                _pred_scores = pred_scores.detach().cpu().float()
                _pred_bboxes = pred_bboxes.detach().cpu().float()
                _anchor_points = anchor_points.cpu().float()
                _gt_labels = gt_labels.cpu().float()
                _gt_bboxes = gt_bboxes.cpu().float()
                _mask_gt = mask_gt.cpu().float()
                _stride_tensor = stride_tensor.cpu().float()
                
                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.formal_assigner(
                            _pred_scores,
                            _pred_bboxes * _stride_tensor,
                            _anchor_points,
                            _gt_labels,
                            _gt_bboxes,
                            _mask_gt)
            
            target_labels = target_labels.cuda()
            target_bboxes = target_bboxes.cuda()
            target_scores = target_scores.cuda()
            fg_mask = fg_mask.cuda()
        # Dynamic release GPU memory
        if step_num % 10 == 0:
            torch.cuda.empty_cache()
        
        # rescale bbox
        target_bboxes /= stride_tensor
        
        # cls loss
        target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.num_classes))
        one_hot_label = F.one_hot(target_labels.long(), self.num_classes + 1)[..., :-1]
        loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)
        
        # avoid devide zero error, devide by zero will cause loss to be inf or nan.
        # if target_scores_sum is 0, loss_cls equals to 0 alson
        try:
            target_scores_sum = target_scores.sum()
            if target_scores_sum > 0:
                loss_cls /= target_scores_sum
        except BaseException as e:
            print(f'Loss ERROR: {e}')
            if not torch.any(torch.isnan(target_scores)):
                target_scores_sum = target_scores.sum()
                if not torch.isnan(target_scores_sum) and target_scores_sum > 0:
                    loss_cls /= target_scores_sum
            else:
                target_scores_sum = 0
        
        # bbox loss
        loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s, target_bboxes,
                                            target_scores, target_scores_sum, fg_mask)
        
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl
        
        loss_items = torch.cat(((self.loss_weight['iou'] * loss_iou).unsqueeze(0),
                                (self.loss_weight['dfl'] * loss_dfl).unsqueeze(0),
                                (self.loss_weight['class'] * loss_cls).unsqueeze(0))).detach()
        
        return loss, loss_items
    
    def preprocess(self, targets, batch_size, scale_tensor):
        targets_list = np.zeros((batch_size, 1, 5)).tolist()
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])
        max_len = max((len(l) for l in targets_list))
        targets = torch.from_numpy(
                np.array(list(map(lambda l: l + [[-1, 0, 0, 0, 0]] * (max_len - len(l)), targets_list)))[:, 1:, :]).to(
                targets.device)
        batch_target = targets[:, :, 1:5].mul_(scale_tensor)
        targets[..., 1:] = xywh2xyxy(batch_target)
        return targets
    
    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            batch_size, n_anchors, _ = pred_dist.shape
            pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1).matmul(
                    self.proj.to(pred_dist.device))
        return dist2bbox(pred_dist, anchor_points)

# 分类loss
class VarifocalLoss(nn.Module):
    def __init__(self):
        super(VarifocalLoss, self).__init__()
    
    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy(pred_score.float(), gt_score.float(), reduction='none') * weight).sum()
        
        return loss

# 四条边 loss
class BboxLoss(nn.Module):
    def __init__(self, num_classes, reg_max, use_dfl=False, iou_type='giou'):
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss = IOUloss(box_format='xyxy', iou_type=iou_type, eps=1e-10)
        self.reg_max = reg_max
        self.use_dfl = use_dfl
    
    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask):
        
        # select positive samples mask
        num_pos = fg_mask.sum()
        if num_pos > 0:
            # iou loss
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes,
                                                  bbox_mask).reshape([-1, 4])
            target_bboxes_pos = torch.masked_select(
                    target_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                    target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     target_bboxes_pos) * bbox_weight
            if target_scores_sum == 0:
                loss_iou = loss_iou.sum()
            else:
                loss_iou = loss_iou.sum() / target_scores_sum
            
            # dfl loss
            if self.use_dfl:
                dist_mask = fg_mask.unsqueeze(-1).repeat(
                        [1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = torch.masked_select(
                        pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                target_ltrb_pos = torch.masked_select(
                        target_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos,
                                         target_ltrb_pos) * bbox_weight
                if target_scores_sum == 0:
                    loss_dfl = loss_dfl.sum()
                else:
                    loss_dfl = loss_dfl.sum() / target_scores_sum
            else:
                loss_dfl = pred_dist.sum() * 0.
        
        else:
            loss_iou = pred_dist.sum() * 0.
            loss_dfl = pred_dist.sum() * 0.
        
        return loss_iou, loss_dfl
    
    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
                pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction='none').view(
                target_left.shape) * weight_left
        loss_right = F.cross_entropy(
                pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction='none').view(
                target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)
