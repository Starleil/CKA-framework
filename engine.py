"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
from torch import nn
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher

from datasets.data_prefetcher_keycan import data_prefetcher_keycan
from collections import defaultdict
import pickle

class PostProcess_for_scores(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_id=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 300, dim=1)
        scores = topk_values
        labels = topk_indexes % out_logits.shape[2]
        if target_id is not None:
            labels_f = labels.flatten()
            if target_id in labels_f:
                indices = torch.where(labels_f == target_id)[0]
                top_indice = indices[torch.topk(labels_f[indices], k=1).indices]
                scores = [scores[0][top_indice]]
                labels = [labels[0][top_indice]]
            else:
                scores = torch.tensor([0.])
                labels = torch.tensor([target_id])
        else:
            scores = torch.tensor([scores[0][0]])
            labels = torch.tensor([labels[0][0]])

        results = [{'scores': s, 'labels': l} for s, l in zip(scores, labels)]

        return results


def train_one_epoch_keycan(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, keycan, max_norm: float = 0, writer=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    prefetcher = data_prefetcher_keycan(data_loader, device, prefetch=True) ############
    _, targets, infos = prefetcher.next()
    infos = infos[0]
    ppfs = PostProcess_for_scores()
    cache_dict = defaultdict(list, {})

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for i in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        vid = infos['video_id'][0]

        if vid not in cache_dict:
            cache_dict[vid] = []
            glb_frms_id = infos['glb_file_id']
            frm_id = infos['frm_id']
            supp_frms_id = infos['supp_frms_id']
            samples, targets = keycan.load_imgs(vid, frm_id, supp_frms_id, glb_frms_id, targets)
            samples = samples.to(str(torch.device(targets[0]['boxes'].device))).unsqueeze(0)
        else:
            glb_frms_id = infos['glb_file_id']
            frm_id = infos['frm_id']
            supp_frms_id = infos['supp_frms_id']
            key_candidate = [d['frm_id'][0] for d in cache_dict[vid] if d['frame_id'] < frm_id]
            for t_idx in supp_frms_id:
                if t_idx in key_candidate:
                    key_candidate.remove(t_idx)

            idx_g = [i for i in range(len(glb_frms_id))]
            idx_k = [i for i in range(len(key_candidate))]
            for ele in glb_frms_id:
                if ele in key_candidate:
                    index_g = glb_frms_id.index(ele)
                    idx_g.remove(index_g)
                    index_k = key_candidate.index(ele)
                    idx_k.remove(index_k)

            if len(key_candidate) >= 3:
                glb_frms_id = key_candidate[:3]
            else:
                for i in range(min(len(idx_k), len(idx_g))):
                    glb_frms_id[idx_g[i]] = key_candidate[idx_k[i]]

            samples, targets = keycan.load_imgs(vid, frm_id, supp_frms_id, glb_frms_id, targets)
            samples = samples.to(str(torch.device(targets['boxes'].device))).unsqueeze(0)
            infos.update({'glb_file_id': glb_frms_id})

        outputs = model(samples)
        #
        results = ppfs(outputs, target_id=int(infos['cat_id'][0]))[0]
        cls_ids = results['labels']
        scores = results['scores']

        infos.update({'scores': scores, 'cls_ids': cls_ids})
        cache_dict[vid].append(infos)
        for k, v in cache_dict.items():
            sorted_v = sorted(v, key=lambda x: x['scores'], reverse=True)
            cache_dict.update({k: sorted_v})

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        #
        writer.add_scalar('loss_sum', 0, global_step=epoch)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if writer is not None:
            writer.add_scalar('loss_sum_Current_epoch', loss_value, global_step=i)
            writer.add_scalar('loss_bbox_Current_epoch', loss_dict_reduced_scaled['loss_bbox'], global_step=i)
            writer.add_scalar('loss_ce_Current_epoch', loss_dict_reduced_scaled['loss_ce'], global_step=i)
            writer.add_scalar('loss_giou_Current_epoch', loss_dict_reduced_scaled['loss_giou'], global_step=i)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets, infos = prefetcher.next()
        if infos is not None:
            infos = infos[0]
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, writer=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for i in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        #
        writer.add_scalar('loss_sum', 0, global_step=epoch)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if writer is not None:
            writer.add_scalar('loss_sum_Current_epoch', loss_value, global_step=i)
            writer.add_scalar('loss_bbox_Current_epoch', loss_dict_reduced_scaled['loss_bbox'], global_step=i)
            writer.add_scalar('loss_ce_Current_epoch', loss_dict_reduced_scaled['loss_ce'], global_step=i)
            writer.add_scalar('loss_giou_Current_epoch', loss_dict_reduced_scaled['loss_giou'], global_step=i)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator

import numpy as np
def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.5f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            # 判断是否传入catId，如果传入就计算指定类别的指标
            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            # 判断是否传入catId，如果传入就计算指定类别的指标
            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info



@torch.no_grad()
def evaluate_each_cat_v1(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, data_coco_lite_path):
    from pathlib import Path
    from pycocotools.coco import COCO

    root = Path(data_coco_lite_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    PATHS = {
        "val": (root / "rawframes", root / f'imagenet_vid_val.json')
    }
    # miccai_buv\rawframes, miccai_buv\imagenet_vid_train_15frames.json
    img_folder, ann_file = PATHS['val']
    co = COCO(ann_file)

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    RST = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        #
        img_path = co.loadImgs(int(targets[0]['image_id']))[0]['file_name'].replace('/','_')
        l = results[0]['labels'] - 1
        # assert l.any() > 0
        info = {
            'img_path': img_path,
            'pred_instances': {
                'scores': results[0]['scores'],
                'labels': l,
                'boxes': results[0]['boxes']
            }
        }
        RST.append(info)

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    with open(os.path.join(output_dir, 'output_online0.5_onlyLIFA.pkl'), 'wb') as file:
        pickle.dump(RST, file)
    # with open(os.path.join(output_dir, 'output.pkl'), 'rb') as file:
    #     data = pickle.load(file)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()

            # calculate COCO info for all classes
            # coco_stats, print_coco = summarize(coco_evaluator.coco_eval['bbox'])
            # calculate voc info for every classes(IoU=0.5)
            voc_map_info_list = []
            category_index = ['benign', 'malignant']
            for i in range(len(category_index)):
                inter_stats, _ = summarize(coco_evaluator.coco_eval['bbox'], catId=i)
                # voc_map_info_list.append("({},{})".format(category_index[i], inter_stats[1]))
                voc_map_info_list.append((category_index[i], inter_stats[1]))
            print_voc = dict(voc_map_info_list)
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            # calculate voc info for every classes(IoU=0.5)
            voc_map_info_list = []
            category_index = ['benign', 'malignant']
            for i in range(len(category_index)):
                inter_stats, _ = summarize(coco_evaluator.coco_eval['bbox'], catId=i)
                # voc_map_info_list.append("({},{})".format(category_index[i], inter_stats[1]))
                voc_map_info_list.append((category_index[i], inter_stats[1]))
            print_voc = dict(voc_map_info_list)
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator, print_voc

@torch.no_grad()
def evaluate_each_cat(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()

            # calculate COCO info for all classes
            # coco_stats, print_coco = summarize(coco_evaluator.coco_eval['bbox'])
            # calculate voc info for every classes(IoU=0.5)
            voc_map_info_list = []
            category_index = ['benign', 'malignant']
            for i in range(len(category_index)):
                inter_stats, _ = summarize(coco_evaluator.coco_eval['bbox'], catId=i)
                # voc_map_info_list.append("({},{})".format(category_index[i], inter_stats[1]))
                voc_map_info_list.append((category_index[i], inter_stats[1]))
            print_voc = dict(voc_map_info_list)
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            # calculate voc info for every classes(IoU=0.5)
            voc_map_info_list = []
            category_index = ['benign', 'malignant']
            for i in range(len(category_index)):
                inter_stats, _ = summarize(coco_evaluator.coco_eval['bbox'], catId=i)
                # voc_map_info_list.append("({},{})".format(category_index[i], inter_stats[1]))
                voc_map_info_list.append((category_index[i], inter_stats[1]))
            print_voc = dict(voc_map_info_list)
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator, print_voc

@torch.no_grad()
def evaluate_each_cat_keycan(model, criterion, postprocessors, data_loader, base_ds, device, keycan, output_dir, data_coco_lite_path):
    from pathlib import Path
    from pycocotools.coco import COCO

    root = Path(data_coco_lite_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    PATHS = {
        "val": (root / "rawframes", root / f'imagenet_vid_val.json')
    }
    # miccai_buv\rawframes, miccai_buv\imagenet_vid_train_15frames.json
    img_folder, ann_file = PATHS['val']
    co = COCO(ann_file)

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    ppfs = PostProcess_for_scores()
    cache_dict = defaultdict(list, {})
    RST = []
    for samples, targets, infos in metric_logger.log_every(data_loader, 10, header):
        _ = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if infos is not None:
            infos = infos[0]

        vid = infos['video_id'][0]
        if vid not in cache_dict:
            cache_dict[vid] = []
            # transform在这里做，coco.py中不再做transform
            # samples = keycan.T(samples, targets)
            glb_frms_id = infos['glb_file_id']
            frm_id = infos['frm_id']
            supp_frms_id = infos['supp_frms_id']
            samples, targets = keycan.load_imgs(vid, frm_id, supp_frms_id, glb_frms_id,
                                                targets)  ############### target from tensor to nomarl
            samples = samples.to(str(torch.device(targets[0]['boxes'].device))).unsqueeze(0)
        else:
            glb_frms_id = infos['glb_file_id']
            frm_id = infos['frm_id']
            supp_frms_id = infos['supp_frms_id']
            # print(glb_frms_id, frm_id, supp_frms_id)
            # print(my_dict1[vid])
            key_candidate = [d['frm_id'][0] for d in cache_dict[vid] if d['frame_id'] < frm_id]
            for t_idx in supp_frms_id:
                if t_idx in key_candidate:
                    key_candidate.remove(t_idx)

            idx_g = [i for i in range(len(glb_frms_id))]
            idx_k = [i for i in range(len(key_candidate))]
            for ele in glb_frms_id:
                if ele in key_candidate:
                    index_g = glb_frms_id.index(ele)
                    idx_g.remove(index_g)
                    index_k = key_candidate.index(ele)
                    idx_k.remove(index_k)

            if len(key_candidate) >= 3:
                glb_frms_id = key_candidate[:3]
            else:
                for i in range(min(len(idx_k), len(idx_g))):
                    glb_frms_id[idx_g[i]] = key_candidate[idx_k[i]]
            # print(glb_frms_id, frm_id, supp_frms_id)
            samples, targets = keycan.load_imgs(vid, frm_id, supp_frms_id, glb_frms_id, targets)
            samples = samples.to(str(torch.device(targets['boxes'].device))).unsqueeze(0)
            infos.update({'glb_file_id': glb_frms_id})

        outputs = model(samples)

        res = ppfs(outputs, target_id=int(infos['cat_id'][0]))[
            0]  # yolo output: Detections matrix nx6 (xyxy, conf, cls), tensor([[196.22995, 122.47235, 392.35941, 258.18585,   0.85984,   0.00000]], device='cuda:0'
        cls_ids = res['labels']
        scores = res['scores']

        infos.update({'scores': scores, 'cls_ids': cls_ids})
        cache_dict[vid].append(infos)
        for k, v in cache_dict.items():
            sorted_v = sorted(v, key=lambda x: x['scores'], reverse=True)
            cache_dict.update({k: sorted_v})
        # print(cache_dict)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        #
        img_path = co.loadImgs(int(targets[0]['image_id']))[0]['file_name'].replace('/', '_')
        l = results[0]['labels'] - 1
        # assert l.any() > 0
        info = {
            'img_path': img_path,
            'pred_instances': {
                'scores': results[0]['scores'],
                'labels': l,
                'boxes': results[0]['boxes']
            }
        }
        RST.append(info)

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    with open(os.path.join(output_dir, 'output_online0.99_keycan.pkl'), 'wb') as file:
        pickle.dump(RST, file)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()

            # calculate COCO info for all classes
            # coco_stats, print_coco = summarize(coco_evaluator.coco_eval['bbox'])
            # calculate voc info for every classes(IoU=0.5)
            voc_map_info_list = []
            category_index = ['benign', 'malignant']
            for i in range(len(category_index)):
                inter_stats, _ = summarize(coco_evaluator.coco_eval['bbox'], catId=i)
                # voc_map_info_list.append("({},{})".format(category_index[i], inter_stats[1]))
                voc_map_info_list.append((category_index[i], inter_stats[1]))
            print_voc = dict(voc_map_info_list)
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            # calculate voc info for every classes(IoU=0.5)
            voc_map_info_list = []
            category_index = ['benign', 'malignant']
            for i in range(len(category_index)):
                inter_stats, _ = summarize(coco_evaluator.coco_eval['bbox'], catId=i)
                # voc_map_info_list.append("({},{})".format(category_index[i], inter_stats[1]))
                voc_map_info_list.append((category_index[i], inter_stats[1]))
            print_voc = dict(voc_map_info_list)
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator, print_voc