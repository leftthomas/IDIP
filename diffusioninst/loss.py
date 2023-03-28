import torch
import torch.nn.functional as F
from detectron2.structures import BitMasks
from mmdet.core.bbox.assigners import SimOTAAssigner
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox.match_costs import FocalLossCost
from torch import nn
from torchvision.ops import generalized_box_iou, sigmoid_focal_loss, box_convert


class SetCriterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cls_weight = cfg.MODEL.DiffusionInst.CLS_WEIGHT
        self.l1_weight = cfg.MODEL.DiffusionInst.L1_WEIGHT
        self.giou_weight = cfg.MODEL.DiffusionInst.GIOU_WEIGHT
        self.mask_weight = cfg.MODEL.DiffusionInst.MASK_WEIGHT
        self.matcher = SimOTAMatcher(self.cls_weight, self.l1_weight, self.giou_weight, self.mask_weight)

    def forward(self, outputs, targets):
        all_cls_loss, all_l1_loss, all_giou_loss, all_mask_loss = 0, 0, 0, 0
        for output in outputs:
            # retrieve the matching between outputs and targets
            indices = self.matcher.assign(output, targets)
            # [B, N, C], [B, N, 4], [B, N, C, 2*S, 2*S]
            pred_logits, pred_boxes, pred_masks = output['pred_logits'], output['pred_boxes'], output['pred_masks']
            b, n, c, t, _ = pred_masks.size()

            num_instances, total_cls_loss, total_l1_loss, total_giou_loss, total_mask_loss = 0, 0, 0, 0, 0
            for i in range(b):
                row_ind, col_ind = indices[i][:, 0], indices[i][:, 1]
                # [K, C], [K, 4], [K, C, 2*S, 2*S]
                logits = torch.index_select(pred_logits[i], dim=0, index=row_ind)
                boxes = torch.index_select(pred_boxes[i], dim=0, index=row_ind)
                masks = torch.index_select(pred_masks[i], dim=0, index=row_ind)
                # [K], [K, 4], [K, H, W]
                gt_classes = torch.index_select(targets[i]['classes'], dim=0, index=col_ind)
                gt_boxes = torch.index_select(targets[i]['boxes'], dim=0, index=col_ind)
                gt_masks = BitMasks(torch.index_select(targets[i]['masks'].tensor, dim=0, index=col_ind))

                image_size, k = targets[i]['image_size'], gt_classes.size(0)
                num_instances += k

                # compute the classification loss with focal loss
                cls_loss = sigmoid_focal_loss(logits, F.one_hot(gt_classes, c).float(), reduction='sum')
                total_cls_loss = total_cls_loss + cls_loss

                # compute the box loss with L1 loss and GIoU loss in normalized coordinates
                norm_boxes = boxes / image_size
                norm_gt_boxes = gt_boxes / image_size
                l1_loss = F.l1_loss(norm_boxes, norm_gt_boxes, reduction='sum')
                total_l1_loss = total_l1_loss + l1_loss

                giou_loss = torch.diag(1 - generalized_box_iou(norm_boxes, norm_gt_boxes)).sum()
                total_giou_loss = total_giou_loss + giou_loss

                # compute the mask loss with dice loss
                masks = masks.sigmoid()
                # [K, 2*S, 2*S]
                gt_masks = gt_masks.crop_and_resize(gt_boxes, t).float()
                masks = masks[torch.arange(k).reshape(k, 1), gt_classes.reshape(k, 1), :, :].reshape(k, -1)
                gt_masks = gt_masks.reshape(k, -1)
                intersection = (masks * gt_masks).sum(dim=-1)
                union = masks.sum(dim=-1) + gt_masks.sum(dim=-1) + 1e-8
                mask_loss = (1 - (2 * intersection / union)).sum()
                total_mask_loss = total_mask_loss + mask_loss

            total_cls_loss = total_cls_loss / num_instances
            total_l1_loss = total_l1_loss / num_instances
            total_giou_loss = total_giou_loss / num_instances
            total_mask_loss = total_mask_loss / num_instances
            all_cls_loss = all_cls_loss + total_cls_loss
            all_l1_loss = all_l1_loss + total_l1_loss
            all_giou_loss = all_giou_loss + total_giou_loss
            all_mask_loss = all_mask_loss + total_mask_loss

        losses = {'cls_loss': self.cls_weight * all_cls_loss / len(outputs),
                  'l1_loss': self.l1_weight * all_l1_loss / len(outputs),
                  'giou_loss': self.giou_weight * all_giou_loss / len(outputs),
                  'mask_loss': self.mask_weight * all_mask_loss / len(outputs)}
        return losses


class SimOTAMatcher(SimOTAAssigner):
    def __init__(self, cost_cls, cost_l1, cost_giou, cost_mask):
        super().__init__()
        self.cls_cost = FocalLossCost(cost_cls, eps=1e-8)
        self.cost_l1 = cost_l1
        self.cost_giou = cost_giou
        self.cost_mask = cost_mask

    @torch.no_grad()
    def assign(self, outputs, targets):
        # [B, N, C], [B, N, 4], [B, N, C, 2*S, 2*S]
        pred_logits, pred_boxes, pred_masks = outputs['pred_logits'], outputs['pred_boxes'], outputs['pred_masks']
        b, n, c, t, _ = pred_masks.size()

        indices = []
        for i in range(b):
            # [N, C], [N, 4], [N, C, 2*S, 2*S]
            logits, boxes, masks = pred_logits[i], pred_boxes[i], pred_masks[i]
            # [M], [M, 4], [M, H, W]
            gt_classes, gt_boxes, gt_masks = targets[i]['classes'], targets[i]['boxes'], targets[i]['masks']
            image_size = targets[i]['image_size']
            m = gt_classes.size(0)

            # filter out the boxes not inside the ground truth boxes
            valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(
                box_convert(boxes, in_fmt='xyxy', out_fmt='cxcywh'), gt_boxes)
            logits, boxes, masks = logits[valid_mask], boxes[valid_mask], masks[valid_mask]

            assert len(logits) > 0, 'No valid boxes in the image'

            # [K, M]
            cls_cost = self.cls_cost(logits, gt_classes)

            # compute the box cost with L1 loss and GIoU loss in normalized coordinates
            norm_boxes = boxes / image_size
            norm_gt_boxes = gt_boxes / image_size
            # [K, M]
            l1_cost = self.cost_l1 * torch.cdist(norm_boxes, norm_gt_boxes, p=1)
            giou_cost = self.cost_giou * (1 - generalized_box_iou(norm_boxes, norm_gt_boxes))

            # compute the mask cost with dice loss
            masks = masks.sigmoid()
            # [M, 2*S, 2*S]
            gt_masks = gt_masks.crop_and_resize(gt_boxes, t).float()
            # [K, M]
            masks = masks.reshape(-1, 1, c, t * t)
            gt_masks = gt_masks.reshape(1, m, 1, -1)
            intersection = (masks * gt_masks).sum(dim=-1)
            union = masks.sum(dim=-1) + gt_masks.sum(dim=-1) + 1e-8
            mask_cost = self.cost_mask * (1 - (2 * intersection / union))
            mask_cost = mask_cost[:, torch.arange(m).reshape(m, 1), gt_classes.reshape(m, 1)].squeeze(dim=-1)

            # final cost matrix
            cost = cls_cost + l1_cost + giou_cost + mask_cost + (~is_in_boxes_and_center) * 100000.0

            pairwise_ious = bbox_overlaps(boxes, gt_boxes, eps=1e-8)
            _, col_ind = self.dynamic_k_matching(cost, pairwise_ious, m, valid_mask)
            row_ind, col_ind = torch.as_tensor(torch.nonzero(valid_mask).squeeze(dim=-1)), torch.as_tensor(col_ind)
            indices.append(torch.stack((row_ind, col_ind), dim=-1).to(cost.device))
        # [B, K, K]
        return indices