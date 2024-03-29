import torch
import torch.nn.functional as F
from detectron2.structures import BitMasks
from mmdet.models.task_modules.assigners import SimOTAAssigner
from mmdet.structures.bbox import bbox_overlaps
from torch import nn
from torchvision.ops import generalized_box_iou, sigmoid_focal_loss, box_convert


class SetCriterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cls_weight = cfg.MODEL.IDIP.CLS_WEIGHT
        self.l1_weight = cfg.MODEL.IDIP.L1_WEIGHT
        self.giou_weight = cfg.MODEL.IDIP.GIOU_WEIGHT
        self.mask_weight = cfg.MODEL.IDIP.MASK_WEIGHT
        self.matcher = SimOTAMatcher(self.cls_weight, self.l1_weight, self.giou_weight)

    def forward(self, outputs, targets, features, mask_head):
        total_cls_loss, total_l1_loss, total_giou_loss, total_mask_loss = 0, 0, 0, 0
        for output in outputs:
            # retrieve the matching between outputs and targets
            indices = self.matcher.assign(output, targets)
            # [B, N, C], [B, N, 4], [B, N, D]
            pred_logits, pred_boxes, proposal_feats = output['pred_logits'], output['pred_boxes'], output[
                'proposal_feat']
            b, n, c = pred_logits.size()

            num_ins, logits, norm_boxes, norm_gt_boxes, masks, gt_labels, gt_masks = 0, [], [], [], [], [], []
            for i in range(b):
                valid_mask, gt_ind = indices[i][0], indices[i][1]
                if len(gt_ind) == 0:
                    # pass the image with empty ground truth
                    continue
                # [N, C], [K, 4], [K, D]
                logit, box, proposal_feat = pred_logits[i], pred_boxes[i][valid_mask], proposal_feats[i][valid_mask]
                feature = [feat[i].unsqueeze(dim=0) for feat in features]
                # [K], [K, 4], [K, H, W]
                gt_class = targets[i]['classes'][gt_ind]
                gt_box = targets[i]['boxes'][gt_ind]
                gt_mask = BitMasks(targets[i]['masks'].tensor[gt_ind])

                label = torch.zeros_like(logit)
                label[valid_mask] = F.one_hot(gt_class, c).float()
                image_size, k = targets[i]['image_size'], gt_class.size(0)
                num_ins += k

                mask = torch.sigmoid(mask_head(feature, box, proposal_feat))
                mask = mask[torch.arange(k, device=mask.device), gt_class, :, :]
                t = mask.size(-1)
                # [K, 2*S, 2*S]
                gt_mask = gt_mask.crop_and_resize(box, t).float()

                logits.append(logit)
                norm_boxes.append(box / image_size)
                norm_gt_boxes.append(gt_box / image_size)
                gt_labels.append(label)
                masks.append(mask)
                gt_masks.append(gt_mask)

            if num_ins > 0:
                # compute the classification loss with focal loss
                cls_loss = sigmoid_focal_loss(torch.cat(logits), torch.cat(gt_labels), reduction='sum')
                # compute the box loss with L1 loss and GIoU loss in normalized coordinates
                l1_loss = F.l1_loss(torch.cat(norm_boxes), torch.cat(norm_gt_boxes), reduction='sum')
                giou_loss = torch.diag(1 - generalized_box_iou(torch.cat(norm_boxes), torch.cat(norm_gt_boxes))).sum()
                # compute the mask loss with dice loss
                masks = torch.cat(masks).reshape(num_ins, -1)
                gt_masks = torch.cat(gt_masks).reshape(num_ins, -1)
                intersection = (masks * gt_masks).sum(dim=-1)
                union = masks.sum(dim=-1) + gt_masks.sum(dim=-1) + 1e-8
                mask_loss = (1 - (2 * intersection / union)).sum()

                total_cls_loss = total_cls_loss + cls_loss / num_ins
                total_l1_loss = total_l1_loss + l1_loss / num_ins
                total_giou_loss = total_giou_loss + giou_loss / num_ins
                total_mask_loss = total_mask_loss + mask_loss / num_ins

        losses = {'cls_loss': self.cls_weight * total_cls_loss / len(outputs),
                  'l1_loss': self.l1_weight * total_l1_loss / len(outputs),
                  'giou_loss': self.giou_weight * total_giou_loss / len(outputs),
                  'mask_loss': self.mask_weight * total_mask_loss / len(outputs)}
        return losses


class SimOTAMatcher(SimOTAAssigner):
    def __init__(self, cost_cls, cost_l1, cost_giou, alpha=0.25, gamma=2):
        super().__init__()
        self.cost_cls = cost_cls
        self.cost_l1 = cost_l1
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def assign(self, outputs, targets):
        # [B, N, C], [B, N, 4]
        pred_logits, pred_boxes = outputs['pred_logits'], outputs['pred_boxes']
        b, n, c = pred_logits.size()

        indices = []
        for i in range(b):
            # [N, C], [N, 4]
            logits, boxes = pred_logits[i], pred_boxes[i]
            # [M], [M, 4]
            gt_classes, gt_boxes = targets[i]['classes'], targets[i]['boxes']
            image_size = targets[i]['image_size']
            m = gt_classes.size(0)
            if m == 0:
                # empty ground truth
                indices.append((torch.zeros(n, device=logits.device).bool(), torch.arange(0, device=logits.device)))
                continue

            # filter out the boxes not inside the ground truth boxes
            valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(
                box_convert(boxes, in_fmt='xyxy', out_fmt='cxcywh'), gt_boxes)
            if valid_mask.sum() == 0:
                # avoid no valid boxes
                valid_mask = torch.ones_like(valid_mask)
                is_in_boxes_and_center = torch.ones(n, m, dtype=torch.bool, device=boxes.device)

            logits, boxes = logits[valid_mask], boxes[valid_mask]

            # [K, M]
            cls_pred = logits.sigmoid()
            neg_cost = -(1 - cls_pred + 1e-8).log() * (1 - self.alpha) * cls_pred.pow(self.gamma)
            pos_cost = -(cls_pred + 1e-8).log() * self.alpha * (1 - cls_pred).pow(self.gamma)
            cls_cost = self.cost_cls * (pos_cost[:, gt_classes] - neg_cost[:, gt_classes])

            # compute the box cost with L1 loss and GIoU loss in normalized coordinates
            norm_boxes = boxes / image_size
            norm_gt_boxes = gt_boxes / image_size
            # [K, M]
            l1_cost = self.cost_l1 * torch.cdist(norm_boxes, norm_gt_boxes, p=1)
            giou_cost = self.cost_giou * (1 - generalized_box_iou(norm_boxes, norm_gt_boxes))

            # final cost matrix
            cost = cls_cost + l1_cost + giou_cost + (~is_in_boxes_and_center) * 100000.0

            pairwise_ious = bbox_overlaps(boxes, gt_boxes, eps=1e-8)
            _, gt_ind = self.dynamic_k_matching(cost, pairwise_ious, m, valid_mask)
            indices.append((valid_mask, gt_ind))
        return indices