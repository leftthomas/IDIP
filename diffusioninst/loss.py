import torch
import torch.nn.functional as F
from detectron2.structures import BitMasks
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchvision.ops import generalized_box_iou, sigmoid_focal_loss


class SetCriterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.MODEL.DiffusionInst.NUM_CLASSES
        self.cls_weight = cfg.MODEL.DiffusionInst.CLS_WEIGHT
        self.l1_weight = cfg.MODEL.DiffusionInst.L1_WEIGHT
        self.giou_weight = cfg.MODEL.DiffusionInst.GIOU_WEIGHT
        self.mask_weight = cfg.MODEL.DiffusionInst.MASK_WEIGHT
        self.matcher = HungarianMatcher(self.cls_weight, self.l1_weight, self.giou_weight, self.mask_weight)

    def forward(self, outputs, targets):
        # retrieve the matching between outputs and targets
        indices = self.matcher(outputs, targets)
        # [B, N, C], [B, N, 4], [B, N, C, 2*S, 2*S]
        pred_logits, pred_boxes, pred_masks = outputs['pred_logits'], outputs['pred_boxes'], outputs['pred_masks']
        b, n, c, t, _ = pred_masks.size()

        num_instances, total_cls_loss, total_l1_loss, total_giou_loss, total_mask_loss = 0, 0, 0, 0, 0
        for i in range(b):
            row_ind, col_ind = indices[i][:, 0], indices[i][:, 1]
            # [M, C], [M, 4], [M, C, 2*S, 2*S]
            logits = torch.index_select(pred_logits[i], dim=0, index=row_ind)
            boxes = torch.index_select(pred_boxes[i], dim=0, index=row_ind)
            masks = torch.index_select(pred_masks[i], dim=0, index=row_ind)
            # [M], [M, 4], [M, H, W]
            gt_classes = torch.index_select(targets[i]['classes'], dim=0, index=col_ind)
            gt_boxes = torch.index_select(targets[i]['boxes'], dim=0, index=col_ind)
            gt_masks = BitMasks(torch.index_select(targets[i]['masks'].tensor, dim=0, index=col_ind))

            image_size, m = targets[i]['image_size'], gt_classes.size(0)
            num_instances += m

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
            # [M, 2*S, 2*S]
            gt_masks = gt_masks.crop_and_resize(gt_boxes, t).float()
            # [M, 2*S*2*S]
            masks = masks.view(m, c, -1)[torch.arange(m).to(masks.device), gt_classes, :]
            gt_masks = gt_masks.view(m, -1)
            intersection = (masks * gt_masks).sum(dim=-1)
            union = masks.sum(dim=-1) + gt_masks.sum(dim=-1) + 1e-8
            mask_loss = (1 - (2 * intersection / union)).sum()
            total_mask_loss = total_mask_loss + mask_loss

        total_cls_loss = total_cls_loss / num_instances
        total_l1_loss = total_l1_loss / num_instances
        total_giou_loss = total_giou_loss / num_instances
        total_mask_loss = total_mask_loss / num_instances
        losses = {'cls_loss': self.cls_weight * total_cls_loss, 'l1_loss': self.l1_weight * total_l1_loss,
                  'giou_loss': self.giou_weight * total_giou_loss, 'mask_loss': self.mask_weight * total_mask_loss}
        return losses


class HungarianMatcher(nn.Module):
    def __init__(self, cost_cls, cost_l1, cost_giou, cost_mask, alpha=0.25, gamma=2):
        super().__init__()
        self.cost_cls = cost_cls
        self.cost_l1 = cost_l1
        self.cost_giou = cost_giou
        self.cost_mask = cost_mask
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        with torch.no_grad():
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

                # compute the classification cost with focal loss
                alpha, gamma = self.alpha, self.gamma
                logits = logits.sigmoid()
                pos_cost = alpha * ((1 - logits) ** gamma) * (-(logits + 1e-8).log())
                neg_cost = (1 - alpha) * (logits ** gamma) * (-(1 - logits + 1e-8).log())
                # [N, M]
                cls_cost = pos_cost[:, gt_classes] - neg_cost[:, gt_classes]

                # compute the box cost with L1 loss and GIoU loss in normalized coordinates
                norm_boxes = boxes / image_size
                norm_gt_boxes = gt_boxes / image_size
                # [N, M]
                l1_cost = torch.cdist(norm_boxes, norm_gt_boxes, p=1)
                giou_cost = 1 - generalized_box_iou(norm_boxes, norm_gt_boxes)

                # compute the mask cost with dice loss
                masks = masks.sigmoid()
                # [M, 2*S, 2*S]
                gt_masks = gt_masks.crop_and_resize(gt_boxes, t).float()
                # [N, M]
                masks = torch.index_select(masks.view(n, c, -1), dim=1, index=gt_classes)
                gt_masks = gt_masks.view(1, m, -1)
                intersection = (masks * gt_masks).sum(dim=-1)
                union = masks.sum(dim=-1) + gt_masks.sum(dim=-1) + 1e-8
                mask_cost = 1 - (2 * intersection / union)

                # final cost matrix
                cost = self.cost_cls * cls_cost + self.cost_l1 * l1_cost + self.cost_giou * giou_cost \
                       + self.cost_mask * mask_cost
                row_ind, col_ind = linear_sum_assignment(cost.cpu())
                row_ind, col_ind = torch.as_tensor(row_ind), torch.as_tensor(col_ind)
                indices.append(torch.stack((row_ind, col_ind), dim=-1).to(cost.device))
        # [B, M, M]
        return indices