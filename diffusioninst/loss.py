import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchvision.ops import generalized_box_iou, sigmoid_focal_loss, box_convert


def dice_loss(x, target):
    n_instance = x.size(0)
    x = x.view(n_instance, -1)
    target = target.view(n_instance, -1)
    intersection = (x * target).sum(dim=-1)
    union = x.sum(dim=-1) + target.sum(dim=-1) + 1e-8
    loss = 1 - (2 * intersection / union)
    return loss


class SetCriterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.MODEL.DiffusionInst.NUM_CLASSES
        self.cls_weight = cfg.MODEL.DiffusionInst.CLS_WEIGHT
        self.l1_weight = cfg.MODEL.DiffusionInst.L1_WEIGHT
        self.giou_weight = cfg.MODEL.DiffusionInst.GIOU_WEIGHT
        self.mask_weight = cfg.MODEL.DiffusionInst.MASK_WEIGHT
        self.matcher = HungarianMatcher(self.cls_weight, self.l1_weight, self.giou_weight, self.mask_weight)

    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs['pred_logits']
        batch_size = len(targets)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        src_logits_list = []
        target_classes_o_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_src_logits = src_logits[batch_idx]
            target_classes_o = targets[batch_idx]['labels']
            target_classes[batch_idx, valid_query] = target_classes_o[gt_multi_idx]

            src_logits_list.append(bz_src_logits[valid_query])
            target_classes_o_list.append(target_classes_o[gt_multi_idx])

        num_boxes = torch.cat(target_classes_o_list).shape[0] if len(target_classes_o_list) != 0 else 1

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], self.num_classes + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout,
                                            device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        src_logits = src_logits.flatten(0, 1)
        target_classes_onehot = target_classes_onehot.flatten(0, 1)
        cls_loss = sigmoid_focal_loss(src_logits, target_classes_onehot)

        loss_ce = torch.sum(cls_loss) / num_boxes
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key 'boxes' containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        src_boxes = outputs['pred_boxes']

        batch_size = len(targets)
        pred_box_list = []
        pred_norm_box_list = []
        tgt_box_list = []
        tgt_box_xyxy_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_image_whwh = targets[batch_idx]['image_size_xyxy']
            bz_src_boxes = src_boxes[batch_idx]
            bz_target_boxes = targets[batch_idx]['boxes']  # normalized (cx, cy, w, h)
            bz_target_boxes_xyxy = targets[batch_idx]['boxes_xyxy']  # absolute (x1, y1, x2, y2)
            pred_box_list.append(bz_src_boxes[valid_query])
            pred_norm_box_list.append(bz_src_boxes[valid_query] / bz_image_whwh)  # normalize (x1, y1, x2, y2)
            tgt_box_list.append(bz_target_boxes[gt_multi_idx])
            tgt_box_xyxy_list.append(bz_target_boxes_xyxy[gt_multi_idx])

        if len(pred_box_list) != 0:
            src_boxes = torch.cat(pred_box_list)
            src_boxes_norm = torch.cat(pred_norm_box_list)  # normalized (x1, y1, x2, y2)
            target_boxes = torch.cat(tgt_box_list)
            target_boxes_abs_xyxy = torch.cat(tgt_box_xyxy_list)
            num_boxes = src_boxes.shape[0]

            losses = {}
            # require normalized (x1, y1, x2, y2)
            loss_bbox = F.l1_loss(src_boxes_norm, box_convert(target_boxes, in_fmt='cxcywh', out_fmt='xyxy'),
                                  reduction='none')
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes

            loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes_abs_xyxy))
            losses['loss_giou'] = loss_giou.sum() / num_boxes
        else:
            losses = {'loss_bbox': outputs['pred_boxes'].sum() * 0,
                      'loss_giou': outputs['pred_boxes'].sum() * 0}

        return losses

    def loss_masks(self, outputs, targets, indices):
        pred_masks = outputs['pred_masks']

        batch_size = len(targets)
        loss_mask = 0
        num_mask = 0
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx][0]
            gt_multi_idx = indices[batch_idx][1]
            if len(gt_multi_idx) == 0:
                continue
            pred_masks = pred_masks[batch_idx]
            bz_target_mask = targets[batch_idx]['masks']
            pred_masks = pred_masks[valid_query]
            gt_masks = bz_target_mask[gt_multi_idx]

            if len(pred_masks) > 0:
                img_h, img_w = pred_masks.size(1) * 8, pred_masks.size(2) * 8
                h, w = gt_masks.size()[1:]
                gt_masks = F.pad(gt_masks, (0, img_w - w, 0, img_h - h), 'constant', 0)
                start = int(4 // 2)
                gt_masks = gt_masks[:, start::8, start::8]
                gt_masks = gt_masks.gt(0.5).float()

                loss_mask += dice_loss(pred_masks, gt_masks).sum()
                num_mask += len(gt_multi_idx)

        if num_mask > 0:
            loss_mask = loss_mask / num_mask
            losses = {'loss_masks': 5 * loss_mask}
        else:
            losses = {'loss_masks': outputs['pred_boxes'].sum() * 0}

        return losses

    def forward(self, outputs, targets):
        # retrieve the matching between outputs and targets
        indices = self.matcher(outputs, targets)
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices))
        losses.update(self.loss_masks(outputs, targets, indices))
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
                giou_cost = -generalized_box_iou(norm_boxes, norm_gt_boxes)

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
        return indices