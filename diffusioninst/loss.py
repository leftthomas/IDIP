import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import generalized_box_iou, sigmoid_focal_loss, box_convert, box_iou


def dice_coefficient(x, target):
    eps = 1e-5
    n_instance = x.size(0)
    x = x.reshape(n_instance, -1)
    target = target.reshape(n_instance, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


class SetCriterion(nn.Module):
    """ This class computes the loss for DiffusionInst.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class, box and mask)
    """
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg.MODEL.DiffusionInst.NUM_CLASSES
        self.matcher = HungarianMatcher()

    def loss_labels(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key 'labels' containing a tensor of dim [nb_target_boxes]
        """
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

                loss_mask += dice_coefficient(pred_masks, gt_masks).sum()
                num_mask += len(gt_multi_idx)

        if num_mask > 0:
            loss_mask = loss_mask / num_mask
            losses = {'loss_masks': 5 * loss_mask}
        else:
            losses = {'loss_masks': outputs['pred_boxes'].sum() * 0}

        return losses

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # retrieve the matching between the outputs of the last layer and the targets
        indices, _ = self.matcher(outputs, targets)
        # compute all the requested losses
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices))
        losses.update(self.loss_masks(outputs, targets, indices))
        return losses


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-k (dynamic) matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_mask: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_mask = cost_mask

    def forward(self, outputs, targets):
        with torch.no_grad():
            bs, num_queries = outputs['pred_logits'].shape[:2]
            out_prob = outputs['pred_logits']
            out_bbox = outputs['pred_boxes']

            indices, matched_ids = [], []
            for batch_idx in range(bs):
                bz_out_prob = out_prob[batch_idx]
                bz_boxes = out_bbox[batch_idx]  # [num_proposals, 4]
                bz_tgt_ids = targets[batch_idx]['gt_classes']
                bz_gtboxs = targets[batch_idx]['gt_boxes']  # [num_gt, 4] (x, y, x, y)
                fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
                    bz_boxes,  # absolute (cx, cy, w, h)
                    bz_gtboxs,  # absolute (x, y, x, y)
                )

                pair_wise_ious = box_iou(bz_boxes, bz_gtboxs)

                # compute the classification cost
                alpha = self.focal_loss_alpha
                gamma = self.focal_loss_gamma
                neg_cost_class = (1 - alpha) * (bz_out_prob ** gamma) * (-(1 - bz_out_prob + 1e-8).log())
                pos_cost_class = alpha * ((1 - bz_out_prob) ** gamma) * (-(bz_out_prob + 1e-8).log())
                cost_class = pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids]

                bz_image_size_out = targets[batch_idx]['image_size_xyxy']
                bz_image_size_tgt = targets[batch_idx]['image_size_xyxy_tgt']

                bz_out_bbox_ = bz_boxes / bz_image_size_out  # normalize (x1, y1, x2, y2)
                bz_tgt_bbox_ = bz_gtboxs / bz_image_size_tgt  # normalize (x1, y1, x2, y2)
                cost_bbox = torch.cdist(bz_out_bbox_, bz_tgt_bbox_, p=1)

                cost_giou = -generalized_box_iou(bz_boxes, bz_gtboxs)

                # Final cost matrix
                cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + 100.0 * (
                    ~is_in_boxes_and_center)
                cost[~fg_mask] = cost[~fg_mask] + 10000.0

                # if bz_gtboxs.shape[0]>0:
                indices_batchi, matched_qidx = self.dynamic_k_matching(cost, pair_wise_ious, bz_gtboxs.shape[0])

                indices.append(indices_batchi)
                matched_ids.append(matched_qidx)

        return indices, matched_ids