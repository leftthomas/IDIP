import random

import torch
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from torch import nn
from torchvision.ops import box_convert, batched_nms

from .head import cosine_schedule, normed_box_to_abs_box, TimeEncoder, DiffusionRoiHead
from .loss import SetCriterion


@META_ARCH_REGISTRY.register()
class DiffusionInst(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.DiffusionInst.NUM_CLASSES
        self.num_proposals = cfg.MODEL.DiffusionInst.NUM_PROPOSALS

        # build backbone
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility
        dim_features = [self.backbone.output_shape()[k].channels for k in self.in_features]
        assert len(set(dim_features)) == 1
        self.dim_features = dim_features[0]

        # build diffusion
        self.num_steps = cfg.MODEL.DiffusionInst.NUM_STEPS
        self.sampling_steps = cfg.MODEL.DiffusionInst.SAMPLING_STEPS
        alphas = cosine_schedule(self.num_steps)
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))

        # build time head
        self.time_head = TimeEncoder(self.dim_features)

        # build RoI head
        strides = [self.backbone.output_shape()[k].stride for k in self.in_features]
        self.roi_head = DiffusionRoiHead(self.dim_features, strides, self.num_classes)

        # build loss criterion
        self.criterion = SetCriterion(cfg)

        self.register_buffer('pixel_mean', torch.as_tensor(cfg.MODEL.PIXEL_MEAN).reshape(3, 1, 1))
        self.register_buffer('pixel_std', torch.as_tensor(cfg.MODEL.PIXEL_STD).reshape(3, 1, 1))
        self.to(self.device)

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        # feature extraction
        src = self.backbone(images.tensor)
        features = [src[f] for f in self.in_features]

        if self.training:
            # [B], [B, N, 4], [B, 1]
            targets, boxes, ts = self.preprocess_target(batched_inputs)
            # [B, 4*D]
            time_emb = self.time_head(ts)
            output = self.roi_head(features, boxes, time_emb)
            loss_dict = self.criterion(output, targets)
            return loss_dict
        else:
            # [N, C], [N, 4], [N, C, 2*S, 2*S]
            pred_logits, pred_boxes, pred_masks = self.ddim_sample(batched_inputs, features)
            results = self.inference(pred_logits, pred_boxes, pred_masks, batched_inputs)
            return results

    def preprocess_image(self, batched_inputs):
        # normalize, pad and batch the input images
        images = [(x['image'].to(self.device) - self.pixel_mean) / self.pixel_std for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)
        return images

    def preprocess_target(self, batched_inputs):
        targets, diffused_boxes, ts = [], [], []
        for x in batched_inputs:
            instances = x['instances'].to(self.device)
            gt_classes = instances.gt_classes
            gt_boxes = instances.gt_boxes.tensor
            gt_masks = instances.gt_masks
            h, w = instances.image_size
            image_size = torch.as_tensor([w, h, w, h], device=self.device).reshape(1, 4)

            crpt_boxes, t = self.prepare_diffusion(gt_boxes, image_size)
            diffused_boxes.append(crpt_boxes)
            ts.append(t)

            targets.append({'classes': gt_classes, 'boxes': gt_boxes, 'masks': gt_masks, 'image_size': image_size})

        return targets, torch.stack(diffused_boxes), torch.stack(ts)

    def prepare_diffusion(self, gt_boxes, image_size):
        # normalize to relative coordinates, and use cxcywh format
        gt_boxes = box_convert(gt_boxes / image_size, in_fmt='xyxy', out_fmt='cxcywh')

        # box padding
        num_gt = len(gt_boxes)
        if num_gt < self.num_proposals:
            # ref DiffusionDet: Diffusion Model for Object Detection
            box_placeholder = normed_box_to_abs_box(torch.randn(self.num_proposals - num_gt, 4, device=self.device))
            box_placeholder = box_convert(box_placeholder, in_fmt='xyxy', out_fmt='cxcywh')
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            # random select
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes

        # operate in [-1, 1] space to keep same with diffusion noise
        x_start = x_start * 2 - 1
        t = torch.randint(0, self.num_steps, (1,), device=self.device)
        noise = torch.randn_like(x_start)
        x_t = self.alphas_cumprod[t].sqrt() * x_start + (1 - self.alphas_cumprod[t]).sqrt() * noise

        # back to absolute coordinates, and use xyxy format
        crpt_boxes = normed_box_to_abs_box(x_t.to(torch.float32), image_size)
        return crpt_boxes, t

    @torch.no_grad()
    def ddim_sample(self, batched_inputs, features):
        assert len(batched_inputs) == 1
        times = reversed(torch.linspace(-1, self.num_steps - 1, self.sampling_steps + 1, device=self.device).long())
        x_t = torch.randn((self.num_proposals, 4), device=self.device)
        h, w = batched_inputs[0]['image'].size()[1:]
        image_size = torch.as_tensor([w, h, w, h], device=self.device).reshape(1, 4)
        for time_now, time_next in zip(times[:-1], times[1:]):
            boxes = normed_box_to_abs_box(x_t, image_size)
            time_emb = self.time_head(time_now.reshape(1, 1))
            output = self.roi_head(features, boxes.unsqueeze(dim=0), time_emb)[-1]
            # [1, N, C], [1, N, 4], [1, N, C, 2*S, 2*S]
            pred_logits, pred_boxes, pred_masks = output['pred_logits'], output['pred_boxes'], output['pred_masks']
            # operate in [-1, 1] space to keep same with diffusion noise
            x_start = box_convert(pred_boxes.squeeze(dim=0) / image_size, in_fmt='xyxy', out_fmt='cxcywh')
            x_start = torch.clamp(x_start, min=0, max=1) * 2 - 1
            pred_noise = (x_t - self.alphas_cumprod[time_now].sqrt() * x_start) / (
                    1 - self.alphas_cumprod[time_now]).sqrt()

            if time_next < 0:
                x_t = x_start
                continue
            # according DDIM to compute x_t of next time step
            alpha, alpha_next = self.alphas_cumprod[time_now], self.alphas_cumprod[time_next]
            sigma = (1 - alpha / alpha_next).sqrt() * ((1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(x_t)
            x_t = (x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise).to(torch.float32)

        return pred_logits.squeeze(0), pred_boxes.squeeze(0), pred_masks.squeeze(0)

    @torch.no_grad()
    def inference(self, pred_logits, pred_boxes, pred_masks, batched_inputs):
        assert len(batched_inputs) == 1
        image_size = batched_inputs[0]['image'].size()[1:]
        result = Instances(image_size)
        t = pred_masks.size(-1)

        pred_logits = torch.sigmoid(pred_logits)
        pred_masks = torch.sigmoid(pred_masks)
        # [N*C]
        labels = torch.arange(self.num_classes, device=self.device).repeat(self.num_proposals)
        # select the top N predictions
        scores, indices = pred_logits.flatten().topk(self.num_proposals, sorted=False)
        classes = labels[indices]
        boxes = pred_boxes.reshape(-1, 1, 4).repeat(1, self.num_classes, 1).reshape(-1, 4)[indices]
        masks = pred_masks.reshape(-1, 1, t, t)[indices]

        # nms
        keep = batched_nms(boxes, scores, classes, 0.5)
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]
        masks = masks[keep]

        # convert to detectron2 needed format
        result.pred_boxes = Boxes(boxes)
        result.scores = scores
        result.pred_classes = classes
        result.pred_masks = masks
        height = batched_inputs[0].get('height', image_size[0])
        width = batched_inputs[0].get('width', image_size[1])
        r = detector_postprocess(result, height, width)
        return [{'instances': r}]