import random
from typing import List

import torch
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes, ImageList, Instances
from torch import nn
from torchvision.ops import box_convert, clip_boxes_to_image

from .head import DetectHead, MaskHead, cosine_schedule
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
        self.register_buffer('betas', cosine_schedule(self.num_steps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - self.alphas_cumprod))

        # build RoI Pooler
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_scales = [1.0 / self.backbone.output_shape()[k].stride for k in self.in_features]
        self.pooler = ROIPooler(pooler_resolution, pooler_scales, sampling_ratio, pooler_type)

        # build detect head
        self.detect_head = DetectHead(cfg, self.dim_features)
        # build mask head
        self.mask_head = MaskHead(cfg, self.dim_features)

        # build loss criterion
        self.criterion = SetCriterion(cfg)

        self.register_buffer('pixel_mean', torch.as_tensor(cfg.MODEL.PIXEL_MEAN).view(3, 1, 1))
        self.register_buffer('pixel_std', torch.as_tensor(cfg.MODEL.PIXEL_STD).view(3, 1, 1))
        self.to(self.device)

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        # feature extraction
        src = self.backbone(images.tensor)
        features = [src[f] for f in self.in_features]

        if self.training:
            # [B], [B, N, 4], [B, N, 4], [B, 1]
            targets, boxes, noises, ts = self.preprocess_target(batched_inputs)
            roi_features = torch.flatten(self.pooler(features, [Boxes(b) for b in boxes]), start_dim=-2)
            # [B, N, D, S*S]
            roi_features = roi_features.view(len(targets), self.num_proposals, self.dim_features, -1)
            # [B, N, C], [B, N, 4]
            pred_logits, pred_boxes = self.detect_head(roi_features, ts, boxes)
            # [B, N, C, 2*S, 2*S]
            pred_masks = self.mask_head(roi_features, ts)
            output = {'pred_logits': pred_logits, 'pred_boxes': pred_boxes, 'pred_masks': pred_masks}
            loss_dict = self.criterion(output, targets)
            return loss_dict
        else:
            results = self.ddim_sample(batched_inputs, features, images)
            return results

    def preprocess_image(self, batched_inputs):
        # normalize, pad and batch the input images
        images = [(x['image'].to(self.device) - self.pixel_mean) / self.pixel_std for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)
        return images

    def preprocess_target(self, batched_inputs):
        targets, diffused_boxes, noises, ts = [], [], [], []
        for x in batched_inputs:
            instances = x['instances'].to(self.device)
            gt_classes = instances.gt_classes
            gt_boxes = instances.gt_boxes.tensor
            gt_masks = instances.gt_masks
            h, w = instances.image_size
            image_size = torch.as_tensor([w, h, w, h], device=self.device).view(1, 4)

            crpt_boxes, noise, t = self.prepare_diffusion(gt_boxes, image_size)
            diffused_boxes.append(crpt_boxes)
            noises.append(noise)
            ts.append(t)

            targets.append({'classes': gt_classes, 'boxes': gt_boxes, 'masks': gt_masks, 'image_size': image_size})

        return targets, torch.stack(diffused_boxes), torch.stack(noises), torch.stack(ts)

    def prepare_diffusion(self, gt_boxes, image_size):
        # normalize to relative coordinates, and use cxcywh format
        gt_boxes = box_convert(gt_boxes / image_size, in_fmt='xyxy', out_fmt='cxcywh')

        # box padding
        num_gt = len(gt_boxes)
        if num_gt < self.num_proposals:
            # ref DiffusionDet: Diffusion Model for Object Detection
            box_placeholder = torch.randn(self.num_proposals - num_gt, 4, device=self.device).div(6).add(0.5)
            box_placeholder = torch.clamp(box_placeholder, min=0, max=1)
            box_placeholder[:, 2:] = torch.clamp(box_placeholder[:, 2:], min=1e-4, max=1.0)
            box_placeholder = box_convert(box_placeholder, in_fmt='cxcywh', out_fmt='xyxy')
            box_placeholder = clip_boxes_to_image(box_placeholder, (1, 1))
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
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

        # back to absolute coordinates, and use xyxy format
        x_t = torch.clamp((x_t + 1) / 2, min=0, max=1)
        x_t[:, 2:] = torch.clamp(x_t[:, 2:], min=1e-4, max=1.0)
        x_t = box_convert(x_t, in_fmt='cxcywh', out_fmt='xyxy')
        x_t = clip_boxes_to_image(x_t, (1, 1))
        crpt_boxes = x_t * image_size
        return crpt_boxes, noise, t

    def model_predictions(self, backbone_feats, images_whwh, x, t):
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        x_boxes = box_convert(x_boxes, in_fmt='cxcywh', out_fmt='xyxy')
        x_boxes = x_boxes * images_whwh[:, None, :]
        outputs_class, outputs_coord, outputs_kernel, mask_feat = self.head(backbone_feats, x_boxes, t, None)

        # torch.Size([6, 1, 500, 80]), torch.Size([6, 1, 500, 153]), torch.Size([1, 8, 200, 304])
        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / images_whwh[:, None, :]
        x_start = box_convert(x_start, in_fmt='xyxy', out_fmt='cxcywh')
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = (self.sqrt_recip_alphas_cumprod[t] * x - x_start) / self.sqrt_recipm1_alphas_cumprod[t]

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_coord, outputs_kernel, mask_feat

    @torch.no_grad()
    def ddim_sample(self, batched_inputs, backbone_feats, images, do_postprocess=True):
        batch = len(batched_inputs)
        shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps = self.num_steps, self.sampling_steps
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        # tensor([ -1., 999.])
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        img = torch.randn(shape, device=self.device)
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            preds, outputs_class, outputs_coord, output_mask = self.model_predictions(backbone_feats, images_whwh, img,
                                                                                      time_cond)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start
            # box renewal
            score_per_image, box_per_image = outputs_class[-1][0], outputs_coord[-1][0]
            threshold = 0.5
            score_per_image = torch.sigmoid(score_per_image)
            value, _ = torch.max(score_per_image, -1, keepdim=False)
            keep_idx = value > threshold
            num_remain = torch.sum(keep_idx)

            pred_noise = pred_noise[:, keep_idx, :]
            x_start = x_start[:, keep_idx, :]
            img = img[:, keep_idx, :]

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            # replenish with randn boxes
            img = torch.cat((img, torch.randn(1, self.num_proposals - num_remain, 4, device=img.device)), dim=1)

        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        box_cls = output['pred_logits']
        box_pred = output['pred_boxes']

        results = self.inference(box_cls, box_pred, outputs_kernel[-1], mask_feat, images.image_sizes)
        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({'instances': r})
            return processed_results

    def inference(self, box_cls, box_pred, kernel, mask_feat, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes
            [1, 500, 153]
            [1, 8, 200, 304]

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        scores = torch.sigmoid(box_cls)
        labels = torch.arange(self.num_classes, device=self.device). \
            unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

        for i, (scores_per_image, box_pred_per_image, image_size, ker, mas) in enumerate(zip(
                scores, box_pred, image_sizes, kernel, mask_feat
        )):
            result = Instances(image_size)
            scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
            labels_per_image = labels[topk_indices]
            ker = ker.view(-1, 1, 153).repeat(1, self.num_classes, 1).view(-1, 153)
            # torch.Size([500, 4])

            box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
            # torch.Size([40000, 4])
            ker = ker[topk_indices]
            box_pred_per_image = box_pred_per_image[topk_indices]
            # torch.Size([500, 4])

            mask_logits = mask_logits.reshape(-1, 1, mas.size(1), mas.size(2)).squeeze(1).sigmoid()
            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            result.pred_masks = mask_logits
            results.append(result)
        return results