import random
from typing import List

import torch
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes, ImageList, Instances
from torch import nn
from torchvision.ops import box_convert

from .head import DetectHead, MaskHead, cosine_schedule
from .loss import SetCriterion


@META_ARCH_REGISTRY.register()
class DiffusionInst(nn.Module):
    """
    Implement DiffusionInst
    """

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
        self.register_buffer("betas", cosine_schedule(self.num_steps))
        self.register_buffer("alphas", 1 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - self.alphas_cumprod))

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

        # loss parameters
        class_weight = cfg.MODEL.DiffusionInst.CLASS_WEIGHT
        giou_weight = cfg.MODEL.DiffusionInst.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DiffusionInst.L1_WEIGHT
        no_object_weight = cfg.MODEL.DiffusionInst.NO_OBJECT_WEIGHT

        # # build criterion
        # matcher = HungarianMatcherDynamicK(
        #     cfg=cfg, cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight, use_focal=self.use_focal
        # )
        self.weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
        self.criterion = SetCriterion(cfg=cfg, num_classes=self.num_classes, eos_coef=no_object_weight)

        self.register_buffer("pixel_mean", torch.as_tensor(cfg.MODEL.PIXEL_MEAN).view(3, 1, 1))
        self.register_buffer("pixel_std", torch.as_tensor(cfg.MODEL.PIXEL_STD).view(3, 1, 1))
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
        """
        images = self.preprocess_image(batched_inputs)
        # feature extraction
        src = self.backbone(images.tensor)
        features = [src[f] for f in self.in_features]

        if self.training:
            targets, boxes, pool_boxes, noises, ts = self.preprocess_target(batched_inputs)
            roi_features = torch.flatten(self.pooler(features, pool_boxes), start_dim=-2)
            pred_logits, pred_boxes = self.detect_head(roi_features, ts, boxes)
            pred_masks = self.mask_head(roi_features, features)
            pred_logits = pred_logits.view(-1, self.num_proposals, self.num_classes)
            pred_boxes = pred_boxes.view(-1, self.num_proposals, 4)
            output = {'pred_logits': pred_logits, 'pred_boxes': pred_boxes, 'pred_masks': pred_masks}
            loss_dict = self.criterion(output, targets)
            for k in loss_dict.keys():
                loss_dict[k] *= self.weight_dict[k]
            return loss_dict

        # prepare proposals
        if not self.training:
            results = self.ddim_sample(batched_inputs, features, images)
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [(x["image"].to(self.device) - self.pixel_mean) / self.pixel_std for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)
        return images

    def preprocess_target(self, batched_inputs):
        targets, diffused_boxes, pool_boxes, noises, ts = [], [], [], [], []
        for x in batched_inputs:
            instances = x["instances"].to(self.device)
            gt_classes = instances.gt_classes
            gt_masks = instances.gt_masks.tensor

            h, w = instances.image_size
            image_size = torch.as_tensor([w, h, w, h], device=self.device).view(1, 4)
            gt_boxes = instances.gt_boxes.tensor

            crpt_boxes, noise, t = self.prepare_diffusion(
                box_convert(gt_boxes / image_size, in_fmt="xyxy", out_fmt="cxcywh"))
            crpt_boxes = crpt_boxes * image_size
            diffused_boxes.append(crpt_boxes)
            pool_boxes.append(Boxes(crpt_boxes))
            noises.append(noise)
            ts.append(t)

            targets.append({"gt_classes": gt_classes, "gt_boxes": gt_boxes, "gt_masks": gt_masks})

        return targets, torch.stack(diffused_boxes), pool_boxes, torch.stack(noises), torch.stack(ts).squeeze(-1)

    def prepare_diffusion(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        """
        # gt box padding
        num_gt = len(gt_boxes)
        if num_gt < self.num_proposals:
            # ref DiffusionDet: Diffusion Model for Object Detection
            box_placeholder = torch.randn(self.num_proposals - num_gt, 4, device=self.device) / 6 + 0.5
            box_placeholder[:, 2:] = torch.clamp(box_placeholder[:, 2:], min=1e-4, max=1)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes

        x_start = x_start * 2 - 1

        # noise sample
        t = torch.randint(0, self.num_steps, (1,), device=self.device)
        noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        x_t = torch.clamp(x_t, min=-1, max=1)

        crpt_boxes = (x_t + 1) / 2
        return crpt_boxes, noise, t

    def predict_noise_from_start(self, x_t, t, x0):
        return ((extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))

    def model_predictions(self, backbone_feats, images_whwh, x, t, x_self_cond=None, clip_x_start=False):
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        x_boxes = _box_cxcywh_to_xyxy(x_boxes)
        x_boxes = x_boxes * images_whwh[:, None, :]
        outputs_class, outputs_coord, outputs_kernel, mask_feat = self.head(backbone_feats, x_boxes, t, None)

        # torch.Size([6, 1, 500, 80]), torch.Size([6, 1, 500, 153]), torch.Size([1, 8, 200, 304])
        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / images_whwh[:, None, :]
        x_start = _box_xyxy_to_cxcywh(x_start)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_coord, outputs_kernel, mask_feat

    @torch.no_grad()
    def ddim_sample(self, batched_inputs, backbone_feats, images, clip_denoised=True, do_postprocess=True):
        batch = images_whwh.shape[0]
        shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta, objective = self.num_steps, self.sampling_steps, self.ddim_sampling_eta, self.objective
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        # tensor([ -1., 999.])
        times = list(reversed(times.int().tolist()))

        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=self.device)

        ensemble_score, ensemble_label, ensemble_coord, ensemble_kernel = [], [], [], []
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            preds, outputs_class, outputs_coord, outputs_kernel, mask_feat = self.model_predictions(backbone_feats,
                                                                                                    images_whwh, img,
                                                                                                    time_cond,
                                                                                                    self_cond,
                                                                                                    clip_x_start=clip_denoised)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if self.box_renewal:  # filter
                # true
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

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if self.box_renewal:  # filter
                # replenish with randn boxes
                img = torch.cat((img, torch.randn(1, self.num_proposals - num_remain, 4, device=img.device)), dim=1)

        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        box_cls = output["pred_logits"]
        box_pred = output["pred_boxes"]

        results = self.inference(box_cls, box_pred, outputs_kernel[-1], mask_feat, images.image_sizes)
        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
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