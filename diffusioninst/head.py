import math
import torch
from detectron2.modeling.box_regression import Box2BoxTransform
from mmdet.core.bbox import bbox2roi
from mmdet.models.roi_heads import SparseRoIHead
from torch import nn
from torchvision.ops import box_convert, clip_boxes_to_image


# ref Improved Denoising Diffusion Probabilistic Models
def cosine_schedule(num_steps, s=0.008):
    # note: must use float64 to avoid numerical error in ddim sample step
    t = torch.linspace(0, num_steps, num_steps + 1, dtype=torch.float64)
    f_t = torch.cos(((t / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_cumprod_t = f_t / f_t[0]
    beta_t = 1 - (alpha_cumprod_t[1:] / alpha_cumprod_t[:-1])
    alpha_t = 1 - torch.clamp(beta_t, min=0, max=0.999)
    return alpha_t


def normed_box_to_abs_box(normed_box, img_size=None):
    normed_box = torch.clamp((normed_box + 1) / 2, min=0, max=1)
    normed_box[:, 2:] = torch.clamp(normed_box[:, 2:], min=1e-4, max=1.0)
    normed_box = box_convert(normed_box, in_fmt='cxcywh', out_fmt='xyxy')
    normed_box = clip_boxes_to_image(normed_box, (1, 1))
    return normed_box if img_size is None else normed_box * img_size


# ref TENER: Adapting Transformer Encoder for Named Entity Recognition
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim_hidden):
        super().__init__()
        self.dim_hidden = dim_hidden

    def forward(self, time):
        half_dim = self.dim_hidden // 2
        embeddings = math.log(10000) / half_dim
        embeddings = torch.exp(torch.arange(half_dim, device=time.device).mul(-embeddings))
        embeddings = time * embeddings[None, :]
        # [B, D]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ref Denoising Diffusion Probabilistic Models
class TimeEncoder(nn.Module):
    def __init__(self, dim_hidden):
        super().__init__()
        self.time_emb = nn.Sequential(SinusoidalPositionEmbeddings(dim_hidden), nn.Linear(dim_hidden, dim_hidden * 4),
                                      nn.GELU(), nn.Linear(dim_hidden * 4, dim_hidden * 4))

    def forward(self, ts):
        time_emb = self.time_emb(ts)
        return time_emb


class DiffusionRoiHead(SparseRoIHead):
    def __init__(self, dim_hidden, strides, num_classes):
        # ref Sparse R-CNN: End-to-End Object Detection with Learnable Proposals
        roi_extractor = dict(type='SingleRoIExtractor',
                             roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
                             out_channels=dim_hidden, featmap_strides=strides)
        bbox_head = dict(type='DIIHead', num_classes=num_classes, in_channels=dim_hidden, dynamic_conv_cfg=dict(
            type='DynamicConv', in_channels=dim_hidden, input_feat_shape=7),
                         loss_cls=dict(type='FocalLoss', use_sigmoid=True))
        mask_head = dict(type='DynamicMaskHead', in_channels=dim_hidden, roi_feat_size=7, conv_out_channels=dim_hidden,
                         num_classes=num_classes, dynamic_conv_cfg=dict(type='DynamicConv', in_channels=dim_hidden,
                                                                        input_feat_shape=7, with_proj=False))
        super().__init__(proposal_feature_channel=dim_hidden, bbox_roi_extractor=roi_extractor,
                         bbox_head=bbox_head, mask_head=mask_head)
        self.transform = Box2BoxTransform(weights=(2.0, 2.0, 1.0, 1.0))
        self.mlps = nn.ModuleList(
            nn.Sequential(nn.SiLU(), nn.Linear(dim_hidden * 4, dim_hidden * 2)) for _ in range(self.num_stages))

    def forward(self, features, boxes, time_emb):
        b, n, _ = boxes.size()
        d = self.proposal_feature_channel
        results = []
        proposals = [boxes[i] for i in range(b)]
        object_feats = None
        for stage in range(self.num_stages):
            # [B*N, 5]
            rois = bbox2roi(proposals)
            roi_extractor = self.bbox_roi_extractor[stage]
            bbox_head = self.bbox_head[stage]
            # [B*N, D, S, S]
            feats = roi_extractor(features, rois)
            object_feats = feats.reshape(b, n, d, -1).mean(-1) if object_feats is None else object_feats

            scale_shift = self.mlps[stage](time_emb)
            # [B, 1, D]
            scale, shift = scale_shift.unsqueeze(dim=1).chunk(2, dim=-1)
            # [B, N, D]
            object_feats = object_feats * (scale + 1) + shift

            # [B, N, C], [B, N, 4], [B, N, D], [B, N, D]
            cls_score, box_delta, object_feats, attn_feats = bbox_head(feats, object_feats)
            # [B*N, 4]
            pred_box = self.transform.apply_deltas(box_delta.reshape(-1, 4), rois[:, 1:])
            proposals = torch.tensor_split(pred_box, b)

            mask_head = self.mask_head[stage]
            # [B*N, C, 2*S, 2*S]
            pred_mask = mask_head(feats, attn_feats)
            results.append({'pred_logits': cls_score, 'pred_boxes': pred_box.reshape(b, n, -1),
                            'pred_masks': pred_mask.reshape(b, n, *pred_mask.shape[1:])})
        return results