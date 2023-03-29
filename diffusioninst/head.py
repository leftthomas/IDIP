import copy

import math
import torch
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from mmdet.models.utils import DynamicConv
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


# ref Sparse R-CNN: End-to-End Object Detection with Learnable Proposals
class BoxHead(nn.Module):
    def __init__(self, dim_hidden, num_classes, box_size, num_heads=8, num_reg_fcs=3, feedforward_channels=2048):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim_hidden, num_heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(dim_hidden)

        self.instance_conv = DynamicConv(dim_hidden, input_feat_shape=box_size)
        self.instance_norm = nn.LayerNorm(dim_hidden)

        self.ffn = nn.Sequential(nn.Linear(dim_hidden, feedforward_channels), nn.ReLU(inplace=True),
                                 nn.Linear(feedforward_channels, dim_hidden))
        self.ffn_norm = nn.LayerNorm(dim_hidden)

        self.cls_fcs = nn.Sequential(nn.Linear(dim_hidden, dim_hidden, bias=False), nn.LayerNorm(dim_hidden),
                                     nn.ReLU(inplace=True))
        self.fc_cls = nn.Linear(dim_hidden, num_classes)

        self.reg_fcs = nn.Sequential(*[nn.Sequential(nn.Linear(dim_hidden, dim_hidden, bias=False),
                                                     nn.LayerNorm(dim_hidden), nn.ReLU(inplace=True)) for _
                                       in range(num_reg_fcs)])
        self.fc_reg = nn.Linear(dim_hidden, 4)

        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim_hidden * 4, dim_hidden * 2))

    def forward(self, roi_feat, proposal_feat, time_emb):
        b, n, d = proposal_feat.size()

        # self attention
        attend_feat = self.attention(proposal_feat, proposal_feat, proposal_feat)[0]
        proposal_feat = self.attention_norm(proposal_feat + attend_feat)

        # instance interactive
        proposal_feat = proposal_feat.reshape(-1, d)
        iic_feat = self.instance_conv(proposal_feat, roi_feat)
        obj_feat = self.instance_norm(proposal_feat + iic_feat)

        # ffn
        ffn_feat = self.ffn(obj_feat)
        obj_feat = self.ffn_norm(obj_feat + ffn_feat).reshape(b, n, -1)

        scale_shift = self.time_mlp(time_emb)
        # [B, 1, D], [B, 1, D]
        scale, shift = scale_shift.unsqueeze(dim=1).chunk(2, dim=-1)
        # [B, N, D]
        fc_feat = obj_feat * (scale + 1) + shift

        cls_feat = self.cls_fcs(fc_feat)
        reg_feat = self.reg_fcs(fc_feat)

        cls_logit = self.fc_cls(cls_feat)
        box_delta = self.fc_reg(reg_feat)

        return cls_logit, box_delta, obj_feat


# ref Instances as Queries
class MaskHead(nn.Module):
    def __init__(self, dim_hidden, num_classes, mask_size, num_convs=4):
        super().__init__()
        self.instance_conv = DynamicConv(dim_hidden, input_feat_shape=mask_size, with_proj=False)
        self.convs = nn.Sequential(*[nn.Conv2d(dim_hidden, dim_hidden, 3, padding=1) for _ in range(num_convs)])
        self.upsample = nn.ConvTranspose2d(dim_hidden, dim_hidden, kernel_size=2, stride=2)
        self.logits = nn.Conv2d(dim_hidden, num_classes, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, roi_feat, proposal_feat):
        b, n, d = proposal_feat.size()
        proposal_feat = proposal_feat.reshape(-1, d)
        iic_feat = self.instance_conv(proposal_feat, roi_feat)
        # [B*N, D, 2*S, 2*S]
        x = iic_feat.permute(0, 2, 1).reshape(roi_feat.size())
        x = self.relu(self.upsample(self.convs(x)))
        # [B, N, C, 2*S, 2*S]
        pred_mask = self.logits(x).reshape(b, n, -1, *x.size()[-2:])
        return pred_mask


class DiffusionRoiHead(nn.Module):
    def __init__(self, dim_hidden, strides, num_classes, box_size, mask_size, box_ratio, mask_ratio, box_type,
                 mask_type, num_stages=6):
        super().__init__()
        self.num_stages = num_stages
        self.num_classes = num_classes
        self.box_extractor = ROIPooler(box_size, strides, box_ratio, box_type)
        self.mask_extractor = ROIPooler(mask_size, strides, mask_ratio, mask_type)
        box_head = BoxHead(dim_hidden, num_classes, box_size)
        self.box_heads = nn.ModuleList([copy.deepcopy(box_head) for _ in range(num_stages)])
        mask_head = MaskHead(dim_hidden, num_classes, mask_size)
        self.mask_heads = nn.ModuleList([copy.deepcopy(mask_head) for _ in range(num_stages)])
        self.transform = Box2BoxTransform(weights=(2.0, 2.0, 1.0, 1.0))
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            # initialize the bias for focal loss
            if p.shape[-1] == self.num_classes:
                nn.init.constant_(p, -math.log((1 - 0.01) / 0.01))

    def forward(self, features, boxes, time_emb):
        b, n, _ = boxes.size()
        results, proposal_feat = [], None
        for stage in range(self.num_stages):
            proposals = [Boxes(b) for b in boxes]
            # [B*N, D, S, S], [B*N, D, 2*S, 2*S]
            roi_box_feat = self.box_extractor(features, proposals)
            roi_mask_feat = self.mask_extractor(features, proposals)

            # [B, N, D]
            proposal_feat = torch.flatten(roi_box_feat, start_dim=-2).mean(-1).reshape(b, n, -1) \
                if proposal_feat is None else proposal_feat
            # [B, N, C], [B, N, 4], [B, N, D]
            pred_logit, pred_delta, proposal_feat = self.box_heads[stage](roi_box_feat, proposal_feat, time_emb)
            # [B, N, 4]
            pred_box = self.transform.apply_deltas(pred_delta.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(b, n, -1)
            boxes = pred_box.detach()

            # [B, N, C, 2*S, 2*S]
            pred_mask = self.mask_heads[stage](roi_mask_feat, proposal_feat)
            results.append({'pred_logits': pred_logit, 'pred_boxes': pred_box, 'pred_masks': pred_mask})
        return results