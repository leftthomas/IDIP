import math
import torch
import torch.nn.functional as F
from detectron2.modeling.box_regression import Box2BoxTransform
from torch import nn


def cosine_schedule(num_steps, s=0.008):
    # ref Improved Denoising Diffusion Probabilistic Models
    t = torch.linspace(0, num_steps, num_steps + 1)
    f_t = torch.cos(((t / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_cumprod_t = f_t / f_t[0]
    beta_t = 1 - (alpha_cumprod_t[1:] / alpha_cumprod_t[:-1])
    return torch.clip(beta_t, 0, 0.999)


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    num_instances = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(
        torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(
                num_instances * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_instances * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(
                num_instances * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_instances)
    return weight_splits, bias_splits


class DynamicGuide(nn.Module):
    def __init__(self, cfg, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_proposals = cfg.MODEL.DiffusionInst.NUM_PROPOSALS
        self.atte = nn.MultiheadAttention(hidden_dim, cfg.MODEL.DiffusionInst.NUM_HEADS, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, roi_features):
        pro_features = roi_features.view(-1, self.num_proposals, self.hidden_dim, roi_features.shape[-1]).mean(-1)
        atte_features = self.atte(pro_features, pro_features, pro_features)[0]
        return self.norm(pro_features + atte_features).view(-1, self.hidden_dim)


class DynamicConv(nn.Module):
    def __init__(self, cfg, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_params = self.hidden_dim ** 2 // 4
        self.linear1 = nn.Linear(hidden_dim, 2 * self.num_params)
        self.norm1 = nn.LayerNorm(hidden_dim // 4)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        self.linear2 = nn.Linear(self.hidden_dim * pooler_resolution ** 2, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, guide_features, roi_features):
        parameters = self.linear1(guide_features)
        param1 = parameters[:, :self.num_params].view(-1, self.hidden_dim, self.hidden_dim // 4)
        param2 = parameters[:, self.num_params:].view(-1, self.hidden_dim // 4, self.hidden_dim)

        roi_features = roi_features.permute(0, 2, 1)
        features = self.relu(self.norm1(torch.bmm(roi_features, param1)))
        features = self.relu(self.norm2(torch.bmm(features, param2)))
        features = self.relu(self.norm3(self.linear2(features.flatten(start_dim=1))))
        return features


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, time):
        half_dim = self.hidden_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DetectHead(nn.Module):
    def __init__(self, cfg, hidden_dim):
        super().__init__()
        num_classes = cfg.MODEL.DiffusionInst.NUM_CLASSES
        self.dynamic_guide = DynamicGuide(cfg, hidden_dim)
        self.dynamic_conv = DynamicConv(cfg, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.time_emb = nn.Sequential(SinusoidalPositionEmbeddings(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 4),
                                      nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim * 4))
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim * 4, hidden_dim * 2))
        self.cls_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, False), nn.LayerNorm(hidden_dim),
                                       nn.ReLU(inplace=True))
        self.reg_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, False), nn.LayerNorm(hidden_dim),
                                       nn.ReLU(inplace=True))
        self.class_logits = nn.Linear(hidden_dim, num_classes)
        self.boxes_delta = nn.Linear(hidden_dim, 4)
        self.transform = Box2BoxTransform(weights=(2.0, 2.0, 1.0, 1.0))

    def forward(self, roi_features, ts, boxes):
        detect_features = self.dynamic_guide(roi_features)
        mixed_features = self.dynamic_conv(detect_features, roi_features)
        obj_features = self.norm1(detect_features + mixed_features)
        obj_features = self.norm2(obj_features + self.linear2(self.relu(self.linear1(obj_features))))
        time_emb = self.time_emb(ts)
        scale_shift = self.time_mlp(time_emb)
        scale_shift = torch.repeat_interleave(scale_shift, len(obj_features) // len(scale_shift), dim=0)
        scale, shift = scale_shift.chunk(2, dim=1)
        fc_feature = obj_features * (scale + 1) + shift

        cls_feature = self.cls_layer(fc_feature)
        reg_feature = self.reg_layer(fc_feature)
        pred_logits = torch.sigmoid(self.class_logits(cls_feature))
        boxes_deltas = self.boxes_delta(reg_feature)
        pred_boxes = self.transform.apply_deltas(boxes_deltas, boxes.view(-1, 4))
        return pred_logits, pred_boxes


class MaskHead(nn.Module):
    def __init__(self, cfg, hidden_dim):
        super().__init__()
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_proposals = cfg.MODEL.DiffusionInst.NUM_PROPOSALS
        self.mask_refine = nn.ModuleList([nn.Sequential(nn.Conv2d(hidden_dim, 128, 3, stride=1, padding=1, bias=False),
                                                        nn.BatchNorm2d(128), nn.ReLU()) for _ in in_features])
        mask_tower = [nn.Sequential(nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128),
                                    nn.ReLU()) for _ in range(4)]
        mask_tower.append(nn.Conv2d(128, 8, kernel_size=1, stride=1))
        self.mask_tower = nn.Sequential(*mask_tower)
        self.dynamic_guide = DynamicGuide(cfg, hidden_dim)
        self.dynamic_conv = DynamicConv(cfg, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.weight_nums = [64, 64, 8]
        self.bias_nums = [8, 8, 1]
        self.controller = nn.Linear(hidden_dim, sum(self.weight_nums) + sum(self.bias_nums))

    def forward(self, roi_features, features):
        for i, x in enumerate(features):
            if i == 0:
                mask_feat = self.mask_refine[i](x)
                target_h, target_w = mask_feat.size()[2:]
            else:
                x_p = self.mask_refine[i](x)
                x_p = F.interpolate(x_p, size=(target_h, target_w), mode='bilinear', align_corners=True)
                mask_feat = mask_feat + x_p
        mask_feat = self.mask_tower(mask_feat)

        mask_features = self.dynamic_guide(roi_features)
        mixed_features = self.dynamic_conv(mask_features, roi_features)
        obj_features = self.norm1(mask_features + mixed_features)
        obj_features = self.norm2(obj_features + self.linear2(self.relu(self.linear1(obj_features))))
        mask_head_params = self.controller(obj_features)
        weights, biases = parse_dynamic_params(mask_head_params, 8, self.weight_nums, self.bias_nums)
        mask_feat_head = mask_feat.unsqueeze(0).repeat(1, self.num_proposals, 1, 1)

        n_layers = len(weights)
        x = mask_feat_head
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=self.num_proposals)
            if i < n_layers - 1:
                x = F.relu(x)
        mask_logits = x
        pred_masks = mask_logits.reshape(-1, 1, mask_feat.size(1), mask_feat.size(2)).squeeze(1).sigmoid()
        return pred_masks