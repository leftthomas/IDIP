import math
import torch
from detectron2.modeling.box_regression import Box2BoxTransform
from torch import nn


def cosine_schedule(num_steps, s=0.008):
    # ref Improved Denoising Diffusion Probabilistic Models
    t = torch.linspace(0, num_steps, num_steps + 1)
    f_t = torch.cos(((t / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_cumprod_t = f_t / f_t[0]
    beta_t = 1 - (alpha_cumprod_t[1:] / alpha_cumprod_t[:-1])
    return torch.clamp(beta_t, min=0, max=0.999)


# ref TENER: Adapting Transformer Encoder for Named Entity Recognition
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, time):
        half_dim = self.hidden_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device).mul(-embeddings))
        embeddings = time * embeddings[None, :]
        # [B, D]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DynamicBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, pooler_resolution):
        super().__init__()
        self.atte = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        # ref Sparse R-CNN: End-to-End Object Detection with Learnable Proposals
        self.num_params = hidden_dim ** 2 // 4
        self.linear1 = nn.Linear(hidden_dim, 2 * self.num_params)
        self.norm1 = nn.LayerNorm(hidden_dim // 4)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_dim * pooler_resolution ** 2, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        # ref Denoising Diffusion Probabilistic Models
        self.time_emb = nn.Sequential(SinusoidalPositionEmbeddings(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 4),
                                      nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim * 4))
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim * 4, hidden_dim * 2))

    def forward(self, roi_features, ts):
        # [B, N, D]
        pro_features = roi_features.mean(-1)
        atte_features = self.atte(pro_features, pro_features, pro_features)[0]
        mixed_features = self.norm(pro_features + atte_features)

        b, n, d = mixed_features.size()
        parameters = self.linear1(mixed_features)
        param1 = parameters[:, :, :self.num_params].view(b, n, d, d // 4)
        param2 = parameters[:, :, self.num_params:].view(b, n, d // 4, d)
        # [B, N, S*S, D]
        roi_features = roi_features.permute(0, 1, 3, 2)
        obj_features = self.relu(self.norm1(torch.matmul(roi_features, param1)))
        obj_features = self.relu(self.norm2(torch.matmul(obj_features, param2)))
        # [B, N, D]
        obj_features = self.relu(self.norm3(self.linear2(obj_features.flatten(start_dim=-2))))

        time_emb = self.time_emb(ts)
        # [B, 2*D]
        scale_shift = self.time_mlp(time_emb)
        # [B, 1, D]
        scale, shift = scale_shift.unsqueeze(dim=1).chunk(2, dim=-1)
        # [B, N, D]
        fc_feature = obj_features * (scale + 1) + shift
        return fc_feature


class DetectHead(nn.Module):
    def __init__(self, cfg, hidden_dim):
        super().__init__()
        num_classes = cfg.MODEL.DiffusionInst.NUM_CLASSES
        num_heads = cfg.MODEL.DiffusionInst.NUM_HEADS
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        self.dynamic_block = DynamicBlock(hidden_dim, num_heads, pooler_resolution)

        self.cls_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, False), nn.LayerNorm(hidden_dim),
                                       nn.ReLU(inplace=True))
        self.reg_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, False), nn.LayerNorm(hidden_dim),
                                       nn.ReLU(inplace=True))
        self.class_logits = nn.Linear(hidden_dim, num_classes)
        self.boxes_delta = nn.Linear(hidden_dim, 4)
        self.transform = Box2BoxTransform(weights=(2.0, 2.0, 1.0, 1.0))

    def forward(self, roi_features, ts, boxes):
        # [B, N, D]
        fc_feature = self.dynamic_block(roi_features, ts)
        b, n, d = fc_feature.size()

        cls_feature = self.cls_layer(fc_feature)
        reg_feature = self.reg_layer(fc_feature)
        # [B, N, C]
        pred_logits = torch.sigmoid(self.class_logits(cls_feature))
        # [B, N, 4]
        boxes_deltas = self.boxes_delta(reg_feature)
        pred_boxes = self.transform.apply_deltas(boxes_deltas.view(-1, 4), boxes.view(-1, 4)).view(b, -1, 4)
        return pred_logits, pred_boxes


class MaskHead(nn.Module):
    def __init__(self, cfg, hidden_dim):
        super().__init__()
        num_heads = cfg.MODEL.DiffusionInst.NUM_HEADS
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        self.num_proposals = cfg.MODEL.DiffusionInst.NUM_PROPOSALS
        self.dynamic_block = DynamicBlock(hidden_dim, num_heads, pooler_resolution)

        self.mask_layer = nn.Sequential(nn.Conv2d(hidden_dim, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                        nn.Conv2d(128, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8),
                                        nn.ReLU(inplace=True), nn.Conv2d(8, 8, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(8), nn.ReLU(inplace=True), nn.Conv2d(8, 1, 1))

    def forward(self, roi_features, ts):
        # [B, N, D]
        fc_feature = self.dynamic_block(roi_features, ts)
        b, n, d = fc_feature.size()
        s = int(roi_features.shape[-1] ** 0.5)

        # [B*N, D, S, S]
        mask_feature = (roi_features + fc_feature.unsqueeze(dim=-1)).view(b * n, d, s, -1)
        # [B, N, S, S]
        pred_masks = torch.sigmoid(self.mask_layer(mask_feature).view(b, n, -1, s))
        return pred_masks