import math
import torch
from detectron2.modeling.box_regression import Box2BoxTransform
from torch import nn
from torchvision.ops import box_convert, clip_boxes_to_image


def cosine_schedule(num_steps, s=0.008):
    # ref Improved Denoising Diffusion Probabilistic Models
    # note: must use float64 to avoid numerical error in ddim sample step
    t = torch.linspace(0, num_steps, num_steps + 1, dtype=torch.float64)
    f_t = torch.cos(((t / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_cumprod_t = f_t / f_t[0]
    beta_t = 1 - (alpha_cumprod_t[1:] / alpha_cumprod_t[:-1])
    alpha_t = 1 - torch.clamp(beta_t, min=0, max=0.999)
    return alpha_t


def normed_box_to_abs_box(normed_box, img_size):
    normed_box = torch.clamp((normed_box + 1) / 2, min=0, max=1)
    normed_box[:, 2:] = torch.clamp(normed_box[:, 2:], min=1e-4, max=1.0)
    normed_box = box_convert(normed_box, in_fmt='cxcywh', out_fmt='xyxy')
    normed_box = clip_boxes_to_image(normed_box, (1, 1))
    return normed_box * img_size


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
    def __init__(self, dim_hidden, num_heads, pooler_resolution):
        super().__init__()
        self.atte = nn.MultiheadAttention(dim_hidden, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim_hidden)
        # ref Sparse R-CNN: End-to-End Object Detection with Learnable Proposals
        self.num_params = dim_hidden ** 2 // 4
        self.linear1 = nn.Linear(dim_hidden, 2 * self.num_params)
        self.norm1 = nn.LayerNorm(dim_hidden // 4)
        self.norm2 = nn.LayerNorm(dim_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(dim_hidden * pooler_resolution ** 2, dim_hidden)
        self.norm3 = nn.LayerNorm(dim_hidden)
        # ref Denoising Diffusion Probabilistic Models
        self.time_emb = nn.Sequential(SinusoidalPositionEmbeddings(dim_hidden), nn.Linear(dim_hidden, dim_hidden * 4),
                                      nn.GELU(), nn.Linear(dim_hidden * 4, dim_hidden * 4))
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim_hidden * 4, dim_hidden * 2))

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
    def __init__(self, num_classes, dim_hidden):
        super().__init__()
        self.cls_layer = nn.Sequential(nn.Linear(dim_hidden, dim_hidden, False), nn.LayerNorm(dim_hidden),
                                       nn.ReLU(inplace=True))
        self.reg_layer = nn.Sequential(nn.Linear(dim_hidden, dim_hidden, False), nn.LayerNorm(dim_hidden),
                                       nn.ReLU(inplace=True))
        self.class_logits = nn.Linear(dim_hidden, num_classes)
        self.boxes_delta = nn.Linear(dim_hidden, 4)
        self.transform = Box2BoxTransform(weights=(2.0, 2.0, 1.0, 1.0))

    def forward(self, time_emb, boxes):
        # [B, N, D]
        b, n, d = time_emb.size()

        cls_feature = self.cls_layer(time_emb)
        reg_feature = self.reg_layer(time_emb)
        # [B, N, C]
        pred_logits = self.class_logits(cls_feature)
        # [B, N, 4]
        boxes_deltas = self.boxes_delta(reg_feature)
        pred_boxes = self.transform.apply_deltas(boxes_deltas.view(-1, 4), boxes.view(-1, 4)).view(b, -1, 4)
        return pred_logits, pred_boxes


class MaskHead(nn.Module):
    def __init__(self, num_classes, dim_hidden):
        super().__init__()
        self.mask_layer = nn.Sequential(nn.Conv2d(dim_hidden, 256, 3, padding=1), nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True), nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
                                        nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                        nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True), nn.Conv2d(128, num_classes, 1))

    def forward(self, time_emb, roi_features):
        # [B, N, D]
        b, n, d = time_emb.size()
        s = int(roi_features.shape[-1] ** 0.5)

        # [B*N, D, S, S]
        mask_feature = (roi_features + time_emb.unsqueeze(dim=-1)).view(b * n, d, s, -1)
        # [B, N, C, 2*S, 2*S]
        pred_masks = self.mask_layer(mask_feature).view(b, n, -1, 2 * s, 2 * s)
        return pred_masks