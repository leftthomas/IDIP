import math
import torch
import torch.nn.functional as F
from detectron2.structures import Instances


def cosine_schedule(num_steps, s=0.008):
    """
    as proposed in Improved Denoising Diffusion Probabilistic Models
    """
    t = torch.linspace(0, num_steps, num_steps + 1)
    f_t = torch.cos(((t / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_cumprod_t = f_t / f_t[0]
    beta_t = 1 - (alpha_cumprod_t[1:] / alpha_cumprod_t[:-1])
    return torch.clip(beta_t, 0, 0.999)


def apply_deltas(deltas, boxes, bbox_weights=(2.0, 2.0, 1.0, 1.0), scale_clamp=math.log(100000 / 16)):
    """
    Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

    Args:
        deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
            deltas[i] represents k potentially different class-specific
            box transformations for the single box boxes[i].
        boxes (Tensor): boxes to transform, of shape (N, 4)
    """
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = bbox_weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=scale_clamp)
    dh = torch.clamp(dh, max=scale_clamp)

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    pred_boxes = torch.zeros_like(deltas)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

    return pred_boxes


def aligned_bilinear(tensor, factor):
    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(tensor, size=(oh, ow), mode='bilinear', align_corners=True)
    tensor = F.pad(tensor, pad=(factor // 2, 0, factor // 2, 0), mode="replicate")
    return tensor[:, :, :oh - 1, :ow - 1]


def clip_mask(Boxes, masks):
    boxes = Boxes.tensor.long()
    assert (len(boxes) == len(masks))
    m_out = []
    k = torch.zeros(masks[0].size()).long().to(boxes.device)
    for i in range(len(masks)):
        mask = masks[i]
        box = boxes[i]
        k[box[1]:box[3], box[0]:box[2]] = 1
        mask *= k
        m_out.append(mask)
        k *= 0
    return torch.stack(m_out)


def resize_instance(results: Instances, output_height: int, output_width: int, mid_size):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.
    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.
    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.
    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    if isinstance(output_width, torch.Tensor):
        # This shape might (but not necessarily) be tensors during tracing.
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)
    masks = results.pred_masks

    pred_global_masks = aligned_bilinear(masks.unsqueeze(1), 8)
    pred_global_masks = pred_global_masks[:, :, :mid_size[0], :mid_size[1]]
    masks = F.interpolate(
        pred_global_masks,
        size=(new_size[0], new_size[1]),
        mode='bilinear',
        align_corners=False).squeeze(1)
    #################################################
    masks.gt_(0.5)
    masks = clip_mask(output_boxes, masks)
    results.pred_masks = masks
    results = results[output_boxes.nonempty()]

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d