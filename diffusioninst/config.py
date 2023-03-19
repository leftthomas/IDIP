from detectron2.config import CfgNode as CN


def add_diffusioninst_config(cfg):
    cfg.MODEL.DiffusionInst = CN()
    cfg.MODEL.DiffusionInst.NUM_CLASSES = 80
    cfg.MODEL.DiffusionInst.NUM_PROPOSALS = 300

    # diffusion
    cfg.MODEL.DiffusionInst.NUM_STEPS = 1000
    cfg.MODEL.DiffusionInst.SAMPLING_STEPS = 1

    # attention
    cfg.MODEL.DiffusionInst.NUM_HEADS = 8

    # loss
    cfg.MODEL.DiffusionInst.CLASS_WEIGHT = 2.0
    cfg.MODEL.DiffusionInst.L1_WEIGHT = 5.0
    cfg.MODEL.DiffusionInst.GIOU_WEIGHT = 2.0
    cfg.MODEL.DiffusionInst.MASK_WEIGHT = 1.0

    # optimizer
    cfg.SOLVER.OPTIMIZER = 'ADAMW'
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0