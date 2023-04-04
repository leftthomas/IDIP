from detectron2.config import CfgNode


def add_diffusioninst_config(cfg):
    cfg.MODEL.DiffusionInst = CfgNode()
    cfg.MODEL.DiffusionInst.NUM_CLASSES = 80
    cfg.MODEL.DiffusionInst.NUM_PROPOSALS = 300

    # diffusion
    cfg.MODEL.DiffusionInst.NUM_STEPS = 1000
    cfg.MODEL.DiffusionInst.SAMPLING_TYPE = 'DDIM'
    cfg.MODEL.DiffusionInst.SAMPLING_STEPS = 1
    cfg.MODEL.DiffusionInst.WITH_DYNAMIC = True

    # loss
    cfg.MODEL.DiffusionInst.CLS_WEIGHT = 2.0
    cfg.MODEL.DiffusionInst.L1_WEIGHT = 5.0
    cfg.MODEL.DiffusionInst.GIOU_WEIGHT = 2.0
    cfg.MODEL.DiffusionInst.MASK_WEIGHT = 5.0
    cfg.MODEL.DiffusionInst.WITH_MASK = False

    # optimizer
    cfg.SOLVER.OPTIMIZER = 'ADAMW'
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0