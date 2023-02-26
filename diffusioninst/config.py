from detectron2.config import CfgNode as CN


def add_diffusioninst_config(cfg):
    """
    Add config for DiffusionInst
    """
    cfg.MODEL.DiffusionInst = CN()
    cfg.MODEL.DiffusionInst.NUM_CLASSES = 80
    cfg.MODEL.DiffusionInst.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.DiffusionInst.NHEADS = 8
    cfg.MODEL.DiffusionInst.DROPOUT = 0.0
    cfg.MODEL.DiffusionInst.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DiffusionInst.ACTIVATION = "relu"
    cfg.MODEL.DiffusionInst.HIDDEN_DIM = 256
    cfg.MODEL.DiffusionInst.NUM_CLS = 1
    cfg.MODEL.DiffusionInst.NUM_REG = 3
    cfg.MODEL.DiffusionInst.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.DiffusionInst.NUM_DYNAMIC = 2
    cfg.MODEL.DiffusionInst.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.DiffusionInst.CLASS_WEIGHT = 2.0
    cfg.MODEL.DiffusionInst.GIOU_WEIGHT = 2.0
    cfg.MODEL.DiffusionInst.L1_WEIGHT = 5.0
    cfg.MODEL.DiffusionInst.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.DiffusionInst.USE_FOCAL = True
    cfg.MODEL.DiffusionInst.ALPHA = 0.25
    cfg.MODEL.DiffusionInst.GAMMA = 2.0
    cfg.MODEL.DiffusionInst.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.DiffusionInst.OTA_K = 5

    # Diffusion
    cfg.MODEL.DiffusionInst.SNR_SCALE = 2.0

    # Inference
    cfg.MODEL.DiffusionInst.USE_NMS = True

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0