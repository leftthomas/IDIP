from detectron2.config import CfgNode


def add_idip_config(cfg):
    cfg.MODEL.IDIP = CfgNode()
    cfg.MODEL.IDIP.NUM_CLASSES = 80
    cfg.MODEL.IDIP.NUM_PROPOSALS = 300

    # diffusion
    cfg.MODEL.IDIP.NUM_STEPS = 1000
    cfg.MODEL.IDIP.SAMPLING_TYPE = 'DDIM'
    cfg.MODEL.IDIP.SAMPLING_STEPS = 1
    cfg.MODEL.IDIP.WITH_DYNAMIC = True

    # loss
    cfg.MODEL.IDIP.CLS_WEIGHT = 2.0
    cfg.MODEL.IDIP.L1_WEIGHT = 5.0
    cfg.MODEL.IDIP.GIOU_WEIGHT = 2.0
    cfg.MODEL.IDIP.MASK_WEIGHT = 5.0

    # optimizer
    cfg.SOLVER.OPTIMIZER = 'ADAMW'
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0