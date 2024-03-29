import itertools
import logging
import os
import weakref
from typing import Any, Dict, List, Set

import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, create_ddp_model, \
    SimpleTrainer
from detectron2.evaluation import COCOEvaluator, verify_results, CityscapesInstanceEvaluator
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from idip import add_idip_config


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger('detectron2')
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = SimpleTrainer(model, data_loader, optimizer)
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, trainer=weakref.proxy(self))
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, 'inference')
        if 'coco' in dataset_name:
            return COCOEvaluator(dataset_name, output_dir=output_folder)
        elif 'cityscapes' in dataset_name:
            return CityscapesInstanceEvaluator(dataset_name)
        else:
            raise NotImplementedError('{} not supported yet'.format(dataset_name))

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if 'backbone' in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == 'full_model'
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x['params'] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == 'SGD':
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == 'ADAMW':
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError('no optimizer type {}'.format(optimizer_type))
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == 'full_model':
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def setup(arg):
    cfg = get_cfg()
    add_idip_config(cfg)
    cfg.merge_from_file(arg.config_file)
    cfg.merge_from_list(arg.opts)
    cfg.freeze()
    default_setup(cfg, arg)
    return cfg


def main(arg):
    cfg = setup(arg)

    if arg.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=arg.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=arg.resume)
    return trainer.train()


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    print('Command Line Args:', args)
    launch(main, args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank, dist_url=args.dist_url,
           args=(args,), )