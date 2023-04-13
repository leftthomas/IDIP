import argparse
import glob
import multiprocessing as mp
import os
import time

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from tqdm import tqdm

from idip import add_idip_config


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else '__unused')
        self.instance_mode = instance_mode
        self.predictor = DefaultPredictor(cfg)
        self.threshold = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST

    def run_on_image(self, image):
        prediction = self.predictor(image)
        instances = prediction['instances']
        instances = instances[instances.scores > self.threshold].to('cpu')
        # convert image from OpenCV BGR format to Matplotlib RGB format
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return prediction, vis_output


def setup_cfg(arg):
    cfg = get_cfg()
    add_idip_config(cfg)
    cfg.merge_from_file(arg.config_file)
    cfg.merge_from_list(arg.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description='Detectron2 demo for builtin configs')
    parser.add_argument('--config-file', metavar='FILE', help='path to config file')
    parser.add_argument('--input', nargs='+', help='A list of space separated input images or a single glob '
                                                   'pattern such as directory/*.jpg')
    parser.add_argument('--output', help='A file or directory to save output visualizations')
    parser.add_argument('--opts', help='Modify config options using the command-line <KEY VALUE> pairs', default=[],
                        nargs=argparse.REMAINDER)
    return parser


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    args = get_parser().parse_args()
    setup_logger(name='fvcore')
    logger = setup_logger()
    logger.info('Arguments: ' + str(args))

    cfgs = setup_cfg(args)

    demo = VisualizationDemo(cfgs)

    if len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, 'The input path(s) was not found'
    for path in tqdm(args.input, disable=not args.output):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format='BGR')
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        logger.info('{}: {} in {:.2f}s'.format(path, 'detected {} instances'.format(len(predictions['instances']))
        if 'instances' in predictions else 'finished', time.time() - start_time))

        if os.path.isdir(args.output):
            out_filename = os.path.join(args.output, os.path.basename(path))
        else:
            assert len(args.input) == 1, 'Please specify a directory with args.output'
            out_filename = args.output
        visualized_output.save(out_filename)