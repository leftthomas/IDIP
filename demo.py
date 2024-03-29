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
    def __init__(self, cfg):
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else '__unused')
        self.predictor = DefaultPredictor(cfg)
        self.threshold = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST

    def run_on_image(self, image):
        prediction = self.predictor(image)
        instances = prediction['instances']
        instances = instances[instances.scores > self.threshold].to('cpu')
        # convert image from OpenCV BGR format to Matplotlib RGB format
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=ColorMode.IMAGE)
        vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return prediction, vis_output


def setup_cfg(arg):
    cfg = get_cfg()
    add_idip_config(cfg)
    cfg.merge_from_file(arg.config_file)
    cfg.merge_from_list(arg.opts)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description='Detectron2 demo for builtin configs')
    parser.add_argument('--config-file', metavar='FILE', help='path to config file')
    parser.add_argument('--input', nargs='+', help='A list of space separated input images or a single glob '
                                                   'pattern such as directory/*.jpg', required=True)
    parser.add_argument('--output', help='A file or directory to save output visualizations', required=True)
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Minimum score for instance predictions to be shown')
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
    bar = tqdm(args.input, disable=not args.output)
    for path in bar:
        # use PIL, to be consistent with evaluation
        img = read_image(path, format='BGR')
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        bar.set_description('{}: {} in {:.2f}s'.format(path, 'detected {} instances'.format(
            len(predictions['instances'])) if 'instances' in predictions else 'finished', time.time() - start_time))

        if os.path.isdir(args.output):
            out_filename = os.path.join(args.output, os.path.basename(path))
        else:
            assert len(args.input) == 1, 'Please specify an existed directory with args.output'
            out_filename = args.output
        visualized_output.save(out_filename)