import cv2 as cv
import numpy as np
import argparse
import json
import os, sys
import pprint

from util import *

def parse_args():
    parser = argparse.ArgumentParser(description='get video meta info')

    parser.add_argument('--cfg', dest='cfg_file', default='./src/cfgs/defatult.yml',
                        help='config file')

    parser.add_argument('--data_folder', default='./data',
                        help='data folder')
    parser.add_argument('--input_video', default='Game_Of_Hunting_EP1_new.mp4',
                        help='input video file')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def video_meta(cfg):
    vcap = cv.VideoCapture(os.path.join(cfg.DATA_FOLDER, cfg.VIDEO_NAME + cfg.VIDEO_EXT))
    if not vcap.isOpened():
        print('> cannot open the video')
        sys.exit(1)

    width = int(vcap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vcap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = vcap.get(cv.CAP_PROP_FPS)

    n_frames = len(os.listdir(os.path.join(cfg.DATA_FOLDER, cfg.FRAMES_FOLDER)))

    vcap.release()

    with open(os.path.join(cfg.DATA_FOLDER, cfg.VIDEO_META.META_JSON), 'w') as f:
        json.dump({
            'video_name': cfg.VIDEO_NAME, 'video_ext': cfg.VIDEO_EXT,
            'width': width, 'height': height, 'fps': fps, 'n_frames': n_frames},
            f, indent=4)

if __name__ == '__main__':
    args = parse_args()

    if args.cfg_file is not None:
        cfg = cfg_from_file(args.cfg_file)

    cfg.DATA_FOLDER = args.data_folder
    cfg.VIDEO_NAME, cfg.VIDEO_EXT = os.path.splitext(args.input_video)

    print('> using config:')
    pprint.pprint(cfg)

    video_meta(cfg)
