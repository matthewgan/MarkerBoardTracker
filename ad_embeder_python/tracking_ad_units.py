import cv2 as cv
import numpy as np
import argparse
import json
import os, sys
import pprint

from util import *
from kalman import *
from kp_track import *

def run_ad_units_tracking(cfg):
    img_width = cfg.VIDEO_WIDTH
    img_height = cfg.VIDEO_HEIGHT

    if os.path.exists(os.path.join(cfg.DATA_FOLDER, cfg.AD_UNITS_TRACKING.MODIFIED_AD_UNITS_JSONS, cfg.AD_UNITS_TRACKING.AD_UNITS_INSTANCES_JSON)):
        with open(os.path.join(cfg.DATA_FOLDER, cfg.AD_UNITS_TRACKING.MODIFIED_AD_UNITS_JSONS, cfg.AD_UNITS_TRACKING.AD_UNITS_INSTANCES_JSON), 'r') as f:
            ad_units = json.load(f)
    else:
        with open(os.path.join(cfg.DATA_FOLDER, cfg.AD_UNITS_EXTRACTION.AD_UNITS_JSONS, cfg.AD_UNITS_EXTRACTION.AD_UNITS_INSTANCES_JSON), 'r') as f:
            ad_units = json.load(f)

    n_instances = ad_units['n_instances']
    instance_start_frame = ad_units['instance_start_frame']
    instance_updated = ad_units['instance_updated']
    ad_units = ad_units['ad_units_instances']
    max_valid_fid = max([int(k) for k in ad_units.keys()])

    assert '%d' % cfg.AD_UNITS_TRACKING.INSTANCE_ID in instance_updated

    pnc=5e-5
    mnc = 1e1
    kalman = [Kalman(pnc=pnc, mnc=mnc), Kalman(pnc=pnc, mnc=mnc), Kalman(pnc=pnc, mnc=mnc), Kalman(pnc=pnc, mnc=mnc)]
    kalman_center = Kalman(pnc=pnc, mnc=mnc)

    kptrack_inst = KPTrack(cfg.AD_UNITS_TRACKING.ESTIMATE_SCALE, cfg.AD_UNITS_TRACKING.ESTIMATE_ROTATION) # key points tracking
    clahe  = cv.createCLAHE(clipLimit=4, tileGridSize=(8,8))

    for fid in range(instance_start_frame['%d' % cfg.AD_UNITS_TRACKING.INSTANCE_ID], cfg.VIDEO_N_FRAMES):

        start = time.time()

        if fid > max_valid_fid:
            break

        if '%d' % fid in ad_units:
            if '%d' % cfg.AD_UNITS_TRACKING.INSTANCE_ID in ad_units['%d' % fid]:
                img = cv.imread(os.path.join(cfg.DATA_FOLDER, cfg.FRAMES_FOLDER, '%d.jpg' % (fid)))
                img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                if cfg.AD_UNITS_TRACKING.HISTOGRAM_EQUAL:
                    img_gray = clahe.apply(img_gray)

                square = np.array(ad_units['%d' % fid]['%d' % cfg.AD_UNITS_TRACKING.INSTANCE_ID], dtype=np.float32).reshape(4, 2)

                n_tracked = 0

                if not kptrack_inst.is_init:
                    square = np.array([cfg.AD_UNITS_TRACKING.TL, cfg.AD_UNITS_TRACKING.TR, cfg.AD_UNITS_TRACKING.BR, cfg.AD_UNITS_TRACKING.BL], dtype=np.float32).reshape(4, 2)
                    kptrack_inst.initialise(img_gray, square[0], square[1], square[2], square[3])
                else:
                    kptrack_inst.process_frame(img_gray)

                    if kptrack_inst.has_result and kptrack_inst.matched_ratio > kptrack_inst.THR_MATCH_CONF:
                        n_tracked = 1
                        square = np.float32([kptrack_inst.tl, kptrack_inst.tr, kptrack_inst.br, kptrack_inst.bl])

                # kalman smoothing
                if cfg.AD_UNITS_TRACKING.KALMAN_SMOOTH:
                    for s in range(4):
                        if kalman[s].is_init:
                            square[s] = kalman[s].update(square[s])
                        else:
                            kalman[s].set_measurement(square[s])

                ad_units['%d' % fid]['%d' % cfg.AD_UNITS_TRACKING.INSTANCE_ID] = square.tolist()

    instance_updated['%d' % cfg.AD_UNITS_TRACKING.INSTANCE_ID] = True
    with open(os.path.join(cfg.DATA_FOLDER, cfg.AD_UNITS_TRACKING.MODIFIED_AD_UNITS_JSONS, cfg.AD_UNITS_TRACKING.AD_UNITS_INSTANCES_JSON), 'w') as f:
        json.dump({'n_instances': n_instances,
            'instance_start_frame': instance_start_frame,
            'instance_updated': instance_updated,
            'ad_units_instances': ad_units},
            f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description='extracting ad units')

    parser.add_argument('--cfg', dest='cfg_file', default='./src/cfgs/defatult.yml',
                        help='config file')

    parser.add_argument('--data_folder', default='./data',
                        help='data folder')
    parser.add_argument('--instance_id', type=int, default=0,
                        help='input instance id to track')
    parser.add_argument('--tl', type=float, nargs=2, metavar=('a', 'b'),
                        default=None,
                        help='ad instance tl (x, y)')
    parser.add_argument('--tr', type=float, nargs=2, metavar=('a', 'b'),
                        default=None,
                        help='ad instance tr (x, y)')
    parser.add_argument('--br', type=float, nargs=2, metavar=('a', 'b'),
                        default=None,
                        help='ad instance br (x, y)')
    parser.add_argument('--bl', type=float, nargs=2, metavar=('a', 'b'),
                        default=None,
                        help='ad instance bl (x, y)')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.cfg_file is not None:
        cfg = cfg_from_file(args.cfg_file)

    cfg.DATA_FOLDER = args.data_folder

    cfg.AD_UNITS_TRACKING.INSTANCE_ID = args.instance_id
    cfg.AD_UNITS_TRACKING.TL = np.array(args.tl, dtype=np.float32)
    cfg.AD_UNITS_TRACKING.TR = np.array(args.tr, dtype=np.float32)
    cfg.AD_UNITS_TRACKING.BR = np.array(args.br, dtype=np.float32)
    cfg.AD_UNITS_TRACKING.BL = np.array(args.bl, dtype=np.float32)

    # get video meta data
    with open(os.path.join(cfg.DATA_FOLDER, cfg.VIDEO_META.META_JSON), 'r') as f:
        video_meta = json.load(f)

    cfg.VIDEO_NAME = video_meta['video_name']
    cfg.VIDEO_EXT = video_meta['video_ext']
    cfg.VIDEO_WIDTH = int(video_meta['width'])
    cfg.VIDEO_HEIGHT = int(video_meta['height'])
    cfg.VIDEO_FPS = float(video_meta['fps'])
    cfg.VIDEO_N_FRAMES = int(video_meta['n_frames'])

    print('> using config:')
    pprint.pprint(cfg)

    # mkdirs
    if not os.path.exists(os.path.join(cfg.DATA_FOLDER, cfg.AD_UNITS_TRACKING.MODIFIED_AD_UNITS_JSONS)):
        os.mkdir(os.path.join(cfg.DATA_FOLDER, cfg.AD_UNITS_TRACKING.MODIFIED_AD_UNITS_JSONS))

    run_ad_units_tracking(cfg)