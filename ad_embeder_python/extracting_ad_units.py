import cv2 as cv
import numpy as np
import argparse
import json
import os, sys
import pprint

from util import *
from ad_unit_detection import *
from filter_ad_units import *

def run_shot_detection(cfg):
    last_hsv = np.array([], dtype=np.float32)
    shots = []
    count = 0

    for fid in range(cfg.VIDEO_N_FRAMES):
        imgPath = os.path.join(cfg.DATA_FOLDER, cfg.FRAMES_FOLDER, '%d.jpg' % (fid))
        img = cv.imread(imgPath)

        curr_hsv = np.array(cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV)), dtype=np.float32)

        if last_hsv.shape[0] > 0:

            delta_hsv = np.mean(np.abs(curr_hsv - last_hsv), axis=(1,2))

            if np.mean(delta_hsv) > cfg.AD_UNITS_EXTRACTION.THRESHOLD_SBD:
                shots.append([count, np.mean(delta_hsv)])

        last_hsv = curr_hsv

        count += 1

    with open(os.path.join(cfg.DATA_FOLDER, cfg.AD_UNITS_EXTRACTION.AD_UNITS_JSONS, cfg.AD_UNITS_EXTRACTION.SHOTS_JSON), 'w') as f:
        json.dump({'shots': np.array(shots, dtype=np.float32).tolist()}, f, indent=4)

def run_ad_unit_detection(cfg):
    ad_units_json = {}

    with open(os.path.join(cfg.DATA_FOLDER, cfg.AD_UNITS_EXTRACTION.AD_UNITS_JSONS, cfg.AD_UNITS_EXTRACTION.SHOTS_JSON), 'r') as f:
        shots = json.load(f)

    img_width = cfg.VIDEO_WIDTH
    img_height = cfg.VIDEO_HEIGHT

    area_lbound = cfg.AD_UNITS_EXTRACTION.THRESHOLD_AREA_MIN * img_height * img_width
    area_rbound = cfg.AD_UNITS_EXTRACTION.THRESHOLD_AREA_MAX * img_height * img_width

    shots = np.array(shots['shots'], dtype=np.float32)
    if shots.shape[0] > 0:
        shots_idx = np.int32(shots[:, 0]).tolist()
    else:
        shots_idx = []

    count = 0
    img1 = cv.imread(os.path.join(cfg.DATA_FOLDER, cfg.FRAMES_FOLDER, '0.jpg'))
    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

    octave = 4
    vis = img1
    squares = find_squares(cfg, img1, octave)
    squares = squares[check_squares(cfg, squares, area_lbound, area_rbound)]

    if squares.shape[0] > 0:
        squares = group_squares(cfg, squares)
        squares = squares[check_squares(cfg, squares, area_lbound, area_rbound)]

    ad_units_json[count] = {'squares': (squares.tolist())} # unit pts

    pre_squares = squares

    use_spatial_propagation = True
    use_temporal_propagation = True
    dis_inst = cv.DISOpticalFlow.create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis_inst.setUseSpatialPropagation(use_spatial_propagation)
    flow = None
    flow_back = None

    for fid in range(1, cfg.VIDEO_N_FRAMES):

        start = time.time()

        img2 = cv.imread(os.path.join(cfg.DATA_FOLDER, cfg.FRAMES_FOLDER, '%d.jpg' % (fid)))

        count += 1

        img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        # flow = cv.calcOpticalFlowFarneback(img1_gray, img2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # dis optical flow
        if flow is not None and use_temporal_propagation:
            #warp previous flow to get an initial approximation for the current flow:
            flow = dis_inst.calc(img1_gray, img2_gray, warp_flow(flow,flow))
            flow_back = dis_inst.calc(img2_gray, img1_gray, warp_flow(flow_back,flow_back))

            flow = (flow - flow_back)/2
            flow_back = -flow
        else:
            flow = dis_inst.calc(img1_gray, img2_gray, None)
            flow_back = dis_inst.calc(img2_gray, img1_gray, None)

            flow = (flow - flow_back)/2
            flow_back = -flow

        squares = find_squares(cfg, img2, octave)
        squares = squares[check_squares(cfg, squares, area_lbound, area_rbound)]

        warped_squares = np.array([], dtype=np.int32).reshape(-1, 4, 2)

        if pre_squares.shape[0] > 0 and count not in shots_idx:
            warped_squares = warp_squares(cfg, pre_squares, flow)
        squares = np.concatenate([squares, warped_squares])
        squares = group_squares(cfg, squares)
        squares = squares[check_squares(cfg, squares, area_lbound, area_rbound)]

        ad_units_json[count] = {'squares': (squares.tolist())} # unit pts

        pre_squares = squares

        img1_gray = img2_gray

    with open(os.path.join(cfg.DATA_FOLDER, cfg.AD_UNITS_EXTRACTION.AD_UNITS_JSONS, cfg.AD_UNITS_EXTRACTION.AD_UNITS_JSON), 'w') as f:
        json.dump({'ad_units': ad_units_json}, f, indent=4)

def run_filter_ad_units(cfg):
    with open(os.path.join(cfg.DATA_FOLDER, cfg.AD_UNITS_EXTRACTION.AD_UNITS_JSONS, cfg.AD_UNITS_EXTRACTION.SHOTS_JSON), 'r') as f:
        shots = json.load(f)

    img_width = cfg.VIDEO_WIDTH
    img_height = cfg.VIDEO_HEIGHT

    with open(os.path.join(cfg.DATA_FOLDER, cfg.AD_UNITS_EXTRACTION.AD_UNITS_JSONS, cfg.AD_UNITS_EXTRACTION.AD_UNITS_JSON), 'r') as f:
        ad_units = json.load(f)
    ad_units = ad_units['ad_units']

    ad_units_filtered, n_instances, instance_start_frame, instance_updated = ad_units_count(cfg, ad_units, shots)

    with open(os.path.join(cfg.DATA_FOLDER, cfg.AD_UNITS_EXTRACTION.AD_UNITS_JSONS, cfg.AD_UNITS_EXTRACTION.AD_UNITS_INSTANCES_JSON), 'w') as f:
        json.dump({'n_instances': n_instances,
            'instance_start_frame': instance_start_frame,
            'instance_updated': instance_updated,
            'ad_units_instances': ad_units_filtered},
            f, indent=4)

    if cfg.DEBUG:
        colors = []
        for i in range(n_instances):
            colors.append(np.random.randint(0,255,(3)).tolist())

        # mkdir
        if not os.path.exists(os.path.join(cfg.DATA_FOLDER, 'debug')):
            os.mkdir(os.path.join(cfg.DATA_FOLDER, 'debug'))

        vwtr = cv.VideoWriter(os.path.join(cfg.DATA_FOLDER, 'debug', 'instances.avi'),
                              cv.VideoWriter_fourcc(*'XVID'),
                              cfg.VIDEO_FPS, (img_width, img_height))

        for fid in range(cfg.VIDEO_N_FRAMES):

            img = cv.imread(os.path.join(cfg.DATA_FOLDER, cfg.FRAMES_FOLDER, '%d.jpg' % (fid)))

            if fid in ad_units_filtered:
                for ins_id in ad_units_filtered[fid]:

                    square = ad_units_filtered[fid][ins_id]
                    img = cv.drawContours(img, np.array([square], dtype=np.int32), -1, colors[ins_id], 2 )
                    img = cv.putText(img, '%d' % ins_id, tuple(square[3]), cv.FONT_HERSHEY_SIMPLEX, 1, colors[ins_id], 2, cv.LINE_AA)

            vwtr.write(img)

        vwtr.release()

def parse_args():
    parser = argparse.ArgumentParser(description='extracting ad units')

    parser.add_argument('--cfg', dest='cfg_file', default='./src/cfgs/defatult.yml',
                        help='config file')

    parser.add_argument('--data_folder', default='./data',
                        help='data folder')

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

    # get video meta data
    with open(os.path.join(cfg.DATA_FOLDER, cfg.VIDEO_META.META_JSON), 'r') as f:
        video_meta = json.load(f)

    cfg.VIDEO_NAME = video_meta['video_name']
    cfg.VIDEO_EXT = video_meta['video_ext']
    cfg.VIDEO_WIDTH = int(video_meta['width'])
    cfg.VIDEO_HEIGHT = int(video_meta['height'])
    cfg.VIDEO_FPS = float(video_meta['fps'])
    cfg.VIDEO_N_FRAMES = int(video_meta['n_frames'])

    # print('> using config:')
    # pprint.pprint(cfg)

    # mkdirs
    if not os.path.exists(os.path.join(cfg.DATA_FOLDER, cfg.AD_UNITS_EXTRACTION.AD_UNITS_JSONS)):
        os.mkdir(os.path.join(cfg.DATA_FOLDER, cfg.AD_UNITS_EXTRACTION.AD_UNITS_JSONS))

    # 1. shot boundary detection
    print('######### 1. shots detection #########')
    run_shot_detection(cfg)

    # 2. ad unit detection
    print('######### 2. ad unit detection #########')
    run_ad_unit_detection(cfg) # costing time

    # 3. ad units filtering -> instances
    print('######### 3. ad units filtering #########')
    run_filter_ad_units(cfg)