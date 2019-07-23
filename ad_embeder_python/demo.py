import cv2 as cv
import numpy as np
import argparse
import json
import os
from tqdm import tqdm

from shot_detection import shot_boundary_detection
from ad_unit_detection import *
from filter_ad_units import *
from ad_embedding import *
from kalman import *
from kp_track import *


def run_shot_detection(args):
    vcap = cv.VideoCapture(os.path.join(args.data_folder, args.input_video))
    if not vcap.isOpened():
        print('> cannot open the video')

    n_frames = vcap.get(cv.CAP_PROP_FRAME_COUNT)
    print('> total frames: %d' % (n_frames))

    scale = min(vcap.get(cv.CAP_PROP_FRAME_WIDTH), vcap.get(cv.CAP_PROP_FRAME_HEIGHT)) / args.short_len
    img_width = int(vcap.get(cv.CAP_PROP_FRAME_WIDTH) / scale)
    img_height = int(vcap.get(cv.CAP_PROP_FRAME_HEIGHT) / scale)

    shots = shot_boundary_detection(vcap, img_width, img_height)
    with open(os.path.join(args.data_folder, args.output_folder, args.shots_json), 'w') as f:
        json.dump({'width': img_width, 'height': img_height, 'shots': np.array(shots, dtype=np.float32).tolist()}, f, indent=4)
    
    vcap.release()

def run_ad_unit_detection(args):
    vcap = cv.VideoCapture(os.path.join(args.data_folder, args.input_video))
    if not vcap.isOpened():
        print('> cannot open the video')
    
    n_frames = int(vcap.get(cv.CAP_PROP_FRAME_COUNT))
    print('> total frames: %d' % (n_frames))

    ad_units_json = {}

    with open(os.path.join(args.data_folder, args.output_folder, args.shots_json), 'r') as f:
        shots = json.load(f)

    img_width = int(shots['width'])
    img_height = int(shots['height'])

    area_lbound = THRESHOLD_AREA_MIN * img_height * img_width
    area_rbound = THRESHOLD_AREA_MAX * img_height * img_width
    
    shots = np.array(shots['shots'], dtype=np.float32)
    shots_idx = np.int32(shots[:, 0]).tolist()

    vwtr = cv.VideoWriter(os.path.join(args.data_folder, args.output_folder, args.output_units_video), cv.VideoWriter_fourcc(*'XVID'),
                          vcap.get(cv.CAP_PROP_FPS), (img_width, img_height))

    count = 0
    ret, img1 = vcap.read()
    if not ret:
        print('> read video error')
    img1 = cv.resize(img1, (img_width, img_height))
    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

    octave = 4
    vis = img1
    squares = find_squares(img1, octave)
    squares = squares[check_squares(squares, area_lbound, area_rbound)]

    if squares.shape[0] > 0:
        squares = group_squares(squares)
        squares = squares[check_squares(squares, area_lbound, area_rbound)]
        
        vis = cv.drawContours( vis, squares, -1, (0, 255, 0), 2 )
    
    ad_units_json[count] = {'squares': (squares.tolist())} # unit pts

    pre_squares = squares
        
    cv.imshow('squares', vis)
    cv.waitKey(1)

    vwtr.write(vis)

    use_spatial_propagation = True
    use_temporal_propagation = True
    dis_inst = cv.DISOpticalFlow.create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis_inst.setUseSpatialPropagation(use_spatial_propagation)
    flow = None
    flow_back = None

    for fid in tqdm(range(n_frames)):
        
        start = time.time()

        ret, img2 = vcap.read()
        if not ret:
            break
        count += 1

        img2 = cv.resize(img2, (img_width, img_height))
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
        
        vis = img2
        squares = find_squares(img2, octave)
        squares = squares[check_squares(squares, area_lbound, area_rbound)]

        warped_squares = np.array([], dtype=np.int32).reshape(-1, 4, 2)

        if pre_squares.shape[0] > 0 and count not in shots_idx:
            warped_squares = warp_squares(pre_squares, flow)
        squares = np.concatenate([squares, warped_squares])
        squares = group_squares(squares)
        squares = squares[check_squares(squares, area_lbound, area_rbound)]

        vis = cv.drawContours( vis, squares, -1, (0, 255, 0), 2 )

        ad_units_json[count] = {'squares': (squares.tolist())} # unit pts

        pre_squares = squares

        cv.imshow('squares', vis)
        key = cv.waitKey(1)
        
        vwtr.write(vis)

        if chr(key & 0xff) == 'q':
            break

        img1_gray = img2_gray

        # print('> image pair %4d: %.3fs' % (count, time.time() - start))

        #cv.imshow('flow', draw_flow(img2_gray, flow))
        #cv.waitKey(1)

    cv.destroyAllWindows()
    vcap.release()
    vwtr.release()

    with open(os.path.join(args.data_folder, args.output_folder, args.ad_units_json), 'w') as f:
        json.dump({'width': img_width, 'height': img_height, 'ad_units': ad_units_json}, f, indent=4)

def run_filter_ad_units(args):
    vcap = cv.VideoCapture(os.path.join(args.data_folder, args.input_video))
    if not vcap.isOpened():
        print('> cannot open the video')
    
    n_frames = int(vcap.get(cv.CAP_PROP_FRAME_COUNT))
    print('> total frames: %d' % (n_frames))

    with open(os.path.join(args.data_folder, args.output_folder, args.shots_json), 'r') as f:
        shots = json.load(f)
    
    img_width = int(shots['width'])
    img_height = int(shots['height'])

    with open(os.path.join(args.data_folder, args.output_folder, args.ad_units_json), 'r') as f:
        ad_units = json.load(f)
    ad_units = ad_units['ad_units']

    ad_units_filtered, n_instances, instance_start_frame = ad_units_count(ad_units, shots)
    max_valid_fid = max([int(k) for k in ad_units_filtered.keys()])

    with open(os.path.join(args.data_folder, args.output_folder, args.ad_units_instances_json), 'w') as f:
        json.dump({'width': img_width, 'height': img_height, 'total_instances': n_instances, \
            'instance_start_frame': instance_start_frame, \
            'ad_units_instances': ad_units_filtered}, f, indent=4)
    
    print('> total instances: %d' % n_instances)
    colors = []
    for i in range(n_instances):
        colors.append(np.random.randint(0,255,(3)).tolist())

    vwtr = cv.VideoWriter(os.path.join(args.data_folder, args.output_folder, args.output_instances_video), cv.VideoWriter_fourcc(*'XVID'),
                          vcap.get(cv.CAP_PROP_FPS), (img_width, img_height))

    count = 0
    for fid in tqdm(range(n_frames)):
        start = time.time()

        ret, img = vcap.read()
        if not ret:
            break
        
        img = cv.resize(img, (img_width, img_height))
        vis = img

        if count in ad_units_filtered:
            for unit in ad_units_filtered[count]:
                ins_id, square = unit
                vis = cv.drawContours( vis, np.array([square], dtype=np.int32), -1, colors[ins_id], 2 )
                vis = cv.putText(vis, '%d' % ins_id, tuple(square[3]), cv.FONT_HERSHEY_SIMPLEX, 1, colors[ins_id], 2, cv.LINE_AA)

        # print('> frame %4d: %d' % (count, len(ad_units_filtered[count]) if count in ad_units_filtered else 0))
        count += 1

        cv.imshow('instances', vis)

        vwtr.write(vis)

        key = cv.waitKey(1)
        if chr(key & 0xff) == 'q':
            break

        if count > max_valid_fid:
            break

    cv.destroyAllWindows()
    vcap.release()
    vwtr.release()

def run_ad_embedding(args):
    ads_img = cv.imread(os.path.join(args.data_folder, args.ad_dir, args.ad_image))

    vcap = cv.VideoCapture(os.path.join(args.data_folder, args.input_video))
    if not vcap.isOpened():
        print('> cannot open the video')
    
    n_frames = int(vcap.get(cv.CAP_PROP_FRAME_COUNT))
    print('> total frames: %d' % (n_frames))

    with open(os.path.join(args.data_folder, args.output_folder, args.ad_units_instances_json), 'r') as f:
        ad_units = json.load(f)

    img_width = int(ad_units['width'])
    img_height = int(ad_units['height'])

    vwtr = cv.VideoWriter(os.path.join(args.data_folder, args.output_folder, 'ads_placing_demo_ins%d_%s.avi' % (args.instance_id, os.path.splitext(args.ad_image)[0])), cv.VideoWriter_fourcc(*'XVID'),
                          vcap.get(cv.CAP_PROP_FPS), (img_width, img_height))

    n_instances = ad_units['total_instances']
    ad_units = ad_units['ad_units_instances']
    max_valid_fid = max([int(k) for k in ad_units.keys()])

    pnc=5e-5
    mnc = 1e1
    kalman = [Kalman(pnc=pnc, mnc=mnc), Kalman(pnc=pnc, mnc=mnc), Kalman(pnc=pnc, mnc=mnc), Kalman(pnc=pnc, mnc=mnc)]
    kalman_center = Kalman(pnc=pnc, mnc=mnc)

    kptrack_inst = KPTrack() # key points tracking
    clahe  = cv.createCLAHE(clipLimit=4, tileGridSize=(8,8))

    count = 0
    for fid in tqdm(range(n_frames)):
        
        start = time.time()

        ret, img = vcap.read()
        if not ret:
            break
        
        img = cv.resize(img, (img_width, img_height))
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # img_gray = clahe.apply(img_gray)
        
        vis = img

        if '%d' % count in ad_units:
            for unit in ad_units['%d' % count]:
                ins_id, square = unit
                if ins_id == args.instance_id:
                    square = np.array(square, dtype=np.float32).reshape(4, 2)
                    n_tracked = 0

                    if not kptrack_inst.is_init:
                        kptrack_inst.initialise(img_gray, square[0], square[1], square[2], square[3])
                    else:
                        kptrack_inst.process_frame(img_gray)
                        
                        if kptrack_inst.has_result and kptrack_inst.matched_ratio > kptrack_inst.THR_MATCH_CONF:
                            n_tracked = 1
                            square = np.float32([kptrack_inst.tl, kptrack_inst.tr, kptrack_inst.br, kptrack_inst.bl])

                    # kalman smoothing
                    for s in range(4):
                        if kalman[s].is_init:
                            square[s] = kalman[s].update(square[s])
                        else:
                            kalman[s].set_measurement(square[s])

                    warped_ads, mask = ads_embedding(vis, ads_img, square)

                    vis[np.where(mask > 0)] = warped_ads[np.where(mask > 0)]
                    
                    vwtr.write(vis)

                    cv.imshow('ads', np.uint8(vis))
                    key = cv.waitKey(1)
                    if chr(key & 0xff) == 'q':
                        break
                
        count += 1
        # print('frame %d: %.3fs' % (count, time.time() - start))

        if count > max_valid_fid:
            break

    cv.destroyAllWindows()
    vcap.release()
    vwtr.release()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--phase', default=1,
                        help='demo phase')

    parser.add_argument('--data_folder', default='./data',
                        help='data folder')
    parser.add_argument('--input_video', default='Game_Of_Hunting_EP1_new.mp4',
                        help='input video file')
    parser.add_argument('--short_len', default=720.0,
                        help='short length of input video')
    parser.add_argument('--shots_json', default='shots.json',
                        help='output shots json file')
    parser.add_argument('--ad_units_json', default='ad_units_pts.json',
                        help='output ad units points json file')
    parser.add_argument('--output_units_video', default='units.avi',
                        help='output ad units video file')
    parser.add_argument('--ad_units_instances_json', default='ad_units_instances.json',
                        help='output ad units instances json file')
    parser.add_argument('--output_instances_video', default='instances.avi',
                        help='output ad units instances video file')

    parser.add_argument('--ad_dir', default='ads',
                        help='ads image folder')
    parser.add_argument('--ad_image', default='surface.png',
                        help='ad image')
    parser.add_argument('--instance_id', type=int, default=13,
                        help='input instance id to place ad')

    args = parser.parse_args()
    args.output_folder = os.path.splitext(os.path.basename(args.input_video))[0]

    print('> Input arguments:')
    for key, val in vars(args).items():
        print('%16s: %s' % (key, val))

    if not os.path.exists(os.path.join(args.data_folder, args.output_folder)):
        os.makedirs(os.path.join(args.data_folder, args.output_folder))

    if int(args.phase) == 1:
        # 1. shot boundary detection
        print('######### 1. shots detection #########')
        run_shot_detection(args)

        # 2. ad unit detection
        print('######### 2. ad unit detection #########')
        run_ad_unit_detection(args) # costing time

        # 3. ad units filtering -> instances
        print('######### 3. ad units filtering #########')
        run_filter_ad_units(args)
    else:
        # 4. ad instance specifying and re-scaning
        # pass

        # 5. new ad embedding
        print('######### 5. ad embedding #########')
        run_ad_embedding(args)