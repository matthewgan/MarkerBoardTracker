import cv2 as cv
import numpy as np
import argparse
import json
import os, sys
from tqdm import tqdm
import pprint
import time

from skimage.transform import match_histograms

from util import *
from tracking_ad_units import run_ad_units_tracking

def ads_embedding(img, ad_img, pts, ref_density, ad_img_density):
    tl, tr, br, bl = pts

    width_A = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
    width_B = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
    max_width = max(int(width_A), int(width_B))
    height_A = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
    height_B = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
    max_height = max(int(height_A), int(height_B))

    ad_img = cv.resize(ad_img, (max_width, max_height), interpolation=cv.INTER_AREA)

    dst = np.array([[0, 0],
                   [ad_img.shape[1] - 1, 0],
                   [ad_img.shape[1] - 1, ad_img.shape[0] - 1],
                   [0, ad_img.shape[0] - 1]], dtype=np.float32)

    M = cv.getPerspectiveTransform(dst, pts)

    # #
    # alpha = 0.85
    M_t = cv.getPerspectiveTransform(pts, dst)
    warped_raw = cv.warpPerspective(img, M_t, dsize=(ad_img.shape[1], ad_img.shape[0]), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)

    alpha = ref_density / ad_img_density
    ad_img = ad_img * alpha
    ad_img[ad_img > 255] = 255
    ad_img =  np.uint8(ad_img)

    # warped_raw = np.uint8(warped_raw * alpha + ad_img * (1 - alpha))
    # ad_img = cv.resize(ad_img, dsize=(0, 0), fx=0.98, fy=0.98, interpolation=cv.INTER_AREA)
    # ad_img_blended = cv.seamlessClone(ad_img, warped_raw,
    #                                   np.ones_like(ad_img, dtype=np.uint8)*255,
    #                                   (int(img.shape[1]/2), int(img.shape[0]/2)),
    #                                   flags=cv.NORMAL_CLONE)

    warped = cv.warpPerspective(ad_img, M, dsize=(img.shape[1], img.shape[0]), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_CONSTANT)
    border = 1
    mask = np.ones_like(ad_img, dtype=np.uint8) * 255
    mask = cv.copyMakeBorder(mask[2*border:, 2*border:, :], border, border, border, border, cv.BORDER_CONSTANT, None, (128, 128, 128))
    mask = cv.warpPerspective(mask, M, dsize=(img.shape[1], img.shape[0]), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    mask = mask/255.
    blended = img * (1 - mask) + warped * mask
    return np.uint8(blended)

def run_ad_embedding(cfg):
    with open(os.path.join(cfg.DATA_FOLDER, cfg.AD_EMBEDDING.ADS_FOLDER, cfg.AD_EMBEDDING.AD_IMAGE + '.json')) as f:
        ad = json.load(f)

    ad_img = cv.imread(os.path.join(cfg.DATA_FOLDER, cfg.AD_EMBEDDING.ADS_FOLDER, cfg.AD_EMBEDDING.AD_IMAGE + ad['ad_ext']))

    img_width = cfg.VIDEO_WIDTH
    img_height = cfg.VIDEO_HEIGHT

    if os.path.exists(os.path.join(cfg.DATA_FOLDER, cfg.AD_UNITS_TRACKING.MODIFIED_AD_UNITS_JSONS, cfg.AD_UNITS_TRACKING.AD_UNITS_INSTANCES_JSON)):
        with open(os.path.join(cfg.DATA_FOLDER, cfg.AD_UNITS_TRACKING.MODIFIED_AD_UNITS_JSONS, cfg.AD_UNITS_TRACKING.AD_UNITS_INSTANCES_JSON), 'r') as f:
            ad_units = json.load(f)
        assert '%d' % cfg.AD_EMBEDDING.INSTANCE_ID in ad_units['instance_updated']

        if not ad_units['instance_updated']['%d' % cfg.AD_EMBEDDING.INSTANCE_ID]:
            cfg.AD_UNITS_TRACKING.INSTANCE_ID = cfg.AD_EMBEDDING.INSTANCE_ID
            square = np.array(ad_units['ad_units_instances']['%d' % ad_units['instance_start_frame']['%d' % cfg.AD_EMBEDDING.INSTANCE_ID]]['%d' % cfg.AD_EMBEDDING.INSTANCE_ID],
                              dtype=np.float32).reshape(4, 2)
            cfg.AD_UNITS_TRACKING.TL = square[0]
            cfg.AD_UNITS_TRACKING.TR = square[1]
            cfg.AD_UNITS_TRACKING.BR = square[2]
            cfg.AD_UNITS_TRACKING.BL = square[3]
            run_ad_units_tracking(cfg)
    else:
        with open(os.path.join(cfg.DATA_FOLDER, cfg.AD_UNITS_EXTRACTION.AD_UNITS_JSONS, cfg.AD_UNITS_EXTRACTION.AD_UNITS_INSTANCES_JSON), 'r') as f:
            ad_units = json.load(f)

        assert '%d' % cfg.AD_EMBEDDING.INSTANCE_ID in ad_units['instance_updated']

        # mkdirs
        if not os.path.exists(os.path.join(cfg.DATA_FOLDER, cfg.AD_UNITS_TRACKING.MODIFIED_AD_UNITS_JSONS)):
            os.mkdir(os.path.join(cfg.DATA_FOLDER, cfg.AD_UNITS_TRACKING.MODIFIED_AD_UNITS_JSONS))

        cfg.AD_UNITS_TRACKING.INSTANCE_ID = cfg.AD_EMBEDDING.INSTANCE_ID

        square = np.array(ad_units['ad_units_instances']['%d' % ad_units['instance_start_frame']['%d' % cfg.AD_EMBEDDING.INSTANCE_ID]]['%d' % cfg.AD_EMBEDDING.INSTANCE_ID],
                            dtype=np.float32).reshape(4, 2)
        cfg.AD_UNITS_TRACKING.TL = square[0]
        cfg.AD_UNITS_TRACKING.TR = square[1]
        cfg.AD_UNITS_TRACKING.BR = square[2]
        cfg.AD_UNITS_TRACKING.BL = square[3]
        run_ad_units_tracking(cfg)

    with open(os.path.join(cfg.DATA_FOLDER, cfg.AD_UNITS_TRACKING.MODIFIED_AD_UNITS_JSONS, cfg.AD_UNITS_TRACKING.AD_UNITS_INSTANCES_JSON), 'r') as f:
            ad_units = json.load(f)

    n_instances = ad_units['n_instances']
    instance_start_frame = ad_units['instance_start_frame']
    instance_updated = ad_units['instance_updated']
    ad_units = ad_units['ad_units_instances']

    vwtr = cv.VideoWriter(os.path.join(cfg.DATA_FOLDER, cfg.OUTPUT_FOLDER, '%s_ins%d_%s.avi' % (cfg.VIDEO_NAME, cfg.AD_EMBEDDING.INSTANCE_ID, cfg.AD_EMBEDDING.AD_IMAGE)),
                          cv.VideoWriter_fourcc(*'XVID'),
                          cfg.VIDEO_FPS, (img_width, img_height))

    end_frame = -1
    ref_density = -1
    ad_img_density = np.mean(ad_img[:])

    for fid in tqdm(range(cfg.VIDEO_N_FRAMES)):

        img = cv.imread(os.path.join(cfg.DATA_FOLDER, cfg.FRAMES_FOLDER, '%d.jpg' % (fid)))

        if '%d' % fid in ad_units:
            if '%d' % cfg.AD_EMBEDDING.INSTANCE_ID in ad_units['%d' % fid]:
                if ref_density < 0:
                    ref_density = np.mean(img[:])
                square = np.array(ad_units['%d' % fid]['%d' % cfg.AD_EMBEDDING.INSTANCE_ID], dtype=np.float32).reshape(4, 2)
                img = ads_embedding(img, ad_img, square, ref_density, ad_img_density)

                # img[np.where(mask > 0)] = warped_ads[np.where(mask > 0)] * cfg.AD_EMBEDDING.EMBEDDING_ALPHA

                end_frame = fid

        vwtr.write(img)

    vwtr.release()

    with open(os.path.join(cfg.DATA_FOLDER, cfg.OUTPUT_FOLDER, '%s_ins%d_%s.json' % (cfg.VIDEO_NAME, cfg.AD_EMBEDDING.INSTANCE_ID, cfg.AD_EMBEDDING.AD_IMAGE)), 'w') as f:
        json.dump({
            'ad_name': ad['ad_name'],
            'ad_url': ad['ad_url'],
            'ad_instance': cfg.AD_EMBEDDING.INSTANCE_ID,
            'start_frame': instance_start_frame['%d' % cfg.AD_EMBEDDING.INSTANCE_ID],
            'end_frame': end_frame
        }, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description='extracting ad units')

    parser.add_argument('--cfg', dest='cfg_file', default='./src/cfgs/defatult.yml',
                        help='config file')

    parser.add_argument('--data_folder', default='./data',
                        help='data folder')
    parser.add_argument('--instance_id', type=int, default=0,
                        help='input instance id to track')
    parser.add_argument('--ads_folder', default='./ads', help="ads folder")
    parser.add_argument('--ad_image', default='surface',
                        help='ad image')

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
    cfg.AD_EMBEDDING.INSTANCE_ID = args.instance_id
    cfg.AD_EMBEDDING.ADS_FOLDER = args.ads_folder
    cfg.AD_EMBEDDING.AD_IMAGE = args.ad_image

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
    if not os.path.exists(os.path.join(cfg.DATA_FOLDER, cfg.OUTPUT_FOLDER)):
        os.mkdir(os.path.join(cfg.DATA_FOLDER, cfg.OUTPUT_FOLDER))

    run_ad_embedding(cfg)