from __future__ import print_function
from collections import namedtuple

import cv2 as cv
import numpy as np
import time

import json

from ad_unit_detection import intersection_over_union

def ad_units_count(cfg, ad_units, shots):    
    shots = np.array(shots['shots'], dtype=np.float32)

    if shots.shape[0] > 0:
        shots_idx = np.int32(shots[:, 0]).tolist()
    else:
        shots_idx = []

    ad_units_instances = []

    pre_squares = np.array([], dtype=np.float32).reshape(-1, 4, 2)

    for i in (range(0, cfg.VIDEO_N_FRAMES)):
        
        #detected = np.array(ad_units['%d' % i]['detected'], dtype=np.int32).reshape(-1, 4, 2)
        #tracked = np.array(ad_units['%d' % i]['tracked'], dtype=np.int32).reshape(-1, 4, 2)
        
        squares = np.array(ad_units['%d' % i]['squares'], dtype=np.int32).reshape(-1, 4, 2)

        if pre_squares.shape[0] <= 0 or i in shots_idx:
            for j in range(squares.shape[0]):
                ad_units_instances.append([(i, j, squares[j].tolist(), 1.0)]) # frame idx, square idx, square, score
        else:
            tmp_n_instances = len(ad_units_instances)
            flags = []

            for ins_i in range(tmp_n_instances):
                f_idx, s_idx, _, _ = ad_units_instances[ins_i][-1]
                if f_idx != i - 1:
                    continue

                max_iou = 0
                max_idx = -1

                boxA = np.array(cv.boundingRect(pre_squares[s_idx]))
                boxA[2] = boxA[0] + boxA[2] - 1
                boxA[3] = boxA[1] + boxA[3] - 1

                for j in range(squares.shape[0]):
                    boxB = np.array(cv.boundingRect(squares[j]))
                    boxB[2] = boxB[0] + boxB[2] - 1
                    boxB[3] = boxB[1] + boxB[3] - 1

                    iou = intersection_over_union(boxA, boxB)
                    if iou > max_iou:
                        max_idx = j
                        max_iou = iou
                
                if max_iou > cfg.AD_UNITS_EXTRACTION.THRESHOLD_LAST_TIMES_IOU:
                    ad_units_instances[ins_i].append((i, max_idx, squares[max_idx].tolist(), max_iou)) # frame idx, square idx, square, score
                    flags.append(max_idx)

            for j in range(squares.shape[0]):
                if j not in flags:
                    ad_units_instances.append([(i, j, squares[j].tolist(), 1.0)]) # frame idx, square idx, square, score
                    
        pre_squares = squares
        
    n_instances = 0
    ad_units_filtered = {}
    instance_start_frame = {}
    instance_updated = {}

    tmp_n_instances = len(ad_units_instances)
    for i in range(tmp_n_instances):
        if len(ad_units_instances[i]) >= cfg.AD_UNITS_EXTRACTION.THRESHOLD_LAST_TIMES:
            for ins_i in range(len(ad_units_instances[i])):
                f_idx, s_idx, square, _ = ad_units_instances[i][ins_i]

                if n_instances not in instance_start_frame:
                    instance_start_frame[n_instances] = f_idx

                    instance_updated[n_instances] = False
                elif instance_start_frame[n_instances] > f_idx:
                    instance_start_frame[n_instances] = f_idx

                if f_idx not in ad_units_filtered:
                    ad_units_filtered[f_idx] = {} # instance idx: square
                
                ad_units_filtered[f_idx][n_instances] = square

            n_instances += 1

    return ad_units_filtered, n_instances, instance_start_frame, instance_updated
