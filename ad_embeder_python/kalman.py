from __future__ import print_function
from collections import namedtuple

import cv2 as cv
import numpy as np
import time

import json

class Kalman(object):
    def __init__(self, pnc=1e-3, mnc=1.0):
        self.kalman = cv.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.eye(2,4, dtype=np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],
                                                 [0,1,0,1],
                                                 [0,0,1,0],
                                                 [0,0,0,1]], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * pnc
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * mnc
        self.is_init = False
    
    def set_measurement(self, init_pt):
        while(True):
            pred = self.update(init_pt)
            if np.sum(np.abs(pred - init_pt)) < 5e-4:
                break

        self.is_init = True

    def update(self, current_pt):
        self.kalman.correct(current_pt)
        return self.kalman.predict()[0:2].reshape(2)



if __name__ == '__main__':

    ads_img_path = 'D:\\workspace\\virtual-ads\\ads1.png'
    ads_img = cv.imread(ads_img_path)

    video_path = 'D:\\workspace\\virtual-ads\\Game_Of_Hunting_EP1_new.mp4'

    short_len = 800.0

    vcap = cv.VideoCapture(video_path)
    if not vcap.isOpened():
        print('cannot open the video')

    scale = min(vcap.get(cv.CAP_PROP_FRAME_WIDTH), vcap.get(cv.CAP_PROP_FRAME_HEIGHT)) / short_len
    img_width = int(vcap.get(cv.CAP_PROP_FRAME_WIDTH) / scale)
    img_height = int(vcap.get(cv.CAP_PROP_FRAME_HEIGHT) / scale)

    ad_units_json_path = 'D:\\workspace\\virtual-ads\\opticalFlow\\opticalFlow\\ad_units_instances.json'

    instance_id = 13

    with open(ad_units_json_path, 'r') as f:
        ad_units = json.load(f)

    n_instances = ad_units['total_instances']
    ad_units = ad_units['ad_units']

    kalman = [Kalman(), Kalman(), Kalman(), Kalman()]

    count = 0
    while(True):
        
        start = time.time()

        ret, img = vcap.read()
        if not ret:
            break
        
        img = cv.resize(img, (img_width, img_height))
        vis = img

        if '%d' % count in ad_units:
            for unit in ad_units['%d' % count]:
                ins_id, square = unit
                if ins_id == instance_id:
                    square = np.array(square, dtype=np.float32).reshape(4, 2)
                    for s in range(4):
                        if kalman[s].is_init:
                            square[s] = kalman[s].update(square[s])
                        else:
                            kalman[s].set_measurement(square[s])

                    vis = cv.drawContours( vis, [square.astype(np.int32)], -1, (255, 0, 0), 2 )

                    cv.imshow('squres', vis)
                    cv.waitKey(42)

        count += 1

    cv.destroyAllWindows()
    vcap.release()
