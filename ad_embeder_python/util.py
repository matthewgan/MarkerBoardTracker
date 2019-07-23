import cv2 as cv
from numpy import math, hstack

import numpy as np
from collections import namedtuple
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        cfg = edict(yaml.load(f))

    return cfg
    
def squeeze_pts(X):
    X = X.squeeze()
    if len(X.shape) == 1:
        X = np.array([X])
    return X

def array_to_int_tuple(X):
    return (int(X[0]), int(X[1]))

def array_to_float_tuple(X):
    return (float(X[0]), float(X[1]))

def L2norm(X):
    return np.sqrt((X ** 2).sum(axis=1))

current_pos = None
tl = None
br = None

def get_rect(im, title='get_rect'):
    mouse_params = {'tl': None, 'br': None, 'current_pos': None,
        'released_once': False}

    cv.namedWindow(title)
    cv.moveWindow(title, 100, 100)

    def onMouse(event, x, y, flags, param):

        param['current_pos'] = (x, y)

        if param['tl'] is not None and not (flags & cv.EVENT_FLAG_LBUTTON):
            param['released_once'] = True

        if flags & cv.EVENT_FLAG_LBUTTON:
            if param['tl'] is None:
                param['tl'] = param['current_pos']
            elif param['released_once']:
                param['br'] = param['current_pos']

    cv.setMouseCallback(title, onMouse, mouse_params)
    cv.imshow(title, im)

    while mouse_params['br'] is None:
        im_draw = np.copy(im)

        if mouse_params['tl'] is not None:
            cv.rectangle(im_draw, mouse_params['tl'],
                mouse_params['current_pos'], (255, 0, 0))

        cv.imshow(title, im_draw)
        _ = cv.waitKey(10)

    cv.destroyWindow(title)

    tl = (min(mouse_params['tl'][0], mouse_params['br'][0]),
        min(mouse_params['tl'][1], mouse_params['br'][1]))
    br = (max(mouse_params['tl'][0], mouse_params['br'][0]),
        max(mouse_params['tl'][1], mouse_params['br'][1]))

    return (tl, br)

def in_rect(keypoints, tl, br):
    if type(keypoints) is list:
        keypoints = keypoints_cv_to_np(keypoints)

    x = keypoints[:, 0]
    y = keypoints[:, 1]

    C1 = x > tl[0]
    C2 = y > tl[1]
    C3 = x < br[0]
    C4 = y < br[1]

    result = C1 & C2 & C3 & C4

    return result

def in_square(keypoints, tl, tr, br, bl):
    if type(keypoints) is list:
        keypoints = keypoints_cv_to_np(keypoints)

    result = np.array([False] * keypoints.shape[0])
    square = np.array([tl, tr, br, bl], dtype=np.float32)
    for i in range(keypoints.shape[0]):
        if cv.pointPolygonTest(square, (keypoints[i, 0], keypoints[i, 1]), False) >= 0:
            result[i] = True
    return result

def keypoints_cv_to_np(keypoints_cv):
    keypoints = np.array([k.pt for k in keypoints_cv])
    return keypoints

def find_nearest_keypoints(keypoints, pos, number=1):
    if type(pos) is tuple:
        pos = np.array(pos)
    if type(keypoints) is list:
        keypoints = keypoints_cv_to_np(keypoints)

    pos_to_keypoints = np.sqrt(np.power(keypoints - pos, 2).sum(axis=1))
    ind = np.argsort(pos_to_keypoints)
    return ind[:number]

def draw_keypoints(keypoints, im, color=(255, 0, 0)):

    for k in keypoints:
        radius = 3  # int(k.size / 2)
        center = (int(k[0]), int(k[1]))

        # Draw circle
        cv.circle(im, center, radius, color)

def track(im_prev, im_gray, keypoints, THR_FB=20):
    lk_params = dict( winSize  = (15, 15),
                     maxLevel = 3,
                     criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01) )

    if type(keypoints) is list:
        keypoints = keypoints_cv_to_np(keypoints)

    num_keypoints = keypoints.shape[0]

    # Status of tracked keypoint - True means successfully tracked
    status = [False] * num_keypoints

    # If at least one keypoint is active
    if num_keypoints > 0:
        # Prepare data for opencv:
        # Add singleton dimension
        # Use only first and second column
        # Make sure dtype is float32
        pts = keypoints[:, None, :2].astype(np.float32)

        # Calculate forward optical flow for prev_location
        nextPts, status, _ = cv.calcOpticalFlowPyrLK(im_prev, im_gray, pts, None, **lk_params)

        # Calculate backward optical flow for prev_location
        pts_back, _, _ = cv.calcOpticalFlowPyrLK(im_gray, im_prev, nextPts, None, **lk_params)

        # Remove singleton dimension
        pts_back = squeeze_pts(pts_back)
        pts = squeeze_pts(pts)
        nextPts = squeeze_pts(nextPts)
        status = status.squeeze()

        # Calculate forward-backward error
        fb_err = np.sqrt(np.power(pts_back - pts, 2).sum(axis=1))

        # Set status depending on fb_err and lk error
        large_fb = fb_err > THR_FB
        status = ~large_fb & status.astype(np.bool)

        nextPts = nextPts[status, :]
        keypoints_tracked = keypoints[status, :]
        keypoints_tracked[:, :2] = nextPts

        flow = np.float32(nextPts - keypoints[status, :2])
    else:
        keypoints_tracked = np.array([])

        flow = np.array([], dtype=np.float32)
    return keypoints_tracked, status, flow

def rotate(pt, rad):
    if(rad == 0):
        return pt

    pt_rot = np.empty(pt.shape)

    s, c = [f(rad) for f in (math.sin, math.cos)]

    pt_rot[:, 0] = c * pt[:, 0] - s * pt[:, 1]
    pt_rot[:, 1] = s * pt[:, 0] + c * pt[:, 1]

    return pt_rot

def br(bbs):

    result = hstack((bbs[:, [0]] + bbs[:, [2]] - 1, bbs[:, [1]] + bbs[:, [3]] - 1))

    return result

def bb2pts(bbs):

    pts = hstack((bbs[:, :2], br(bbs)))

    return pts

def order_points(pts):
    # pts: 4 corners' points

    cnt = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)

    cnt[0] = pts[np.argmin(s)]
    cnt[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    cnt[1] = pts[np.argmin(diff)]
    cnt[3] = pts[np.argmax(diff)]

    return cnt