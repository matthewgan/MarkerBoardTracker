from __future__ import print_function
from collections import namedtuple

import cv2 as cv
import numpy as np
import time

import json

'''
    square detection
'''

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def edge_ratio(pts):
    if len(pts) != 4:
        print('must be quadrilateral')
        return 0

    dist = [cv.norm(pts[0] - pts[1]),
            cv.norm(pts[1] - pts[2]),
            cv.norm(pts[2] - pts[3]),
            cv.norm(pts[3] - pts[0])]

    return np.min(dist) / np.max(dist)

def check_squares(cfg, contours, area_lbound, area_rbound):
    keep = []
    n_contours = contours.shape[0]

    for i in range(n_contours):
        if area_lbound < cv.contourArea(contours[i]) < area_rbound and cv.isContourConvex(contours[i]):
            max_cos = np.max([angle_cos( contours[i][k], contours[i][(k + 1) % 4], contours[i][(k + 2) % 4] ) for k in range(4)])

            if edge_ratio(contours[i]) > cfg.AD_UNITS_EXTRACTION.THRESHOLD_EDGE_RATIO and max_cos < cfg.AD_UNITS_EXTRACTION.THRESHOLD_ANGLE:
                keep.append(i)
    
    return keep

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

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv.Canny(image, lower, upper)

	# return the edged image
	return edged

def find_squares(cfg, img, octave):
    
    for oct in range(octave):
        if oct == 0:
            pyr_img = cv.pyrUp(img)
            squares = find_squares_impl(cfg, pyr_img) * (2.0 ** (oct - 1))
        else:
            pyr_img = cv.pyrDown(pyr_img)
            _squares = find_squares_impl(cfg, pyr_img) * (2.0 ** (oct - 1))
            
            squares = np.concatenate([squares, _squares])
    
    return np.int32(squares)

def find_squares_impl(cfg, img):

    squares = []
    for gray in cv.split(img):
        
        bin = auto_canny(gray)
        bin = cv.dilate(bin, None)

        contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            cnt_len = cv.arcLength(cnt, True)
            cnt = cv.approxPolyDP(cnt, cfg.AD_UNITS_EXTRACTION.EPSILON * cnt_len, True)
            
            if len(cnt) == 4:
                squares.append(order_points(np.asarray(cnt).reshape(-1, 2)))
    return np.float32(squares).reshape(-1, 4, 2)


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res

def warp_squares(cfg, squares, flow):
    
    warped_squares = []
    for s in squares:
        ws = []
        
        for pt in s:
            fx, fy = flow[pt[1], pt[0]]
            if fx > cfg.AD_UNITS_EXTRACTION.THRESHOLD_FLOW or fy > cfg.AD_UNITS_EXTRACTION.THRESHOLD_FLOW:
                break
            pt[0] = max(0, min(pt[0] + fx, flow.shape[1] - 1))
            pt[1] = max(0, min(pt[1] + fy, flow.shape[0] - 1))
            ws.append(np.int32(pt))

        if len(ws) == 4:
            warped_squares.append(ws)
    
    return np.asarray(warped_squares, dtype=np.int32).reshape(-1, 4, 2)

def group_squares(cfg, squares):
    n_squares = squares.shape[0]
    if n_squares < 2:
        return squares

    rects = np.zeros((n_squares, 4), dtype=np.float32)

    for i in range(n_squares):
        rects[i] = cv.boundingRect(squares[i])

    rects[:, 2] = rects[:, 0] + rects[:, 2] - 1
    rects[:, 3] = rects[:, 1] + rects[:, 3] - 1

    ious = np.zeros((n_squares, n_squares), dtype=np.float32)
    for i in range(n_squares):
        ious[i, i] = 1.0

        for j in range(i + 1, n_squares):
            ious[i, j] = intersection_over_union(rects[i], rects[j])
            ious[j, i] = ious[i, j]

    groups = np.zeros(n_squares, dtype=np.int32) - 1 # group number: 0-indexed
    n_groups = 0

    for i in range(n_squares):
        if groups[i] < 0:
            groups[i] = n_groups
        
            for j in range(i + 1, n_squares):
                if ious[i, j] > cfg.AD_UNITS_EXTRACTION.THRESHOLD_NMS_IOU:
                    groups[j] = n_groups
                    groups[np.where(ious[j, :] > cfg.AD_UNITS_EXTRACTION.THRESHOLD_NMS_IOU)] = n_groups

            n_groups += 1

    result_squares = np.zeros((n_groups, 4, 2), dtype=np.float32)

    for i in range(n_groups):
        result_squares[i] = np.mean(squares[np.where(groups == i)], axis=0)
        
    return np.int32(result_squares)


def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    w = max(0.0, xB - xA + 1)
    h = max(0.0, yB - yA + 1)
    inter_area = w * h

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(boxA_area + boxB_area - inter_area)

    # return the intersection over union value
    return iou