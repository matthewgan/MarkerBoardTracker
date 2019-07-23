import cv2 as cv
import itertools
from numpy import array, zeros, vstack, hstack, math, nan, argsort, median, \
    argmax, isnan, append
import scipy.cluster
import scipy.spatial
import time

import numpy as np
import util

class KPTrack(object):

    DESC_LENGTH = 512
    
    THR_OUTLIER = 20
    THR_CONF = 0.75 # absolute distance
    THR_RATIO = 0.8 # distance ratio

    THR_MATCH_CONF = 0.1 # matching confidence

    MIN_MATCH_COUNT = 10 # minmum number of points to find homography

    # estimate_scale = True
    # estimate_rotation = True

    def __init__(self, estimate_scale, estimate_rotation):
        self.matched_ratio = 0
        self.estimate_scale = estimate_scale
        self.estimate_rotation = estimate_rotation
        self.is_init = False

    def initialise(self, im_gray0, tl, tr, br, bl):

        # Initialise detector, descriptor, matcher
        self.detector = cv.BRISK_create()
        self.descriptor = cv.xfeatures2d.LATCH_create(bytes=64)
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False) # set crossCheck to False to use knnMatch

        # Get initial keypoints in whole image
        keypoints_cv = self.detector.detect(im_gray0)

        # Remember keypoints that are in the rectangle as selected keypoints
        ind = util.in_square(keypoints_cv, tl, tr, br, bl)
        self.selected_keypoints_cv = list(itertools.compress(keypoints_cv, ind))
        self.selected_keypoints_cv, self.selected_features = self.descriptor.compute(im_gray0, self.selected_keypoints_cv)
        selected_keypoints = util.keypoints_cv_to_np(self.selected_keypoints_cv)
        num_selected_keypoints = len(self.selected_keypoints_cv)

        if num_selected_keypoints == 0:
            raise Exception('No keypoints found in selection')

        # Remember keypoints that are not in the rectangle as background keypoints
        background_keypoints_cv = list(itertools.compress(keypoints_cv, ~ind))
        background_keypoints_cv, background_features = self.descriptor.compute(im_gray0, background_keypoints_cv)
        _ = util.keypoints_cv_to_np(background_keypoints_cv)

        # Assign each keypoint a class starting from 1, background is 0
        self.selected_classes = array(list(range(num_selected_keypoints))) + 1
        background_classes = zeros(len(background_keypoints_cv))

        # Stack background features and selected features into database
        self.features_database = vstack((background_features, self.selected_features))

        # Same for classes
        self.database_classes = hstack((background_classes, self.selected_classes))

        # Get all distances between selected keypoints in squareform
        pdist = scipy.spatial.distance.pdist(selected_keypoints)
        self.squareform = scipy.spatial.distance.squareform(pdist)

        # Get all angles between selected keypoints
        angles = np.empty((num_selected_keypoints, num_selected_keypoints))
        for k1, i1 in zip(selected_keypoints, list(range(num_selected_keypoints))):
            for k2, i2 in zip(selected_keypoints, list(range(num_selected_keypoints))):

                # Compute vector from k1 to k2
                v = k2 - k1

                # Compute angle of this vector with respect to x axis
                angle = math.atan2(v[1], v[0])

                # Store angle
                angles[i1, i2] = angle

        self.angles = angles

        # Find the center of selected keypoints
        center = np.mean(selected_keypoints, axis=0)

        # Remember the rectangle coordinates relative to the center
        self.center_to_tl = np.array(tl) - center
        self.center_to_tr = np.array(tr) - center
        self.center_to_br = np.array(br) - center
        self.center_to_bl = np.array(bl) - center

        # Calculate springs of each keypoint
        self.springs = selected_keypoints - center

        # Set start image for tracking
        self.im_prev = im_gray0

        # Make keypoints 'active' keypoints
        self.active_keypoints = np.copy(selected_keypoints)

        # Attach class information to active keypoints
        self.active_keypoints = hstack((selected_keypoints, self.selected_classes[:, None]))

        # Remember number of initial keypoints
        self.initial_keypoints = np.copy(self.active_keypoints)
        self.num_initial_keypoints = len(self.selected_keypoints_cv)

        self.matched_ratio = 0.0

        self.tl = util.array_to_float_tuple(tl)
        self.tr = util.array_to_float_tuple(tr)
        self.br = util.array_to_float_tuple(br)
        self.bl = util.array_to_float_tuple(bl)

        self.is_init = True

    def estimate(self, keypoints, flow):

        center = array((nan, nan))
        scale_estimate = nan
        med_rot = nan

        # At least 2 keypoints are needed for scale
        if keypoints.size > 1:

            # Extract the keypoint classes
            keypoint_classes = keypoints[:, 2].squeeze().astype(np.int)

            # Retain singular dimension
            if keypoint_classes.size == 1:
                keypoint_classes = keypoint_classes[None]

            # Sort
            ind_sort = argsort(keypoint_classes)
            keypoints = keypoints[ind_sort]
            keypoint_classes = keypoint_classes[ind_sort]

            # Get all combinations of keypoints
            all_combs = array([val for val in itertools.product(list(range(keypoints.shape[0])), repeat=2)])

            # But exclude comparison with itself
            all_combs = all_combs[all_combs[:, 0] != all_combs[:, 1], :]

            # Measure distance between allcombs[0] and allcombs[1]
            ind1 = all_combs[:, 0]
            ind2 = all_combs[:, 1]

            class_ind1 = keypoint_classes[ind1] - 1
            class_ind2 = keypoint_classes[ind2] - 1

            duplicate_classes = class_ind1 == class_ind2

            if not all(duplicate_classes):
                ind1 = ind1[~duplicate_classes]
                ind2 = ind2[~duplicate_classes]

                class_ind1 = class_ind1[~duplicate_classes]
                class_ind2 = class_ind2[~duplicate_classes]

                pts_allcombs0 = keypoints[ind1, :2]
                pts_allcombs1 = keypoints[ind2, :2]

                # This distance might be 0 for some combinations,
                # as it can happen that there is more than one keypoint at a single location
                dists = util.L2norm(pts_allcombs0 - pts_allcombs1)

                original_dists = self.squareform[class_ind1, class_ind2]

                scalechange = dists / original_dists

                # Compute angles
                angles = np.empty((pts_allcombs0.shape[0]))

                v = pts_allcombs1 - pts_allcombs0
                angles = np.arctan2(v[:, 1], v[:, 0])

                original_angles = self.angles[class_ind1, class_ind2]

                angle_diffs = angles - original_angles

                # Fix long way angles
                long_way_angles = np.abs(angle_diffs) > math.pi

                angle_diffs[long_way_angles] = angle_diffs[long_way_angles] - np.sign(angle_diffs[long_way_angles]) * 2 * math.pi

                scale_estimate = median(scalechange)
                if not self.estimate_scale:
                    scale_estimate = 1;

                med_rot = median(angle_diffs)
                if not self.estimate_rotation:
                    med_rot = 0;

                keypoint_class = keypoints[:, 2].astype(np.int)
                votes = keypoints[:, :2] - scale_estimate * (util.rotate(self.springs[keypoint_class - 1], med_rot))

                # Remember all votes including outliers
                self.votes = votes

                # Compute pairwise distance between votes
                pdist = scipy.spatial.distance.pdist(votes)
                
                # Compute linkage between pairwise distances
                linkage = scipy.cluster.hierarchy.linkage(pdist)

                # Perform hierarchical distance-based clustering
                T = scipy.cluster.hierarchy.fcluster(linkage, self.THR_OUTLIER, criterion='distance')

                # Count votes for each cluster
                cnt = np.bincount(T)  # Dummy 0 label remains

                # Get largest class
                Cmax = argmax(cnt)

                # Identify inliers (=members of largest class)
                inliers = T == Cmax
                # inliers = med_dists < THR_OUTLIER

                # Remember outliers
                self.outliers = keypoints[~inliers, :]

                # Stop tracking outliers
                keypoints = keypoints[inliers, :]
                flow = flow[inliers, :]

                # Remove outlier votes
                votes = votes[inliers, :]

                # Compute object center
                center = np.mean(votes, axis=0)

        return (center, scale_estimate, med_rot, keypoints, flow)

    def estimate_homography(self, src_keypoints_cv, src_features, dst_keypoints_cv, dst_features):
        knn_matches = self.matcher.knnMatch(src_features, dst_features, 2)

        ratio_thresh = 0.75
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        #-- Localize the object
        src_points = np.empty((len(good_matches),2), dtype=np.float32)
        dst_points = np.empty((len(good_matches),2), dtype=np.float32)
        for i in range(len(good_matches)):
            #-- Get the keypoints from the good matches
            src_points[i,0] = src_keypoints_cv[good_matches[i].queryIdx].pt[0]
            src_points[i,1] = src_keypoints_cv[good_matches[i].queryIdx].pt[1]
            dst_points[i,0] = dst_keypoints_cv[good_matches[i].trainIdx].pt[0]
            dst_points[i,1] = dst_keypoints_cv[good_matches[i].trainIdx].pt[1]
        
        H, status = cv.findHomography(src_points, dst_points, cv.LMEDS)
        status = status.squeeze()
        return H, status.astype(np.bool)

    def process_frame(self, im_gray):
        tracked_keypoints, status, flow = util.track(self.im_prev, im_gray, self.active_keypoints, 1.0)
                
        (center, scale_estimate, rotation_estimate, tracked_keypoints, flow) = self.estimate(tracked_keypoints, flow)

        # creat mask
        mask = np.zeros_like(im_gray, dtype=np.uint8)
        if tracked_keypoints.size > 0 and not any(isnan(center)):
            tl = util.array_to_float_tuple(center + scale_estimate * util.rotate(self.center_to_tl[None, :], rotation_estimate).squeeze())
            tr = util.array_to_float_tuple(center + scale_estimate * util.rotate(self.center_to_tr[None, :], rotation_estimate).squeeze())
            br = util.array_to_float_tuple(center + scale_estimate * util.rotate(self.center_to_br[None, :], rotation_estimate).squeeze())
            bl = util.array_to_float_tuple(center + scale_estimate * util.rotate(self.center_to_bl[None, :], rotation_estimate).squeeze())

            cv.fillPoly(mask, np.array([tl, tr, br, bl], dtype=np.int32).reshape(-1, 4, 2), (255, 255, 255))
            mask = cv.dilate(mask, np.ones((19 ,19), np.uint8), iterations=2)
            
            # Detect keypoints, compute descriptors
            keypoints_cv = self.detector.detect(im_gray, mask=mask)
        else:
            # Detect keypoints, compute descriptors
            keypoints_cv = self.detector.detect(im_gray)
        keypoints_cv, features = self.descriptor.compute(im_gray, keypoints_cv)

        # # double check for transformation estimation
        # mask = np.zeros_like(self.im_prev, dtype=np.uint8)
        # cv.fillPoly(mask, np.array([self.tl, self.tr, self.br, self.bl], dtype=np.int32).reshape(-1, 4, 2), (255, 255, 255))
        # keypoints_cv_prev = self.detector.detect(self.im_prev, mask=mask)
        # keypoints_cv_prev, features_prev = self.descriptor.compute(self.im_prev, keypoints_cv_prev)
        
        # H, status = self.estimate_homography(keypoints_cv_prev, features_prev, keypoints_cv, features)
        
        # if H is not None:
        #     tl = util.array_to_float_tuple(cv.perspectiveTransform(np.float32(self.tl).reshape(1, 1, -1), H).reshape(-1))
        #     tr = util.array_to_float_tuple(cv.perspectiveTransform(np.float32(self.tr).reshape(1, 1, -1), H).reshape(-1))
        #     br = util.array_to_float_tuple(cv.perspectiveTransform(np.float32(self.br).reshape(1, 1, -1), H).reshape(-1))
        #     bl = util.array_to_float_tuple(cv.perspectiveTransform(np.float32(self.bl).reshape(1, 1, -1), H).reshape(-1))

        # Create list of active keypoints
        active_keypoints = zeros((0, 3))

        # Get all matches for selected features
        if not any(isnan(center)):
            selected_matches_all = self.matcher.knnMatch(features, self.selected_features, len(self.selected_features))

        # import pdb
        # pdb.set_trace()

        # For each keypoint and its descriptor
        matched_ratio = 0.0
        if len(keypoints_cv) > 0:
            transformed_springs = scale_estimate * util.rotate(self.springs, -rotation_estimate)
            for i in range(len(keypoints_cv)):

                # Retrieve keypoint location
                location = np.array(keypoints_cv[i].pt)

                # If structural constraints are applicable
                if not any(isnan(center)):

                    # Compute distances to initial descriptors
                    matches = selected_matches_all[i]
                    distances = np.array([m.distance for m in matches])
                    # Re-order the distances based on indexing
                    idxs = np.argsort(np.array([m.trainIdx for m in matches]))
                    distances = distances[idxs]

                    # Convert distances to confidences
                    confidences = 1 - distances / self.DESC_LENGTH

                    # Compute the keypoint location relative to the object center
                    relative_location = location - center

                    # Compute the distances to all springs
                    displacements = util.L2norm(transformed_springs - relative_location)

                    # For each spring, calculate weight
                    weight = displacements < self.THR_OUTLIER  # Could be smooth function

                    combined = weight * confidences

                    classes = self.selected_classes

                    # Sort in descending order
                    sorted_conf = argsort(combined)[::-1]  # reverse

                    # Get best and second best index
                    bestInd = sorted_conf[0]
                    secondBestInd = sorted_conf[1]

                    # Compute distance ratio according to Lowe
                    ratio = (1 - combined[bestInd] + 1e-8) / (1 - combined[secondBestInd] + 1e-8)
                    
                    # Extract class of best match
                    keypoint_class = classes[bestInd]

                    # If distance ratio is ok and absolute distance is ok and keypoint class is not background
                    if ratio < self.THR_RATIO and combined[bestInd] > self.THR_CONF and keypoint_class != 0:
                        matched_ratio += 1
                        # Add keypoint to active keypoints
                        new_kpt = append(location, keypoint_class)

                        # Check whether same class already exists
                        if active_keypoints.size > 0:
                            same_class = np.nonzero(active_keypoints[:, 2] == keypoint_class)
                            active_keypoints = np.delete(active_keypoints, same_class, axis=0)

                        active_keypoints = append(active_keypoints, array([new_kpt]), axis=0)
        
        # If some keypoints have been tracked
        if tracked_keypoints.size > 0:

            # Extract the keypoint classes
            tracked_classes = tracked_keypoints[:, 2]

            # If there already are some active keypoints
            if active_keypoints.size > 0:

                # Add all tracked keypoints that have not been matched
                associated_classes = active_keypoints[:, 2]
                missing = ~np.in1d(tracked_classes, associated_classes)
                active_keypoints = append(active_keypoints, tracked_keypoints[missing, :], axis=0)

            # Else use all tracked keypoints
            else:
                active_keypoints = tracked_keypoints

        # Update object state estimate
        _ = active_keypoints
        self.center = center
        self.scale_estimate = scale_estimate
        self.rotation_estimate = rotation_estimate
        self.tracked_keypoints = tracked_keypoints
        self.active_keypoints = active_keypoints
        self.keypoints_cv = keypoints_cv
        _ = time.time()

        self.bb = array([nan, nan, nan, nan])

        self.has_result = False
        if not any(isnan(self.center)) and self.active_keypoints.shape[0] > self.num_initial_keypoints / 10:
            self.has_result = True

            min_x = min((tl[0], tr[0], br[0], bl[0]))
            min_y = min((tl[1], tr[1], br[1], bl[1]))
            max_x = max((tl[0], tr[0], br[0], bl[0]))
            max_y = max((tl[1], tr[1], br[1], bl[1]))

            self.tl = tl
            self.tr = tr
            self.bl = bl
            self.br = br

            self.bb = np.array([min_x, min_y, max_x - min_x, max_y - min_y])

        self.matched_ratio = matched_ratio / len(self.selected_features)
        self.im_prev = im_gray