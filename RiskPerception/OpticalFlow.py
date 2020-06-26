#!/usr/bin/env python
__author__ = "Miguel Costa"
__email__ = "mncosta(at)isr(dot)tecnico(dot)ulisboa(dot)pt"


from __future__ import print_function
import cv2
import numpy as np
import math
import argparse
import json

from RiskPerception.CONFIG import  CLAHE_clipLimit, \
                    CLAHE_tileGridSize,\
                    OF_MINIMUM_DIST,\
                    OF_DISTANCES_USE_GLOBAL,\
                    OF_CALC_INTERVALS
from RiskPerception.Objects import getOFWeightFromObjects


def optical_flow_interesting_points(frame_gray1, feature_params):
    height, width,  = frame_gray1.shape

    good_points = list()

    # Divide image into 4x4 grid so OF points are better
    for i in range(4):
        for j in range(4):
            # Get grayscale images

            ROI_frame1 = frame_gray1[i * height // 4:(i + 1) * height // 4,
                   j * width // 4:(j + 1) * width // 4]


            # Apply equalization to grayscale to improve corner detection
            clahe = cv2.createCLAHE(clipLimit=CLAHE_clipLimit,
                                    tileGridSize=CLAHE_tileGridSize)
            roiCLAHE1 = clahe.apply(ROI_frame1)

            # Get corners
            pAux1 = cv2.goodFeaturesToTrack(roiCLAHE1,
                                            mask=None,
                                            **feature_params)

            # Transform roiCLAHE coordinates to frame_gray coordinates
            if pAux1 is not None:
                pAux1 = np.add(pAux1, [j * width / 4, i * height / 4])

                for point in pAux1:
                    good_points.append(point)

    return good_points


def compute_optical_flow(frame_gray1, frame_gray2, points, LK_params):
    p1, st1, err1 = cv2.calcOpticalFlowPyrLK(frame_gray1,
                                             frame_gray2,
                                             points,
                                             None,
                                             **LK_params)

    return p1[st1 == 1], points[st1 == 1]


def calcDistance(x0, y0, x1, y1):
    """Return the euclidean distance given two points."""
    return math.sqrt( (x0-x1)**2 + (y0-y1)**2 )


def getWeightFromOFDistance(dist, dist_avg, dist_max):
    """Returns the weight associated with the vector magnitude."""
    distanceWeight = 1

    # Work with global distances or with annulus regions
    if not OF_DISTANCES_USE_GLOBAL:
        if dist < OF_MINIMUM_DIST:
            dist = 0

    # Work the the logaritm of the distances
    dist = math.log(dist + 1)
    dist_avg = math.log(dist_avg + 1)
    dist_max = math.log(dist_max + 1)

    # Initialize the intervals
    A = 2 * dist_avg / 6
    B = 3 * dist_avg / 6
    C = 4 * dist_avg / 6
    D = dist_avg + 2 * ((dist_max - dist_avg) / 6)
    E = dist_avg + 3 * ((dist_max - dist_avg) / 6)
    F = dist_avg + 4 * ((dist_max - dist_avg) / 6)

    # Associate the weight to the given interval
    if (0 <= dist < A):
        distanceWeight = 0.1
    elif (A <= dist < B):
        distanceWeight = 0.75
    elif (B <= dist < C):
        distanceWeight = 1
    elif (C <= dist < D):
        distanceWeight = 1
    elif (D <= dist < E):
        distanceWeight = 1
    elif (E <= dist < F):
        distanceWeight = 1
    elif (F <= dist <= dist_max):
        distanceWeight = 0.75
    else:
        pass
    # print 'else', dist, dist_max

    # Return the weight
    return distanceWeight


def drawOF(mask,
           vectorStart,
           vectorEnd,
           lineExtensionStart = None,
           lineExtensionEnd = None,
           OFcolor = np.array([0,255,0])):
    """Draws the Optical Flow on a frame."""

    # OF vector line
    cv2.line(mask,
             (int(vectorStart[0]), int(vectorStart[1])),
             (int(vectorEnd[0]), int(vectorEnd[1])),
             OFcolor, 1)
    # OF vector direction
    cv2.circle(mask,
               (int(vectorStart[0]), int(vectorStart[1])),
               1, OFcolor, -1)

    # Draw the line extension of the optical flow vector
    if lineExtensionStart is not None and lineExtensionEnd is not None:
        # Line extension
        cv2.line(mask, lineExtensionStart,lineExtensionEnd, OFcolor, 1)

    return


def get_interval_metrics_for_optical_flow(new_points, old_points, FOE,
                                          dist_intervals):
    intervals_avg_distances = np.zeros(4)
    intervals_max_distances = np.zeros(4)
    intervals = []
    for i in range(OF_CALC_INTERVALS):
        intervals.append(np.array([]))

    for i, (new, old) in enumerate(zip(new_points, old_points)):
        a1, b1 = new.ravel()
        c1, d1 = old.ravel()

        distToFOE = calcDistance((a1, b1), FOE)

        for j in range(dist_intervals.shape[0] - 1):
            if dist_intervals[j] < distToFOE < dist_intervals[j + 1]:
                break

        intervals[j] = np.append(intervals[j],
                                 math.sqrt((a1 - c1) ** 2 + (b1 - d1) ** 2))

    for i in range(OF_CALC_INTERVALS):
        intervals_avg_distances[i] = np.sum(intervals[i]) / intervals[i].shape[
            0]
        try:
            intervals_max_distances[i] = max(intervals[i])
        except ValueError:
            intervals_max_distances[i] = 0

    return intervals, intervals_avg_distances, intervals_max_distances



def get_lines_weights_from_optical_flow(new_points,
                                        old_points,
                                        width,
                                        height,
                                        FOE,
                                        objects,
                                        framenbr,
                                        intervals,
                                        intervals_avg_distances,
                                        intervals_max_distances,
                                        mask):
    a_i = np.array([])
    b_i = np.array([])
    c_i = np.array([])
    w_i = np.array([])

    OF = np.array([]).reshape(0, 4)

    aux = max(calcDistance(FOE, (0, 0)), calcDistance(FOE, (0, height)),
              calcDistance(FOE, (width, 0)), calcDistance(FOE, (width, height)))
    dist_intervals = np.arange(0, aux + 1, aux / OF_CALC_INTERVALS)

    distance_weights = []
    object_weights = []

    for i, (new, old) in enumerate(zip(new_points, old_points)):
        a1, b1 = new.ravel()
        c1, d1 = old.ravel()

        OF = np.vstack((OF, np.array([a1, b1, c1, d1])))

        if (a1 - c1) == 0:
            continue
        a = float(b1 - d1) / float(a1 - c1)
        b = -1
        c = (b1) - a * a1

        lengthLine = math.sqrt((a1 - c1) ** 2 + (b1 - d1) ** 2)

        distToFOE = calcDistance((a1, b1), FOE)
        for j in range(dist_intervals.shape[0] - 1):
            if dist_intervals[j] < distToFOE < dist_intervals[j + 1]:
                break

        distance_weight = getWeightFromOFDistance(lengthLine,
                                                  intervals_avg_distances[j],
                                                  intervals_max_distances[j])
        distance_weights.append(distance_weight)

        # Objects contribute with e^-(weight of the object)
        object_weight = math.exp(-1 * getOFWeightFromObjects(objects,
                                                            (a1,b1),
                                                            framenbr))
        object_weights.append(object_weight)

        weight = distance_weight * object_weight

        denominator = float(a ** 2 + 1)
        a_i = np.append(a_i, a / denominator)
        b_i = np.append(b_i, b / denominator)
        c_i = np.append(c_i, c / denominator)
        w_i = np.append(w_i, [weight])

        color1 = np.array(
            [[0, 0, 255], [0, 255, 0], [255, 0, 0], [100, 100, 100]])

        if distance_weight == 0.1:
            aux = color1[0]
        elif distance_weight == 0.3:
            aux = color1[3]
        elif distance_weight == 0.75:
            aux = color1[2]
        elif distance_weight == 1:
            aux = color1[1]

        drawOF(mask, (a1, b1), (c1, d1), OFcolor=aux)

    return mask, a_i, b_i, c_i, w_i, OF, distance_weights, object_weights


def parse_args():
    """Parse the arguments of the code."""

    # Current possible arguments:
    parser = argparse.ArgumentParser(description='Optical Flow')
    parser.add_argument('--video',
                        dest='video',
                        help='Video file')
    parser.add_argument('--of',
                        dest='of',
                        help='Optical flow file of the given video')
    parser.add_argument('--of_weights',
                        dest='of_weights',
                        help='Optical flow weights file of the given video')
    parser.add_argument('--function',
                        dest='f',
                        help='function to use')

    args = parser.parse_args()
    return args


def draw_of(args):
    of_file = args.of
    video = cv2.VideoCapture(args.video)
    frame_idx = 0

    with open(of_file) as inp_file:
        for line in inp_file:
            data = json.loads(line)
            for i in range(frame_idx, data['frame_number']):
                ret, frame = video.read()
                frame_idx += 1
            if ret is not False:
                old_points = data['old_points']
                new_points = data['new_points']

                for i in xrange(len(old_points)):
                    drawOF(frame,
                           old_points[i],
                           new_points[i],
                           lineExtensionStart=None,
                           lineExtensionEnd=None,
                           OFcolor=np.array([0, 255, 0]))
                cv2.imshow('test_of', frame)
                cv2.waitKey(0)




def draw_of_weights():
    pass


def optical_flow_get_function(function):
    functions = {
        'draw_of': draw_of,
        'draw_of_weights': draw_of_weights
    }
    try:
        f = functions[function]
        return f
    except KeyError:
        return None

if __name__ == '__main__':
    args = parse_args()

    f = optical_flow_get_function(args.f)
    if f is not None:
        f(args)
