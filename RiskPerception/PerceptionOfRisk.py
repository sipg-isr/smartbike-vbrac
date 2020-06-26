from __future__ import print_function
import argparse
import cv2
import sys
import numpy as np
import os
import json
import subprocess

import re

from timeit import default_timer as timer

from RiskPerception.Objects import readObjectsCSV
from RiskPerception.RiskClassifier import  train_classifier,\
                            divide_image_into_regions,\
                            create_risk_labels,\
                            classify_risk
from RiskPerception.OpticalFlow import optical_flow_interesting_points, \
                        compute_optical_flow, \
                        get_lines_weights_from_optical_flow,\
                        get_interval_metrics_for_optical_flow, \
                        calcDistance
from RiskPerception.FocusOfExpansion import huber_loss_optimization, \
    iterative_improve_on_object_weights, \
    points_history, \
    initialize_points_history, \
    generate_weights, \
    compute_avg_FOE, l2_norm_optimization

from RiskPerception.CONFIG import  EXPONENTIAL_DECAY_NBR_WEIGHTS,\
                    RISK_ZONE_SCORE_HIGH,\
                    RISK_ZONE_SCORE_MEDIUM,\
                    RISK_ZONE_SCORE_LOW,\
                    OF_CALC_INTERVALS,\
                    OF_FEATURE_PARAMS,\
                    SKIP_N_FRAMES

class ContinueError(Exception): pass


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def parse_args():
    """Parse the arguments of the code."""

    # Current possible arguments:
    #    --video_file (video file to use)
    #    --objects_csv (objects file of the given video)
    #    --start (skip # of frames at start of video)

    parser = argparse.ArgumentParser(description='Video Processing')
    parser.add_argument('--video_file',
                        dest='video_file',
                        help='Video file to process')
    parser.add_argument('--objects_csv',
                        dest='objects_csv',
                        help='Objects file of the given video')
    parser.add_argument('--classifier_data_folder',
                        dest='classifier_data_folder',
                        help='Folder containing data to train the classifier')
    parser.add_argument('--frame_start',
                        dest='frame_start',
                        type=int,
                        default=0,
                        help='Start process at frame \'start\'')

    args = parser.parse_args()
    return args


def __SetupGlobals__((width,height) = (None, None)):
    """Setup global variables"""

    global OF_LK_PARAMS

    if width is not None and height is not None:

        auxWindowSize = int((width - 364.6) / float(3.297))
        OF_LK_PARAMS = dict(winSize=(auxWindowSize, auxWindowSize),
                            maxLevel=1,
                            criteria=(cv2.TERM_CRITERIA_EPS |
                                      cv2.TERM_CRITERIA_COUNT, 10, 0.03))


if __name__ == '__main__':

    print('\n\n\t\tPerception Of Risk video analysis\n'
          '===================================================================')
    print('\n\n')

    args = parse_args()

    print('\rReading Objects from file...',end='')
    start = timer()
    objects = readObjectsCSV(args.objects_csv)
    end = timer()
    if objects is not None:
        print('\r',bcolors.OKGREEN, 'OK', bcolors.ENDC,
          'Reading Objects from file.')
    else:
        print('\r', bcolors.FAIL, 'NOK', bcolors.ENDC,
              'Reading Objects from file.')
        sys.exit(-1)
    print('===================================================================')
    print('\tReading took', round(end - start, 5), 'seconds.')

    print('\n')

    print('\rTraining Classifier with PathOccupation and Distance metrics...',
          end='')
    signaturesPathOccupation = None
    signaturesPathOccupationClass = None
    signaturesDistance = None
    signaturesDistanceClass = None
    start = timer()
    return_getter = train_classifier(args.classifier_data_folder)
    end = timer()
    if return_getter is not None:
        print('\r', bcolors.OKGREEN, 'OK', bcolors.ENDC,
            'Training Classifier with PathOccupation and Distance metrics.')
        signaturesPathOccupation = return_getter[0]
        signaturesPathOccupationClass = return_getter[1]
        signaturesDistance = return_getter[2]
        signaturesDistanceClass = return_getter[3]
    else:
        print('\r', bcolors.FAIL, 'NOK', bcolors.ENDC,
              'Training Classifier with PathOccupation and Distance metrics.')
        sys.exit(-1)
    print('===================================================================')
    print('\tTraining took', round(end - start, 5), 'seconds.')

    print('\n')

    print('\rChecking video...', end='')
    cap = cv2.VideoCapture(args.video_file)
    ret, frame1 = cap.read()
    if ret == False:
        print('\r', bcolors.FAIL, 'NOK', bcolors.ENDC,
              'Checking video.')
        sys.exit()
    else:
        print('\r', bcolors.OKGREEN, 'OK', bcolors.ENDC,
              'Checking video.')
    height, width, channels = frame1.shape
    print('===================================================================')

    print('\n')

    print('Starting to read video.')
    print('===================================================================')

    cap = cv2.VideoCapture(args.video_file)
    frame_idx = 0

    if args.frame_start > 0:
        while frame_idx < args.frame_start:
            ret, frame1 = cap.read()
            if not ret:
                print(bcolors.FAIL, 'ERR', bcolors.ENDC,
                      'Reading frame (', frame_idx,').')
                sys.exit()
            frame_idx += 1

    print(bcolors.HEADER+bcolors.BOLD, args.frame_start, bcolors.ENDC,
          'frames skipped at start of video\n')
    print('===================================================================')

    print('\rInitializing parameters...', end='')
    ret1, frame1 = cap.read()
    if not ret1:
        print(bcolors.FAIL, 'ERR', bcolors.ENDC,
                      'Reading frame (', frame_idx,').')
        sys.exit(-1)
    ret2, frame2 = cap.read()
    if not ret2:
        print(bcolors.FAIL, 'ERR', bcolors.ENDC,
                      'Reading frame (', frame_idx,').')
        sys.exit()

    cmd = ['ffmpeg', '-i', args.video_file, '-map', '0:v:0', '-c', 'copy',
           '-f', 'null', '-']
    p1 = subprocess.Popen(cmd, stderr=subprocess.PIPE)
    p1_out_list = p1.stderr.read().split()
    for i, p in enumerate(p1_out_list):
        if p == 'frame=':
            total_number_of_frames = p1_out_list[i + 1]
            break

        elif 'frame=' in p:
            total_number_of_frames = int(re.search(r'\d+', p).group())
            break
    else:
        total_number_of_frames = None
    height, width, channels = frame1.shape

    __SetupGlobals__(width=width, height=height)


    points = initialize_points_history(width, height)
    weights = generate_weights(EXPONENTIAL_DECAY_NBR_WEIGHTS)

    filename_aux = os.path.splitext(args.video_file)[0]
    print ('Creating output files:')
    print('Optical Flow:'.ljust(25),
          bcolors.UNDERLINE, filename_aux+'_of.txt', bcolors.ENDC)
    of_output_file = open(filename_aux+'_of.txt','w')
    print('Optical Flow Weights:'.ljust(25),
          bcolors.UNDERLINE, filename_aux + '_of_weights.txt', bcolors.ENDC)
    of_weights_output_file = open(filename_aux + '_of_weights.txt', 'w')
    print('Events:'.ljust(25),
          bcolors.UNDERLINE, filename_aux + '_events.txt', bcolors.ENDC)
    events_output_file = open(filename_aux+'_events.txt','w')
    print('Focus of Expansion:'.ljust(25),
          bcolors.UNDERLINE, filename_aux + '_foe.txt', bcolors.ENDC)
    foe_output_file = open(filename_aux+'_foe.txt','w')
    print('Risk:'.ljust(25),
          bcolors.UNDERLINE, filename_aux + '_risk.txt', bcolors.ENDC)
    risk_output_file = open(filename_aux+'_risk.txt','w')
    print('Risk Descriptor:'.ljust(25),
          bcolors.UNDERLINE, filename_aux + '_risk_descriptor.txt', bcolors.ENDC)
    risk_descriptor_output_file = open(filename_aux+'_risk_descriptor.txt','w')
    print('Static Objects:'.ljust(25),
          bcolors.UNDERLINE, filename_aux + '_static_objects.txt', bcolors.ENDC)
    static_objects_output_file = open(filename_aux+'_static_objects.txt','w')

    print('\r', bcolors.OKGREEN, 'OK', bcolors.ENDC, 'Initializing parameters.')
    print('===================================================================')

    while cap.isOpened():
        try:
            start_total = timer()
            frame_idx += 1
            print(bcolors.HEADER + bcolors.BOLD, frame_idx, bcolors.ENDC,
                  '[', round(frame_idx/float(total_number_of_frames)*100,2),
                  '%]')

            # Transform images to grey
            frame_gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame_gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # - - - Compute Optical flow - - -
            start = timer()
            of_points = optical_flow_interesting_points(frame_gray1,
                                                     OF_FEATURE_PARAMS)
            if of_points is None:
                raise ContinueError
            new_points, old_points = compute_optical_flow(frame_gray1,
                                                          frame_gray2,
                                                          np.array(
                                                              of_points).astype(
                                                              np.float32),
                                                          OF_LK_PARAMS)
            of_data = {'frame_number': frame_idx,
                       'new_points': new_points.tolist(),
                       'old_points': old_points.tolist(),
                      }
            of_output_file.write(json.dumps(of_data)+'\n')

            try:
                FOE_avg_point
            except NameError:
                FOE = (width / 2., height / 2.)
            else:
                FOE = FOE_avg_point

            mask = frame1.copy()

            aux = max(calcDistance(FOE, (0, 0)),
                      calcDistance(FOE, (0, height)),
                      calcDistance(FOE, (width, 0)),
                      calcDistance(FOE, (width, height)))
            dist_intervals = np.arange(0, aux + 1, aux / OF_CALC_INTERVALS)

            intervals, intervals_avg_distances, intervals_max_distances = \
                get_interval_metrics_for_optical_flow(new_points,
                                                      old_points,
                                                      FOE,
                                                      dist_intervals)
            end = timer()
            print('\tCompute OF took'.ljust(40),
                  round(end - start, 5), 'seconds.')

            # - - - Compute Optical flow weights - - -
            start = timer()
            mask, a_i, b_i, c_i, w_i, OFVectors, w_distances, w_objects \
                = get_lines_weights_from_optical_flow(  new_points,
                                                        old_points,
                                                        width,
                                                        height,
                                                        FOE,
                                                        objects,
                                                        frame_idx,
                                                        intervals,
                                                        intervals_avg_distances,
                                                        intervals_max_distances,
                                                        mask)
            of_weights_data = {'frame_number': frame_idx,
                               'distance_weights': w_distances,
                               'object_weights': w_objects
                              }
            of_weights_output_file.write(json.dumps(of_weights_data) + '\n')
            end = timer()
            print('\tCompute OF Weights took'.ljust(40),
                  round(end - start, 5), 'seconds.')

            # - - - Compute Focus of Expansion - - -
            start = timer()
            result = huber_loss_optimization(a_i,b_i,c_i,w_i)

            if result[0] is None and result[1] is None:
                result = l2_norm_optimization(a_i,b_i,c_i,w_i)

            foe_data = {'frame_number': frame_idx,
                        'foe_huber_loss': result
                       }
            foe_output_file.write(json.dumps(foe_data) + '\n')
            end = timer()
            print('\tCompute FOE took'.ljust(40),
                  round(end - start, 5), 'seconds.')

            # - - - Improve Focus of Expansion estimation - - -
            start = timer()
            result = iterative_improve_on_object_weights(huber_loss_optimization,
                                                         OFVectors, objects,
                                                         frame_idx, FOE, result,
                                                         dist_intervals,
                                                         intervals_avg_distances,
                                                         intervals_max_distances)

            points = points_history(points, np.array([result[0], result[1]]))
            FOE_avg_point = compute_avg_FOE(weights, points)
            end = timer()
            print('\tCompute FOE Improvements took'.ljust(40), round(end - start, 5),
                  'seconds.')

            # - - - Compute Risk Descriptor - - -
            start = timer()
            a,b1,b2,c1,c2 = divide_image_into_regions(FOE_avg_point, width, height)
            regions = np.array([a, b1, b2, c1, c2])
            riskScores = np.array([RISK_ZONE_SCORE_HIGH, RISK_ZONE_SCORE_MEDIUM,
                                   RISK_ZONE_SCORE_MEDIUM, RISK_ZONE_SCORE_LOW,
                                   RISK_ZONE_SCORE_LOW])

            firstSignature = create_risk_labels(objects, frame_idx, regions,
                                              riskScores, width, height,
                                              method=1).flatten()

            risk_descriptor_data = {'frame_number': frame_idx,
                                    'risk_descriptor': firstSignature.tolist()
                                   }
            risk_descriptor_output_file.write(json.dumps(risk_descriptor_data)+'\n')
            end = timer()
            print('\tCompute Risk Descriptor'.ljust(40), round(end - start, 5),
                  'seconds.')

            # - - - Compute Risk estimation - - -
            start = timer()
            riskPathOccupation, riskDistance = \
                        classify_risk(firstSignature,
                                      signaturesPathOccupation,
                                      signaturesPathOccupationClass,
                                      signaturesDistance,
                                      signaturesDistanceClass)
            risk_data = {'frame_number': frame_idx,
                         'risk_path_occupation': str(riskPathOccupation),
                         'risk_distance': str(riskDistance),
                        }
            risk_output_file.write(json.dumps(risk_data) + '\n')
            end = timer()
            print('\tCompute Risk'.ljust(40), round(end - start, 5),
                  'seconds.')

            print(bcolors.BOLD,
                  '\tFinal EMDPathOccupation:'.ljust(25), riskPathOccupation,
                  bcolors.ENDC)
            print(bcolors.BOLD,
                  '\tFinal EMDDistance:'.ljust(25),riskDistance,
                  bcolors.ENDC)

            # - - - Overall frame metrics - - -
            end_total = timer()
            print('\tOverall computation took'.ljust(40),
                  round(end_total - start_total, 5), 'seconds.\n')

            raise ContinueError
        except ContinueError:
            frame1 = frame2
            for i in range(SKIP_N_FRAMES):
                ret, frame2 = cap.read()
                if not ret:
                    print(bcolors.FAIL, 'ERR', bcolors.ENDC,
                      'Reading frame (', frame_idx,').')
                    sys.exit()
                frame_idx += 1
            frame_idx -= 1 # Because we increment at the begin of while

    print('===================================================================')
    cap.release()
    cv2.destroyAllWindows()
