#!/usr/bin/env python
__author__ = "Miguel Costa"
__email__ = "mncosta(at)isr(dot)tecnico(dot)ulisboa(dot)pt"

from __future__ import print_function
import numpy as np
import os
import cv2
from timeit import default_timer as timer
from pyemd import emd

from RiskPerception.CONFIG import  DISTANCE_MATRIX_PATH_OCCUPATION,\
                    DISTANCE_MATRIX_DISTANCE,\
                    CLASSIFIER_RISK_NUMBER_SUBREGIONS_PER_REGION,\
                    RISK_OBJECT_SCORE_CAR,\
                    RISK_OBJECT_SCORE_BUS,\
                    RISK_OBJECT_SCORE_MOTORBIKE,\
                    RISK_OBJECT_SCORE_BICYCLE,\
                    RISK_OBJECT_SCORE_PERSON,\
                    RISK_SUBREGIONS_SCORES,\
                    OBJECT_MINIMUM_HEIGHT_IN_PROJECTION_PERCENTAGE,\
                    OBJECT_MINIMUM_HEIGHT_IN_PROJECTION_THRESHOLD


def train_classifier(data_path, PathOccupation = True, Distance=True):
    """Train Classifier with manually classified images."""

    signaturesPathOccupation = np.array([]).reshape(0, 25)
    signaturesPathOccupationClass = np.array([]).reshape(0, 1)

    signaturesDistance = np.array([]).reshape(0, 25)
    signaturesDistanceClass = np.array([]).reshape(0, 1)

    for i in range(1, 4):
        if PathOccupation:
            foldername = os.path.join(data_path, 'PathOccupied', str(i))
            for filename in os.listdir(foldername):
                if os.path.isfile(os.path.join(foldername, filename)):
                    my_data = np.genfromtxt(os.path.join(foldername, filename),
                                            delimiter=',')
                    if len(my_data.shape) == 1:
                        my_data = np.array([my_data])
                    signaturesPathOccupation = np.vstack \
                        ((signaturesPathOccupation, my_data))
                    className = np.repeat(str(i), my_data.shape[0]).reshape \
                        (my_data.shape[0], 1)
                    signaturesPathOccupationClass = np.vstack \
                        ((signaturesPathOccupationClass, className))
        if Distance:
            foldername = os.path.join(data_path, 'Distance', str(i))
            for filename in os.listdir(foldername):
                if os.path.isfile(os.path.join(foldername, filename)):
                    my_data = np.genfromtxt(os.path.join(foldername, filename),
                                            delimiter=',')
                    if len(my_data.shape) == 1:
                        my_data = np.array([my_data])
                    signaturesDistance = np.vstack((signaturesDistance,
                                                    my_data))
                    className = np.repeat(str(i), my_data.shape[0]).reshape \
                        (my_data.shape[0], 1)
                    signaturesDistanceClass = np.vstack(
                        (signaturesDistanceClass, className))


    if PathOccupation and Distance:
        return signaturesPathOccupation, signaturesPathOccupationClass, \
               signaturesDistance, signaturesDistanceClass
    elif PathOccupation and not Distance:
        return signaturesPathOccupation, signaturesPathOccupationClass
    elif not PathOccupation and Distance:
        return signaturesDistance, signaturesDistanceClass
    else:
        return None


def divide_image_into_regions(FOE, width, height):
    """Divides the image into risk zones, considereing the found FOE."""

    def createA():
        x1 = FOE[0] + width / 25.0
        x2 = FOE[0] - width / 25.0
        y1 = FOE[1] - height / 16.0
        y2 = FOE[1] - height / 16.0

        x4 = FOE[0] + width / 4.0
        x3 = FOE[0] - width / 4.0
        y4 = height
        y3 = height

        return np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    def createB(A):
        x2 = A[1][0]
        y2 = A[1][1]

        x5 = 0
        y5 = height

        x3 = A[2][0]
        y3 = A[2][1]

        b1 = np.array([[x2, y2], [x5, y5], [x3, y3]])

        x2 = A[0][0]
        y2 = A[0][1]

        x5 = width
        y5 = height

        x3 = A[3][0]
        y3 = A[3][1]

        b2 = np.array([[x2, y2], [x5, y5], [x3, y3]])

        return b1, b2

    def createC(B1, B2):
        x2 = B1[0][0]
        y2 = B1[0][1]

        x5 = B1[1][0]
        y5 = B1[1][1]

        x6 = x2
        y6 = y2 - (height - y2) / 8.

        x8 = x6 - (width - x6) / 8.
        y8 = y6

        x7 = 0
        y7 = FOE[1] - (height - FOE[1]) / 8.

        c1 = np.array([[x2, y2], [x5, y5], [x7, y7], [x8, y8], [x6, y6]])

        x2 = B2[0][0]
        y2 = B2[0][1]

        x5 = B2[1][0]
        y5 = B2[1][1]

        x6 = x2
        y6 = y2 - (height - y2) / 8.

        x8 = x6 + (width - x6) / 8.
        y8 = y6

        x7 = width
        y7 = FOE[1] - (height - FOE[1]) / 8.

        c2 = np.array([[x2, y2], [x5, y5], [x7, y7], [x8, y8], [x6, y6]])

        return c1, c2

    A = createA()
    B1, B2 = createB(A)
    C1, C2 = createC(B1, B2)

    return A, B1, B2, C1, C2


def findObjectInZone(object, zone, (width, height), subregion=None,
                     zoneNumber=None, frame=None):
    """Find the percentage of the object that is inside a given zone."""
    aux1 = np.zeros((height, width))
    aux2 = np.zeros_like(aux1)

    # If it is to calculate a subregion:
    if subregion is not None:
        subRegionNumber = subregion
        aux3 = np.zeros_like(aux1)
        topCutOff = min(zone[:, 1])
        interval = float(
            height - topCutOff) / CLASSIFIER_RISK_NUMBER_SUBREGIONS_PER_REGION
        topCut = subregion * interval + topCutOff
        botCut = (subregion + 1) * interval + topCutOff

        subRegion = np.array([[int(float(0)), int(topCut)],
                              [int(float(width)), int(topCut)],
                              [int(float(width)), int(float(botCut))],
                              [int(float(0)), int(float(botCut))]])

        cv2.fillConvexPoly(aux3, subRegion, color=255)

    # Only consider projection of the object on the floor

    if abs(int(float(object[2])) - int(
            float(object[4]))) > OBJECT_MINIMUM_HEIGHT_IN_PROJECTION_THRESHOLD:
        objectHeightMaxY = max(int(float(object[2])), int(float(object[4])))
        objectCut = max(
            objectHeightMaxY - OBJECT_MINIMUM_HEIGHT_IN_PROJECTION_PERCENTAGE * abs(
                int(float(object[2])) - int(float(object[4]))),
            OBJECT_MINIMUM_HEIGHT_IN_PROJECTION_THRESHOLD)
        # Create bounding box from object
        objectBBox = np.array([[int(float(object[1])), int(objectHeightMaxY)],
                               [int(float(object[1])), int(objectCut)],
                               [int(float(object[3])), int(objectCut)],
                               [int(float(object[3])), int(objectHeightMaxY)]])
    else:
        # Create bounding box from object
        objectBBox = np.array([[int(float(object[1])), int(float(object[2]))],
                               [int(float(object[1])), int(float(object[4]))],
                               [int(float(object[3])), int(float(object[4]))],
                               [int(float(object[3])), int(float(object[2]))]])

    # Create shapes
    cv2.fillConvexPoly(aux1, zone.astype('int32'), color=255)
    cv2.fillConvexPoly(aux2, objectBBox, color=255)

    # If subregion, calculate subregion
    if subregion is not None:
        aux1 = cv2.bitwise_and(aux3, aux1)

    # Check intersection between shapes
    result = cv2.bitwise_and(aux2, aux1)

    # Find the contours to calculate areas after
    _, contours1, hierarchy1 = cv2.findContours(
        cv2.threshold(result.astype(np.uint8), 100, 255, 0)[1], cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)
    _, contours, hierarchy = cv2.findContours(
        cv2.threshold(aux2.astype(np.uint8), 100, 255, 0)[1], cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)
    _, contoursZone, hierarchyZone = cv2.findContours(
        cv2.threshold(aux1.astype(np.uint8), 100, 255, 0)[1], cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)
    # contours1, hierarchy1 = cv2.findContours( cv2.threshold(result.astype(np.uint8),100,255, 0)[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours( cv2.threshold(aux2.astype(np.uint8),100,255, 0)[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contoursZone, hierarchyZone = cv2.findContours( cv2.threshold(aux1.astype(np.uint8),100,255, 0)[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours1 and contours:
        # Area of intersection
        area1 = cv2.contourArea(contours1[0])
        # Area of object
        area2 = cv2.contourArea(contours[0])
        # Area of image
        area3 = width * height
        # Area of zone/subzone
        area4 = cv2.contourArea(contoursZone[0])
        # Return ratio between areas
        return True, area1, area2, area3, area4
    # If theres no intersection then return 0
    return False, None, None, None, None


def create_risk_labels(objects,
                       framenbr,
                       riskZones,
                       riskZoneScores,
                       width,
                       height,
                       method = 0,
                       frame = None):
    """Returns a descriptor of risk for each sub-region on the image."""

    # Get objects in the frame
    objectsInFrame = objects[objects[:, 0] == str(framenbr)]
    # Define SubRegion Risk Scores
    riskSubZonesScores = RISK_SUBREGIONS_SCORES

    # Initiate some variables
    situationLabel = []

    # For each existent zone/region
    for i, zone in enumerate(riskZones):
        # Initiate some variables
        regionVect = np.zeros(CLASSIFIER_RISK_NUMBER_SUBREGIONS_PER_REGION)
        # For each existent sub-zone/sub-region
        for j in range(CLASSIFIER_RISK_NUMBER_SUBREGIONS_PER_REGION):
            # For each object in the frame
            for object in objectsInFrame:
                # Score given to each object type
                def objectTypeToRisk(x):
                    return {
                        ' car': RISK_OBJECT_SCORE_CAR,
                        ' bus': RISK_OBJECT_SCORE_BUS,
                        ' motorbike': RISK_OBJECT_SCORE_MOTORBIKE,
                        ' bicycle': RISK_OBJECT_SCORE_BICYCLE,
                        ' person': RISK_OBJECT_SCORE_PERSON,
                        'car': RISK_OBJECT_SCORE_CAR,
                        'bus': RISK_OBJECT_SCORE_BUS,
                        'motorbike': RISK_OBJECT_SCORE_MOTORBIKE,
                        'bicycle': RISK_OBJECT_SCORE_BICYCLE,
                        'person': RISK_OBJECT_SCORE_PERSON
                    }.get(x, 0)

                # Find if the object is inside the zone
                st, areaIntersection, areaObject, areaImage, areaZone = findObjectInZone(
                    object, zone, (width, height), subregion=j)

                # Initiate some variables
                riskScore = 0

                # If the object is inside the zone
                if st:
                    # Compute the risk of the object in the zone with regards to:
                    # 		- type of object
                    # 		- confidence score outputted by the NN
                    # 		- risk score of the region
                    # 		- risk score of the sub-region
                    # 		- ratio of object in the zone to :
                    #			- Method 0 :total area of object
                    #			- Method 0 :total area of zone

                    if method == 0:
                        riskScore = objectTypeToRisk(object[5]) * \
                                    float(object[6]) * \
                                    riskZoneScores[i] * \
                                    riskSubZonesScores[j] * \
                                    areaIntersection / float(areaObject)

                    elif method == 1:
                        riskScore = objectTypeToRisk(object[5]) * \
                                    float(object[6]) * \
                                    riskZoneScores[i] * \
                                    riskSubZonesScores[j] * \
                                    areaIntersection / float(areaZone)

                    # Add all object risk scores of one sub-region
                    regionVect[j] = regionVect[j] + riskScore
        # Append vector descriptors of different regions
        situationLabel.append(regionVect)

    # Return risk descriptor
    return np.array(situationLabel)


def classify_risk(firstSignature,
                  signaturesPathOccupation,
                  signaturesPathOccupationClass,
                  signaturesDistance,
                  signaturesDistanceClass,
                  PathOccupation=True,
                  Distance=True):

    returning = []

    if PathOccupation:
        minEMD = float("inf")
        minEMDTypePO = ''
        record = []

        for i, secondSignature in enumerate(signaturesPathOccupation):
            earthMoversDistancePathOccupation = emd(firstSignature,
                                                    secondSignature,
                                                    DISTANCE_MATRIX_PATH_OCCUPATION)
            record.append((earthMoversDistancePathOccupation,
                           signaturesPathOccupationClass[i, 0]))
            if earthMoversDistancePathOccupation < minEMD:
                minEMD = earthMoversDistancePathOccupation
                minEMDTypePO = signaturesPathOccupationClass[i]
        record = np.sort(
            np.array(record, dtype=[('dist', float), ('class', 'S10')]),
            order='dist')
        returning.append(minEMDTypePO)


    if Distance:
        minEMD = float("inf")
        minEMDTypeD = ''
        record = []
        for i, secondSignature in enumerate(signaturesDistance):
            earthMoversDistanceDistance = emd(firstSignature, secondSignature,
                                              DISTANCE_MATRIX_DISTANCE)
            record.append(
                (earthMoversDistanceDistance, signaturesDistanceClass[i, 0]))
            if earthMoversDistanceDistance < minEMD:
                # print entrou
                minEMD = earthMoversDistanceDistance
                minEMDTypeD = signaturesDistanceClass[i]
        record = np.sort(
            np.array(record, dtype=[('dist', float), ('class', 'S10')]),
            order='dist')
        returning.append(minEMDTypeD)

    return returning



if __name__ == '__main__':

    print('\n\nTraining Classifier with PathOccupation and Distance metrics.\n'
          '===================================================================')

    start = timer()
    signaturesPathOccupation, \
    signaturesPathOccupationClass, \
    signaturesDistance, \
    signaturesDistanceClass = train_classifier('../PoR_train_classifier/')
    end = timer()

    print('Training took', round(end-start, 5), 'seconds.')
    print('Results:')
    print('PathOccupation: '.ljust(20), signaturesPathOccupationClass.shape[0],
          'trained instances.')
    print('Distance: '.ljust(20), signaturesDistanceClass.shape[0],
          'trained instances.')

    print('\n\n')
