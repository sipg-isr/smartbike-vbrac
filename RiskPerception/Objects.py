#!/usr/bin/env python
__author__ = "Miguel Costa"
__email__ = "mncosta(at)isr(dot)tecnico(dot)ulisboa(dot)pt"

from __future__ import print_function
import numpy as np
import os
from timeit import default_timer as timer

from RiskPerception.CONFIG import OBJECT_MINIMUM_SCORE


def readObjectsCSV(fileName, minScore=0.6):
    """Read the objects file and pass everything to an array."""

    # See if the file is usable (*.csv) and pass contents to an array 'objects'
    try:
        filename, file_extension = os.path.splitext(fileName)
    except AttributeError:
        return None

    if file_extension == '.csv':
        objects = np.genfromtxt(fileName,
                                delimiter=',',
                                dtype=str)
    else:
        objects = np.genfromtxt(filename +"_labels.csv",
                                delimiter=',',
                                dtype=str)


    # Make it that every line of the *.csv corresponds to row of the array
    objects = np.array([list(x) for x in objects])

    # Save only objects that the score is above the chosen minimum score (so
    # that searching objects is faster)
    objects = objects[objects[: ,-1].astype(np.float) >= minScore,:]

    # Return the objects
    return objects


def allowedObject(typeOfObject):
    """Returns True if the object type is allowed, False otherwise."""
    if typeOfObject == 'car':
        return True
    elif typeOfObject == 'person':
        return True
    elif typeOfObject == 'bicycle':
        return True
    elif typeOfObject == 'bus':
        return True
    elif typeOfObject == 'motorbike':
        return True
    elif typeOfObject == ' car':
        return True
    elif typeOfObject == ' person':
        return True
    elif typeOfObject == ' bicycle':
        return True
    elif typeOfObject == ' bus':
        return True
    elif typeOfObject == ' motorbike':
        return True
    else:
        return False


def getObjectWeight(typeOfObject, objectsScore):
    """Return the score of the object if it is allowed."""
    # If the Object type is allowed, return its score
    if allowedObject(typeOfObject):
        return objectsScore
    else:
        return 1


def getOFWeightFromObjects(objects, point, framenbr = None):
    """Return the weight of a given OF if theres an object in the OF."""

    # Is possible that a point coorresponds to one or more objects, so
    # store all weights to the diferent objects
    if framenbr is None:
        weight = 1;
        detectedObject = False
        for object in objects:
            # If the object score is above the chosen minimum
            if float(object[6]) > OBJECT_MINIMUM_SCORE:
                # Check if theres an object in the OF point
                if (float(object[1]) <= float(point[0]) <= float(object[3])) and (float(object[2]) <= float(point[1]) <= float(object[4])):
                    # Calculate the weight
                    weight = weight * getObjectWeight(object[5], float(object[6]))
                    detectedObject = True
        if detectedObject:
            if weight == 1:
                return 0
            else:
                return weight
        else:
            return 0
    else:
        weight = 1;
        detectedObject = False
        for object in objects[objects[:,0]==str(framenbr)]:
            # If the object score is above the chosen minimum
            if float(object[6]) > OBJECT_MINIMUM_SCORE:
                # Check if theres an object in the OF point
                if (float(object[1]) <= float(point[0]) <= float(object[3])) and (float(object[2]) <= float(point[1]) <= float(object[4])):
                    # Calculate the weight
                    weight = weight * getObjectWeight(object[5], float(object[6]))
                    detectedObject = True
        if detectedObject:
            if weight == 1:
                return 0
            else:
                return weight
        else:
            return 0


if __name__ == '__main__':

    print('\n\nReading Objects from file.\n'
          '===================================================================')

    start = timer()
    objects = readObjectsCSV(
        '/mnt/smartbyke/videoMiguelLisboa/GOPR0057_labels.csv', minScore=0.7)
    end = timer()

    print('Reading took', round(end-start, 5), 'seconds.')
    print('Results:')
    print('Number of objects read:'.ljust(20), objects.shape[0],
          'over', max(objects[:,0].astype(np.int)), 'frames, with',
          np.unique(objects[:,0].astype(np.int)).shape[0],
          'different frames.')

    print('\n\n')

