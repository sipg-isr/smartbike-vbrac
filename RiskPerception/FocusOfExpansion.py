#!/usr/bin/env python
__author__ = "Miguel Costa"
__email__ = "mncosta(at)isr(dot)tecnico(dot)ulisboa(dot)pt"

import numpy as np
from sklearn.linear_model import HuberRegressor
import math
from random import randint
import cvxopt as cvx

from RiskPerception.OpticalFlow import getWeightFromOFDistance, calcDistance
from RiskPerception.Objects import getOFWeightFromObjects

from RiskPerception.CONFIG import  CVX_SUPRESS_PRINT,\
                    HUBER_LOSS_EPSILON,\
                    RANSAC_MINIMUM_DATAPOINTS,\
                    RANSAC_NUMBER_ITERATIONS, \
                    RANSAC_MINIMUM_RATIO_INLIERS,\
                    RANSAC_MINIMUM_ERROR_ANGLE,\
                    RANSAC_RATIO_INCREASE_ETA,\
                    ITERATIVE_OBJECT_WEIGHTS_ITERATIONS,\
                    MAXIMUM_INLIERS_ANGLE,\
                    EXPONENTIAL_DECAY_NBR_WEIGHTS,\
                    EXPONENTIAL_DECAY_INITIAL,\
                    EXPONENTIAL_DECAY_TAU


def l1_norm_optimization(a_i, b_i, c_i, w_i=None):
    """Solve l1-norm optimization problem."""

    cvx.solvers.options['show_progress'] = not CVX_SUPRESS_PRINT
    # Non-Weighted optimization:
    if w_i is None:
        # Problem must be formulated as sum |P*x - q|
        P = cvx.matrix([[cvx.matrix(a_i)], [cvx.matrix(b_i)]])
        q = cvx.matrix(c_i * -1)

        # Solve the l1-norm problem
        u = cvx.l1.l1(P, q)

        # Get results
        x0, y0 = u[0], u[1]

    # Weighted optimization:
    else:
        # Problem must be formulated as sum |P*x - q|
        P = cvx.matrix([[cvx.matrix(np.multiply(a_i, w_i))],
                        [cvx.matrix(np.multiply(b_i, w_i))]])
        q = cvx.matrix(np.multiply(w_i, c_i * -1))

        # Solve the l1-norm problem
        u = cvx.l1.l1(P, q)

        # Get results
        x0, y0 = u[0], u[1]

    # return resulting point
    return (x0, y0)


def l2_norm_optimization(a_i, b_i, c_i, w_i=None):
    """Solve l2-norm optimization problem."""

    # Non-Weighted optimization:
    if w_i is None:
        aux1 = -2 * ((np.sum(np.multiply(b_i, b_i))) * (
            np.sum(np.multiply(a_i, a_i))) / float(
            np.sum(np.multiply(a_i, b_i))) - (np.sum(np.multiply(a_i, b_i))))
        aux2 = 2 * ((np.sum(np.multiply(b_i, b_i))) * (
            np.sum(np.multiply(a_i, c_i))) / float(
            np.sum(np.multiply(a_i, b_i))) - (np.sum(np.multiply(b_i, c_i))))

        x0 = aux2 / float(aux1)

        y0 = (-(np.sum(np.multiply(a_i, c_i))) - (
            np.sum(np.multiply(a_i, a_i))) * x0) / float(
            np.sum(np.multiply(a_i, b_i)))

    # Weighted optimization:
    else:
        aux1 = -2 * ((np.sum(np.multiply(np.multiply(b_i, b_i), w_i))) * (
            np.sum(np.multiply(np.multiply(a_i, a_i), w_i))) / float(
            np.sum(np.multiply(np.multiply(a_i, b_i), w_i))) - (
                         np.sum(np.multiply(np.multiply(a_i, b_i), w_i))))
        aux2 = 2 * ((np.sum(np.multiply(np.multiply(b_i, b_i), w_i))) * (
            np.sum(np.multiply(np.multiply(a_i, c_i), w_i))) / float(
            np.sum(np.multiply(np.multiply(a_i, b_i), w_i))) - (
                        np.sum(np.multiply(np.multiply(b_i, c_i), w_i))))

        x0 = aux2 / float(aux1)

        y0 = (-(np.sum(np.multiply(np.multiply(a_i, c_i), w_i))) - (
            np.sum(np.multiply(np.multiply(a_i, a_i), w_i))) * x0) / float(
            np.sum(np.multiply(np.multiply(a_i, b_i), w_i)))

    # return resulting point
    return (x0, y0)


def huber_loss_optimization(a_i, b_i, c_i, w_i=None):
    """Solve Huber loss optimization problem."""

    for k in range(5):
        try:
            # Non-Weighted optimization:
            if w_i is None:

                huber = HuberRegressor(fit_intercept=True, alpha=0.0,
                                       max_iter=100, epsilon=HUBER_LOSS_EPSILON)

                X = -1 * np.concatenate(
                    (a_i.reshape(a_i.shape[0], 1),
                     b_i.reshape(b_i.shape[0], 1)), axis=1)
                y = c_i

                huber.fit(X, y)

                # Get results
                x0, y0 = huber.coef_ + np.array([0., 1.]) * huber.intercept_

            # Weighted optimization:
            else:
                huber = HuberRegressor(fit_intercept=True, alpha=0.0,
                                       max_iter=100, epsilon=HUBER_LOSS_EPSILON)

                X = -1 * np.concatenate(
                    (a_i.reshape(a_i.shape[0], 1),
                     b_i.reshape(b_i.shape[0], 1)), axis=1)
                y = c_i
                sampleWeight = w_i

                huber.fit(X, y, sample_weight=sampleWeight)

                # Get results
                x0, y0 = huber.coef_ + np.array([0., 1.]) * huber.intercept_
        except ValueError:
            pass
        else:
            # return resulting point
            return x0, y0
    else:
        return None, None


def select_subset(OFVectors):
    """Select a subset of a given set."""
    subset = np.array([]).reshape(0, 4)
    for i in range(RANSAC_MINIMUM_DATAPOINTS):
        idx = randint(0, (OFVectors.shape)[0] - 1)
        subset = np.vstack((subset, np.array([OFVectors[idx]])))
    return subset


def fit_model(subset):
    """Return a solution for a given subset of points."""
    # Initialize some empty variables
    a_i = np.array([])
    b_i = np.array([])
    c_i = np.array([])

    # Save the lines coeficients of the form a*x + b*y + c = 0 to the variables
    for i in range(subset.shape[0]):
        a1, b1, c1, d1 = subset[i]

        pt1 = (a1, b1)
        # So we don't divide by zero
        if (a1 - c1) == 0:
            continue
        a = float(b1 - d1) / float(a1 - c1)
        b = -1
        c = (b1) - a * a1

        denominator = float(a ** 2 + 1)

        a_i = np.append(a_i, a / denominator)
        b_i = np.append(b_i, b / denominator)
        c_i = np.append(c_i, c / denominator)

    # Solve a optimization problem with Minimum Square distance as a metric
    (x0, y0) = l2_norm_optimization(a_i, b_i, c_i)

    # Return FOE
    return (x0, y0)


def get_intersect_point(a1, b1, c1, d1, x0, y0):
    """Get the point on the lines that passes through (a1,b1) and (c1,d1) and s closest to the point (x0,y0)."""
    a = 0
    if (a1 - c1) != 0:
        a = float(b1 - d1) / float(a1 - c1)
    c = b1 - a * a1

    # Compute the line perpendicular to the line of the OF vector that passes throught (x0,y0)
    a_aux = 0
    if a != 0:
        a_aux = -1 / a
    c_aux = y0 - a_aux * x0

    # Get intersection of the two lines
    x1 = (c_aux - c) / (a - a_aux)
    y1 = a_aux * x1 + c_aux

    return (x1, y1)


def find_angle_between_lines(x0, y0, a1, b1, c1, d1):
    """Finds the angle between two lines."""

    # Line 1 : line that passes through (x0,y0) and (a1,b1)
    # Line 2 : line that passes through (c1,d1) and (a1,b1)

    angle1 = 0
    angle2 = 0
    if (a1 - x0) != 0:
        angle1 = float(b1 - y0) / float(a1 - x0)
    if (a1 - c1) != 0:
        angle2 = float(b1 - d1) / float(a1 - c1)

    # Get angle in degrees
    angle1 = math.degrees(math.atan(angle1))
    angle2 = math.degrees(math.atan(angle2))

    ang_diff = angle1 - angle2
    # Find angle in the interval [0,180]
    if math.fabs(ang_diff) > 180:
        ang_diff = ang_diff - 180

    # Return angle between the two lines
    return ang_diff


def find_inliers_outliers(x0, y0, OFVectors):
    """Find set of inliers and outliers of a given set of optical flow vectors and the estimated FOE."""
    # Initialize some varaiables
    inliers = np.array([])
    nbr_inlier = 0

    # Find inliers with the angle method

    # For each vector
    for i in range((OFVectors.shape)[0]):
        a1, b1, c1, d1 = OFVectors[i]
        # Find the angle between the line that passes through (x0,y0) and (a1,b1) and the line that passes through (c1,d1) and (a1,b1)
        ang_diff = find_angle_between_lines((x0, y0), (a1, b1, c1, d1))
        # If the angle is below a certain treshold consider it a inlier
        if -RANSAC_MINIMUM_ERROR_ANGLE < ang_diff < RANSAC_MINIMUM_ERROR_ANGLE:
            # Increment number of inliers and add save it
            nbr_inlier += 1
            inliers = np.append(inliers, i)
    # Compute the ratio of inliers to overall number of optical flow vectors
    ratioInliersOutliers = float(nbr_inlier) / (OFVectors.shape)[0]

    # Return set of inliers and ratio of inliers to overall set
    return inliers, ratioInliersOutliers


def RANSAC(OFVectors):
    """Estimate the FOE of a set of optical flow (OF) vectors using RANSAC."""
    # Initialize some variables
    savedRatio = 0
    FOE = (0, 0)
    inliersModel = np.array([])

    # Repeat iterations for a number of times
    for i in range(RANSAC_NUMBER_ITERATIONS):
        # Randomly initial select OF vectors
        subset = select_subset(OFVectors)
        # Estimate a FOE for the set of OF vectors
        (x0, y0) = fit_model(subset)
        # Find the inliers of the set for the estimated FOE
        inliers, ratioInliersOutliers = find_inliers_outliers((x0, y0), OFVectors)
        # If ratio of inliers is bigger than the previous iterations, save current solution
        if savedRatio < ratioInliersOutliers:
            savedRatio = ratioInliersOutliers
            inliersModel = inliers
            FOE = (x0, y0)
        # If ratio is acceptable, stop iterating and return the found solution
        if savedRatio > RANSAC_MINIMUM_RATIO_INLIERS and RANSAC_MINIMUM_RATIO_INLIERS != 0:
            break

    # Return the estimated FOE, the found inliers ratio and the set of inliers
    return FOE, savedRatio, inliersModel


def RANSAC_ImprovedModel(OFVectors):
    """Estimate the FOE of a set of optical flow (OF) vectors using a form of RANSAC method."""
    # Initialize some variables
    FOE = (0, 0)
    savedRatio = 0
    inliersModel = np.array([])

    # Repeat iterations for a number of times
    for i in range(RANSAC_NUMBER_ITERATIONS):
        # Randomly select initial OF vectors
        subset = select_subset(OFVectors)
        # Estimate a FOE for the set of OF vectors
        (x0, y0) = fit_model(subset)
        # Find the inliers of the set for the estimated FOE
        inliers, ratioInliersOutliers = find_inliers_outliers((x0, y0),
                                                             OFVectors)
        # Initialize some varaibles
        iter = 0
        ratioInliersOutliers_old = 0
        # While the ratio of inliers keeps on increasing
        while ((inliers.shape)[
                   0] != 0 and ratioInliersOutliers - ratioInliersOutliers_old > RANSAC_RATIO_INCREASE_ETA):
            # Repeat iterations for a number of times
            if iter > RANSAC_NUMBER_ITERATIONS:
                break
            iter += 1
            # Select a new set of OF vectors that are inliers tot he estimated FOE
            for i in range((inliers.shape)[0]):
                subset = np.vstack(
                    (subset, np.array([OFVectors[int(inliers[i])]])))
            # Estimate a FOE for the new set of OF vectors
            (x0, y0) = fit_model(subset)
            # Save the previous iteration ratio if inliers
            ratioInliersOutliers_old = ratioInliersOutliers
            # Find the inliers of the set for the estimated FOE
            inliers, ratioInliersOutliers = find_inliers_outliers((x0, y0),
                                                                OFVectors)

            # If ratio of inliers is bigger than the previous iterations, save current solution
            if savedRatio < ratioInliersOutliers:
                savedRatio = ratioInliersOutliers
                inliersModel = inliers
                FOE = (x0, y0)
            # If ratio is acceptable, stop iterating and return the found solution
            if savedRatio > RANSAC_MINIMUM_RATIO_INLIERS and RANSAC_MINIMUM_RATIO_INLIERS != 0:
                break
        # If ratio is acceptable, stop iterating and return the found solution
        if savedRatio > RANSAC_MINIMUM_RATIO_INLIERS and RANSAC_MINIMUM_RATIO_INLIERS != 0:
            break

    # Return the estimated FOE, the found inliers ratio and the set of inliers
    return FOE, savedRatio, inliersModel


def vectorOFRightDirection(OFvect, FOE):
    """Returns True if OF vector is pointing away from the FOE, False otherwise."""
    # Get points of optical flow vector
    a1, b1, c1, d1 = OFvect

    # If left side of FOE
    if a1 <= FOE[0]:
        if c1 <= a1:
            return False
    # If right side of FOE
    else:
        if c1 >= a1:
            return False
    return True


def improveOFObjectsWeights(OF, objects, framenbr, FOE, currResult,
                            dist_intervals=None, dist_avg_int=None,
                            dist_max_int=None):
    """Iteratively check weights coming from objects."""

    staticObjects = objects[objects[:, 0] == str(framenbr)].copy()
    staticObjects[:, 6] = 0

    a_i = np.array([])
    b_i = np.array([])
    c_i = np.array([])
    w_i = np.array([])

    (x0, y0) = currResult

    for i in range((OF.shape)[0]):
        a1, b1, c1, d1 = OF[i]

        # So we don't divide by zero
        if (a1 - c1) == 0:
            continue
        a = float(b1 - d1) / float(a1 - c1)
        b = -1
        c = (b1) - a*a1

        lengthLine = math.sqrt((a1-c1)**2 + (b1-d1)**2)

        distToFOE = calcDistance((a1, b1), FOE)
        for j in range(dist_intervals.shape[0] - 1):
            if dist_intervals[j] < distToFOE < dist_intervals[j + 1]:
                break
        distance_weight = (
            getWeightFromOFDistance((lengthLine), (dist_avg_int[j]),
                                    (dist_max_int[j])))

        if getOFWeightFromObjects(objects, (a1, b1), framenbr) != 0:
            for object in staticObjects:
                if (float(object[1]) <= float(a1) <= float(object[3])) and (
                        float(object[2]) <= float(b1) <= float(object[4])):
                    if (-MAXIMUM_INLIERS_ANGLE <
                            find_angle_between_lines((x0,y0), (a1, b1, c1,d1)) <
                        MAXIMUM_INLIERS_ANGLE) and \
                                vectorOFRightDirection((a1, b1, c1, d1), FOE):
                        object_weight = 1
                        object[6] = str(float(object[6]) + 1)
                    else:
                        object_weight = 0
                        object[6] = str(float(object[6]) - 1)

        else:
            object_weight = 1

        weight = distance_weight * object_weight

        denominator = float(a ** 2 + 1)

        a_i = np.append(a_i, a / denominator)
        b_i = np.append(b_i, b / denominator)
        c_i = np.append(c_i, c / denominator)
        w_i = np.append(w_i, [weight])

    return a_i, b_i, c_i, w_i, staticObjects


def iterative_improve_on_object_weights(optimization_method, OF, objects,
                                        framenbr, FOE, curr_result,
                                        dist_intervals, dist_avg_int,
                                        dist_max_int):
    for i in range(ITERATIVE_OBJECT_WEIGHTS_ITERATIONS):
        a_i, b_i, c_i, w_i, staticObjects = \
            improveOFObjectsWeights(OF,
                                    objects,
                                    framenbr,
                                    FOE,
                                    curr_result,
                                    dist_intervals=dist_intervals,
                                    dist_avg_int=dist_avg_int,
                                    dist_max_int=dist_max_int)

        (x0, y0) = optimization_method(a_i, b_i, c_i, w_i)
        if x0 is None and y0 is None:
            return curr_result

        return (x0, y0)


def negative_exponential_decay(x, initial = None, tau = None):
    """Returns the value of a negative exponential [f(x) = No * e^-(t*x)	]."""
    if initial is None:
        initial = EXPONENTIAL_DECAY_INITIAL
    if tau is None:
        tau = EXPONENTIAL_DECAY_TAU
    return initial * math.exp(-1 * tau * x)


def generate_weights(nbr_weights):
    """Generate negative exponential weights."""
    weights = np.array([])
    for i in range(nbr_weights):
        weights = np.append(weights, negative_exponential_decay(i))
    return weights


def points_history(points, newPoint):
    """Refresh the points history. Delete the oldest point and add the new point."""
    points = np.delete(points, (points.shape)[1] - 1, axis=1)
    points = np.insert(points, 0, newPoint, axis=1)

    return points


def initialize_points_history(width, height, nbr_points = None):
    """Initialize the point history memory."""
    if nbr_points is None:
        nbr_points = EXPONENTIAL_DECAY_NBR_WEIGHTS
    points = np.array([[],[]])
    for i in range(nbr_points):
        # Initialize with points corresponding to the center of the image
        points = np.insert(points, 0, [width/2.0,height/2.0],axis=1)
    return points


def compute_avg_FOE(weights, points):
    """Computes the weighted mean of points given the weights."""
    num_x = 0
    num_y = 0
    den = 0
    # for each point, calculate the x and y mean
    for i in range(points.shape[1]):
        num_x = num_x + weights[i] * points[0][i]
        num_y = num_y + weights[i] * points[1][i]
        den = den + weights[i]
    # return the weighted means of x and y
    return np.array([num_x/den, num_y/den])


if __name__ == '__main__':
    pass
