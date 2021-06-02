#!/usr/bin/python3
NAME = 'katera_cepiva'

import os
import matplotlib.pyplot as plt
import math
import csv
import random
import numpy as np
import rospy
import operator
import requests

from finale.srv import *

prediction = 'n.a.'
PATH1 = os.path.dirname(os.path.realpath(__file__)) + '/training2.data'
PATH2 = os.path.dirname(os.path.realpath(__file__)) + '/test2.data'

#knn classifier

# calculation of euclidead distance
def calculateEuclideanDistance(variable1, variable2, length):
    distance = 0
    for x in range(length):
        distance += pow(variable1[x] - variable2[x], 2)
    return math.sqrt(distance)


# get k nearest neigbors
def kNearestNeighbors(training_feature_vector, testInstance, k):
    distances = []
    length = len(testInstance)
    for x in range(len(training_feature_vector)):
        dist = calculateEuclideanDistance(testInstance,
                training_feature_vector[x], length)
        distances.append((training_feature_vector[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# votes of neighbors
def responseOfNeighbors(neighbors):
    all_possible_neighbors = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in all_possible_neighbors:
            all_possible_neighbors[response] += 1
        else:
            all_possible_neighbors[response] = 1
    sortedVotes = sorted(all_possible_neighbors.items(),
                         key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


# Load image feature data to training feature vectors and test feature vector
def loadDataset(
    filename,
    filename2,
    training_feature_vector=[],
    test_feature_vector=[],
    ):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(2):
                dataset[x][y] = float(dataset[x][y])
            training_feature_vector.append(dataset[x])

    with open(filename2) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(2):
                dataset[x][y] = float(dataset[x][y])
            test_feature_vector.append(dataset[x])


def knn(training_data, test_data):
    training_feature_vector = []  # training feature vector
    test_feature_vector = []  # test feature vector
    loadDataset(training_data, test_data, training_feature_vector, test_feature_vector)
    classifier_prediction = []  # predictions
    k = 3  # K value of k nearest neighbor
    for x in range(len(test_feature_vector)):
        neighbors = kNearestNeighbors(training_feature_vector, test_feature_vector[x], k)
        result = responseOfNeighbors(neighbors)
        classifier_prediction.append(result)
    return classifier_prediction[0]		        
        

def prepoznaj(req):
    #if os.path.isfile(PATH1) and os.access(PATH1, os.R_OK):
        #print ('training data is ready, classifier is loading...')
    #with open(PATH2, 'w') as myfile:
        #myfile.write(req.tabela)
    url = req.url
    r = requests.get(url, allow_redirects=True)
    open(PATH1, 'wb').write(r.content)
    pathTraining = PATH1
    f = open(PATH2, "w")
    f.write(str(req.prvaStevilka) + "," + str(req.drugaStevilka))
    f.close()
    pathTest = PATH2
    prediction = knn(pathTraining, pathTest)
    print('Best suited vaccine is:', prediction)
    return prediction

def prepoznaj_barvo():
    rospy.init_node(NAME)
    s = rospy.Service('finale/prepoznaj_cepivo', KateraCepiva, prepoznaj)
    # spin() keeps Python from exiting until node is shutdown
    rospy.spin()

if __name__ == "__main__":
    prepoznaj_barvo()