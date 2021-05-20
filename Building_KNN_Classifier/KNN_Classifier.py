#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Name: Parvez Golam
# Date: 02/03/2021
#  KNN Classifier on training and test datasets related to 
#  red variants of the Portuguese "Vinho Verde" wine.
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def minmaxNormalize(dataset):
    '''Rescales the dataset into the range between 0 and 1.
    '''
    for col in dataset:
        min_value = min(dataset[col])
        max_value = max(dataset[col])
        for i in range(len(dataset[col])):
            dataset[col][i] = (dataset[col][i] - min_value) / (max_value - min_value)

    return dataset
    
def standardNormalize(dataset):
    '''Rescales the dataset based on Standard Normalization
    '''
    for col in dataset:
        mean_value = np.mean(dataset[col])
        std_dev = np.std(dataset[col])
        for i in range(len(dataset[col])):
            dataset[col][i] = (dataset[col][i] - mean_value) / std_dev

    return dataset

def euclideanDistance(query_point, train_sample):
    return np.sqrt( np.sum( np.square( query_point - train_sample) ) ) 

def manhattanDistance(query_point, train_sample):
    return np.sum( np.absolute( query_point - train_sample) )

def calculateDistance(dataSet, query_point, distance_type):
    ''' Returns distance calculated based on 'diatance_type'
        (Manhattan/Eucledian)
    '''
    distance = np.zeros(len(dataSet))

    for i in range(len(dataSet)):
        if distance_type == EUCLEDIAN :
            d = euclideanDistance( query_point, dataSet.iloc[i])
        else:
            d = manhattanDistance( query_point, dataSet.iloc[i])
        distance[i] = d
    
    return distance

def knn(k, dis_typ, x_train, x_test, y_train, y_test):
    ''' Based on the value of 'k' and distance type'dis_typ' predicts 
        class level using KNN classifier,
        and returns accuracy(%) of the algrothm.
    '''
    #count of the number of correct classifications
    correctClassifications = 0

    # classify each instance from the test feature dataset in turn
    for num in range(0, len(x_test)):
        test_label  = y_test.iloc[num]

        # Get distance of test query point and train features
        distances = calculateDistance(x_train, x_test.iloc[num], dis_typ) 

        inds = distances.argsort() 
        neighbors = y_train[inds][:k]

        # most frequent class in K neighbors 
        predict_label = most_frequent(neighbors)
        #predict_label = mode( neighbors )[0][0] 
    
        if test_label == predict_label:
            correctClassifications += 1

    # Accurary(%) of Knn
    accuracy = ( (correctClassifications / len(x_test) ) * 100 )

    return accuracy 

def most_frequent(df):
    ''' Finds most frequent value from'df'
    ''' 
    count1, count2 = 0, 0 
    for ele in df:
        if ele == 1 :
            count1 += 1 
        else:
            count2 += 1

    # store the element with higher count
    result = 1 if count1 > count2 else -1

    return result

def Weightedknn(k, dist_typ, x_train, x_test, y_train, y_test):
    ''' Based on the value of 'k' and distance type 'dist_typ' predicts
        class label using Weighted-KNN classifer,
        with inverse distance squared as distance weighted metric
        and returns accuracy(%) of the algrothm.
    '''
    # Count of the number of correct classifications
    correctClassifications = 0

    # classify each instance from the test feature dataset in turn
    for num in range(0, len(x_test)):
        test_label  = y_test.iloc[num]

        # Get distance of test query point and train features
        distances = calculateDistance(x_train, x_test.iloc[num], dist_typ)

        inds = distances.argsort()
        neighbors = y_train[inds][:k]

        weight1 = 0 # weighted sum of group 1 
        weight2 = 0 # weighted sum of group -1 

        # weight clculation -inverse distance squared
        for i in neighbors.index :
            if neighbors[i] == 1 : 
                weight1 +=  1 / np.square(distances[i])
            elif neighbors[i] == -1 :  
                weight2 +=  1 / np.square(distances[i]) 

        predict_label = 1 if weight1>weight2 else -1
    
        if test_label == predict_label:
            correctClassifications += 1

    # Accurary(%) of Weighted-Knn
    accuracyWeightedModel = ( (correctClassifications / len(x_test) ) * 100 )

    return accuracyWeightedModel 

def getFeatureSet(data_frame):
    return data_frame[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides","free sulfur dioxide",
                       "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]

def getLabelSet(data_frame):
    return data_frame["Quality"]

def loadFiles():
    # read the dataset training and test data
    df_train = pd.read_csv("wine-data-project-train.csv")
    df_test  = pd.read_csv("wine-data-project-test.csv")

    return df_train, df_test


def main(classifier, dist_typ, flag_norm ):
    '''
    Evaluates the acuuray of KNN/Weighted-KNN based on 'knn_type', 
    distance type 'dist_typ'(Eucledian or Manhattan),
    on scaled or unscaled data
    for differnt values of k.
    '''
    # train & test data
    df_train, df_test = loadFiles()

    # Features - test and training dataset
    test_feature =  getFeatureSet(df_test)
    train_feature = getFeatureSet(df_train)

    # Labels - test and training set
    test_label = getLabelSet(df_test)
    train_label = getLabelSet(df_train)

    # Normalize -Features dataset
    if flag_norm == MINMAX_NORM : # MinMax Normalization
        test_feature = minmaxNormalize(test_feature)
        train_feature = minmaxNormalize(train_feature)
    elif flag_norm == Z_NORM : # Z-Score Normalization
        test_feature = standardNormalize(test_feature)
        train_feature = standardNormalize(train_feature)

    allResults = []

    # Get accuracy of Knn and Weighted-Knn
    # for all the odd values of k between 2 and 40
    for k in range(3, 40, 2):
        
        if classifier == WKNN :   # Weighted-Knn
            accuracyWeightedModel = Weightedknn(k, dist_typ,\
                train_feature, test_feature, train_label, test_label)
            allResults.append(accuracyWeightedModel)

        else:                           # Knn
            accuracy = knn(k, dist_typ, train_feature,test_feature,\
                train_label, test_label)
            allResults.append(accuracy)

    sns.set_style("darkgrid")
    plt.plot( list(range(3, 40, 2)), allResults)
    plt.show()
    

#------------ Driver program
# Constants
WKNN = "Weighted-Knn"
KNN = "Knn"
MANHATTAN = 'M'
EUCLEDIAN = 'E'
MINMAX_NORM = 'MinMax'
Z_NORM = 'Z-Score'
NO_NORM = 'No'


classifier, distance_type, normalization = WKNN, MANHATTAN , Z_NORM
# Based on the combination selected get the accuracy graph of the classifier
main( classifier,distance_type, normalization)
