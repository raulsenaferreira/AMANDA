import numpy as np
from source import classifiers
from source import metrics
from source import util



def start(**kwargs):
    dataValues = kwargs["dataValues"]
    dataLabels = kwargs["dataLabels"]
    initialLabeledData = kwargs["initialLabeledData"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    classes = kwargs["classes"]
    batches = kwargs["batches"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    p = kwargs["excludingPercentage"]
    K = kwargs["K_variation"]
    clfName = kwargs["clfName"]
    densityFunction='gmmBIC'
    distanceMetric = 'mahalanobis'

    print("METHOD: {} as classifier and GMM with BIC and Mahalanobis as core support extraction".format(clfName))
    usePCA=False
    arrAcc = []
    arrX = []
    arrY = []
    arrUt = []
    arrYt = []
    arrClf = []
    arrPredicted = []
    initialDataLength = 0
    finalDataLength = initialLabeledData #round((initialLabeledDataPerc)*sizeOfBatch)
    # ***** Box 1 *****
    #Initial labeled data
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    #Starting the process
    for t in range(batches):
        #print("Step: ", t)
        initialDataLength=finalDataLength
        finalDataLength=finalDataLength+sizeOfBatch
        # ***** Box 2 *****
        Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)

        # ***** Box 3 *****
        #predicted = classifiers.classify(X, y, Ut, K, classes, clfName)
        clf = classifiers.labelPropagation(X, y, K)
        # for decision boundaries plot
        arrClf.append(clf)
        arrX.append(X)
        arrY.append(y)
        arrUt.append(np.array(Ut))
        arrYt.append(yt)
        predicted = clf.predict(Ut)
        arrPredicted.append(predicted)

        # Evaluating classification
        arrAcc.append(metrics.evaluate(yt, predicted))
        
        # ***** Box 4 *****
        indexesByClass = util.slicingClusteredData(y, classes)
        bestModelSelectedByClass = util.loadBestModelByClass(X, indexesByClass)
        
        # ***** Box 5 *****
        predictedByClass = util.slicingClusteredData(predicted, classes)
        selectedIndexes = util.mahalanobisCoreSupportExtraction(Ut, predictedByClass, bestModelSelectedByClass, p)
        #selectedIndexes = np.hstack([selectedIndexes[0],selectedIndexes[1]])
        stackedIndexes=selectedIndexes[0]
        for i in range(1, len(selectedIndexes)):
            stackedIndexes = np.hstack([stackedIndexes,selectedIndexes[i]])
        selectedIndexes =  stackedIndexes   
        
        # ***** Box 6 *****
        X, y = util.selectedSlicedData(Ut, predicted, selectedIndexes)

        
    return "COMPOSE GMM", arrAcc, X, y, arrX, arrY, arrUt, arrYt, arrClf, arrPredicted