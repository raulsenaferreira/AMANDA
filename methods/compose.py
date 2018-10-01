import numpy as np
from source import util
from source import metrics
from source import classifiers



def start(**kwargs):
    dataValues = kwargs["dataValues"]
    dataLabels = kwargs["dataLabels"]
    initialLabeledDataPerc = kwargs["initialLabeledDataPerc"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    usePCA = kwargs["usePCA"]
    classes = kwargs["classes"]
    batches = kwargs["batches"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    CP=kwargs["CP"]
    alpha=kwargs["alpha"]
    K = kwargs["K_variation"]
    
    print("METHOD: Cluster and label as classifier and Alpha-Shape as core support extraction")

    initialDataLength = 0
    finalDataLength = round((initialLabeledDataPerc)*sizeOfBatch)
    arrAcc = []
    
    # ***** Box 1 *****
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    initialDataLength=finalDataLength
    finalDataLength=sizeOfBatch
    #Starting the process
    for t in range(batches):
        #print("Step: ", t)
        # ***** Box 2 *****
        Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)

        # ***** Box 3 *****
        predicted = classifiers.clusterAndLabel(X, y, Ut, K, classes)
        instances = np.vstack([X, Ut])
        labelsInstances = np.hstack([y, predicted])
        # Evaluating classification
        arrAcc.append(metrics.evaluate(yt, predicted))
        
        # ***** Box 4 & Box 5 *****
        threshold = int( len(instances)*(1-CP) )
        indexesByClass = util.slicingClusteredData(labelsInstances, classes)
        selectedPointsByClass, selectedIndexesByClass = util.loadGeometricCoreExtractionByClass(instances, indexesByClass, alpha, threshold)
        
        # ***** Box 6 *****
        X = np.vstack([selectedPointsByClass[0], selectedPointsByClass[1]])
        y = np.hstack([labelsInstances[selectedIndexesByClass[0]], labelsInstances[selectedIndexesByClass[1]]])
        #X, y = util.selectedSlicedData(selectedPointsByClass, selectedIndexesByClass, labelsInstances)
        initialDataLength=finalDataLength
        finalDataLength+=sizeOfBatch   
    #metrics.finalEvaluation(arrAcc)
    
    return "COMPOSE", arrAcc, X, y