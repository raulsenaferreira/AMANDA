import numpy as np
from source import classifiers
from source import metrics
from source import util



def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]


def makeAccuracy(arrAllAcc, arrTrueY):
    arrAcc = []
    ini = 0
    end = ini
    for predicted in arrAllAcc:
        predicted = np.asarray(predicted)
        predicted = predicted.flatten()
        batchSize = len(predicted)
        ini=end
        end=end+batchSize

        yt = arrTrueY[ini:end]
        arrAcc.append(metrics.evaluate(yt, predicted))
        
    return arrAcc


def start(**kwargs):
    dataValues = kwargs["dataValues"]
    dataLabels = kwargs["dataLabels"]
    initialLabeledData = kwargs["initialLabeledData"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    classes = kwargs["classes"]
    K = kwargs["K_variation"]
    batches = kwargs["batches"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    excludingPercentage = kwargs["excludingPercentage"]
    clfName = kwargs["clfName"]
    densityFunction = kwargs["densityFunction"]
    poolSize = kwargs["poolSize"]
    isBatchMode = kwargs["isBatchMode"]

    print("METHOD: Incremental {0} as classifier".format(clfName))
    usePCA=False
    arrAcc = []
    arrX = []
    arrY = []
    arrUt = []
    arrYt = []
    arrClf = []
    arrPredicted = []
    Ut = []
    predicted = []
    initialDataLength = 0
    finalDataLength = initialLabeledData
    # ***** Box 1 *****
    #Initial labeled data
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    clf = classifiers.classifier(X, y, K, clfName)

    if isBatchMode:
        for t in range(batches):
            clf.fit(X, y)

            initialDataLength=finalDataLength
            finalDataLength=finalDataLength+sizeOfBatch
                       
            Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
            
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
            # incremental
            X, y = np.vstack([X, Ut]), np.hstack([y, predicted])
    else:
        inst = X
        labels = y
        remainingX , remainingY = util.loadLabeledData(dataValues, dataLabels, finalDataLength, len(dataValues), usePCA)
        
        for Ut, yt in zip(remainingX, remainingY):
            predicted = clf.predict(Ut.reshape(1, -1))
            arrAcc.append(predicted)
            inst = np.vstack([inst, Ut])
            labels = np.hstack([labels, predicted])

            # for decision boundaries plot
            arrClf.append(clf)
            arrX.append(X)
            arrY.append(y)
            arrUt.append(Ut)
            arrYt.append(yt)
            arrPredicted.append(predicted)
            
            clf = classifiers.classifier(inst, labels, K, clfName)
            
        arrAcc = split_list(arrAcc, batches)
        arrAcc = makeAccuracy(arrAcc, remainingY)
        arrYt = split_list(arrYt, batches)
        arrPredicted = split_list(arrPredicted, batches)
    
    return "Incremental SSL", arrAcc, X, y, arrX, arrY, arrUt, arrYt, arrClf, arrPredicted