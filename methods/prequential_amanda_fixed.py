import numpy as np
from source import classifiers
from source import metrics
from source import util
from scipy.spatial.distance import euclidean


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
    
    print("METHOD: {} as classifier and {} as core support extraction with cutting data method".format(clfName, densityFunction))
    usePCA=False
    arrAcc = []
    arrX = []
    arrY = []
    arrUt = []
    arrYt = []
    arrClf = []
    arrPredicted = []
    initialDataLength = 0
    excludingPercentage = 1-excludingPercentage
    finalDataLength = initialLabeledData #round((initialLabeledDataPerc)*sizeOfBatch)
    reset = True

    # ***** Box 1 *****
    #Initial labeled data
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    clf = classifiers.classifier(X, y, K, clfName) #O(nd+kn)
    if isBatchMode:
        for t in range(batches):
            #print("passo: ",t)
            initialDataLength=finalDataLength
            finalDataLength=finalDataLength+sizeOfBatch
            #print(initialDataLength)
            #print(finalDataLength)
            
            Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)

            # for decision boundaries plot
            arrClf.append(clf)
            arrX.append(X)
            arrY.append(y)
            arrUt.append(np.array(Ut))
            arrYt.append(yt)

            #classifies
            predicted = clf.predict(Ut)
            
            arrPredicted.append(predicted)
            # Evaluating classification
            arrAcc.append(metrics.evaluate(yt, predicted))
            
            # ***** Box 4 *****
            #pdfs from each new points from each class applied on new arrived points
            allInstances = []
            allLabels = []
            if reset == True:
                #Considers only the last distribution (time-series like)
                pdfsByClass = util.pdfByClass(Ut, yt, classes, densityFunction)#O(nmd)
            else:
                #Considers the past and actual data (concept-drift like)
                allInstances = np.vstack([X, Ut])
                allLabels = np.hstack([y, yt])
                pdfsByClass = util.pdfByClass(allInstances, allLabels, classes, densityFunction)
                
            selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, excludingPercentage)#O(n log(n) c)
        
            # ***** Box 6 *****
            if reset == True:
                #Considers only the last distribution (time-series like)
                X, y = util.selectedSlicedData(Ut, yt, selectedIndexes)
            else:
                #Considers the past and actual data (concept-drift like)
                X, y = util.selectedSlicedData(allInstances, allLabels, selectedIndexes)#O(n)

            #training
            clf = classifiers.classifier(X, y, K, clfName) #O(nd+kn)
    else:
        inst = []
        labels = []
        clf = classifiers.classifier(X, y, K, clfName)
        remainingX , remainingY = util.loadLabeledData(dataValues, dataLabels, finalDataLength, len(dataValues), usePCA)
        reset = False
        
        for Ut, yt in zip(remainingX, remainingY):
            predicted = clf.predict(Ut.reshape(1, -1))[0]
            arrAcc.append(predicted)
            inst.append(Ut)
            labels.append(predicted)

            # for decision boundaries plot
            arrClf.append(clf)
            arrX.append(X)
            arrY.append(y)
            arrUt.append(Ut)
            arrYt.append(yt)
            arrPredicted.append(predicted)
            
            if len(inst) == poolSize:
                inst = np.asarray(inst)
                '''pdfsByClass = util.pdfByClass(inst, labels, classes, densityFunction)
                selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, excludingPercentage)
                X, y = util.selectedSlicedData(inst, labels, selectedIndexes)
                clf = classifiers.classifier(X, y, K, clfName)
                inst = []
                labels = []'''
                if reset == True:
                    #Considers only the last distribution (time-series like)
                    pdfsByClass = util.pdfByClass(inst, labels, classes, densityFunction)
                else:
                    #Considers the past and actual data (concept-drift like)
                    allInstances = np.vstack([X, inst])
                    allLabels = np.hstack([y, labels])
                    pdfsByClass = util.pdfByClass(allInstances, allLabels, classes, densityFunction)

                selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, excludingPercentage)

                if reset == True:
                    #Considers only the last distribution (time-series like)
                    X, y = util.selectedSlicedData(inst, labels, selectedIndexes)
                else:
                    #Considers the past and actual data (concept-drift like)
                    X, y = util.selectedSlicedData(allInstances, allLabels, selectedIndexes)

                clf = classifiers.classifier(X, y, K, clfName)
                inst = []
                labels = []
            
        arrAcc = split_list(arrAcc, batches)
        arrAcc = makeAccuracy(arrAcc, remainingY)
        arrYt = split_list(arrYt, batches)
        arrPredicted = split_list(arrPredicted, batches)

    # returns accuracy array and last selected points
    return "AMANDA (Fixed)", arrAcc, X, y, arrX, arrY, arrUt, arrYt, arrClf, arrPredicted