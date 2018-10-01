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
    #initialLabeledDataPerc = kwargs["initialLabeledDataPerc"]
    initialLabeledData = kwargs["initialLabeledData"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    usePCA = kwargs["usePCA"]
    classes = kwargs["classes"]
    K = kwargs["K_variation"]
    batches = kwargs["batches"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    excludingPercentage = kwargs["excludingPercentage"]
    clfName = kwargs["clfName"]
    densityFunction = kwargs["densityFunction"]
    poolSize = kwargs["poolSize"]
    isBatchMode = kwargs["isBatchMode"]
    
    print("METHOD: SVM as classifier and boundary remover and {} as CSE with cutting data method".format(densityFunction))

    arrAcc = []
    initialDataLength = 0
    excludingPercentage = 1-excludingPercentage
    finalDataLength = initialLabeledData #round((initialLabeledDataPerc)*sizeOfBatch)
    # ***** Box 1 *****
    #Initial labeled data
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    if isBatchMode:
        for t in range(batches):
            #print("passo: ",t)
            initialDataLength=finalDataLength
            finalDataLength=finalDataLength+sizeOfBatch
            #print(initialDataLength)
            #print(finalDataLength)
            # ***** Box 2 *****
            Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
            
            # ***** Box 3 *****
            #predicted = classifiers.classify(X, y, Ut, K, classes, clfName)
            clf = classifiers.svmClassifier(X, y)
            #predicted = clf.predict(Ut)
            predicted = classifiers.classify(X, y, Ut, K, classes, clfName)
            # Evaluating classification
            arrAcc.append(metrics.evaluate(yt, predicted))

            # ***** Box 4 *****
            #removing boundaryPoints for the next batch
            Ut, predicted = util.removeBoundaryPoints(clf.support_, Ut, predicted)

            #pdfs from each new points from each class applied on new arrived points
            pdfsByClass = util.pdfByClass(Ut, predicted, classes, densityFunction)

            # ***** Box 5 *****
            selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, excludingPercentage)

            # ***** Box 6 *****
            X, y = util.selectedSlicedData(Ut, predicted, selectedIndexes)
    else:
        indexUt=finalDataLength
        clf = classifiers.knn(X, y, K)
        inst = np.copy(X)
        labels = np.copy(y)
        remainingX , remainingY = util.loadLabeledData(dataValues, dataLabels, finalDataLength, len(dataValues), usePCA)
        
        for Ut, yt in zip(remainingX, remainingY):
            Ut = Ut.reshape(1, -1)
            predicted = clf.predict(Ut)
            arrAcc.append(predicted)

            clfSVM = classifiers.svmClassifier(np.vstack([inst, Ut]), np.hstack([labels, predicted]))
            #print(clfSVM.support_, indexUt)
            if indexUt in clfSVM.support_:
                '''
                print("Alerta de intruso na fronteira!")
                print(len(X))
                print(len(Ut))
                print("===============================")'''
                pass
            else:
                #print("Instances= ",inst)
                #print("New Instance",Ut)
                inst = np.vstack([inst, Ut])
                labels = np.hstack([labels, predicted])
            #print(len(inst))
            if len(inst) == poolSize:
                inst = np.asarray(inst)
                
                #Ut, predicted = util.removeBoundaryPoints(clfSVM.support_, Ut, predicted)

                pdfsByClass = util.pdfByClass(inst, labels, classes, densityFunction)
                selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, excludingPercentage)
                X, y = util.selectedSlicedData(inst, labels, selectedIndexes)
                clf = classifiers.knn(X, y, K)
                inst = np.copy(X)
                labels = np.copy(y)
            
        arrAcc = split_list(arrAcc, 100)
        arrAcc = makeAccuracy(arrAcc, remainingY)

    # returns accuracy array and last selected points
    return "SVM removing boundary + GMM", arrAcc, X, y