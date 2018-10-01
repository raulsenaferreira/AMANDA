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


_SQRT2 = np.sqrt(2)
def hellinger(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / _SQRT2


def cuttingPercentage(Xt_1, Xt, t=None):
    res = []
    
    for i in range(Xt_1.shape[1]):
        P = Xt_1[:, i]
        Q = Xt[:, i]
        bins = int(np.sqrt(len(Xt_1)))
        hP = np.histogram(P+(-np.min(P)), bins=bins)
        hQ = np.histogram(Q+(-np.min(Q)), bins=bins)
        res.append(hellinger(hP[1], hQ[1]))
    
    H = np.mean(res)
    alpha = _SQRT2-H
    #print(t, H, alpha)
    #if alpha < 0:
    #    alpha *= -1
    
    if alpha > 0.9:
        alpha = 0.9
    elif alpha < 0.5:
        alpha = 0.5
    return 1-alpha #percentage of similarity


def cuttingPercentage2(Xt_1, Xt, t=None):
    res = []
    reset = False
    for i in range(Xt_1.shape[1]):
        P = Xt_1[:, i]
        Q = Xt[:, i]
        bins = int(np.sqrt(len(Xt_1)))
        hP = np.histogram(P+(-np.min(P)), bins=bins)
        hQ = np.histogram(Q+(-np.min(Q)), bins=bins)
        res.append(hellinger(hP[1], hQ[1]))

    H = np.mean(res)
    lowerBound = np.power(H, 2)
    upperBound = np.sqrt(2)*H

    similarity = 1-H/upperBound #1 - (((100 * res)/x)/100)#(100 - ((100 * res)/x))/100
    middle = abs(upperBound - lowerBound)
    #print(t, H, lowerBound, middle, similarity)
      
    if lowerBound > upperBound:
        #print(t, res, similarity)
        similarity = abs(middle-H)
        reset = True
    else:
        similarity = H
        reset = False

    #similarity = 0.5+((H / upperBound))
    
    if similarity > 0.9:
        similarity = 0.9
    elif similarity < 0.5:
        similarity = 0.5
    
    #print("step {}, similarity = {}, reset = {} ".format(t, similarity, reset))
    return 1-similarity, reset #percentage of dissimilarity


def cuttingPercentageByClass(Xt_1, Xt, yt_1, yt, classes, t=None):
    x = np.sqrt(2)
    reset = False

    hellinger_distance_by_class = {}
    similarityByClass = {}

    indexes_Xt_1_ByClass = util.slicingClusteredData(yt_1, classes)
    indexes_Xt_ByClass = util.slicingClusteredData(yt, classes)    

    for c in classes:
        res = []
        for i in range(Xt_1.shape[1]):
            P = Xt_1[indexes_Xt_1_ByClass[c], i]
            Q = Xt[indexes_Xt_ByClass[c], i]

            bins = int(np.sqrt(len(indexes_Xt_1_ByClass[c])))

            hP = np.histogram(P+(-np.min(P)), bins=bins)
            hQ = np.histogram(Q+(-np.min(Q)), bins=bins)
            res.append(hellinger(hP[1], hQ[1]))

        res = np.mean(res)
        similarity = 1 - (((100 * res)/x)/100)#(100 - ((100 * res)/x))/100
        #print(t,res, similarity)
        if similarity < 0:
            reset = True
        elif similarity > 0:
            reset = False

        similarity = 0.5+((res / x)/10)
        if similarity > 0.9:
            similarity = 0.9

        similarityByClass.update({c: similarity})
        #print(t,c,similarity)

    return similarityByClass, reset #percentage of similarity


def start(**kwargs):
    dataValues = kwargs["dataValues"]
    dataLabels = kwargs["dataLabels"]
    initialLabeledData = kwargs["initialLabeledData"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    classes = kwargs["classes"]
    K = kwargs["K_variation"]
    batches = kwargs["batches"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    clfName = kwargs["clfName"]
    densityFunction = kwargs["densityFunction"]
    poolSize = kwargs["poolSize"]
    isBatchMode = kwargs["isBatchMode"]
    
    print("METHOD: {} as classifier and {} and Hellinger distance as dynamic CSE".format(clfName, densityFunction))
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
    clf = classifiers.classifier(X, y, K, clfName)#O(nd+kn)
    reset = True
    if isBatchMode:
        for t in range(batches):
            #print("passo: ",t)
            initialDataLength=finalDataLength
            finalDataLength=finalDataLength+sizeOfBatch
            #print(initialDataLength)
            #print(finalDataLength)
            # ***** Box 2 *****
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
            
            # ***** Box 4 *****
            excludingPercentage = cuttingPercentage(X, Ut, t)
            #excludingPercentageByClass, reset = cuttingPercentageByClass(X, Ut, y, predicted, classes, t)
            allInstances = []
            allLabels = []
            
            # ***** Box 5 *****
            if reset == True:
                #Considers only the last distribution (time-series like)
                pdfsByClass = util.pdfByClass(Ut, predicted, classes, densityFunction)#O(n^{2}d)
            else:
                #Considers the past and actual data (concept-drift like)
                allInstances = np.vstack([X, Ut])
                allLabels = np.hstack([y, yt])
                pdfsByClass = util.pdfByClass(allInstances, allLabels, classes, densityFunction)
                
            selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, excludingPercentage)#O(n log(n) c)
            #selectedIndexes = util.compactingDataDensityBased(pdfsByClass, excludingPercentageByClass)
            #print(t, excludingPercentage)
            # ***** Box 6 *****
            if reset == True:
                #Considers only the last distribution (time-series like)
                X, y = util.selectedSlicedData(Ut, yt, selectedIndexes)#O(n)
            else:
                #Considers the past and actual data (concept-drift like)
                X, y = util.selectedSlicedData(allInstances, allLabels, selectedIndexes)

            clf = classifiers.classifier(X, y, K, clfName)#O(nd+kn)
    else:
        t=0
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
            
            #new approach
            if len(inst) == poolSize:
                inst = np.array(inst)
                excludingPercentage = cuttingPercentage(X, inst, t)
                t+=1
                '''if excludingPercentage < 0:
                    #print("negative, reseting points")
                    excludingPercentage = 0.5 #default
                    reset = True
                else:
                    reset = False
                '''
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
    return "AMANDA (Dynamic)", arrAcc, X, y, arrX, arrY, arrUt, arrYt, arrClf, arrPredicted