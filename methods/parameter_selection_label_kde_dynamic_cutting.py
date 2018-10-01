import numpy as np
from source import metrics
from source import util
from source import classifiers
from sklearn.base import BaseEstimator, ClassifierMixin




def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] for i in range(wanted_parts) ]


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
class batthacharyya(BaseEstimator, ClassifierMixin):

    def __init__(self, K=1, sizeOfBatch=100, batches=50, poolSize=100, isBatchMode=True, initialLabeledData=50):
        #super(proposed_gmm_core_extraction,self).__init__()
        self.sizeOfBatch = sizeOfBatch
        self.batches = batches
        self.initialLabeledData=initialLabeledData
        #self.classes=[0, 1]
        self.usePCA=False
        #used only by gmm and cluster-label process
        self.densityFunction='kde' #optimizedkde, kde, gmm
        self.K = K
        self.clfName = 'label'
        self.poolSize = poolSize
        self.isBatchMode = isBatchMode
        
        #print("{} excluding percecntage".format(excludingPercentage))    
    
    def get_params(self, deep=True):
        return {"K":self.K, "sizeOfBatch":self.sizeOfBatch, "batches":self.batches, "poolSize":self.poolSize, "isBatchMode":self.isBatchMode}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
            
    def fit(self, dataValues, dataLabels=None):
        arrAcc = []
        classes = list(set(dataLabels))
        initialDataLength = 0
        finalDataLength = self.initialLabeledData

        # ***** Box 1 *****
        #Initial labeled data
        X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, self.usePCA)
        if self.isBatchMode:
            for t in range(self.batches):
                #print("passo: ",t)
                initialDataLength=finalDataLength
                finalDataLength=finalDataLength+self.sizeOfBatch
                #print(initialDataLength)
                #print(finalDataLength)
                # ***** Box 2 *****            
                Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, self.usePCA)
                
                # ***** Box 3 *****
                #predicted = classifiers.classify(X, y, Ut, self.K, classes, self.clfName)
                predicted = classifiers.classify(X, y, Ut, self.K, classes, self.clfName)
                # Evaluating classification
                arrAcc.append(metrics.evaluate(yt, predicted))

                # ***** Box 4 *****
                #pdfs from each new points from each class applied on new arrived points
                #pdfsByClass = util.pdfByClass2(Ut, predicted, classes)
                pdfsByClass = util.pdfByClass(Ut, predicted, classes, self.densityFunction)
                instancesXByClass, instancesUtByClass = util.unifyInstancesByClass(X, y, Ut, predicted, classes)

                # ***** Box 5 *****
                keepPercentageByClass = util.getBhattacharyyaScoresByClass(instancesXByClass, instancesUtByClass, classes)
                #keepPercentage = util.getBhattacharyyaScores(instancesUtByClass)

                #selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, keepPercentage)
                selectedIndexes = util.compactingDataDensityBased3(pdfsByClass, keepPercentageByClass)
                # ***** Box 6 *****
                X, y = util.selectedSlicedData(Ut, predicted, selectedIndexes)
        else:
            inst = []
            labels = []
            clf = classifiers.knn(X, y, self.K)
            remainingX , remainingY = util.loadLabeledData(dataValues, dataLabels, finalDataLength, len(dataValues), self.usePCA)
            
            for Ut, yt in zip(remainingX, remainingY):
                predicted = clf.predict(Ut.reshape(1, -1))
                arrAcc.append(predicted)
                inst.append(Ut)
                labels.append(predicted)
                
                if len(inst) == self.poolSize:
                    inst = np.asarray(inst)
                    pdfsByClass = util.pdfByClass(inst, labels, classes, self.densityFunction)
                    selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, self.excludingPercentage)
                    X, y = util.selectedSlicedData(inst, labels, selectedIndexes)
                    clf = classifiers.knn(X, y, self.K)
                    inst = []
                    labels = []
                
            arrAcc = split_list(arrAcc, 100)
            arrAcc = makeAccuracy(arrAcc, remainingY)   
            
     
        # returns accuracy array and last selected points
        self.threshold_ = arrAcc
        return self
    
    def predict(self):
        try:
            getattr(self, "threshold_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        return self.threshold_
    
    def score(self, X, y=None):
        accuracies = self.predict()
        N = len(accuracies)
        #print(self.K, self.excludingPercentage, sum(accuracies)/N)
        return sum(accuracies)/N
