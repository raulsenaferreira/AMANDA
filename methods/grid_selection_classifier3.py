import numpy as np
from source import metrics
from source import util
from source import classifiers
from sklearn.base import BaseEstimator, ClassifierMixin



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


class run(BaseEstimator, ClassifierMixin):

    def __init__(self, p=0.2, K=1, sizeOfBatch=100, batches=50, initialLabeledData=50):
        self.sizeOfBatch = sizeOfBatch
        self.batches = batches
        self.initialLabeledData=initialLabeledData
        self.usePCA=False
        self.p = p
        self.K = K
        
        #print("{} excluding percecntage".format(excludingPercentage))    
    
    def get_params(self, deep=True):
        return {"p" : self.p, "K": self.K, "sizeOfBatch":self.sizeOfBatch, "batches":self.batches}
    
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
        
        for t in range(self.batches):
            #print("passo: ",t)
            initialDataLength=finalDataLength
            finalDataLength=finalDataLength+self.sizeOfBatch
            
            # ***** Box 2 *****            
            Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, self.usePCA)
            
            # ***** Box 3 *****
            clf = classifiers.labelPropagation(X, y, self.K)
            #classifiers.classifier(X, y, self.K, self.clfName)
            
            predicted = clf.predict(Ut)
            # Evaluating classification
            arrAcc.append(metrics.evaluate(yt, predicted))

            # ***** Box 4 *****
            #pdfs from each new points from each class applied on new arrived points
            indexesByClass = util.slicingClusteredData(y, classes)
            bestModelSelectedByClass = util.loadBestModelByClass(X, indexesByClass)
            
            # ***** Box 5 *****
            predictedByClass = util.slicingClusteredData(predicted, classes)
            #p% smallest distances per class, based on paper
            selectedIndexes = util.mahalanobisCoreSupportExtraction(Ut, predictedByClass, 
                bestModelSelectedByClass, self.p)
            #selectedIndexes = np.hstack([selectedIndexes[0],selectedIndexes[1]])
            stackedIndexes=selectedIndexes[0]
            for i in range(1, len(selectedIndexes)):
                stackedIndexes = np.hstack([stackedIndexes,selectedIndexes[i]])
            selectedIndexes =  stackedIndexes
            
            # ***** Box 6 *****
            X, y = util.selectedSlicedData(Ut, predicted, selectedIndexes)
           
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
