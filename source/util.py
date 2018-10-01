import numpy as np
from scipy.spatial.distance import mahalanobis
from math import floor
import matplotlib.pyplot as plt
import random
from methods import alpha_shape
from source import classifiers



def loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA):
    X = np.copy(dataValues[initialDataLength:finalDataLength])
    y = np.copy(dataLabels[initialDataLength:finalDataLength])
    if usePCA:
        X = classifiers.pca(X, 2)

    return X, y


def selectedSlicedData(instances, labelsInstances, selectedIndexes):
    instances = np.array(instances)
    labelsInstances = np.array(labelsInstances)
    X = instances[selectedIndexes]
    y = labelsInstances[selectedIndexes]

    return X, y


def loadGeometricCoreExtractionByClass(instances, indexesByClass, alpha, threshold):
    selectedPointsByClass = {}
    selectedIndexesByClass = {}

    for c in indexesByClass:
        points = instances[indexesByClass[c]]
        inst, indexes, edge_points = alpha_shape.alpha_compaction(points, alpha, threshold)
        selectedPointsByClass[c] = inst
        selectedIndexesByClass[c] = indexes

    return selectedPointsByClass, selectedIndexesByClass


def solve(m1,m2,s1,s2):
    x1 = (s1*s2*np.sqrt((-2*np.log(s1/s2)*s2**2)+2*s1**2*np.log(s1/s2)+m2**2-2*m1*m2+m1**2)+m1*s2**2-m2*s1**2)/(s2**2-s1**2)
    x2 = -(s1*s2*np.sqrt((-2*np.log(s1/s2)*s2**2)+2*s1**2*np.log(s1/s2)+m2**2-2*m1*m2+m1**2)-m1*s2**2+m2*s1**2)/(s2**2-s1**2)
    return x1,x2


def plotDistributionss(distributions):
    i=0
    #ploting
    fig = plt.figure()
    handles = []
    colors = ['magenta', 'cyan']
    classes = ['Class 1', 'Class 2']
    ax = fig.add_subplot(121)

    for k, v in distributions.items():
        points = distributions[k]
        points = np.array(points)
        #print(points)
        handles.append(ax.scatter(points[:, 0], points[:, 1], color=colors[i], s=5, edgecolor='none'))
        i+=1

    ax.legend(handles, classes)

    plt.show()


def baseClassifier(instancesToPredict, classifier):
    return classifier.predict(instancesToPredict)


def initializingData(X, y):
    c1=[]
    c2=[]
    for i in range(len(y)):
        if y[i]==0:
            c1.append(X[i])
        else:
            c2.append(X[i])

    return c1, c2


def loadDensitiesByClass(oldInstances, newInstances, allInstances, oldIndexesByClass, newIndexesByClass, densityFunction):
    previousPdfByClass = {}
    currentPdfByClass = {}
    numClasses = len(newIndexesByClass)

    for c in newIndexesByClass:
        oldPdfs = [-1] * len(oldInstances)
        newPdfs = [-1] * len(newInstances)

        oldPoints = oldInstances[oldIndexesByClass[c]]
        newPoints = newInstances[newIndexesByClass[c]]

        clf = densityFunction(allInstances, numClasses)
        oldPdfsByPoints = np.exp(clf.score_samples(oldPoints))
        newPdfsByPoints = np.exp(clf.score_samples(newPoints))

        a = 0
        for i in oldIndexesByClass[c]:
            oldPdfs[i]=oldPdfsByPoints[a]
            a+=1
        previousPdfByClass[c] = oldPdfs

        a = 0
        for i in newIndexesByClass[c]:
            newPdfs[i]=newPdfsByPoints[a]
            a+=1
        currentPdfByClass[c] = newPdfs

    return previousPdfByClass, currentPdfByClass


def loadDensitiesByClass2(instances, allInstances, indexesByClass, densityFunction):
    pdfsByClass = {}
    numClasses = len(indexesByClass)

    for c, indexes in indexesByClass.items():
        pdfs = [-1] * len(instances)
        points = instances[indexes]
        pdfsByPoints = densityFunction(points, allInstances, numClasses)
        a = 0
        for i in indexes:
            pdfs[i]=pdfsByPoints[a]
            a+=1
        pdfsByClass[c] = pdfs

    return pdfsByClass


#Slicing instances according to their inferred clusters
def slicingClusteredData(clusters, classes):
    indexes = {}
    for c in range(len(classes)):
        indexes[classes[c]]=[i for i in range(len(clusters)) if clusters[i] == c]
        #if len(indexes[classes[c]]) < 1:
            #choose one index randomly if the array is empty
            #indexes[classes[c]] = [random.randint(0,len(clusters)-1)]
            #print("Empty array for class ", c)

    return indexes


#Cutting data for next iteration
def compactingDataDensityBased4(densities, criteria):
    selectedIndexes=[]

    for k in densities:
        arrPdf = densities[k]
        cutLine = max(arrPdf)*criteria
        #a = [i for i in range(len(arrPdf)) if arrPdf[i] != -1 and arrPdf[i] >= cutLine ]
        a = [i for i in range(len(arrPdf)) if arrPdf[i] >= cutLine ]
        if len(a) < criteria*len(arrPdf):
            #a=[i for i in range(len(arrPdf)) if arrPdf[i] != -1 and arrPdf[i] >= cutLine*criteria]
            a=[i for i in range(len(arrPdf)) if arrPdf[i] >= cutLine*criteria]
        selectedIndexes.append(a)

    stackedIndexes=selectedIndexes[0]

    for i in range(1, len(selectedIndexes)):
        stackedIndexes = np.hstack([stackedIndexes,selectedIndexes[i]])

    return stackedIndexes


#Cutting data for next iteration
def compactingDataDensityBased3(densitiesByClass, criteriaByClass):
    selectedIndexes=[]

    for c in densitiesByClass:
        arrPdf = np.array(densitiesByClass[c])
        numSelected = int(np.floor((1-criteriaByClass[c])*len(arrPdf)))
        ind = (-arrPdf).argsort()[:numSelected]
        selectedIndexes.append(ind)

    stackedIndexes=selectedIndexes[0]

    for i in range(1, len(selectedIndexes)):
        stackedIndexes = np.hstack([stackedIndexes,selectedIndexes[i]])

    return stackedIndexes


def cuttingDataByIntersection3(x, x2, y):

    #getting intersection
    m1 = np.mean(x)
    std1 = np.std(x)
    m2 = np.mean(x2)
    std2 = np.std(x2)

    r = solve(m1,m2,std1,std2)[0]

    if np.min(x) < np.min(x2):
        #print("D1 < D2")
        indX = x[:,0]>r
        indX2 = x2[:,0]<r
    else:
        #print("D2 < D1")
        indX = x[:,0]<r
        indX2 = x2[:,0]>r

    y=np.array(y)
    return x[indX], y[indX]


def getDistributionIntersection(X, Ut, indexesByClass, predictedByClass, densityFunction):
    pdfX = {}
    pdfUt = {}
    nComponents = 2

    for c, indexes in indexesByClass.items():
        arrX = []
        arrU = []
        oldPoints = X[indexes]
        newPoints = Ut[predictedByClass[c]]
        GMMX = densityFunction(oldPoints, nComponents)
        GMMU = densityFunction(newPoints, nComponents)

        for i in range(len(newPoints)):
            x = GMMX.predict_proba(newPoints[i])[0]
            x = np.array(x)
            x = x.reshape(1,-1)
            arrU.append(x)
        for i in range(len(oldPoints)):
            u = GMMU.predict_proba(oldPoints[i])[0]
            u = np.array(u)
            u = u.reshape(1, -1)
            arrX.append(u)

        pdfUt[c] = arrU
        pdfX[c] = arrX

    plotDistributionss(pdfX)
    plotDistributionss(pdfUt)


def loadBestModelByClass(X, indexesByClass):
    bestModelForClass = {}

    for c, indexes in indexesByClass.items():
        points = X[indexes]
        bestModelForClass[c] = classifiers.gmmWithBIC(points, X)

    return bestModelForClass


def mahalanobisCoreSupportExtraction(Ut, indexesPredictedByClass, bestModelSelectedByClass, p):
    inf = 1e6
    selectedMinIndexesByClass={}

    for c in bestModelSelectedByClass:
        distsByComponent = []
        precisions = bestModelSelectedByClass[c].precisions_ #inverse of covariance matrix
        means = bestModelSelectedByClass[c].means_
        pointIndexes = indexesPredictedByClass[c]

        for i in range(len(means)):
            dists = []
            v = means[i]
            VI = precisions[i]

            for k in pointIndexes:
                u = Ut[k]
                dist = mahalanobis(u, v, VI)
                dists.append(dist)

            distsByComponent.append(dists)

        distsByComponent = np.array(distsByComponent)
        selectedMinIndexesByClass[c] = [inf]*len(Ut)

        for j in range(len(pointIndexes)):
            vals = distsByComponent[:, j]
            i = vals.argmin()
            selectedMinIndexesByClass[c][pointIndexes[j]] = distsByComponent[i][j]

        #p% smallest distances per class, based on paper
        p = floor(p*len(selectedMinIndexesByClass[c]))
        #p = 70
        selectedMinIndexesByClass[c] = np.array(selectedMinIndexesByClass[c])
        selectedMinIndexesByClass[c] = selectedMinIndexesByClass[c].argsort()[:p]
    #print(len(selectedMinIndexesByClass))
    return selectedMinIndexesByClass


def pdfByClass(instances, labels, classes, densityFunction):
    indexesByClass = slicingClusteredData(labels, classes)
    pdfsByClass = {}
    numClasses = len(classes)
    
    for c, indexes in indexesByClass.items():
        if len(indexes) > 0:
            pdfs = [-1] * len(instances)
            #print("class {} = {} points".format(c, len(indexes)))
            #print(indexes)
            #print(instances)
            points = instances[indexes]
            #points from a class, all points, number of components
            if densityFunction=='gmm':
                pdfsByPoints = classifiers.gmmWithPDF(points, instances, numClasses)
            elif densityFunction=='bayes':
                pdfsByPoints = classifiers.bayesianGMM(points, instances, numClasses)
            elif densityFunction=='kde':
                pdfsByPoints = classifiers.kde(points, instances)
            a = 0
            for i in indexes:
                if pdfsByPoints[a] != -1:
                    pdfs[i]=pdfsByPoints[a]
                a+=1
            pdfsByClass[c] = pdfs

    return pdfsByClass


def pdfByClass2(pastInstances, pastLabels, instances, labels, classes, densityFunction):
    indexesByClass = slicingClusteredData(labels, classes)
    pastIndexesByClass = slicingClusteredData(pastLabels, classes)

    pdfsByClass = {}
    numClasses = len(classes)
    
    for c in classes:
        allIndexesByClass = np.hstack([pastIndexesByClass[c], indexesByClass[c]])
        if len(allIndexesByClass) > 0:
            pdfs = [-1] * len(allIndexesByClass)
            #print("class {} = {} points".format(c, len(indexes)))
            #print(indexes)
            #print(instances)
            XByClass = pastInstances[pastIndexesByClass[c]]
            UtByClass = instances[indexesByClass[c]]
            #print(len(indexesByClass[c]))
            allInstancesByClass = np.vstack([XByClass, UtByClass])
            #points from a class, all points, number of components
            if densityFunction=='kde':
                pdfsByPointsX = classifiers.kde(XByClass, allInstancesByClass)
                pdfsByPointsUt = classifiers.kde(UtByClass, allInstancesByClass)
                pdfsByPoints = np.hstack([pdfsByPointsX, pdfsByPointsUt])
            a = 0
            for i in range(len(allIndexesByClass)):
                pdfs[i]=pdfsByPoints[a]
                a+=1
            pdfsByClass[c] = pdfs

    return pdfsByClass


def pdfByClass3(X, y, Ut, predicted, classes, criteria):
    resultingX = []
    resultingY = []
    #instancesXByClass, instancesUtByClass = unifyInstancesByClass(X, y, Ut, predicted, classes)
    labels = np.hstack([y, predicted])
    allInstances = np.vstack([X, Ut])
    selectedIndexesByClass = {}
    numClasses = len(classes)
    for c in classes:
        
        indexesX=[i for i in range(len(y)) if y[i] == c]
        indexesUt=[i for i in range(len(predicted)) if predicted[i] == c]
        #print("Size of Class and points inside X: ", c, len(indexesX))
        #print("Size of Class and points inside Ut: ", c, len(indexesUt))
        #choose one index randomly if the array is empty
        '''if len(indexesX) < 1:
            indexesX = [random.randint(0,len(y)-1)]
        if len(indexesUt) < 1:
            indexesUt = [random.randint(0,len(predicted)-1)]'''

        XByClass = X[indexesX]
        UtByClass = Ut[indexesUt]

        allInstancesByClass = np.vstack([XByClass, UtByClass])
        
        pdfsByPointsX = classifiers.gmmWithPDF(XByClass, allInstancesByClass, numClasses)
        pdfsByPointsUt = classifiers.gmmWithPDF(UtByClass, allInstancesByClass, numClasses)

        selectedIndexesByClass[c] = compactingDataDensityBased(np.hstack([pdfsByPointsX, pdfsByPointsUt]), criteria)
        
    resultingX = allInstances[selectedIndexesByClass[0]]
    resultingY = labels[selectedIndexesByClass[0]]
    for c in range(1, len(classes)):
        resultingX = np.vstack([resultingX, allInstances[selectedIndexesByClass[c]]])
        resultingY = np.hstack([resultingY, labels[selectedIndexesByClass[c]]])

    return resultingX, resultingY


def compactingDataDensityBased(densities, criteriaByClass, reverse=False):
    selectedIndexes=[]

    for k in densities:
        arrPdf = np.array(densities[k])
        numSelected = int(np.floor((1-criteriaByClass[k])*len(arrPdf)))
        if reverse:
            ind = (arrPdf).argsort()[:numSelected]
        else:
            ind = (-arrPdf).argsort()[:numSelected]
        selectedIndexes.append(ind)

    stackedIndexes=selectedIndexes[0]
    #print(0 ,len(selectedIndexes[0]))

    for i in range(1, len(selectedIndexes)):
        stackedIndexes = np.hstack([stackedIndexes,selectedIndexes[i]])
        #print(i, len(selectedIndexes[i]))

    return stackedIndexes


#Cutting data for next iteration
def compactingDataDensityBased2(densities, criteria, reverse=False):
    selectedIndexes=[]

    for k in densities:
        arrPdf = np.array(densities[k])
        numSelected = int(np.floor(criteria*len(arrPdf)))
        if reverse:
            ind = (arrPdf).argsort()[:numSelected]
        else:
            ind = (-arrPdf).argsort()[:numSelected]
        selectedIndexes.append(ind)

    stackedIndexes=selectedIndexes[0]
    #print(0 ,len(selectedIndexes[0]))

    for i in range(1, len(selectedIndexes)):
        stackedIndexes = np.hstack([stackedIndexes,selectedIndexes[i]])
        #print(i, len(selectedIndexes[i]))

    return stackedIndexes


def bhattacharyya (h1, h2):
    def normalize(h):
        for i in range(len(h)):
            if h[i]<0:
                h[i]=(h[i]*-1)+10
            else:
                h[i]=h[i]+10
                
        h = h / np.sum(h)
        #print(h)
        return h

    return 1 - np.sum(np.sqrt(np.multiply(normalize(h1), normalize(h2))))


def getBhattacharyyaScores(instancesByClass):
    scoresByClass = {}
    means= []
    for c, instances in instancesByClass.items():
        # generate and output scores
        scores = [];
        for i in range(len(instances)):
            score = [];
            for j in range(len(instances)):
                score.append( bhattacharyya(instances[i], instances[j]) );
            scores.append(score);
        scoresByClass[c]=scores
        #print(np.mean(scores))
        means.append(np.mean(scores))
    #print(np.mean(means))
    #return scoresByClass
    return np.mean(means)


def getBhattacharyyaScoresByClass(X, Ut, classes):
#def getBhattacharyyaScoresByClass(X, y, Ut, predicted, classes):
    scoresByClass = {}
    '''
    means= []
    for c, instances in instancesByClass.items():
        # generate and output scores
        scores = [];
        for i in range(len(instances)):
            score = [];
            for j in range(len(instances)):
                score.append( bhattacharyya(instances[i], instances[j]) );
            scores.append(score);
        scoresByClass[c]=scores
        #print(np.mean(scores))
        means.append(np.mean(scores))
    #print(np.mean(means))
    #return scoresByClass
    return np.mean(means)
    '''
    penalty = 0.3
    for c in classes:
        limit = min(len(X[c]),len(Ut[c]))
        score = []
        for i in range(limit):
            score.append( bhattacharyya(X[c][i], Ut[c][i]) )

        mean = 1-np.mean(score)-penalty
        
        if mean > 0.95:
            scoresByClass[c] = 0.95
        #elif mean < 0.7:
            #scoresByClass[c] = 0.7
        else:
            scoresByClass[c] = mean
        #print(c, scoresByClass[c])
    return scoresByClass


def compactingDataScoreBased(scores, criteria):
    cut = 1-criteria
    selectedIndexes=[]
    i=0
    for k in scores:
        arrScores = np.array(scores[k])
        numSelected = int(np.ceil(cut*len(arrScores)))
        numSelected = 10
        ind = (arrScores).argsort()[:numSelected]
        ind = np.delete(ind, i)
        selectedIndexes.append(ind)
        i+=1

    stackedIndexes=selectedIndexes[0]
    #print(selectedIndexes[0])
    #print(selectedIndexes[1])
    for i in range(1, len(selectedIndexes)):
        stackedIndexes = np.hstack([stackedIndexes,selectedIndexes[i]])

    return stackedIndexes


def unifyInstancesByClass(X, y, Ut, predicted, classes):
    allInstancesByClass = {}
    indexesYByClass = slicingClusteredData(y, classes)
    indexesYtByClass = slicingClusteredData(predicted, classes)
    instancesXByClass = {}
    instancesUtByClass = {}
    numClasses = len(classes)
    #print("{} instances".format(len(instances)))
    for c, indexes in indexesYByClass.items():
        instancesXByClass[c] = X[indexesYByClass[c]]
    for c, indexes in indexesYtByClass.items():
        instancesUtByClass[c] = Ut[indexesYtByClass[c]]
    '''for c in range(numClasses):
        allInstancesByClass[c] = np.vstack([instancesXByClass[c], instancesUtByClass[c]])
    return allInstancesByClass
    '''
    return instancesXByClass, instancesUtByClass


def getIntersection(X, y, Ut, predicted, classes):
    intersectionByClass = {}
    for c in classes:
        instancesXByClass = X[[i for i in range(len(y)) if y[i]==c]]
        instancesUtByClass = Ut[[i for i in range(len(predicted)) if predicted[i]==c]]
        intersectionByClass[c] = return_intersection(instancesXByClass, instancesUtByClass)
    return intersectionByClass


def removeBoundaryPoints(supportIndexes, X, y):
    dif = [i for i in range(0,len(X))]
    dif = np.setdiff1d(dif, supportIndexes)
    
    return X[dif], y[dif]