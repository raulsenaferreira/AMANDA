import numpy as np
from sklearn import mixture
from sklearn.neighbors.kde import KernelDensity
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.decomposition import PCA
from collections import Counter
from source import util
from sklearn.semi_supervised import label_propagation
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.cluster import Birch
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.naive_bayes import BernoulliNB



def pca(X, numComponents):
    pca = PCA(n_components=numComponents)
    pca.fit(X)
    PCA(copy=True, iterated_power='auto', n_components=numComponents, random_state=None, svd_solver='auto', tol=0.0, whiten=False)
    
    return pca.transform(X)
       
    
def naiveBayes(X, y):
    return GaussianNB().fit(X, y)


def kMeans(X, k):
    return KMeans(n_clusters=k).fit(X)


def labelPropagation(X, y, K):
    return label_propagation.LabelSpreading(kernel='knn', n_neighbors=K, alpha=1).fit(X, y)


def SGDClassifier(X, y):
    return linear_model.SGDClassifier().fit(X, y)


def SAG(X, y):
    return LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X.shape[0]).fit(X, y)


def svmClassifier(X, y):
    #clf=svm.SVC()
    '''
    if isImbalanced:
        clf = svm.SVC(C=1.0, cache_size=200, kernel='linear', class_weight='balanced', coef0=0.0,
            decision_function_shape=None, degree=3, gamma='auto', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
        return clf.fit(X, y)
    else:'''
    '''clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape=None, degree=4, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    '''
    clf = svm.SVC(gamma=2, C=1)
    return clf.fit(X, y)
    

def kmeans_matlab(X, k, mode='auto', centroids=[]):
    if mode=='auto':
        return KMeans(n_clusters=k).fit(X).cluster_centers_
    elif mode=='start':
        return KMeans(n_clusters=k, init=centroids).fit(X).cluster_centers_


def knn_classify(training_data, labels, test_instance):
    predicted_label = nearest = None
    best_distance = np.inf
    tam = np.shape(training_data)[0]
    for i in range(tam):
        compare_data = training_data[i, :]
        
        distance = np.sqrt(np.sum(np.power((test_instance - compare_data), 2))) #euclidean distance
        if distance < best_distance:
            best_distance = distance
            predicted_label = labels[i]
            nearest = compare_data
    #print("nearest manually",nearest)
    #print("predicted manually",predicted_label)
    return predicted_label, best_distance, nearest


def knn_scargc(X, y, Ut):
    '''clf = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    predicted_label = clf.predict(Ut.reshape(1, -1))'''
    best_distance, ind = NearestNeighbors(n_neighbors=1).fit(X).kneighbors(Ut.reshape(1, -1))
    nearest = X[ind]
    clf = NearestCentroid(metric='euclidean').fit(X, y)
    predicted_label = clf.predict(Ut.reshape(1, -1))
    #print(clf.centroids_) #exemplo [[ 0.25940611 -0.02868181] [ 5.450457    5.40674248]]
    #print("nearest scikit",nearest[0][0])
    #print("predicted scikit",predicted_label)
    return predicted_label, best_distance, nearest[0]


def libsvmtrain(y, X):
    #based on libsvm lib for matlab: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
    '-t 2 -g 1 -r 10 -b 1 -q' # parameters from original matlab implementation
    clf = svm.SVC(C=1.0, cache_size=100, class_weight=None, coef0=10, degree=3, gamma=1, kernel='rbf',
        probability=True, shrinking=True, tol=0.001, verbose=False)
    return clf.fit(X, y)


def libsvmpredict(Ut, clf):
    #based on libsvm lib for matlab: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
    '-b 1 -q' # parameters from original matlab implementation
    return clf.predict(Ut)


def mClassification(X, y, threshold, K=None):
    brc = Birch(n_clusters=K, threshold=threshold,compute_labels=True)
    return brc.partial_fit(X)
    #brc = Birch(threshold=0.1, n_clusters=None)
    #return brc.fit(X)


def randomForest(X, y):
    num_trees = 100
    max_features = np.ndim(X)
    return RandomForestClassifier(n_estimators=num_trees, max_features=max_features).fit(X, y)
    

def knn(X, y, K):
    return KNeighborsClassifier(n_neighbors=K, algorithm = 'brute').fit(X, y)


def gmmWithBIC(X, allPoints):
    ctype = 'full' #'spherical', 'tied', 'diag', 'full'
    best_gmm = False
    lowest_bic = np.infty
    bic = []
    n_components_range = [1, 2, 3, 5, 10, 15, 20]
    lenPoints = len(X)
    numComponentsChosen = 0

    for numComponents in n_components_range:
        #print(lenPoints)
        if lenPoints >= numComponents:
            GMM = mixture.GaussianMixture(n_components=numComponents, covariance_type=ctype)
            GMM.fit(allPoints)
            actualBIC = GMM.bic(X)
            bic.append(actualBIC)

            if actualBIC < lowest_bic:
                lowest_bic = actualBIC
                best_gmm = GMM
                numComponentsChosen = numComponents
    #return pdfs of best GMM model
    if best_gmm != False:
        #print("Best number of components: ",numComponentsChosen)
        return best_gmm
    else:
        #print("gmmWithBIC: Error to choose the best GMM model")
        # Default:  As cited in original article
        GMM = mixture.GaussianMixture(n_components=2, covariance_type=ctype)
        GMM.fit(allPoints)
        return GMM


def gmm(points, numComponents):
    clf = mixture.GaussianMixture(n_components=numComponents, covariance_type='full')
    clf.fit(points)
    return clf


def gmmWithPDF(points, allPoints, numComponents):
    if len(allPoints) < numComponents:
        numComponents=len(allPoints)
    clf = mixture.GaussianMixture(n_components=numComponents, covariance_type='full')
    clf.fit(allPoints)
    return np.exp(clf.score_samples(points))   
    

def bayesianGMM(points, allPoints, numComponents):
    if len(allPoints) < numComponents:
        numComponents=len(allPoints)
    clf = BayesianGaussianMixture(
        weight_concentration_prior_type="dirichlet_process", weight_concentration_prior=100000,
        n_components=2 * numComponents, reg_covar=0, init_params='random',
        max_iter=1500, mean_precision_prior=.8,
        random_state=2)
    clf.fit(allPoints)
    return np.exp(clf.score_samples(points))


def kde(points, allPoints):
    kernel = KernelDensity(kernel='gaussian', bandwidth=0.4).fit(allPoints)
    pdfs = np.exp(kernel.score_samples(points))
    
    return pdfs


def majorityVote(clusteredData, clusters, y):
    kPredicted = []
    
    for i in range(len(clusteredData)):
        group = clusteredData[i]
        ind = np.where(clusters==group)[0]
        #label = np.squeeze(np.asarray(y[ind]))
        label=y[ind]
        voting = Counter(label).most_common(1)[0][0]
        kPredicted.append(voting)
    
    return kPredicted


def clusterAndLabel(X, y, Ut, classes):
    labels=[]
    initK = len(classes)
    #arrPredicted=np.array([-1]*len(Ut))
    arrPredicted = []
    lenPoints = len(X)
    #print(len(Ut))
    #print(lenPoints)
    #K varies in 5 values
    for k in range(initK, 5+initK):
        if lenPoints >= k:
            #print("lllll")
            kmeans = kMeans(X, k)
            clusters = kmeans.labels_
            clusteredData = kmeans.predict(Ut)
            #arrPredicted=np.vstack([arrPredicted, majorityVote(clusteredData, clusters, y)])
            arrPredicted.append(majorityVote(clusteredData, clusters, y))
    
    arrPredicted = np.array(arrPredicted)
    #print(arrPredicted)
    #print(len(Ut))
    for j in range(len(Ut)):
        #print(arrPredicted[:, j])
        labels.append(Counter(arrPredicted[:, j]).most_common(1)[0][0])
    #print(labels)
    return labels


def classifier(X, y, K, clf):
        if clf=='svm':
            #print("Using SVM")
            return svmClassifier(X, y)
        elif clf=='cl':
            #print("Using cluster and label")
            return clusterAndLabel(X, y)
        elif clf=='rf':
            #print("Using Random Forest")
            return randomForest(X, y)
        elif clf=='knn':
            #print("Using K-Nearest Neighbors")
            return knn(X, y, K)
        elif clf == 'lp':
            #print("Using Label Propagation")
            return labelPropagation(X, y, K)
        elif clf == 'nb':
            #print("Using Naive Bayes")
            return naiveBayes(X, y)
        elif clf == 'sgd':
            #print("Using Stochastic Gradient Descent")
            return SGDClassifier(X, y)