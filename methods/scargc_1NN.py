from source import classifiers
from source import metrics
from source import util
import numpy as np
from scipy.stats import itemfreq



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

  print("METHOD: SCARGC with 1-NN")
  usePCA=False
  arrAcc = []
  arrX = []
  arrY = []
  arrUt = []
  arrYt = []
  arrClf = []
  arrPredicted = []
  initialDataLength = 0
  finalDataLength = initialLabeledData

  #Initial labeled data
  X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
  #unlabeled data used for the test phase
  Ut, yt = util.loadLabeledData(dataValues, dataLabels, finalDataLength, len(dataValues), usePCA)

  centroids_ant = []
  tmp_cent = []
  
  #first centroids
  if K == len(classes): #for unimodal case, the initial centroid of each class is the mean of each feature
    for cl in range(len(classes)):
      tmp_cent = np.median(X[y==classes[cl], 0], axis=0)
      for atts in range(1, X.shape[1]):
        aux = np.median(X[y==classes[cl], atts], axis=0)
        tmp_cent = np.hstack([tmp_cent, aux])
      
      centroids_ant = np.vstack([centroids_ant, tmp_cent]) if len(centroids_ant) > 0 else tmp_cent

    e = np.array(classes)[np.newaxis]
    centroids_ant = np.hstack([centroids_ant, e.T])
    
  else: #for multimodal case, the initial centroids are estimated by kmeans
    centroids_ant = classifiers.kmeans_matlab(X, K)
    #associate labels for first centroids
    centroids_ant_lab = []
    centroids_ant_lab, a, b = classifiers.knn_classify(X, y, centroids_ant[0,:])
    #centroids_ant_lab, a, b = classifiers.knn_scargc(X, y, centroids_ant[0,:])

    for core in range(1,centroids_ant.shape[0]):
      pred_lab, a, b = classifiers.knn_classify(X, y, centroids_ant[core,:])
      #pred_lab, a, b = classifiers.knn_scargc(X, y, centroids_ant[core,:])
      centroids_ant_lab = np.vstack([centroids_ant_lab, pred_lab])
    
    centroids_ant = np.hstack([centroids_ant, centroids_ant_lab])
    
  pool_data = []
  predicted, a, b = classifiers.knn_classify(X, y, Ut[0,:])
  #predicted, a, b = classifiers.knn_scargc(X, y, Ut[0,:])
  pool_data = np.hstack([Ut[0,:], predicted])
  
  arrYt.append(yt[0])
  arrPredicted.append(predicted)
  arrAcc.append(predicted)

  for i in range(1,len(yt)):
    #classify each stream's instance with 1NN classifier
    predicted, a, b = classifiers.knn_classify(X, y, Ut[i,:])
    #predicted, a, b = classifiers.knn_scargc(X, y, Ut[i,:])
    arrX.append(X)
    arrY.append(y)
    arrUt.append(Ut[i,:])
    arrYt.append(yt[i])
    arrPredicted.append(predicted)
    arrAcc.append(predicted)
    
    aux = np.hstack([Ut[i,:], predicted])
    pool_data = np.vstack([pool_data, aux]) if len(pool_data) > 0 else aux
    
    if pool_data.shape[0] == poolSize:
      centroids_cur = classifiers.kmeans_matlab(pool_data[:,0:-1], K, 'start', centroids_ant[-K:,:-1])
      clab, a, nearest = classifiers.knn_classify(centroids_ant[:,:-1], centroids_ant[:,-1], centroids_cur[0,:])
      #clab, a, nearest = classifiers.knn_scargc(centroids_ant[:,:-1], centroids_ant[:,-1], centroids_cur[0,:])
      cent_labels = clab
      intermed = np.hstack([np.median(np.vstack([nearest, centroids_cur[0,:]]), axis=0), clab])
      
      for p in range(1, centroids_cur.shape[0]):
        clab, a, nearest = classifiers.knn_classify(centroids_ant[:,:-1], centroids_ant[:,-1], centroids_cur[p,:])
        #clab, a, nearest = classifiers.knn_scargc(centroids_ant[:,:-1], centroids_ant[:,-1], centroids_cur[p,:])
        aux = np.hstack([np.median(np.vstack([nearest, centroids_cur[p,:]]), axis=0), clab])
        intermed = np.vstack([intermed, aux])
        cent_labels = np.vstack([cent_labels, clab])
       
      centroids_cur = np.hstack([centroids_cur, cent_labels])
      centroids_ant = intermed
      
      pred, a, b = classifiers.knn_classify(np.vstack([centroids_cur[:,:-1], centroids_ant[:,:-1]]), np.hstack([centroids_cur[:,-1], centroids_ant[:,-1]]), pool_data[0,0:-1])
      #pred, a, b = classifiers.knn_scargc(np.vstack([centroids_cur[:,:-1], centroids_ant[:,:-1]]), np.hstack([centroids_cur[:,-1], centroids_ant[:,-1]]), pool_data[0,0:-1])
      new_pool = np.hstack([pool_data[0,0:-1] ,pred])

      for p in range(1, pool_data.shape[0]):
        pred, a, b = classifiers.knn_classify(np.vstack([centroids_cur[:,:-1], centroids_ant[:,:-1]]), np.hstack([centroids_cur[:,-1], centroids_ant[:,-1]]), pool_data[p,0:-1])
        #pred, a, b = classifiers.knn_scargc(np.vstack([centroids_cur[:,:-1], centroids_ant[:,:-1]]), np.hstack([centroids_cur[:,-1], centroids_ant[:,-1]]), pool_data[p,0:-1])
        new_pool = np.vstack([new_pool, np.hstack([pool_data[p,0:-1], pred])])
        
      concordant_labels = np.nonzero(pool_data[:,-1] == new_pool[:,-1])[0]
      
      if len(concordant_labels)/poolSize < 1 or len(y) < pool_data.shape[0]:
        pool_data[:,-1] = new_pool[:,-1]
        centroids_ant = np.vstack([centroids_cur, intermed])
        
        X = pool_data[:,0:-1]
        y = pool_data[:,-1]
                 
      pool_data = []
    
  # Evaluating classification
  arrAcc = split_list(arrAcc, batches)
  arrAcc = makeAccuracy(arrAcc, arrYt)
  arrYt = split_list(arrYt, batches)
  arrPredicted = split_list(arrPredicted, batches)
  
  return "SCARGC 1NN", arrAcc, X, y, arrX, arrY, arrUt, arrYt, arrClf, arrPredicted