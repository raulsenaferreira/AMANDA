import numpy as np
from source import classifiers
from source import metrics
from source import util
from keras.models import Sequential
from keras.layers import Dense, Activation  
from keras.layers import LSTM



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

    print("METHOD: Sliding LSTM as classifier (Long-short term memory)")
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
    in_out_neurons = K
    hidden_neurons = 300
    # ***** Box 1 *****
    #Initial labeled data
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    
    model = Sequential()  
    model.add(LSTM(hidden_neurons, input_shape=(len(X), in_out_neurons)))
    #model.add(Flatten())
    model.add(Dense(32))#hidden_neurons, input_shape=(len(X), in_out_neurons)  
    
    model.compile(loss="mean_squared_error", optimizer="rmsprop")  

    if isBatchMode:
        for t in range(batches):
            # sliding 
            model.fit(X, y, epochs=10, validation_split=0.05)

            initialDataLength=finalDataLength
            finalDataLength=finalDataLength+sizeOfBatch
            #print(initialDataLength)
            #print(finalDataLength)
                     
            Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
            
            # for decision boundaries plot
            arrClf.append(model)
            arrX.append(X)
            arrY.append(y)
            arrUt.append(np.array(Ut))
            arrYt.append(yt)
            predicted = model.predict(Ut)
            arrPredicted.append(predicted)
            # Evaluating classification
            arrAcc.append(metrics.evaluate(yt, predicted))
            
            X, y = Ut, yt
    else:
        inst = []
        labels = []
        remainingX , remainingY = util.loadLabeledData(dataValues, dataLabels, finalDataLength, len(dataValues), usePCA)
        model.fit(X, y, epochs=10, validation_split=0.05)
        for Ut, yt in zip(remainingX, remainingY):
            predicted = model.predict(Ut.reshape(1, -1))
            arrAcc.append(predicted)
            inst.append(Ut)
            labels.append(yt)

            # for decision boundaries plot
            arrClf.append(model)
            arrX.append(X)
            arrY.append(y)
            arrUt.append(Ut)
            arrYt.append(yt)
            arrPredicted.append(predicted)
            
            if len(inst) == poolSize:
                inst = np.asarray(inst)
                clf = model.fit(inst, labels, epochs=10, validation_split=0.05)
                inst = []
                labels = []
            
        arrAcc = split_list(arrAcc, batches)
        arrAcc = makeAccuracy(arrAcc, remainingY)
        arrYt = split_list(arrYt, batches)
        arrPredicted = split_list(arrPredicted, batches)
    
    return "LSTM (with true labels)", arrAcc, X, y, arrX, arrY, arrUt, arrYt, arrClf, arrPredicted