import numpy as np
'''
Rotating checkerboard problem with a linear drift. 
The CB rotates over a range of 0 - 2pi with 0-pi being a recurring environment at pi-2pi.
Adapted from ConceptDriftData 2011 by Gregory Ditzler, Ryan Elwell, and Robi Polikar.
Original version written by Ludmila Kuncheva.
'''

def generateData(side, a, N, T):
    xTrain = {}
    yTrain = {}
    xTest = {}
    yTest = {}
    
    def gendatcb(N,a,alpha):
        # N data points, uniform distribution,
        # checkerboard with side a, rotated at alpha
        d = np.random.rand(N,2)
        d_transformed = [d[:,0]*np.cos(alpha)-d[:,1]*np.sin(alpha), d[:,0]*np.sin(alpha)+d[:,1]*np.cos(alpha)]
        
        s = np.ceil(d_transformed[0]/a)+np.floor(d_transformed[1]/a)
        labd = 2-np.mod(s,2)
        labd = labd - 1
        return d, labd, d_transformed
    
    def CBDAT(a,alpha,N):
        X,Y, d_transformed = gendatcb(N,a,alpha)
        #X = np.transpose(X)
        #Y = np.transpose(Y)
        return X, Y
    
    for t in range(T):
        xTrain[t],yTrain[t] = CBDAT(side,a[t],N)
        xTest[t],yTest[t] = CBDAT(side,a[t],N)
    
    #X = np.array(xTrain[0])
    #A = np.where(yTrain[0] > 0)
    #print(X[0, A])
    #print(xTrain[0])
    #print(yTrain[0])
    '''
    for i in range(0, T):
        A = np.where(yTrain[i] > 0)
        B = np.where(yTrain[i] < 1)
        X = np.array(xTrain[i])
        #print(X[1,:])
        plt.scatter(X[0, A], X[1, A], color="r")
        plt.scatter(X[0, B], X[1, B], color="g")
        plt.title("Checkerboard")
        plt.show()
    '''
    return xTrain, yTrain