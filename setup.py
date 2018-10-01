import numpy as np
import pandas as pd
import checkerboard
from source import util
 


def countInstances(datasetID, dataLabels):
    classes = list(set(dataLabels))
    inst = util.slicingClusteredData(dataLabels, classes)
    for i in range(len(inst)):
        print("{}: class {} -> {} instances.".format(datasetID, i, len(inst[i])))


#artificial datasets                
def loadCDT(path, sep):
    description = "One Class Diagonal Translation. 2 Dimensional data."
    
    dataValues = pd.read_csv(path+'sinthetic'+sep+'1CDT.txt',sep = ",", header=None)
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    #countInstances("1CDT", dataLabels)
    return dataValues, dataLabels, description


def loadCHT(path, sep):
    description = "One Class Horizontal Translation. 2 Dimensional data."

    dataValues = pd.read_csv(path+'sinthetic'+sep+'1CHT.txt',sep = ",", header=None)
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    #countInstances("1CHT", dataLabels)
    return dataValues, dataLabels, description


def load2CDT(path, sep):
    description = "Two Classes Diagonal Translation. 2 Dimensional data"
    
    dataValues = pd.read_csv(path+'sinthetic'+sep+'2CDT.txt',sep = ",", header=None)
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    #countInstances("2CDT", dataLabels)
    return dataValues, dataLabels, description


def load2CHT(path, sep):
    description = "Two Classes Horizontal Translation. 2 Dimensional data."
    
    dataValues = pd.read_csv(path+'sinthetic'+sep+'2CHT.txt',sep = ",", header=None)
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    #countInstances("2CHT", dataLabels)
    return dataValues, dataLabels, description


def loadUG_2C_2D(path, sep):
    description = "Two Bidimensional Unimodal Gaussian Classes." 
    
    dataValues = pd.read_csv(path+'sinthetic'+sep+'UG_2C_2D.txt',sep = ",", header=None)
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    #countInstances("UG_2C_2D", dataLabels)
    return dataValues, dataLabels, description


def loadUG_2C_3D(path, sep):
    description = "Artificial Two 3-dimensional Unimodal Gaussian Classes."
    
    dataValues = pd.read_csv(path+'sinthetic'+sep+'UG_2C_3D.txt',sep = ",", header=None)
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 3]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:3]
    #print(dataValues)
    #countInstances("UG_2C_3D", dataLabels)
    return dataValues, dataLabels, description


def loadUG_2C_5D(path, sep):
    description = "Two 5-dimensional Unimodal Gaussian Classes."
    
    dataValues = pd.read_csv(path+'sinthetic'+sep+'UG_2C_5D.txt',sep = ",", header=None)
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 5]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:5]
    #print(dataValues)
    #countInstances("UG_2C_5D", dataLabels)
    return dataValues, dataLabels, description


def loadMG_2C_2D(path, sep):
    description = "Two Bidimensional Multimodal Gaussian Classes."
    
    dataValues = pd.read_csv(path+'sinthetic'+sep+'MG_2C_2D.txt',sep = ",", header=None)
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    #countInstances("MG_2C_2D", dataLabels)
    return dataValues, dataLabels, description


def loadFG_2C_2D(path, sep):
    description = "Two Bidimensional Classes as Four Gaussians."

    dataValues = pd.read_csv(path+'sinthetic'+sep+'FG_2C_2D.txt',sep = ",", header=None)
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    #countInstances("FG_2C_2D", dataLabels)
    return dataValues, dataLabels, description


def loadGEARS_2C_2D(path, sep):
    description = "Two Rotating Gears (Two classes. Bidimensional)."

    dataValues = pd.read_csv(path+'sinthetic'+sep+'GEARS_2C_2D.txt',sep = ",", header=None)
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    #countInstances("GEARS_2C_2D", dataLabels)
    return dataValues, dataLabels, description


def loadCSurr(path, sep):
    description = "One Class Surrounding another Class. Bidimensional."

    dataValues = pd.read_csv(path+'sinthetic'+sep+'1CSurr.txt',sep = ",", header=None)
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    #countInstances("1CSurr", dataLabels)
    return dataValues, dataLabels, description


def load5CVT(path, sep):
    description = "Five Classes Vertical Translation. Bidimensional."

    dataValues = pd.read_csv(path+'sinthetic'+sep+'5CVT.txt',sep = ",", header=None)
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    #countInstances("5CVT", dataLabels)
    return dataValues, dataLabels.astype(int), description


def load4CR(path, sep):
    description = 'Four Classes Rotating Separated. Bidimensional.'

    dataValues = pd.read_csv(path+'sinthetic'+sep+'4CR.txt',sep = ",", header=None)
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    #countInstances("4CR", dataLabels)
    return dataValues, dataLabels.astype(int), description


def load4CRE_V1(path, sep):
    description = 'Four Classes Rotating with Expansion V1. Bidimensional.'

    dataValues = pd.read_csv(path+'sinthetic'+sep+'4CRE-V1.txt',sep = ",", header=None)
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataLabels)
    #countInstances("4CRE-V1", dataLabels)
    return dataValues, dataLabels.astype(int), description


def load4CRE_V2(path, sep):
    description = 'Four Classes Rotating with Expansion V2. Bidimensional.'

    dataValues = pd.read_csv(path+'sinthetic'+sep+'4CRE-V2.txt',sep = ",", header=None)
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    #countInstances("4CRE-V2", dataLabels)
    return dataValues, dataLabels.astype(int), description

def load4CE1CF(path, sep):
    description = 'Four Classes Expanding and One Class Fixed. Bidimensional.'

    dataValues = pd.read_csv(path+'sinthetic'+sep+'4CE1CF.txt',sep = ",", header=None)
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 2]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:2]
    #print(dataValues)
    #countInstances("4CE1CF", dataLabels)
    return dataValues, dataLabels.astype(int), description

def loadCheckerBoard(path, sep, T=100, N=2000):
    description = 'Rotated checkerboard dataset. Rotating 2*PI.'

    #Test sets: Predicting N instances by step. T steps. Two classes.
    '''
    Same parameters from original work
    '''
    a = np.linspace(0,2*np.pi,T)
    side = 0.25
    
    auxV, auxL = checkerboard.generateData(side, a, N, T)
    #print(dV[2][0])#tempo 2 da classe 0
    dataLabels = auxL[0]
    dataValues = auxV[0]
    
    for i in range(1, T):
        dataLabels = np.hstack([dataLabels, auxL[i]])
        dataValues = np.vstack([dataValues, auxV[i]])

    #countInstances("checkerboard", dataLabels)
    return dataValues, dataLabels, description

#real datasets
def loadKeystroke(path, sep):
    description = 'Keyboard patterns database. 10 features. 4 classes.'

    dataValues = pd.read_csv(path+'real'+sep+'keystroke'+sep+'keystroke.txt',sep = ",", header=None)
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 10]
    dataLabels = dataLabels-1
    dataValues = dataValues[:,0:10]
    #print(dataValues)
    #countInstances("keystroke", dataLabels)
    return dataValues, dataLabels.astype(int), description


def loadElecData(path, sep):
    description = 'Electricity data. 7 features. 2 classes.'

    dataValues = pd.read_csv(path+'real'+sep+'elecdata'+sep+'electricity_dataset.csv',sep = ",", header=None)
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 7]
    dataValues = dataValues[:, 2:7]
    #print(dataValues)
    #countInstances("elecdata", dataLabels)
    return dataValues, dataLabels.astype(int), description


def loadNOAADataset(path, sep):
    description = 'NOAA dataset. Eight  features. Two classes.'

    #Test sets: Predicting 365 instances by step. 50 steps. Two classes.
    '''
    NOAA dataset:
    Eight  features  (average temperature, minimum temperature, maximum temperature, dew
    point,  sea  level  pressure,  visibility,  average wind speed, maximum  wind  speed)
    are  used  to  determine  whether  each  day  experienced  rain  or no rain.
    '''
    dataValues = pd.read_csv(path+'real'+sep+'noaa'+sep+'noaa_dataset.csv',sep = ",", header=None)
    dataValues = pd.DataFrame.as_matrix(dataValues)
    dataLabels = dataValues[:, 8]
    dataValues = dataValues[:,0:8]
    #dataLabels = np.squeeze(np.asarray(dataLabels))
    #countInstances("NOAA", dataLabels)
    return dataValues, dataLabels.astype(int), description


def loadSCARGCBoxplotResults(path, sep):
    print('Results from SCARGC algorithm (for boxplot and accuracy timelime).')
    path1 = path+'results_scargc'+sep+'setting_1'+sep
    path2 = path+'results_scargc'+sep+'setting_2'+sep
    arrFiles = ['1CDT', '2CDT', '1CHT', '2CHT', '4CR', '4CRE-V1', '4CRE-V2', '5CVT', '1CSurr', 
    '4CE1CF', 'UG_2C_2D', 'MG_2C_2D', 'FG_2C_2D', 'UG_2C_3D', 'UG_2C_5D', 'GEARS_2C_2D', 'keystroke']
    accuracies_1={}
    accuracies_2={}

    for i in range(len(arrFiles)):
        accuracies_1[arrFiles[i]] = np.loadtxt(path1+arrFiles[i]+'.txt')
        accuracies_2[arrFiles[i]] = np.loadtxt(path2+arrFiles[i]+'.txt')
    
    accuracies_1['noaa'] = np.loadtxt(path1+'noaa_data_matlab.csv')
    accuracies_1['elec2'] = np.loadtxt(path1+'elec2_matlab.csv')
    accuracies_2['noaa'] = np.loadtxt(path2+'noaa_data_matlab.txt')
    accuracies_2['elec2'] = np.loadtxt(path2+'elec2_matlab.txt')

    return accuracies_1, accuracies_2


def loadLevelIwBoxplotResults(path, sep):
    pathAcc = path+'results_level_iw'+sep+'acc'+sep
    pathF1 = path+'results_level_iw'+sep+'f1'+sep
    arrFiles = ['1CDT', '2CDT', '1CHT', '2CHT', '4CR', '4CRE-V1', '4CRE-V2', '5CVT', '1CSurr', 
    '4CE1CF', 'UG_2C_2D', 'MG_2C_2D', 'FG_2C_2D', 'UG_2C_3D', 'UG_2C_5D', 'GEARS_2C_2D', 'keystroke']
    accuracies={}
    F1s={}
    time={}

    for i in range(len(arrFiles)):
        accuracies[arrFiles[i]] = np.loadtxt(pathAcc+arrFiles[i]+'.txt')
        F1s[arrFiles[i]] = np.loadtxt(pathF1+arrFiles[i]+'.txt')-1
    
    accuracies['noaa'] = np.loadtxt(pathAcc+'noaa_data_matlab.csv')
    accuracies['elec2'] = np.loadtxt(pathAcc+'elec2_matlab3.csv')
    F1s['noaa'] = np.loadtxt(pathF1+'noaa_data_matlab.csv')-1
    F1s['elec2'] = np.loadtxt(pathF1+'elec2_matlab3.csv')-1

    databaseNames = ['keystroke', 'noaa', 'elec2', '1CDT', '2CDT', '1CHT', '2CHT', '4CR', '4CRE-V1', '4CRE-V2', '5CVT', '1CSurr', '4CE1CF', 'UG_2C_2D', 'MG_2C_2D', 'FG_2C_2D', 'UG_2C_3D', 'UG_2C_5D', 'GEARS_2C_2D']
    file = np.loadtxt(path+'results_level_iw'+sep+'processingTimes.txt')
    
    for i in range(len(file)):
        time[databaseNames[i]] = file[i]

    #print("1CDT ==> Acc:{}; F1:{}; Time:{}".format(np.mean(accuracies['1CDT']), np.mean(F1s['1CDT']), time['1CDT']))
    return accuracies, F1s, time