import numpy as np
from source import classifiers
from source import metrics
from source import util
from source import plotFunctions
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from matplotlib import animation



def startAnimation(algorithmName, arrX, arrY, arrUt, arrYt, arrClf):
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    #ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
    #line, = ax.plot([], [], lw=2)
    #fig, ax = plt.subplots()
    
    # initialization function: plot the background of each frame
    def init():
        '''X = arrX[0]
        y = arrY[0]
        clf = arrClf[0]
        #decision boundaries
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        contour = plt.contourf(xx, yy, Z, alpha=0.4)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')'''
        scatter = plt.scatter([], [], s=20, edgecolor='k')
        return scatter,
        
    
    # animation function.  This is called sequentially
    def animate(i):
        X = arrX[i]
        y = arrY[i]
        clf = arrClf[i]
        Ut = arrUt[i]
        yt = arrYt[i] #gabarito
        
        #decision boundaries
        x_min, x_max = Ut[:, 0].min() - 1, Ut[:, 0].max() + 1
        y_min, y_max = Ut[:, 1].min() - 1, Ut[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        contour = plt.contourf(xx, yy, Z, alpha=0.4)
        scatter = plt.scatter(Ut[:, 0], Ut[:, 1], c=yt, s=30)
        cores = plt.scatter(X[:, 0], X[:, 1], c=y, s=50, marker ='v', edgecolor='k')
        plt.title("Batch {}".format(i+1))
        plt.show()
        return scatter,
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=100, interval=10000, blit=True)
    print('saving video in... results/1CSurr_animation_'+algorithmName+'.mp4')
    anim.save('results/1CSurr_animation_'+algorithmName+'.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()


def run(dataValues, dataLabels, datasetDescription, isBinaryClassification, isImbalanced, experiments, batches, labeledData, isBatchMode, poolSize, plot_anim, externalResults={}):
    listOfAccuracies = []
    listOfMethods = []
    listOfMCCs = []
    listOfF1sMacro = []
    listOfF1sMicro = []
    listOfTimeExecutions = []
    avgAccuracies = []
    
    sizeOfBatch = int((len(dataLabels)-labeledData)/batches)
    arrYt = dataLabels[labeledData:]
    arrYt = [ arrYt[i::batches] for i in range(batches) ]
    
    print(datasetDescription)
    if isBatchMode:
        print("{} batches of {} instances".format(batches, sizeOfBatch))
    else:
        print("Stream mode with pool size = {}".format(poolSize))
    print("\n\n")

    #F1Type = 'macro' #for balanced datasets with 2 classes
    #if isImbalanced or not isBinaryClassification:
    #    F1Type='micro'
    
    for name, e in experiments.items():
        try:
            CoreX = []
            CoreY = []
            accTotal = []
            accuracies=[]
            classes = list(set(dataLabels))#getting all possible classes in data

            start = timer()
            #accuracy per step
            algorithmName, accuracies, CoreX, CoreY, arrX, arrY, arrUt, arrYt, arrClf, arrPredicted = e.method.start(
                dataValues=dataValues, dataLabels=dataLabels, classes=classes, densityFunction=e.densityFunction, 
                batches=batches, sizeOfBatch = sizeOfBatch, initialLabeledData=labeledData, excludingPercentage=e.excludingPercentage, 
                K_variation=e.K_variation, clfName=e.clfName, poolSize=poolSize, isBatchMode=isBatchMode)
            end = timer()
            averageAccuracy = np.mean(accuracies)

            #elapsed time per step
            elapsedTime = end - start
            
            accTotal.append(averageAccuracy)

            arrF1Macro = metrics.F1(arrYt, arrPredicted, 'macro')
            arrF1Micro = metrics.F1(arrYt, arrPredicted, 'micro')
            listOfAccuracies.append(accuracies)
            listOfMethods.append(algorithmName)
            listOfF1sMacro.append(arrF1Macro)
            listOfF1sMicro.append(arrF1Micro)
            listOfTimeExecutions.append(elapsedTime)
            
            print("Execution time: ", elapsedTime)
            
            if isBinaryClassification:
                arrMCC = metrics.mcc(arrYt, arrPredicted)
                listOfMCCs.append(arrMCC)
                print("Average MCC: ", np.mean(arrMCC))

            print("Average error:", 100-averageAccuracy)
            print("Average macro-F1: {}".format(np.mean(arrF1Macro)))
            print("Average micro-F1: {}".format(np.mean(arrF1Micro)))
            plotFunctions.finalEvaluation(accuracies, batches, algorithmName)
            plotFunctions.plotF1(arrF1Macro, batches, algorithmName)
            plotFunctions.plotF1(arrF1Micro, batches, algorithmName)
            avgAccuracies.append(np.mean(accuracies))
            
            #print data distribution in step t
            initial = (batches*sizeOfBatch)-sizeOfBatch
            final = initial + sizeOfBatch
            #plotFunctions.plot(dataValues[initial:final], dataLabels[initial:final], CoreX, CoreY, batches)
            print("\n\n")
            if plot_anim == True:
                startAnimation(algorithmName, arrX, arrY, arrUt, arrYt, arrClf)
        except Exception as e:
            print(e)
            raise e
        
    
    # Beginning of external results plottings
    for extResult in externalResults:
        print("Method: {}".format(extResult['name']))
        print("Execution time: ", elapsedTime)
        
        if isBinaryClassification:
            MCCs = metrics.mcc(arrYt, extResult['predictions'])
            print("Average MCC: ", np.mean(MCCs))
            listOfMCCs.append(MCCs)

        arrF1External = metrics.F1(arrYt, extResult['predictions'], 'macro')
        print("Average macro-F1: {}".format(np.mean(arrF1External)))
        arrF1External = metrics.F1(arrYt, extResult['predictions'], 'micro')
        print("Average micro-F1: {}".format(np.mean(arrF1External)))

        plotFunctions.finalEvaluation(extResult['accuracies'], batches, extResult['name'])
        plotFunctions.plotF1(arrF1External, batches, extResult['name'])
        
        listOfMethods.append(extResult['name'])
        listOfAccuracies.append(extResult['accuracies'])
        
        listOfF1sMacro.append(arrF1External)
        listOfF1sMicro.append(arrF1External)
        listOfTimeExecutions.append(extResult['time'])
        avgAccuracies.append(np.mean(extResult['accuracies']))
    # End of external results plottings

    plotFunctions.plotBoxplot('acc', listOfAccuracies, listOfMethods)
    
    if isBinaryClassification:
        plotFunctions.plotBoxplot('mcc', listOfMCCs, listOfMethods)

    plotFunctions.plotBoxplot('macro-f1', listOfF1sMacro, listOfMethods)
    plotFunctions.plotBoxplot('micro-f1', listOfF1sMicro, listOfMethods)
    plotFunctions.plotAccuracyCurves(listOfAccuracies, listOfMethods)
    plotFunctions.plotBars(listOfTimeExecutions, listOfMethods)
    plotFunctions.plotBars2(avgAccuracies, listOfMethods)
    plotFunctions.plotBars3(avgAccuracies, listOfMethods)
    plotFunctions.plotBars4(avgAccuracies[0], avgAccuracies, listOfMethods)