arrAccSCARGC, finalAccSCARGC = accSCARGC(path, sep, 'UG_2C_5D', steps)
    
    #sinthetic
    dataValues, dataLabels, description = setup.loadUG_2C_5D(path, sep)










    experiments[4] = Experiment(proposed_gmm_core_extraction, 2, 0.9, "kde", poolSize, isBatchMode)

    '''
    Proposed method 2 (Intersection between two distributions + GMM)
    '''
    #experiments[5] = Experiment(intersection)
    
    '''Proposed method 4 (classifying and removing boundaries points with SVM)'''
    experiments[7] = Experiment(proposed_gmm_decision_boundaries, 7, None, "kde", poolSize, isBatchMode)
    experiments[99] = Experiment(improved_intersection, 7, None, "kde", poolSize, isBatchMode)