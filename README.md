## AMANDA: Semi-supervised approach for learning on environments with gradual drift and extreme verification latency


You can test the framework for artificial datasets through a jupyter notebook in Datasets/benchmarking/sinthetic/BATCH_MODE or STREAM_MODE

You can test the framework for real datasets through a jupyter notebook in Datasets/benchmarking/real/BATCH_MODE or STREAM_MODE

### Project structure

Here are the structure of the project

#### Folders

**data** = All datasets (real and sinthetic) applied to this work. It contains data from experiments from LevelIW and SCARGC algorithms.

**experiments**: 
>>>> **Algorithms benchmarking** = Contains the optmization algorithms for choosing the parameters for the algorithms

>>>> **Datasets benchmarking** = Contains contains all experiments for the sinthetic and real datasets. Each file represents an experiment using all algorithms over a specific dataset.
            
>>>> **Test cases** = Contains the plots for the decision boundaries regarding the AMANDA classifier and the most significative datasets
            
**methods** = Contains all the algorithms applied for this work, including the customized gridsearch for the AMANDA fixed and dynamic versions. It also contains the methods applied for computing the prequential tests for all algorithms.

**results/old results** = All experiment results ordered by dataset

**source** = Contains the following files:
>>>> **classifiers.py** = Call / customization of several classifiers contained in the scikit library, including statistical methods such as GMM and KDE

>>>> **metrics** = Implementation of all evaluation metrics applied in this work

>>>> **plotFunctions.py** = Implementation for several types of plots

>>>> **utils.py** = Implementation of auxiliary functions used with the main algorithms from this work

#### Files

**checkerboard.py** = Implementation of checkerboard dataset
**setup.py** = File that loads all datasets from the folder data

P.S.: **We are refining and cleaning the code. It is not a final version.**
