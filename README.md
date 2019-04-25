## AMANDA: Semi-supervised approach for learning on environments with gradual drift and extreme verification latency

You can test the framework installing the Jupyter notebook in your machine and going to the specific file and executing the code (CTRL+ENTER) 

for artificial datasets go to Datasets/benchmarking/sinthetic/BATCH_MODE

for real datasets go to Datasets/benchmarking/real/BATCH_MODE

## Project structure

Here are the structure of the project

### Folders

**data** = All datasets (real and sinthetic) applied to this work. It contains data from experiments from LevelIW and SCARGC algorithms.

**experiments**: 
>>>> **Algorithms benchmarking** = Contains the optmization algorithms for choosing the parameters for the algorithms

>>>> **Datasets benchmarking** = Contains contains all experiments for the sinthetic and real datasets. Each file represents an experiment using all algorithms over a specific dataset.
                        
**methods** = Contains all the algorithms applied for this work, including the customized gridsearch for the AMANDA fixed and dynamic versions. It also contains the methods applied for computing the prequential tests for all algorithms.

**results** = All experiment results ordered by dataset

>>>> **dynamic** = Results for all datasets using AMANDA-DCP (results from 5 different classifiers tested in this work)

>>>> **fixed** = Results for all datasets using AMANDA-FCP (results from 5 different classifiers tested in this work)

**source** = Contains the following files:
>>>> **classifiers.py** = Call / customization of several classifiers contained in the scikit library, including statistical methods such as GMM and KDE

>>>> **metrics** = Implementation of all evaluation metrics applied in this work

>>>> **plotFunctions.py** = Implementation for several types of plots

>>>> **utils.py** = Implementation of auxiliary functions used with the main algorithms from this work

### Files

**checkerboard.py** = Implementation of checkerboard dataset
**setup.py** = File that loads all datasets from the folder data

P.S.: **We are refining and cleaning the code. It is not a final version.**
