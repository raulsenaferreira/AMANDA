import os
import sys
sys.path.append(os.getcwd()+"\\source")

from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from math import sqrt
import numpy as np


def evaluate(y_actual, y_predicted):
    return round(accuracy_score(y_actual, y_predicted), 4)*100
    

def F1(arr_y_true, arr_y_predicted, average):
	arrF1 = []
	m = min(len(arr_y_true), len(arr_y_predicted))

	for y_true, y_pred in zip(arr_y_true, arr_y_predicted):
		y_true, y_pred = y_true[:m], y_pred[:m]
		arrF1.append(f1_score(y_true, y_pred, average=average))

	return arrF1


def mcc(arr_y_true, arr_y_predicted):
	arrMcc = []
	m = min(len(arr_y_true), len(arr_y_predicted))

	for y_true, y_predicted in zip(arr_y_true, arr_y_predicted):
		y_true, y_predicted = y_true[:m], y_predicted[:m]
		arrMcc.append(matthews_corrcoef(y_true, y_predicted))

	return arrMcc