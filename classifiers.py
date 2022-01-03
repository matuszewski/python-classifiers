#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://archive.ics.uci.edu/ml/machine-learning-databases/00472/
#
# sapphire dragon
# 
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import numpy as np
from sklearn.metrics import accuracy_score
import csv


# ustawianie dokladnosci wszystkich metod na 0
dtc_tree_acc = 0
rfc_acc = 0
l_acc = 0
s_acc = 0
kNB_acc = 0

dataset_filename = 'caesarian-sections-dataset.csv'
dataset_file = open(dataset_filename)
dataset = list(csv.reader(dataset_file, delimiter=','))

# ustawianie legedny (nagłówka) oraz danych
legend = dataset[0]
data = dataset[1:]

print(legend)
print('-')

for record in data:
    print(record)


X = []
Y = []
