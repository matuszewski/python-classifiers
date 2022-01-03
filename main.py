#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

# badany zbior zbiorow
jX = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],[166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42],
     [181, 85, 43], [168, 77, 41]]

# tabela Y bedzie uczyla program klasyfikacji
jY = ['x', 'x', 'y', 'y', 'x', 'x', 'y', 'y', 'y', 'x', 'x', 'y']


dataset_filename = 'ewp_dsh_zgony_po_szczep.csv'
dataset_file = open(dataset_filename)
dataset = list(csv.reader(dataset_file, delimiter=';'))

# ustawianie legedny (nagłówka) oraz danych
legend = dataset[0]
data = dataset[1:]

X = []
Y = []

for record in data:

    # zamieniamy wartosci K i M na 1 i 0 w zaleznosci od plci: kobieta = 1, mezczyzna = 0
    if record[3] == "K":
        record[3] = 1
    else:
        record[3] = 0

    # dodajemy do listy X po 3 atrybuty: wiek, czyBylyWspolistniejace, zmieniamy typowanie
    X.append([record[3], int(record[4]), int(record[6])])

    # czy byl zgon
    Y.append(record[-1])

for element in X:
    print(element)


# jaka jest szansa, ze czlowiek o podanych informacjach umarl na covid?:

test_data = [[190, 70, 43], [154, 75, 38], [181, 65, 40], [168, 75, 41]]
test_labels = ['x', 'y', 'x', 'y']
