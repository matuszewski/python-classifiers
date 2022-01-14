#!/usr/bin/env python
# -*- coding: utf-8 -*-

# docstring
"""
Program for choosing best method of mathematical classification based on provided dataset.
"""

# DATASET https://archive.ics.uci.edu/ml/machine-learning-databases/00472/
from os import system
import csv
import numpy as np
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ustawianie dokladnosci wszystkich metod na 0
dtcAcc = 0
rfcAcc = 0
lrAcc = 0
svcAcc = 0
knbAcc = 0

Datafile = 'datasets/caesarian-sections.csv'
dataset_file = open(Datafile)
dataset = list(csv.reader(dataset_file, delimiter=','))

# ustawianie legedny (nagłówka) oraz danych
legend = dataset[0]
data = dataset[1:]

X = []
Y = []

for record in data:
    #___________wiek____________liczba dzieci___czas przyjazdu__czy byly problemy kard.
    X.append([  int(record[0]), int(record[1]), int(record[2]), int(record[3])  ])

    #___________cesarskie ciecie
    Y.append(   int(record[4])  )

# wyswietlenie danych
#print(X)
#print('-' * 30)
#print(Y)
#print('-' * 30)

# dane testowe (manualnie)
test_data = [[25, 1, 0, 2, 0], [26, 2, 0, 1, 0], [33, 3, 3, 1], [24, 1, 0, 0]]

# dane testowe (wyodrebnienie 25% - albo wlasnej ilosci)
test_data = [[22,1,0,2], [26,2,0,1], [26,2,1,1],
             [28,1,0,2], [22,2,0,1], [26,1,1,0],
             [27,2,0,1], [32,3,0,1], [28,2,0,1],
             [27,1,1,1], [36,1,0,1], [33,1,1,0],
             [23,1,1,1], [20,1,0,1], [29,1,2,0],
             [25,1,2,0], [25,1,0,1], [20,1,2,2],
             [37,3,0,1], [24,1,2,0]]
test_labels = [0,1,0,0,1,0,0,1,0,1,0,1,0,0,1,0,0,1,1,1]

LiczbaTestow = 100

for i in range(liczba_testow):
    # metoda klasyfikatora drzewa decyzyjnego
    dtcClf = tree.DecisionTreeClassifier()
    dtcClf = dtcClf.fit(X, Y)
    dtcPrediction = dtcClf.predict(test_data)
    #print(dtc_prediction)

    # metoda klasyfikatora decyzyjnego lasu losowego
    rfcClf = RandomForestClassifier(n_estimators=100)
    rfcClf.fit(X, Y)
    rfcPrediction = rfcClf.predict(test_data)
    #print(rfc_prediction)

    # metoda regresji logistycznej
    lrClf = LogisticRegression(solver='lbfgs')
    lrClf.fit(X, Y)
    lrPrediction = lrClf.predict(test_data)
    #print(lr_prediction)

    # metoda wektorow podporowych SVC
    svcClf = SVC(gamma='scale')
    svcClf.fit(X, Y)
    svcPrediction = svcClf.predict(test_data)
    #print(svc_prediction)

    # metoda klasyfikatora k-najblizszych sąsiadów
    knbClf = KNeighborsClassifier()
    knbClf.fit(X, Y)
    knbPrediction = knbClf.predict(test_data)
    #print(knb_prediction)

    # określanie dokładności klasyfikacji
    dtcAcc += accuracy_score(dtcPrediction, test_labels)
    rfcAcc += accuracy_score(rfcPrediction, test_labels)
    lrAcc += accuracy_score(lrPrediction, test_labels)
    svcAcc += accuracy_score(svcPrediction, test_labels)
    knbAcc += accuracy_score(knbPrediction, test_labels)

dokladnosc = np.array([dtcAcc, rfcAcc, lrAcc, svcAcc, knbAcc])
maxAcc = np.argmax(dokladnosc)

podsumowanie = {
    "Drzewo decyzyjne": dtcAcc,
    "Decyzyjny las losowy": rfcAcc,
    "Regresja logistyczna": lrAcc,
    "Metoda wektorow podporowych SVC": svcAcc,
    "k-Najbliższych sasiadów": knbAcc,
}

print("{:<40} {:<12}".format('KLASYFIKATOR', 'SKUTECZNOSC'))
print('-'*55)
for element in podsumowanie.items():
    klasyfikator, skutecznosc = element
    print("{:<35} {:<12}".format(klasyfikator, skutecznosc))

klasyfikatory = ['Drzewo decyzyjne',
                 'Decyzyjny las losowy',
                 'Regresja logistyczna',
                 'SVC',
                 'Najblizsi sasiedzi']

print("Najlepszy klasyfiaktor do tego problemu: " + klasyfikatory[maxAcc])

print("liczba kompletow danych: " + str(len(X)))
print("liczba iteracji: " + str(LiczbaTestow))
