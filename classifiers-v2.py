#!/usr/bin/env python
# -*- coding: utf-8 -*-

# DATASET SOURCE: https://archive.ics.uci.edu/ml/machine-learning-databases/00472/
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import numpy as np
from sklearn.metrics import accuracy_score
import csv
from os import system


# ustawianie dokladnosci wszystkich metod na 0
dtc_acc = 0
rfc_acc = 0
lr_acc = 0
svc_acc = 0
knb_acc = 0

dataset_filename = 'datasets/caesarian-sections.csv'
dataset_file = open(dataset_filename)
dataset = list(csv.reader(dataset_file, delimiter=','))

# ustawianie legedny (nagłówka) oraz danych
legend = dataset[0]
data = dataset[1:]

X = []
Y = []

for record in data:    
    #___________wiek____________liczba dzieci___czas przyjazdu__czy byly problemy kardiologiczne____
    X.append([  int(record[0]), int(record[1]), int(record[2]), int(record[3])  ])
    
    #___________cesarskie ciecie____________________________________________________________________
    Y.append(   int(record[4])  )

# wyswietlenie danych 
#print(X)
#print('-' * 30)
#print(Y)
#print('-' * 30)


# dane testowe (manualnie)
test_data = [[25, 1, 0, 2, 0], [26, 2, 0, 1, 0], [33, 3, 3, 1], [24, 1, 0, 0]]

# dane testowe (wyodrebnienie 25% - albo wlasnej ilosci)
test_data = [[22,1,0,2], [26,2,0,1], [26,2,1,1], [28,1,0,2], [22,2,0,1], [26,1,1,0], [27,2,0,1], [32,3,0,1], [28,2,0,1], [27,1,1,1], [36,1,0,1], [33,1,1,0], [23,1,1,1], [20,1,0,1], [29,1,2,0], [25,1,2,0], [25,1,0,1], [20,1,2,2], [37,3,0,1], [24,1,2,0]]
test_labels = [0,1,0,0,1,0,0,1,0,1,0,1,0,0,1,0,0,1,1,1]

liczba_testow = 100
for i in range(liczba_testow):
    # metoda klasyfikatora drzewa decyzyjnego
    dtc_clf = tree.DecisionTreeClassifier()
    dtc_clf = dtc_clf.fit(X, Y)
    dtc_prediction = dtc_clf.predict(test_data)
    #print(dtc_prediction)

    # metoda klasyfikatora decyzyjnego lasu losowego
    rfc_clf = RandomForestClassifier(n_estimators=100)
    rfc_clf.fit(X, Y)
    rfc_prediction = rfc_clf.predict(test_data)
    #print(rfc_prediction)

    # metoda regresji logistycznej
    lr_clf = LogisticRegression(solver='lbfgs')
    lr_clf.fit(X, Y)
    lr_prediction = lr_clf.predict(test_data)
    #print(lr_prediction)

    # metoda wektorow podporowych SVC
    svc_clf = SVC(gamma='scale')
    svc_clf.fit(X, Y)
    svc_prediction = svc_clf.predict(test_data)
    #print(svc_prediction)

    # metoda klasyfikatora k-najblizszych sąsiadów
    knb_clf = KNeighborsClassifier()
    knb_clf.fit(X, Y)
    knb_prediction = knb_clf.predict(test_data)
    #print(knb_prediction)

    # określanie dokładności klasyfikacji
    dtc_acc += accuracy_score(dtc_prediction, test_labels)
    rfc_acc += accuracy_score(rfc_prediction, test_labels)
    lr_acc += accuracy_score(lr_prediction, test_labels)
    svc_acc += accuracy_score(svc_prediction, test_labels)
    knb_acc += accuracy_score(knb_prediction, test_labels)
    
    system('cls')

dokładnosc = np.array([dtc_acc, rfc_acc, lr_acc, svc_acc, knb_acc])
max_acc = np.argmax(dokładnosc)

#print(f'Drzewo decyzyjne {dtc_acc}')
#print(f'Decyzyjny las losowy {rfc_acc}')
#print(f'Regresja logistyczna {lr_acc}')
#print(f'SVC {svc_acc}')
#print(f'Najbliżsi sąsiedzi {knb_acc}')

## Python program to print the data
podsumowanie = {
    "Drzewo decyzyjne": dtc_acc,
    "Decyzyjny las losowy": rfc_acc,
    "Regresja logistyczna": lr_acc,
    "Metoda wektorów podporowych SVC": svc_acc,
    "k-Najbliższych sąsiadów": knb_acc,
}

print("{:<40} {:<12}".format('KLASYFIKATOR', 'SKUTECZNOŚĆ'))
print('-'*55)
for element in podsumowanie.items():
    klasyfikator, skutecznosc = element
    print("{:<35} {:<12}".format(klasyfikator, skutecznosc))


klasyfikatory = ['Drzewo decyzyjne', 'Decyzyjny las losowy', 'Regresja logistyczna', 'SVC', 'Najbliżsi sąsiedzi']
print(f'\nNajlepszy klasyfiaktor do tego problemu: {klasyfikatory[max_acc]}')

print(f'liczba kompletow danych: {len(X)}')
print(f'liczba iteracji: {liczba_testow}')
