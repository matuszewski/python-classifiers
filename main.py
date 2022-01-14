#!/usr/bin/env python
# -*- coding: utf-8 -*-

# dataset used: https://archive.ics.uci.edu/ml/machine-learning-databases/00472/
# test change02
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

dataset_filename = 'ewp_dsh_zgony_po_szczep.csv'
dataset_file = open(dataset_filename)
dataset = list(csv.reader(dataset_file, delimiter=';'))

# ustawianie legedny (nagłówka) oraz danych
legend = dataset[0]
data = dataset[1:]

X = []
Y = []

# usuwanie rekordow gdzie brak danych o wieku
number = -1
for record in data:
    number += 1
    # usuwamy wszystkie rekordy gdzie nie jest wpisany wiek - brak danych
    if record[4] == '':
        data.pop(number)

# usuwanie rekordow gdzie plec jest nieznana
number = -1
for record in data:
    number += 1
    if record[3] == "nieznana":
        data.pop(number)

# zamieniamy wartosci K i M na 1 i 0 w zaleznosci od plci: kobieta = 1, mezczyzna = 0
number = -1
for record in data:
    number += 1

    if record[3] == 'K':
        data[number][3] = 1
    elif record[3] == 'M':
        data[number][3] = 0


# dodawanie danych do wlasciwej tablicy dwuwymiarowej:
for record in data:
    
    # dodajemy do listy X po 3 atrybuty: wiek, czyBylyWspolistniejace, zmieniamy typowanie
    X.append([record[3], int(float(record[4])), int(record[6])])

    # dodajemy do listy Y czy byl zgon (tak: 1, nie: 2)
    Y.append(record[-1])


#for element in X:
#    print(element)


# jaka jest szansa, ze czlowiek o podanych informacjach umarl na covid?:

test_data = [[1, 65, 0], [0, 30, 1], [1, 56, 1], [0, 79, 3]]
test_labels = [1, 0, 1, 0]



for i in range(100):
    # Drzewa decyzyjne
    dtc_clf = tree.DecisionTreeClassifier()
    dtc_clf = dtc_clf.fit(X, Y)
    dtc_prediction = dtc_clf.predict(test_data)
    # print(dtc_prediction)

    # Decyzyjny las losowy
    rfc_clf = RandomForestClassifier(n_estimators=100)
    rfc_clf.fit(X, Y)
    rfc_prediction = rfc_clf.predict(test_data)
    # print(rfc_prediction)

    # Regresja logistyczna
    l_clf = LogisticRegression(solver='lbfgs')
    l_clf.fit(X, Y)
    l_prediction = l_clf.predict(test_data)
    # print(l_prediction)

    # SVC - Metoda wektorow podporowych
    s_clf = SVC(gamma='scale')
    s_clf.fit(X, Y)
    s_prediction = s_clf.predict(test_data)
    # print(s_prediction)

    # k - najblizszych sasiadow
    kNB_clf = KNeighborsClassifier()
    kNB_clf.fit(X, Y)
    kNB_prediction = kNB_clf.predict(test_data)
    # print(kNB_prediction)

    # okreslanie dokladnosci
    dtc_tree_acc += accuracy_score(dtc_prediction, test_labels)
    rfc_acc += accuracy_score(rfc_prediction, test_labels)
    l_acc += accuracy_score(l_prediction, test_labels)
    s_acc += accuracy_score(s_prediction, test_labels)
    kNB_acc += accuracy_score(kNB_prediction, test_labels)
    pass

print("Drzewo decyzyjne " + dtc_tree_acc)
print("Decyzyjny las losowy " + rfc_acc)
print("Regresja logistyczna " + l_acc)
print("SVC " + s_acc)
print("Najbliżsi sąsiedzi " + kNB_acc)

dokladnosc = np.array([dtc_tree_acc, rfc_acc, l_acc, s_acc, kNB_acc])
max_acc = np.argmax(dokladnosc)

klasyfikatory = ['Drzewo decyzyjne', 'Decyzyjny las losowy', 'Regresja logistyczna', 'SVC', 'Najbliżsi sąsiedzi']
print('\n' + klasyfikatory[max_acc] + " jest najlepszym klasyfikatorem problemu.\n")