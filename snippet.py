
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

print(f'Drzewo decyzyjne {dtc_tree_acc}')
print(f'Decyzyjny las losowy {rfc_acc}')
print(f'Regresja logistyczna {l_acc}')
print(f'SVC {s_acc}')
print(f'Najbliżsi sąsiedzi {kNB_acc}')


dokładnosc = np.array([dtc_tree_acc, rfc_acc, l_acc, s_acc, kNB_acc])
max_acc = np.argmax(dokładnosc)

klasyfikatory = ['Drzewo decyzyjne', 'Decyzyjny las losowy', 'Regresja logistyczna', 'SVC', 'Najbliżsi sąsiedzi']
print('\n' + klasyfikatory[max_acc] + ' jest najlepszym klasyfikatorem problemu.\n')