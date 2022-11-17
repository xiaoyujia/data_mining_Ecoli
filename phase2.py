import numpy as np
import pandas as pd
import sklearn.ensemble
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
import csv



def phase2():
    #import test dataset
    ecoli_test = pd.read_csv('Ecoli_test.csv')

    #import training dataset
    ecoli = pd.read_csv('Ecoli.csv')
    ecoli_raw_data = ecoli.drop(columns=['Target (Col 107)'])
    ecoli_target = ecoli['Target (Col 107)']


    ecoli_num = ecoli_raw_data.filter(like='Num')
    ecoli_nom = ecoli_raw_data.filter(like='Nom')


    #preprocessing
    #use mean method to fill up the NAN values in dataset for numberical
    imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer_mean = imputer_mean.fit(ecoli_num)
    ecoli_num= imputer_mean.transform(ecoli_num)

    # use mean method to fill up the NAN values in dataset for nominal
    imputer_most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputer_most_frequent = imputer_most_frequent.fit(ecoli_nom)
    ecoli_nom = imputer_most_frequent.transform(ecoli_nom)

    #put the data back together after imputation
    ecoli_data = ecoli_num
    ecoli_data = np.append(ecoli_data, ecoli_nom, axis = 1)

    #train test split
    x_train, x_test, y_train, y_test = train_test_split(ecoli_data, ecoli_target, test_size=0.2)

    #voting use different models
    '''
    dt_model = tree.DecisionTreeClassifier(max_depth=4)
    rf_model = RandomForestClassifier(max_depth=9)
    knn_model = KNeighborsClassifier(n_neighbors=10)
    '''
    dt_model = tree.DecisionTreeClassifier()
    rf_model = RandomForestClassifier()
    knn_model = KNeighborsClassifier()


    voting_model = VotingClassifier(estimators=[
                                                ('dt', dt_model),
                                                ('rf', rf_model),
                                                ('knn', knn_model)
                                                ])

    voting_model.fit(x_train, y_train)

    predict = voting_model.predict(ecoli_test)

    #calculate f1
    if (len(predict) == len(ecoli_target)):
        print("dt_f1: ", f1_score(ecoli_target, dt_model.fit(x_train, y_train).predict(ecoli_data)))
        print("rf_f1: ", f1_score(ecoli_target, rf_model.fit(x_train, y_train).predict(ecoli_data)))
        print("knn_f1: ", f1_score(ecoli_target, knn_model.fit(x_train, y_train).predict(ecoli_data)))
        print('F1: ', f1_score(ecoli_target, predict))
        print('CV: ', cross_val_score(voting_model, ecoli_data, ecoli_target))
        print('mean: ', np.mean(cross_val_score(voting_model, ecoli_data, ecoli_target)))




    with open('s4725500.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)

        for p in predict:
            writer.writerow([p])
        print('csv file created')
        f.close()



phase2()