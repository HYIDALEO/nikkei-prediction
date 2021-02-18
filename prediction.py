import math

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
import sklearn.neighbors as neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.utils import shuffle

market_df = pd.read_csv('E:/hw1_data/ex_daily.txt',index_col=0)

ex_daily_DATA = []
ex_daily_VALUE = []
nikkei_daily_DATA = []
nikkei_daily_VALUE = []
sp_daily_caldt = []
sp_daily_sprtrn = []


def read_ex_daily():
    f = open('E:/hw1_data/ex_daily.txt', "r");
    line = f.readline()
    line = f.readline()

    while line:
        line = line[:-1]
        ele = line.split()
        ex_daily_DATA.append(ele[0].replace("-",""))
        ex_daily_VALUE.append(float(ele[1]))
        line = f.readline()
    f.close()
#    print(ex_daily_DATA)
#    print(ex_daily_VALUE)

def read_nikkei_daily():
    f = open('E:/hw1_data/nikkei_daily.txt', "r");
    line = f.readline()
    line = f.readline()

    while line:
        line = line[:-1]
        ele = line.split()

        data = ele[0]
        data = data.replace("Jan", "01")
        data = data.replace("Feb", "02")
        data = data.replace("Mar", "03")
        data = data.replace("Apr", "04")
        data = data.replace("May", "05")
        data = data.replace("Jun", "06")
        data = data.replace("Jul", "07")
        data = data.replace("Aug", "08")
        data = data.replace("Sep", "09")
        data = data.replace("Oct", "10")
        data = data.replace("Nov", "11")
        data = data.replace("Dec", "12")
        year = data[6:10]
        month = data[3:5]
        day = data[0:2]
        data = year + month + day
        nikkei_daily_DATA.append(data)
        nikkei_daily_VALUE.append(float(ele[1]))
        line = f.readline()
    f.close()

# first input
def read_sp_daily():
    f = open('E:/hw1_data/sp_daily.txt', "r");
    line = f.readline()
    line = f.readline()

    while line:
        line = line[:-1]
        ele = line.split()
        sp_daily_caldt.append(ele[0])
        sp_daily_sprtrn.append(float(ele[1]))
        line = f.readline()
    f.close()

def get_sp_daily_input_list():
    sp_daily_input_list = []
    sp_daily_input_test_list = []
    jan2005 = sp_daily_caldt.index("20050104")
    dec2007 = sp_daily_caldt.index("20071228")

    for count in range(jan2005,dec2007+1):
        if sp_daily_caldt[count] in nikkei_daily_DATA and sp_daily_caldt[count] in ex_daily_DATA:
            sp_daily_input_list.append(sp_daily_sprtrn[count])

    #共有第一天，20120724，下标2907
    # 共有的最后一天，20121019，下标，2969

    for count in range(2908,2970):
        if sp_daily_caldt[count] in nikkei_daily_DATA and sp_daily_caldt[count] in ex_daily_DATA:
            sp_daily_input_test_list.append(sp_daily_sprtrn[count])
    return sp_daily_input_list,sp_daily_input_test_list


def get_nikkei_increase_list():
    nikkei_increase_list = []
    jan2005 = nikkei_daily_DATA.index("20050104")
    dec2007 = nikkei_daily_DATA.index("20071228")

    nikkei_increase_test_list = []
    for count in range(jan2005,dec2007+1):
        if nikkei_daily_DATA[count] in sp_daily_caldt and nikkei_daily_DATA[count] in ex_daily_DATA:
            if nikkei_daily_VALUE[count] > nikkei_daily_VALUE[count-1]:
                nikkei_increase_list.append(1)
            elif nikkei_daily_VALUE[count] < nikkei_daily_VALUE[count-1]:
                nikkei_increase_list.append(-1)

    #共有第一天，20120724，下标2907
    # 共有的最后一天，20121019，下标，2969
 #   print(nikkei_daily_DATA.index("20120724"))
 #   print(nikkei_daily_DATA.index("20121019"))

 #   print('now')
    for count in range(7021,7082):
        if nikkei_daily_DATA[count] in sp_daily_caldt and nikkei_daily_DATA[count] in ex_daily_DATA:
            if nikkei_daily_VALUE[count] > nikkei_daily_VALUE[count-1]:
                nikkei_increase_test_list.append(1)
            elif nikkei_daily_VALUE[count] < nikkei_daily_VALUE[count-1]:
                nikkei_increase_test_list.append(-1)

    return nikkei_increase_list,nikkei_increase_test_list

def get_ex_daily_dif_list():
    ex_daily_dif_list = []
    ex_daily_dif_test_list = []
    jan2005 = ex_daily_DATA.index("20050104")
    dec2007 = ex_daily_DATA.index("20071228")
    for count in range(jan2005,dec2007+1):
        if ex_daily_DATA[count] in sp_daily_caldt and ex_daily_DATA[count] in nikkei_daily_DATA:
            ex_daily_dif_list.append(math.log(ex_daily_VALUE[count - 1])-math.log(ex_daily_VALUE[count - 2]))

    # print(ex_daily_DATA.index("20120724"))
    # print(ex_daily_DATA.index("20121019"))
    #
    # print('now')
    for count in range(10427,10488):
        if ex_daily_DATA[count] in sp_daily_caldt and ex_daily_DATA[count] in nikkei_daily_DATA:
            ex_daily_dif_test_list.append(math.log(ex_daily_VALUE[count - 1])-math.log(ex_daily_VALUE[count - 2]))


    return ex_daily_dif_list,ex_daily_dif_test_list




def logi(X_train,y_train,X_test,y_test):
    l_y_train = []
    for count in range(0, len(y_train)):
        if y_train[count] == -1:
            l_y_train.append(0)
        else:
            l_y_train.append(1)

    l_y_test = []
    for count in range(0, len(y_test)):
        if y_test[count] == -1:
            l_y_test.append(0)
        else:
            l_y_test.append(1)
    y_train = l_y_train
    y_test = l_y_test

    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)
    logit = sm.Logit(y_train, X_train)
    results = logit.fit()

    results.summary()
    test_predictions = results.predict(X_test)

    prediction_array = np.array(test_predictions>0.5, dtype=int)
    print(len(prediction_array))
    print('The model made', np.sum(prediction_array == y_test) / len(y_test), '% correct predictions on the TEST SET.')

def myLDA(X_train,y_train,X_test,y_test):
    lda_clf = LDA(solver='lsqr')
    results = lda_clf.fit(X_train, y_train)
    testing_pred = results.predict(X_test)
#    pred_accu = sum(testing_pred == y_test) / len(y_test)
    print(testing_pred)
#    prediction_array = np.array(testing_pred > 0.5, dtype=float)
#    print(confusion_matrix(y_test, prediction_array))
    print(testing_pred == y_test)
    rsum = 0
    for count in range(0,len(y_test)):
        if y_test[count] == testing_pred[count]:
            rsum += 1


    print('The model made', rsum / len(y_test), 'correct predictions on the testing dataset.')

def myCLF(X_train, y_train,X_test,y_test,tree_depth):
    N_size = np.shape(X_train)[0]
    Dot = np.linspace(0, N_size, 10 + 1)
    Bound = [int(i) for i in Dot]

    clf = DecisionTreeClassifier(max_depth=tree_depth)
    X_temp = X_train.copy()
    y_temp = y_train.copy()
    Tr_error = []
    Va_error = []
    Va_acc = []
    for j in range(10):
        X_Va_df = X_temp[Bound[j]:Bound[j + 1]]
        y_Va_df = y_temp[Bound[j]:Bound[j + 1]]

        Index_tr = np.arange(0, Bound[j]).tolist() + np.arange(Bound[j + 1], N_size).tolist()

        X_Tr_df = X_temp[Index_tr, :]
        y_Tr_df = y_temp[Index_tr, :]
        tree_est = clf.fit(X_Tr_df, y_Tr_df)

        Y_Tr_pred = clf.predict(X_Tr_df)
        Y_Va_pred = clf.predict(X_Va_df)

        mse_tr = np.mean((Y_Tr_pred - y_Tr_df) ** 2)
        mse_va = np.mean((Y_Va_pred - y_Va_df) ** 2)
        Va_error.append(mse_va)
        Tr_error.append(mse_tr)


    print("For CLF with max_depth =", tree_depth)
    print("the MSE for train set =", np.mean(Tr_error))
    print("the MSE for test set =", np.mean(Va_error))

def clfpre(X_train, y_train,X_test,y_test):
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    rsum = 0
    for count in range(0, len(y_test)):
        if y_test[count] == y_pred[count]:
            rsum += 1

    print('The classification tree made', rsum / len(y_test), 'correct predictions on the testing dataset.')

def mySVMLinear(X_train,y_train,X_test,y_test,value_C):
    N_size = np.shape(X_train)[0]
    Dot = np.linspace(0, N_size, 10 + 1)
    Bound = [int(i) for i in Dot]


    X_temp = X_train.copy()
    y_temp = y_train.copy()
    Tr_error = []
    Va_error = []
    for j in range(10):
        svc0 = SVC(kernel='linear', C=value_C)
        X_Va_df = X_temp[Bound[j]:Bound[j + 1]]
        y_Va_df = y_temp[Bound[j]:Bound[j + 1]]

        Index_tr = np.arange(0, Bound[j]).tolist() + np.arange(Bound[j + 1], N_size).tolist()

        X_Tr_df = X_temp[Index_tr, :]
        y_Tr_df = y_temp[Index_tr, :]

        svc0.fit(X_Tr_df, y_Tr_df)

        Y_Tr_pred = svc0.predict(X_Tr_df)
        Y_Va_pred = svc0.predict(X_Va_df)

        mse_tr = np.mean((Y_Tr_pred - y_Tr_df) ** 2)
        mse_va = np.mean((Y_Va_pred - y_Va_df) ** 2)
        Va_error.append(mse_va)
        Tr_error.append(mse_tr)


    print("For linear SVM with C =", value_C)
    print("the MSE for train set =", np.mean(Tr_error))
    print("the MSE for test set =", np.mean(Va_error))

def SVMLinearpre(X_train,y_train,X_test,y_test):
    svm0 = SVC(kernel='linear',  C=0.1)
    svm0.fit(X_train, y_train)
    y_pred = svm0.predict(X_test)
    rsum = 0
    for count in range(0, len(y_test)):
        if y_test[count] == y_pred[count]:
            rsum += 1

    print('The linear SVM made', rsum / len(y_test), 'correct predictions on the testing dataset.')


def SVMpre(X_train,y_train,X_test,y_test):
    svm0 = SVC(kernel='rbf', gamma=0.001, C=0.0005)
    svm0.fit(X_train, y_train)
    y_pred = svm0.predict(X_test)
    rsum = 0
    for count in range(0, len(y_test)):
        if y_test[count] == y_pred[count]:
            rsum += 1

    print('The Radical SVM made', rsum / len(y_test), 'correct predictions on the testing dataset.')

def mySVMRadial(X_train,y_train,X_test,y_test,value_C,value_gamma):
    N_size = np.shape(X_train)[0]
    Dot = np.linspace(0, N_size, 10 + 1)
    Bound = [int(i) for i in Dot]

    svm0 = SVC(kernel='rbf', gamma=value_gamma, C=value_C)
    X_temp = X_train.copy()
    y_temp = y_train.copy()
    Tr_error = []
    Va_error = []
    for j in range(10):
        X_Va_df = X_temp[Bound[j]:Bound[j + 1]]
        y_Va_df = y_temp[Bound[j]:Bound[j + 1]]

        Index_tr = np.arange(0, Bound[j]).tolist() + np.arange(Bound[j + 1], N_size).tolist()

        X_Tr_df = X_temp[Index_tr, :]
        y_Tr_df = y_temp[Index_tr, :]

        svm0.fit(X_Tr_df, y_Tr_df)

        Y_Tr_pred = svm0.predict(X_Tr_df)
        Y_Va_pred = svm0.predict(X_Va_df)

        mse_tr = np.mean((Y_Tr_pred - y_Tr_df) ** 2)
        mse_va = np.mean((Y_Va_pred - y_Va_df) ** 2)
        Va_error.append(mse_va)
        Tr_error.append(mse_tr)


    print("For Radical SVM with C =", value_C," gamma = ", value_gamma)

    print("the MSE for train set =", np.mean(Tr_error))
    print("the MSE for test set =", np.mean(Va_error))
    return np.mean(Va_error)



def main():
    read_ex_daily()
    read_nikkei_daily()
    read_sp_daily()
    nikkei_increase_list,nikkei_increase_test_list = get_nikkei_increase_list()
    sp_daily_input_list,sp_daily_input_test_list = get_sp_daily_input_list()
    ex_daily_dif_list,ex_daily_dif_test_list = get_ex_daily_dif_list()

    # sp_daily_input_list,ex_daily_dif_list
    # dif多两个数据
    X_train = np.array([[sp_daily_input_list[count],ex_daily_dif_list[count]]for count in range(len(sp_daily_input_list)-60)])
    X_test = np.array([[sp_daily_input_list[count],ex_daily_dif_list[count]]for count in range(len(sp_daily_input_list)-60,len(sp_daily_input_list))])


    X_name = ['sp', 'ex']
 #   X_train = pd.DataFrame(data=X_train,index=[X_name])

    y_train = np.array([nikkei_increase_list[:len(nikkei_increase_list)-60]]).reshape(-1, 1)
    y_test =  np.array([nikkei_increase_list[len(nikkei_increase_list)-60:]]).reshape(-1, 1)
    all_train = np.array([[sp_daily_input_list[count],ex_daily_dif_list[count],nikkei_increase_list[count]]for count in range(len(sp_daily_input_list))])
    print(len(y_train))
    print(len(y_test))


 #   X_test = np.array(
 #       [[sp_daily_input_test_list[count], ex_daily_dif_test_list[count]] for count in range(len(sp_daily_input_test_list))])

#    y_test = np.array([nikkei_increase_test_list]).reshape(-1, 1)

    all_test = np.array(
        [[sp_daily_input_test_list[count], ex_daily_dif_test_list[count],nikkei_increase_test_list[count]] for count in range(len(sp_daily_input_test_list))])




#    logi(X_train, y_train,X_test,y_test)
#    myLDA(X_train,y_train,X_test,y_test)
#    maxDepth = [3, 6, 8, 10, 15, 25]
#    for dep in maxDepth:
#       myCLF(X_train,y_train,X_test,y_test,dep)
#    C = [0.1,1,10,100,200,500,1000]
#    for c in C:
#       mySVMLinear(X_train, y_train, X_test, y_test,c)
#    C = [0.01, 0.1, 1, 10, 20, 50]
#    sig = [0.0005,0.001,0.01,0.05,0.1,0.5,1,10,100,1000,2000,3000]
#    mse = 2
#    for c in C:
#        for s in sig:
#            res = mySVMRadial(X_train, y_train, X_test, y_test,c,s)
#            if res<mse :
#                mse = res
#                print(c,s)
#    SVMpre(X_train, y_train, X_test, y_test)
#    SVMLinearpre(X_train, y_train, X_test, y_test)
#    clfpre(X_train, y_train, X_test, y_test)
main()