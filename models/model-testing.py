import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import threading

def load_data():
    digits = datasets.load_digits()
    dataset = digits['data']
    targets = digits['target']
    return digits, dataset, targets

def view_digit(digits, index):
    plt.imshow(digits.images[index], cmap = plt.cm.gray_r, interpolation = 'nearest')
    plt.title(str(digits.target[index]))
    plt.show()

def init_svm():
    svc = svm.SVC(gamma=0.001, C = 100.)
    return svc

def fit_model(model, dataset, targets, training_size):
    model.fit(dataset[:training_size], targets[:training_size])

def test_model(model, dataset, targets, training_size):
    predictions = model.predict(dataset[training_size+1:])
    #print(predictions, targets[1791:])
    return accuracy_score(predictions, targets[training_size+1:])

def svc_classify(dataset, targets, training_size):
    svc = init_svm()
    fit_model(svc, dataset, targets, training_size)
    accuracy = test_model(svc, dataset, targets, training_size)
    return accuracy

def init_dtc():
    dtc = DecisionTreeClassifier(criterion="gini")
    return dtc

def dtc_classify(dataset, targets, training_size):
    dtc = init_dtc()
    fit_model(dtc, dataset, targets, training_size)
    accuracy = test_model(dtc, dataset, targets, training_size)
    return accuracy

def init_rfc():
    rfc = RandomForestClassifier(n_estimators=150)
    return rfc

def rfc_classify(dataset, targets, training_size):
    rfc = init_rfc()
    fit_model(rfc, dataset, targets, training_size)
    accuracy = test_model(rfc, dataset, targets, training_size)
    return accuracy

def do_svc(dataset, targets):
    global svc_data, input_data
    for i in range (10, 1800, 10):
        training_size = i
        svc = svc_classify(dataset, targets, training_size)
        svc_data.append(svc)
        input_data.append(training_size)

def do_dtc(dataset, targets):
    global dtc_data
    for i in range (10, 1800, 10):
        training_size = i
        dtc = dtc_classify(dataset, targets, training_size)
        dtc_data.append(dtc)

def do_rfc(dataset, targets):
    global rfc_data
    for i in range (10, 1800, 10):
        training_size = i
        rfc = rfc_classify(dataset, targets, training_size)
        rfc_data.append(rfc)

if __name__ == "__main__":
    global svc_data, dtc_data, rfc_data, input_data
    digits, dataset, targets = load_data()
    training_size = 1600
    # view_digit(digits, 17)
    # print(svc_classify(dataset, targets, training_size))
    # print(dtc_classify(dataset, targets, training_size))
    # print(rfc_classify(dataset, targets, training_size))
    
    
    svc_data = list()
    dtc_data = list()
    rfc_data = list()
    input_data = list()
    # for i in range (10, 1800, 10):
    #     training_size = i
    #     svc = svc_classify(dataset, targets, training_size)
    #     dtc = dtc_classify(dataset, targets, training_size)
    #     rfc = rfc_classify(dataset, targets, training_size)
    #     svc_data.append(svc)
    #     dtc_data.append(dtc)
    #     rfc_data.append(rfc)
    #     input_data.append(training_size)
    t1 = threading.Thread(target = lambda: do_svc(dataset=dataset, targets=targets))
    t2 = threading.Thread(target = lambda: do_dtc(dataset=dataset, targets=targets))
    t3 = threading.Thread(target = lambda: do_rfc(dataset=dataset, targets=targets))
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()
    
    plt.rcParams["figure.figsize"] = [7.00, 7.00]
    plt.rcParams["figure.autolayout"] = True
    plt.xlim(0,1800)
    plt.ylim(0, 1.0)
    plt.scatter(input_data,svc_data, label='SVC')
    plt.scatter(input_data,dtc_data, label='DTC')
    plt.scatter(input_data,rfc_data, label='RFC')
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    dataframe = pd.DataFrame(np.array([input_data, svc_data, dtc_data, rfc_data]),
                             columns=["input", "svc", "dtc", "rfc"])
    dataframe.to_csv('/figures/out.csv')



    

    

