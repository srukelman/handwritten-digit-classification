import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

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

def do_nonlinreg(input_data, model_data):
    normalizer = tf.keras.layers.Normalization(input_shape=[1,], axis = None)
    normalizer.adapt(np.array(input_data))
    dnn_model = build_and_compile_model(normalizer)
    trained_model = train_model(dnn_model, model_data, input_data)
    x = tf.linspace(0.0, 1800, 1801)
    y = trained_model.predict(x)
    return x,y
    

def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation = 'relu'),
        layers.Dense(64, activation = 'relu'),
        layers.Dense(1)
    ])
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
    return model

def train_model(dnn_model, model_data, input_data):
    history = dnn_model.fit(
        np.array(input_data),
        np.array(model_data),
        validation_split = 0.2,
        verbose = 0, epochs = 100)
    return dnn_model

if __name__ == "__main__":
    digits, dataset, targets = load_data()
    training_size = 1600
    view_digit(digits, 17)
    print(svc_classify(dataset, targets, training_size))
    print(dtc_classify(dataset, targets, training_size))
    print(rfc_classify(dataset, targets, training_size))
    plt.rcParams["figure.figsize"] = [7.00, 7.00]
    plt.rcParams["figure.autolayout"] = True
    plt.xlim(0,1800)
    plt.ylim(0, 1.0)
    
    svc_data = list()
    dtc_data = list()
    rfc_data = list()
    input_data = list()
    for i in range (10, 1800, 10):
        training_size = i
        svc = svc_classify(dataset, targets, training_size)
        dtc = dtc_classify(dataset, targets, training_size)
        rfc = rfc_classify(dataset, targets, training_size)
        svc_data.append(svc)
        dtc_data.append(dtc)
        rfc_data.append(rfc)
        input_data.append(training_size)
    plt.scatter(input_data,svc_data, label='SVC')
    plt.scatter(input_data,dtc_data, label='DTC')
    plt.scatter(input_data,rfc_data, label='RFC')
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    svc_x, svc_y = do_nonlinreg(input_data, svc_data)
    dtc_x, dtc_y = do_nonlinreg(input_data, dtc_data)
    rfc_x, rfc_y = do_nonlinreg(input_data, rfc_data)
    plt.rcParams["figure.figsize"] = [7.00, 7.00]
    plt.rcParams["figure.autolayout"] = True
    plt.xlim(0,1800)
    plt.ylim(0, 1.0)
    plt.scatter(input_data,svc_data, label='SVC')
    plt.scatter(input_data,dtc_data, label='DTC')
    plt.scatter(input_data,rfc_data, label='RFC')
    plt.plot(svc_x, svc_y, label="SVC")
    plt.plot(dtc_x, dtc_y, label="DTC")
    plt.plot(rfc_x, rfc_y, label="RFC")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    

    

