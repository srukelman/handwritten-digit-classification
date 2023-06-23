import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
        layers.Dense(64, activation = 'tanh'),
        layers.Dense(64, activation = 'relu'),
        layers.Dense(64, activation = 'relu'),
        # layers.Dense(64, activation = 'relu'),
        # layers.Dense(64, activation = 'relu'),
        #layers.LeakyReLU(),
        layers.Dense(1)
    ])
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
    return model

def train_model(dnn_model, model_data, input_data):
    history = dnn_model.fit(
        np.array(input_data),
        np.array(model_data),
        validation_split = 0.2,
        shuffle = True,

        verbose = 0, epochs = 100)
    return dnn_model


if __name__ == "__main__":
    dataframe = pd.read_csv('out.csv')
    input_data = dataframe['input_data']
    svc_data = dataframe['svc']
    dtc_data = dataframe['dtc']
    rfc_data = dataframe['rfc']
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