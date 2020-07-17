import numpy as np 
import pandas as pd 
import datetime
import re
import os, os.path
import time
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf

def calculate_model_size(model):
    print(model.summary())
    var_sizes = [
      np.product(list(map(int, v.shape))) * v.dtype.size
      for v in model.trainable_variables
      ]
    print("Model size:", sum(var_sizes) / 1024, "KB")

def makeModels(modelType,samples):
    if modelType =='lstm':
        model = tf.keras.Sequential([
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(22),
                input_shape=(samples, 3)),  # output_shape=(batch, 44)
            tf.keras.layers.Dense(4, activation="sigmoid")  # (batch, 4)
        ])


    elif modelType == 'cnn':
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, (4, 3),padding="same",activation="relu",
                                input_shape=(samples, 3, 1)),  # output_shape=(batch, 128, 3, 8)
            tf.keras.layers.MaxPool2D((3, 3)),  # (batch, 42, 1, 8)
            tf.keras.layers.Dropout(0.1),  # (batch, 42, 1, 8)
            tf.keras.layers.Conv2D(16, (4, 1), padding="same",activation="relu"),  # (batch, 42, 1, 16)
            tf.keras.layers.MaxPool2D((3, 1), padding="same"),  # (batch, 14, 1, 16)
            tf.keras.layers.Dropout(0.1),  # (batch, 14, 1, 16)
            tf.keras.layers.Flatten(),  # (batch, 224)
            tf.keras.layers.Dense(16, activation="relu"),  # (batch, 16)
            tf.keras.layers.Dropout(0.1),  # (batch, 16)
            tf.keras.layers.Dense(4, activation="softmax")  # (batch, 4)
        ])
    
    return model


def train_lstm(model, epochs_lstm, batch_size, tensor_train_set, tensor_val_set, tensor_test_set, val_set,test_set):
    tensor_train_set_lstm = tensor_train_set.batch(batch_size).repeat()
    tensor_val_set_lstm = tensor_val_set.batch(batch_size)
    tensor_test_set_lstm = tensor_test_set.batch(batch_size)

    model.fit(
        tensor_train_set_lstm,
        epochs=epochs_lstm,
        validation_data=tensor_val_set_lstm,
        steps_per_epoch=200,
        validation_steps=int((len(val_set) - 1) / batch_size + 1))


    loss_lstm, acc_lstm = model.evaluate(tensor_test_set_lstm)
    pred_lstm = np.argmax(model.predict(tensor_test_set_lstm), axis=1)
    confusion_lstm = tf.math.confusion_matrix(
        labels=tf.constant(test_set['gesture'].to_numpy()),
        predictions=tf.constant(pred_lstm),
        num_classes=4)

    #print(confusion_lstm)
    #print("Loss {}, Accuracy {}".format(loss_lstm, acc_lstm))
    model.save('lstm_model.h5') 
    return model, confusion_lstm, loss_lstm, acc_lstm

converter = tf.lite.TFLiteConverter.from_keras_model(lstm_model)
lstm_tflite_model = converter.convert()

open("lstm_model.tflite", "wb").write(lstm_tflite_model)

converter = tf.lite.TFLiteConverter.from_keras_model(lstm_model)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.target_spec.supported_types = [tf.float16]
lstm_opt_tflite_model = converter.convert()
# Save the model to disk
open("lstm_model_quantized.tflite", "wb").write(lstm_opt_tflite_model)

basic_model_size = os.path.getsize("lstm_model.tflite")
print("Basic model is %d bytes" % basic_model_size)
quantized_model_size = os.path.getsize("lstm_model_quantized.tflite")
print("Quantized model is %d bytes" % quantized_model_size)
difference = basic_model_size - quantized_model_size
print("Difference is %d bytes" % difference)

"""---------------------------------------------------------------------------"""

def reshape_function(data, label):
    reshaped_data = tf.reshape(data, [-1, 3, 1])
    return reshaped_data, label

def train_CNN(model, epochs_cnn, batch_size, tensor_train_set, tensor_test_set, tensor_val_set, val_set, test_set):
    tensor_train_set_cnn = tensor_train_set.map(reshape_function)
    tensor_test_set_cnn = tensor_test_set.map(reshape_function)
    tensor_val_set_cnn = tensor_val_set.map(reshape_function)

    tensor_train_set_cnn = tensor_train_set_cnn.batch(batch_size).repeat()
    tensor_test_set_cnn = tensor_test_set_cnn.batch(batch_size)
    tensor_val_set_cnn = tensor_val_set_cnn.batch(batch_size)

    model.fit(
        tensor_train_set_cnn,
        epochs=epochs_cnn,
        validation_data=tensor_val_set_cnn,
        steps_per_epoch=300,
        validation_steps=int((len(val_set) - 1) / batch_size + 1))


    loss_cnn, acc_cnn = cnn_model.evaluate(tensor_test_set_cnn)
    pred_cnn = np.argmax(cnn_model.predict(tensor_test_set_cnn), axis=1)
    confusion_cnn = tf.math.confusion_matrix(
        labels=tf.constant(test_set['gesture'].to_numpy()),
        predictions=tf.constant(pred_cnn),
        num_classes=4)


    #print(confusion_cnn)
    #print("Loss {}, Accuracy {}".format(loss_cnn, acc_cnn))
    model.save('cnn_model.h5') 
    return model, confusion_cnn, loss_cnn, acc_cnn


converter = tf.lite.TFLiteConverter.from_keras_model(cnn_model)
cnn_tflite_model = converter.convert()

open("cnn_model.tflite", "wb").write(cnn_tflite_model)


converter = tf.lite.TFLiteConverter.from_keras_model(cnn_model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
cnn_opt_tflite_model = converter.convert()
# Save the model to disk
open("cnn_model_quantized.tflite", "wb").write(cnn_opt_tflite_model)

basic_model_size = os.path.getsize("cnn_model.tflite")
print("Basic model is %d bytes" % basic_model_size)
quantized_model_size = os.path.getsize("cnn_model_quantized.tflite")
print("Quantized model is %d bytes" % quantized_model_size)
difference = basic_model_size - quantized_model_size
print("Difference is %d bytes" % difference)


## Install xxd if it is not available
#!apt-get -qq install xxd
## Save the file as a C source file
#!xxd -i cnn_model_quantized.tflite > cnn_opt_model.cc
## Print the source file
#!cat /cnn_opt_model.cc


if __name__=='__main__':
    #load in data sets
    trainingData = pd.read_csv('processed_train_set.csv',converters={'acceleration': eval})
    testingData = pd.read_csv('processed_test_set.csv',converters={'acceleration': eval})
    validationData = pd.read_csv('processed_val_set.csv',converters={'acceleration': eval})

    #Lets make the models
    sample_len = len(trainingData['acceleration'][0])
    lstm_model = makeModels('lstm',sample_len)
    cnn_model = makeModels('cnn',sample_len)

    #convert the datasets into tensors
    tensor_train_set = tf.data.Dataset.from_tensor_slices(
        (np.array(trainingData['acceleration'].tolist(),dtype=np.float64),
        trainingData['gesture'].tolist()))
    tensor_test_set = tf.data.Dataset.from_tensor_slices(
        (np.array(testingData['acceleration'].tolist(),dtype=np.float64),
        testingData['gesture'].tolist()))
    tensor_val_set = tf.data.Dataset.from_tensor_slices(
        (np.array(validationData['acceleration'].tolist(),dtype=np.float64),
        validationData['gesture'].tolist()))


    epochs = 20
    batch_size = 192

    calculate_model_size(lstm_model)
    calculate_model_size(cnn_model)

    lstm_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])

    cnn_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])

    lstm_model, confusion_lstm, loss_lstm, acc_lstm = train_lstm(lstm_model, epochs, batch_size, tensor_train_set, tensor_val_set, tensor_test_set, validationData, testingData)
    cnn_model, confusion_cnn, loss_cnn, acc_cnn = train_CNN(cnn_model, epochs, batch_size, tensor_train_set, tensor_test_set, tensor_val_set, validationData, testingData)

    print(confusion_lstm)
    print("Loss {}, Accuracy {}".format(loss_lstm, acc_lstm))
    print(confusion_cnn)
    print("Loss {}, Accuracy {}".format(loss_cnn, acc_cnn))

    



