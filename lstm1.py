# -*- coding: utf-8 -*-
# %%
from __future__ import print_function

import time
import warnings
import numpy as np
import time
import matplotlib.pyplot as plt
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

warnings.filterwarnings("ignore")


def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'rb').read()
    data = f.split(str.encode('\n'))

    # for i in range(len(data)):
    #     data[i] = int(data[i])

    for i in range(len(data)):
        if i > 5:
            data[i] = (int(data[i])+data[i-1]+data[i-2]+data[i-3]+data[i-4])/5
        else:
            data[i] = int(data[i])

    print('data len:', len(data))
    print('sequence len:', seq_len)

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        if sum([int(index > 73 + i * 288 and index < 264 + i * 288) for i in range(47520//288)]) > 0:
            # 得到长度为seq_len+1的向量，最后一个作为label
            result.append(data[index: index + sequence_length])

    print('result len:', len(result))
    print('result shape:', np.array(result).shape)
    print(result[:1])

    if normalise_window:
        result, jilu = normalise_windows(result)

    print(result[:1])
    print('normalise_windows result shape:', np.array(result).shape)

    result = np.array(result)

    # 划分train、test
    row = round(0.9 * result.shape[0])
    # print(row)
    # input()
    train = result[:row, :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[row:, :-1]
    y_test = result[row:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test, jilu]


def normalise_windows(window_data):
    jilu = []
    normalised_data = []
    for window in window_data:  # window shape (sequence_length L ,)  即(51L,)
        jilu.append(window[0])
        normalised_window = [((float(p) / float(window[0])) - 1)
                             for p in window]
        normalised_data.append(normalised_window)
    return normalised_data, jilu


def build_model(layers,layer):  #layers [1,50,100,1]
    model = Sequential()

    model.add(LSTM(input_dim=layers[0],output_dim=layers[1],return_sequences=True))
    model.add(Dropout(0.2))

    for i in range(layer):
        if i == layer - 1:
            model.add(LSTM(layers[2],return_sequences=False))
            model.add(Dropout(0.2))
        else:
            model.add(LSTM(layers[2],return_sequences=True))
            model.add(Dropout(0.2))            

    model.add(Dense(output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model

# 直接全部预测


def predict_point_by_point(model, data):
    predicted = model.predict(data)
    print('predicted shape:', np.array(predicted).shape)  # (412L,1L)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

# 滚动预测


def predict_sequence_full(model, data, window_size):  # data X_test
    curr_frame = data[0]  # (50L,1L)
    predicted = []
    for i in range(len(data)):
        # x = np.array([[[1],[2],[3]], [[4],[5],[6]]])  x.shape (2, 3, 1) x[0,0] = array([1])  x[:,np.newaxis,:,:].shape  (2, 1, 3, 1)
        # np.array(curr_frame[newaxis,:,:]).shape (1L,50L,1L)
        predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        # numpy.insert(arr, obj, values, axis=None)
        curr_frame = np.insert(
            curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted


# window_size = seq_len
def predict_sequences_multiple(model, data, window_size, prediction_len):
    prediction_seqs = []
    for i in range(len(data)//prediction_len):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(
                curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


def plot_results(predicted_data, true_data, filename):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.savefig(filename+'.png')
    # plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.savefig('plot_results_multiple.png')
    # plt.show()

def main(layer):

    global_start_time = time.time()
    epochs = 30
    seq_len = 50

    print('> Loading data... ')

    X_train, y_train, X_test, y_test, jilu = load_data(
        'sp500.csv', seq_len, True)

    print('X_train shape:', X_train.shape)  # (3709L, 50L, 1L)
    print('y_train shape:', y_train.shape)  # (3709L,)
    print('X_test shape:', X_test.shape)  # (412L, 50L, 1L)
    print('y_test shape:', y_test.shape)  # (412L,)

    print('> Data Loaded. Compiling...')

    model = build_model([1, 50, 100, 1],layer)

    model.fit(X_train, y_train, batch_size=512,
              nb_epoch=epochs, validation_split=0.05)

    multiple_predictions = predict_sequences_multiple(
        model, X_test, seq_len, prediction_len=50)
    print('multiple_predictions shape:', np.array(
        multiple_predictions).shape)  # (8L,50L)

    full_predictions = predict_sequence_full(model, X_test, seq_len)
    print('full_predictions shape:', np.array(
        full_predictions).shape)  # (412L,)

    point_by_point_predictions = predict_point_by_point(model, X_test)
    print('point_by_point_predictions shape:', np.array(
        point_by_point_predictions).shape)  # (412L)

    print('Training duration (s) : ', time.time() - global_start_time)

    plot_results_multiple(multiple_predictions, y_test, 50)
    plot_results(full_predictions, y_test, 'full_predictions')
    plot_results(point_by_point_predictions, y_test,
                 'point_by_point_predictions')

    real_pre = []
    real_data = []
    for i in range(len(point_by_point_predictions)):
        real_pre.append((point_by_point_predictions[i]+1) * jilu[28191+i])
        real_data.append((y_test[i]+1) * jilu[28191+i])

    real_data = np.array(real_data)
    real_pre = np.array(real_pre)

    np.savetxt('n1.csv', real_data)
    np.savetxt('n2.csv', real_pre)

    error = 0
    for i in range(len(point_by_point_predictions)):
        error += abs(point_by_point_predictions[i]-y_test[i])/(y_test[i]+1.1)
    error /= len(point_by_point_predictions)
    print('error1', error)
    # %%
    error = 0
    for i in range(len(point_by_point_predictions)):
        error += abs(real_data[i]-real_pre[i])/real_data[i]

    error /= len(point_by_point_predictions)
    print('error2', error)
    # %%
    print('~')
    plt.figure()
    plt.plot(real_data[:200])
    plt.plot(real_pre[:200])
    # plt.show()

    return error


if __name__ == '__main__':
    for i in range(1,20):
        error = main(i)
        f = open('mutilayer.txt','a')
        f.write("{},{},{}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),i,error))
        f.close()
