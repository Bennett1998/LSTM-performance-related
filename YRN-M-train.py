import IPython
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from kerastuner import HyperModel, RandomSearch


def create_interval_dataset(dataset, look_back=12):
    """
    :param dataset: input array of time intervals
    :param look_back: each training set time step length
    :return: convert an array of values into a dataset matrix.
    """
    data_list = []
    for i in range(len(dataset) - look_back):
        data_list.append(dataset.iloc[i:i + look_back + 1, :])
    dataset = np.array([df.values for df in data_list])
    return dataset


def split_x_y(data, item_index, time_steps):
    start_index = - (time_steps + 1)

    inputs = data[:, start_index:-1, :]
    outputs = data[:, -1, item_index]

    return inputs, outputs


def data_load(data_path):
    raw_data = pd.read_csv(data_path)
    train_list = []
    valid_list = []
    test_list = []
    for stcd in raw_data['STCD'].unique():
        stcd_data = raw_data[raw_data['STCD'] == stcd].iloc[:, 2:]

        x_y_data = create_interval_dataset(stcd_data)
        np.random.seed(10)
        np.random.shuffle(x_y_data)
        n = len(x_y_data)
        train_list.append(x_y_data[0:int(n * 0.8)])
        valid_list.append(x_y_data[int(n * 0.8):int(n * 0.9)])
        test_list.append(x_y_data[int(n * 0.9):])

    train_data = np.vstack(train_list)
    valid_data = np.vstack(valid_list)
    test_data = np.vstack(test_list)
    return train_data, valid_data, test_data


def data_split_norm(train_data, valid_data, test_data, item_index, time_steps):
    # 分离features和labels
    train_x, train_y = split_x_y(train_data, item_index, time_steps)
    valid_x, valid_y = split_x_y(valid_data, item_index, time_steps)
    test_x, test_y = split_x_y(test_data, item_index, time_steps)

    train_mean = np.mean(train_x)
    train_std = np.std(train_x)

    train_x = ((train_x - train_mean) / train_std)
    valid_x = ((valid_x - train_mean) / train_std)
    test_x = ((test_x - train_mean) / train_std)
    return train_mean, train_std, train_x, valid_x, test_x, train_y, valid_y, test_y


def calc_nse_cc(obs: np.array, sim: np.array) -> float:

    """Calculate Nash-Sutcliff-Efficiency.
    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: NSE CC value.
    """
    # only consider time steps, where observations are available
    sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # check for NaNs in observations
    sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    denominator = np.sum((obs - np.mean(obs)) ** 2)

    numerator = np.sum((sim - obs) ** 2)
    nse_val = 1 - numerator / denominator

    son = np.sum((obs - np.mean(obs)) * (sim - np.mean(sim)))
    SSR = np.sum((sim - np.mean(sim)) ** 2)
    mother = math.sqrt(denominator * SSR)

    cc_val = son / mother

    return nse_val, cc_val


class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


def fvu_error(y_true, y_pred):
    y = tf.math.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    x = tf.math.reduce_sum(tf.square(y_pred - y_true))

    return tf.divide(x, y)


def my_plot(y1, y2, y1_label, y2_label, x_label, y_label, plot_path, figure_size=None):
    if figure_size:
        plt.figure(figsize=figure_size)
    plt.plot(y1, linewidth=2, label=y1_label, marker='s')
    plt.plot(y2, linewidth=2, label=y2_label, marker='o')
    plt.legend(loc='upper right')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(plot_path, dpi=300)
    plt.close()


def build_model(input_shape, neurons):
    inputs = keras.Input(shape=input_shape)

    if len(neurons) >=2 :
        x = keras.layers.LSTM(neurons[0], return_sequences=True, activation='relu')(inputs)
        x = keras.layers.Dropout(0.5)(x)
        if len(neurons) == 3:
            x = keras.layers.LSTM(neurons[1], return_sequences=True, activation='relu')(x)
            x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.LSTM(neurons[-1], activation='relu')(x)
    else:
        x = keras.layers.LSTM(neurons[0], activation='relu')(inputs)

    outputs = keras.layers.Dense(1, activation='relu')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  loss=fvu_error,
                  metrics=[tf.keras.metrics.RootMeanSquaredError(),
                           tf.keras.metrics.MeanAbsolutePercentageError(),
                           ]
                  )
    return model

def training(train_data, valid_data, test_data, item, item_index, time_step, neurons, main_path, item_path, save_name):
    train_mean, train_std, train_x, valid_x, test_x, train_y, valid_y, test_y = data_split_norm(train_data, valid_data, test_data, item_index, time_step)

    for repeat in range(1, 101):
        repeat_path = os.path.join(item_path, str(repeat))
        if not os.path.exists(repeat_path):
            os.mkdir(repeat_path)

        model = build_model(input_shape=train_x.shape[-2:], neurons=neurons)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=15,
                                             restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001),
        ]
        history = model.fit(train_x, train_y, epochs=200, batch_size=256,
                            validation_data=(valid_x, valid_y),
                            callbacks=callbacks,
                            verbose=2)

        models_path = os.path.join(repeat_path, "models")
        if not os.path.exists(models_path):
            os.mkdir(models_path)
        model_name = item + "_" + str(repeat) + "_model.h5"
        model_path = os.path.join(models_path, model_name)
        model.save(model_path)

        plot_path_1 = os.path.join(repeat_path, "loss_epoch.png")
        plot_path_2 = os.path.join(repeat_path, "measure_predict.png")
        csv_path = os.path.join(repeat_path, "measure_predict.csv")

        history = history.history

        ylabel_1 = 'loss[$' + item + '$]'
        my_plot(history['loss'], history['val_loss'], 'Train', 'Valid', 'Epoch',
                ylabel_1, plot_path_1)

        test_predictions = model.predict(test_x)
        if item is 'PH':
            ylabel_2 = item
        elif item is "WT":
            ylabel_2 = item + '(℃)'
        else:
            ylabel_2 = item + '(mg/L)'
        my_plot(test_predictions, test_y, 'predict', 'measure', 'Test set',
                ylabel_2, plot_path_2, figure_size=(20, 6))

        valid_predictions = model.predict(valid_x).flatten()
        test_predictions = test_predictions.flatten()

        train_fvu, train_rmse, train_mre = history['loss'][-1], history['root_mean_squared_error'][-1], history["mean_absolute_percentage_error"][-1]

        valid_fvu, valid_rmse, valid_mre = model.evaluate(valid_x, valid_y, verbose=0)
        valid_nse, valid_cc = calc_nse_cc(valid_y, valid_predictions)
        test_fvu, test_rmse, test_mre = model.evaluate(test_x, test_y, verbose=0)
        test_nse, test_cc = calc_nse_cc(test_y, test_predictions)

        measure_predict_data = pd.DataFrame()
        measure_predict_data['measure'] = test_y
        measure_predict_data['predict'] = test_predictions
        measure_predict_data.to_csv(csv_path, encoding='utf-8')
        hidden_neurons = '_'.join([str(x) for x in neurons])

        keras.backend.clear_session()

        model_info = {'output': item,
                      "time_steps": str(time_step),
                      'hidden_layer_neurons': hidden_neurons,
                      'repeat': str(repeat),
                      'train_mean': train_mean,
                      "train_std": train_std,
                      'train_fvu': train_fvu,
                      'train_rmse': train_rmse,
                      'train_mre': train_mre,
                      "valid_fvu": valid_fvu,
                      'valid_nse': valid_nse,
                      'valid_rmse': valid_rmse,
                      'valid_mre': valid_mre,
                      'valid_cc': valid_cc,
                      "test_fvu": test_fvu,
                      'test_nse': test_nse,
                      'test_rmse': test_rmse,
                      'test_mre': test_mre,
                      'test_cc': test_cc
                      }
        result = pd.DataFrame()
        result = result.append(model_info, ignore_index=True)
        # model_info = pd.DataFrame(model_info, columns=['output', "time_steps", 'hidden_layer_neurons', "repeat",
        #                                              "train_mean", "train_stcd", 'train_fvu', 'train_rmse', 'train_mre',
        #                                              "valid_fvu", 'valid_nse', 'valid_rmse', 'valid_mre', 'valid_cc',
        #                                              "test_fvu", 'test_nse', 'test_rmse', 'test_mre', 'test_cc'])

        results_path = os.path.join(main_path, save_name)

        # lock.acquire()

        if not os.path.exists(results_path):
            result.to_csv(results_path, encoding='utf-8')
        else:
            result.to_csv(results_path, mode='a', header=False)


if __name__ == '__main__':

    item_dict = {"WT": 0, "PH": 1, "DOX": 2, "CODMN": 3, "BOD5": 4, "NH3N": 5, "TP": 6}
    item_name_list = ["WT", "PH", "DO", "COD", "BOD", "AN", "TP"]

    neurons_1 = {"WT": [8, 8, 16], "PH": [12, 16, 8], "DOX": [12, 8, 20],
                 "CODMN": [16, 16, 16], "BOD5": [16, 12, 16], "NH3N": [8, 16, 12],
                 "TP": [16, 12, 12]}
    neurons_2 = {"WT": [8, 12, 20], "PH": [4, 20, 12], "DOX": [8, 20, 12],
                 "CODMN": [8, 4, 8], "BOD5": [4, 8, 12], "NH3N": [8, 12, 12],
                 "TP": [20, 8, 12]}
    neurons_3 = {'WT': [12, 8, 16], 'PH': [20, 8, 16], 'DOX': [12, 16, 12],
                 'CODMN': [16, 8, 12], 'BOD5': [16, 12, 20], 'NH3N': [4, 12, 8],
                 'TP': [20, 16, 4]}
    neurons_4 = {'WT': [4, 12, 12], 'PH': [8, 20, 16], 'DOX': [16, 0, 20],
                 'CODMN': [16, 8, 8], 'BOD5': [6, 16, 16], 'NH3N': [12, 16, 16],
                 'TP': [20, 4, 20]}
    neurons_5 = {'WT': [16, 4, 12], 'PH': [8, 16, 16], 'DOX': [8, 16, 8],
                 'CODMN': [12, 4, 16], 'BOD5': [16, 12, 16], 'NH3N': [16, 4, 8],
                 'TP': [16, 16, 8]}
    neurons_6 = {'WT': [16, 12, 20], 'PH': [8, 16, 20], 'DOX': [16, 20, 16],
                 'CODMN': [20, 16, 12], 'BOD5': [16, 20, 12], 'NH3N': [16, 20, 16],
                 'TP': [16, 12, 16]}
    neurons_7 = {'WT': [4, 20, 8], 'PH': [20, 8, 12], 'DOX': [20, 8, 12],
                 'CODMN': [8, 20, 16], 'BOD5': [12, 8, 20], 'NH3N': [20, 4, 8],
                 'TP': [20, 16, 4]}
    neurons_8 = {'WT': [20, 16, 12], 'PH': [12, 16, 16], 'DOX': [20, 8, 20],
                 'CODMN': [8, 12, 8], 'BOD5': [12, 20, 12], 'NH3N': [20, 12, 12],
                 'TP': [16, 8, 20]}
    neurons_9 = {"WT": [12, 16, 16], "PH": [12, 8, 12], "DOX": [20, 16, 16],
                 "CODMN": [20, 16, 16], "BOD5": [4, 16, 8], "NH3N": [20, 8, 8],
                 "TP": [20, 16, 12]}
    neurons_10 = {'WT': [16], 'PH': [12, 12, 16], 'DOX': [16, 4, 8],
                  'CODMN': [20, 16, 16], 'BOD5': [4, 20, 16], 'NH3N': [16, 12, 4],
                  'TP': [20, 8, 8]}

    neurons_11 = {'WT': [8, 12, 12], 'PH': [20, 20, 16], 'DOX': [8, 16, 20],
                  'CODMN': [20, 20, 8], 'BOD5': [16, 4, 20], 'NH3N': [4, 12, 16],
                  'TP': [16, 20, 20]}
    neurons_12 = {"WT": [8, 12, 8], "PH": [20, 16, 16], "DOX": [4, 8, 20],
                  "CODMN": [16, 16, 8], "BOD5": [8, 12, 16], "NH3N": [4, 4, 20],
                  "TP": [12, 8, 16]}

    data_path = "../data/data_2.csv"
    main_path = "repeat_train"

    save_name = "a_repeat_train.csv"

    if not os.path.exists(main_path):
        os.mkdir(main_path)


    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_virtual_device_configuration(
            physical_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=700),])
        print("Successfully!")
    except:
        print('Cannot modify the virtual devices once they have been initialized.')


    for time_step in range(1, 13):

        neurons_name = "neurons_" + str(time_step)
        item_neurons = eval(neurons_name)

        train_data, valid_data, test_data = data_load(data_path)

        month_path = os.path.join(main_path, str(time_step))
        if not os.path.exists(month_path):
            os.mkdir(month_path)

        # po_2 = Pool(2)
        # lock = Lock()

        for item, item_index in item_dict.items():
            neurons = item_neurons[item]
            item_name = item_name_list[item_index]

            item_path = os.path.join(month_path, item)
            if not os.path.exists(item_path):
                os.mkdir(item_path)
            training(train_data, valid_data, test_data, item_name, item_index, time_step, neurons, main_path, item_path, save_name)


            # po_2.apply_async(training, (train_data, valid_data, test_data, item_name, item_index, time_step, neurons, main_path, item_path, save_name))


        # po_2.close()
        # po_2.join()
