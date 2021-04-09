import IPython
import math
import os
import win32gui
import win32con
import win32clipboard as w
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from kerastuner import HyperModel, RandomSearch



def warn_by_qq(message):
	import win32gui
	import win32con
	import win32clipboard as w
	#窗口名字
	name = "Bennett"
	#获取窗口句柄
	handle = win32gui.FindWindow(None, name)
	#发送的消息
	msg = message + "完毕!"
	#将测试消息复制到剪切板中
	w.OpenClipboard()
	w.EmptyClipboard()
	w.SetClipboardData(win32con.CF_UNICODETEXT, msg)
	w.CloseClipboard()
	#填充消息
	win32gui.SendMessage(handle, 770, 0, 0)
	#回车发送消息
	win32gui.SendMessage(handle, win32con.WM_KEYDOWN, win32con.VK_RETURN, 0)


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


# a
def split_x_y(data, item_index, time_steps):
    start_index = - (time_steps + 1)

    inputs = data[:, start_index:-1, :]
    outputs = data[:, -1, item_index]

    return inputs, outputs
# c
# def split_x_y(data, item_index, time_steps):
#     start_index = - (time_steps + 1)

#     inputs = data[:, start_index:-1, [item_index]]
#     outputs = data[:, -1, item_index]

#     return inputs, outputs

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
    return train_x, valid_x, test_x, train_y, valid_y, test_y


def data_split_norm_2(train_data, valid_data, test_data, item_index, time_steps):
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


class MyHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        physical_devices = tf.config.experimental.list_physical_devices('GPU')

        try:
            tf.config.experimental.set_virtual_device_configuration(
                physical_devices[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=700)])
            print("Successfully!")
        except:
            print('Cannot modify the virtual devices once they have been initialized.')

    def build(self, hp):
        model = keras.Sequential()
        model.add(keras.Input(shape=self.input_shape))
        for i in range(hp.Int('num_layers', min_value=0, max_value=2)):

            model.add(keras.layers.LSTM(hp.Int('units_' + str(i), min_value=4, max_value=20, step=4),
                                        activation='relu', return_sequences=True))
            model.add(keras.layers.Dropout(0.5))

        model.add(keras.layers.LSTM(hp.Int('units_n', min_value=4, max_value=20, step=4),
                                    activation='relu'))

        model.add(keras.layers.Dense(1, activation='relu'))

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                      loss=fvu_error,
                      metrics=[tf.keras.metrics.RootMeanSquaredError(),
                               tf.keras.metrics.MeanAbsolutePercentageError(),
                               ]
                      )
        return model


def hyper_tuner(log_path, time_step, train_data, valid_data, test_data, item, item_index, item_name):
    keras.backend.clear_session()

    train_x, valid_x, test_x, train_y, valid_y, test_y = data_split_norm(train_data, valid_data, test_data, item_index, time_step)
    
    hyper_model = MyHyperModel(input_shape=train_x.shape[-2:])

    project_name = item_name + "_" + str(time_step)

    tuner = RandomSearch(
        hyper_model,
        objective='val_loss',
        max_trials=150,
        executions_per_trial=10,
        directory=log_path,
        project_name=project_name)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.0001),
        ClearTrainingOutput()
    ]

    tuner.search_space_summary()

    tuner.search(train_x, train_y, epochs=200, batch_size=256,
                 validation_data=(valid_x, valid_y),
                 callbacks=callbacks,
                 verbose=2)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    try:
        hidden_layer_1 = best_hps.get('units_0')
    except:
        hidden_layer_1 = 0
    try:
        hidden_layer_2 = best_hps.get('units_1')
    except:
        hidden_layer_2 = 0

    model_info = {'output': item,
                  "time_step": time_step,
                  'hidden_layer_1': hidden_layer_1,
                  'hidden_layer_2': hidden_layer_2,
                  'hidden_layer_3': best_hps.get('units_n'),
                  }

    # 清除tuner
    del tuner

    return model_info


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
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_virtual_device_configuration(
            physical_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=700),])
        print("Successfully!")
    except:
        print('Cannot modify the virtual devices once they have been initialized.')
    	
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
    train_mean, train_std, train_x, valid_x, test_x, train_y, valid_y, test_y = data_split_norm_2(train_data, valid_data, test_data, item_index, time_step)

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
        results_path = os.path.join(main_path, save_name)

        if not os.path.exists(results_path):
            result.to_csv(results_path, encoding='utf-8')
        else:
            result.to_csv(results_path, mode='a', header=False)
