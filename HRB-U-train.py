import os
import pandas as pd
from utilies import warn_by_qq, data_load, training


def main():

    item_dict = {'PH': 0, 'DO': 1, 'CODMN': 2, 'BOD': 3, 'AN': 4, 'TP': 5, 'CODCR': 6}
    item_name_list = ['PH', 'DO', 'CODMN', 'BOD', 'AN', 'TP', 'CODCR']


    neuron_data = pd.read_csv("c_tune/c_tune.csv", header=None, names=["output", "time_step", "hidden_layer_1", "hidden_layer_2", "hidden_layer_3"])
    for time_step in neuron_data["time_step"].unique():
        locals()["neurons_" + str(int(time_step))] = {}
        time_data = neuron_data[neuron_data["time_step"] == time_step]

        for i in time_data.index:
            row = time_data.loc[i,:]

            eval("neurons_" + str(int(time_step)))[row["output"]] = []
            for hidden_neuron in ["hidden_layer_1", "hidden_layer_2", "hidden_layer_3"]:
                if row[hidden_neuron]:
                    eval("neurons_" + str(int(time_step)))[row["output"]].append(int(row[hidden_neuron]))
        # print(time_step)
        # print(eval("neurons_" + str(int(time_step))))   


    data_path = "ziya.csv"
    main_path = "repeat_train"
    save_name = "c_repeat_train.csv"

    if not os.path.exists(main_path):
        os.mkdir(main_path)


    for time_step in range(1, 13):

        neurons_name = "neurons_" + str(time_step)
        item_neurons = eval(neurons_name)

        train_data, valid_data, test_data = data_load(data_path)

        month_path = os.path.join(main_path, str(time_step))
        if not os.path.exists(month_path):
            os.mkdir(month_path)

        for item, item_index in item_dict.items():
            neurons = item_neurons[item]
            item_name = item_name_list[item_index]

            item_path = os.path.join(month_path, item)
            if not os.path.exists(item_path):
                os.mkdir(item_path)
            training(train_data, valid_data, test_data, item_name, item_index, time_step, neurons, main_path, item_path, save_name)
    message = "模型重复训练"
    warn_by_qq(message)


if __name__ == '__main__':
    main()
