import os
import pandas as pd
from utilies import warn_by_qq, data_load, hyper_tuner


def main():

    item_dict = {'PH': 0, 'DO': 1, 'CODMN': 2, 'BOD': 3, 'AN': 4, 'TP': 5, 'CODCR': 6}
    item_name_list = ['PH', 'DO', 'CODMN', 'BOD', 'AN', 'TP', 'CODCR']

    data_path = "ziya.csv"
    log_path = "alog"
    save_name = "a_tune.csv"
    main_path = "a_tune"
    # if not os.path.exists(main_path):
    #     os.mkdir(main_path)
    train_data, valid_data, test_data = data_load(data_path)

    for time_steps in range(1, 13):

        for item, item_index in item_dict.items():
            item_name = item_name_list[item_index]
            results = pd.DataFrame()

            model_information = hyper_tuner(log_path, time_steps, train_data, valid_data, test_data, item, item_index, item_name)

            results = results.append(model_information, ignore_index=True)

            results = results[['output', "time_step",'hidden_layer_1', 'hidden_layer_2', 'hidden_layer_3']]

            information_path = os.path.join(main_path, save_name)

            if not os.path.exists(information_path):
                results.to_csv(information_path, encoding='utf-8', index=False)
            else:
                results.to_csv(information_path, mode='a', header=False, index=False)
                
    message = "模型调参"
    warn_by_qq(message)


if __name__ == '__main__':
    main()
