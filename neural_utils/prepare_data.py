import pandas as pd


def prepare_data(langs: list):
    data = pd.read_csv('C:/Users/Clown/PycharmProjects/language_recognition_model/data/sentences.csv',
                       sep='\t',
                       encoding='utf8',
                       index_col=0,
                       names=['lang', 'text'])
    data = data[data['lang'].isin(langs)]
    data = data.sample(frac=1)  # shuffle

    data_len = len(data)
    trb1 = 0
    trb2 = int(data_len * 0.8)
    tsb1 = trb2 + 1
    tsb2 = data_len - 1

    data_train = data[trb1:trb2]
    data_test = data[tsb1:tsb2]

    return data_train, data_test
