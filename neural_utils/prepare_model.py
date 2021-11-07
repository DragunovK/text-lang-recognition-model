from sklearn.preprocessing import LabelEncoder

from neural_utils.build_model import build_model
from neural_utils.prepare_vocabulary import prepare_vocabulary
from neural_utils.prepare_data import prepare_data
from neural_utils.prepare_xy import prepare_xy


def prepare_model(langs: list):
    data_train, data_test = prepare_data(langs)
    vocabulary = prepare_vocabulary(data_train, langs)
    encoder = LabelEncoder()
    encoder.fit(langs)
    x_train, y_train = prepare_xy(vocabulary, data_train, encoder)
    x_test, y_test = prepare_xy(vocabulary, data_test, encoder)
    model, test_info = build_model(x_train, y_train, x_test, y_test, encoder)
    return vocabulary, model, test_info
