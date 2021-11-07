import pandas as pd
from keras.utils import np_utils
from sklearn.feature_extraction.text import CountVectorizer


def prepare_xy(vocabulary, data, encoder):
    vectorizer = CountVectorizer(analyzer='char',
                                 ngram_range=(3, 3),
                                 vocabulary=vocabulary)

    corpus = data['text']
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    features = pd.DataFrame(data=X.toarray(), columns=feature_names)
    f_min = features.min()
    f_max = features.max()
    features = (features - f_min) / (f_max - f_min)
    features['lang'] = list(data['lang'])

    y_ = encoder.transform(features['lang'])
    y = np_utils.to_categorical(y_)
    x = features.drop('lang', axis=1)

    return x, y
