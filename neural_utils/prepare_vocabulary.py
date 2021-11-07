from sklearn.feature_extraction.text import CountVectorizer


def prepare_vocabulary(data, langs):
    features = set()

    for lang in langs:
        corpus = data[data.lang == lang]['text']
        vectorizer = CountVectorizer(analyzer='char',
                                     max_features=200,
                                     ngram_range=(3, 3))
        _ = vectorizer.fit_transform(corpus)
        f = vectorizer.get_feature_names_out()
        features.update(f)

    vocabulary = dict()
    for i, f in enumerate(features):
        vocabulary[f] = i

    return vocabulary

