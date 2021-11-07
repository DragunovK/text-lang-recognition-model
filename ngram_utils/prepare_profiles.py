from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def prepare_profiles(langs: list):
    data = pd.read_csv('data/sentences.csv',
                       sep='\t',
                       encoding='utf8',
                       index_col=0,
                       names=['lang', 'text'])

    result = dict()

    for lang in langs:
        corpus = data[data.lang == lang]['text']
        vectorizer = CountVectorizer(analyzer='char',
                                     ngram_range=(5, 5),
                                     max_features=300)
        vectorizer.fit_transform(corpus)
        ngrams = vectorizer.get_feature_names_out()
        lang_profile = dict()
        for i, ng in enumerate(ngrams):
            lang_profile[ng] = i
        result[lang] = lang_profile

    return result
