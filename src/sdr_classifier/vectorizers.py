'''
    Developed by
    Nobal Niraula, Boeing Research & Technology
    Daniel Whyatt, Boeing Research & Technology
    Hai Nguyen, Boeing Enterprise Safety Data Analytics
'''

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack


class Vectorizers:

    @staticmethod
    def fit_vectorizers(train_text, options=["words", "characters"], stopwords=set()):
        exp_vectorizers = {}
        if "words" in options:
            word_count_vect = CountVectorizer(analyzer='word', ngram_range=(1, 2), stop_words=list(stopwords))
            train_counts = word_count_vect.fit_transform(train_text)
            word_tfidf_transformer = TfidfTransformer()
            word_tfidf_transformer.fit_transform(train_counts)
            exp_vectorizers["words"] = (word_count_vect, word_tfidf_transformer)

        if "characters" in options:
            char_count_vect = CountVectorizer(analyzer='char', ngram_range=(3, 3))
            train_counts = char_count_vect.fit_transform(train_text)
            char_tfidf_transformer = TfidfTransformer()
            char_tfidf_transformer.fit(train_counts)
            exp_vectorizers["characters"] = (char_count_vect, char_tfidf_transformer)
        return exp_vectorizers

    @staticmethod
    def transform_with_vectorizers(X, vectorizers):
        X_wordfeatures, X_charfeatures, X_semanticfeatures, X_universalfeatures = None, None, None, None
        if "words" in vectorizers:
            word_counter, word_tfidf = vectorizers["words"]
            X_counts = word_counter.transform(X)
            X_wordfeatures = word_tfidf.transform(X_counts)

        if "characters" in vectorizers:
            char_counter, char_tfidf = vectorizers["characters"]
            X_counts = char_counter.transform(X)
            X_charfeatures = char_tfidf.transform(X_counts)

        X_transformed = hstack([x for x in [X_wordfeatures, X_charfeatures, X_semanticfeatures, X_universalfeatures] if x is not None])

        return X_transformed