import nltk
import pandas as pd

from empath import Empath

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

# from collections import defaultdict


nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


class NlpPipeline:
    def __init__(self):
        self.stemmer = SnowballStemmer("english")
        self.stop_words = set(stopwords.words("english"))
        self.lexicon = Empath()

    def preprocess(self, db: list[str]):
        db = [self.tokenize_and_stem(d) for d in db]

        return db

    def tokenize_and_stem(self, text: str):
        """Tokenize, stem, and remove stop words in a document using NLTK."""

        words = word_tokenize(text, "english")
        words = [self.stemmer.stem(word) for word in words]
        words = [
            word.lower()
            for word in words
            if word.isalpha() and word.lower() not in self.stop_words
        ]

        return words

    def stats(self, db: list[list[str]], vocabulary: list[str] = None):
        """Compute word counts and TF-IDF for a list of tokenized documents."""
        corpus = [" ".join(d) for d in db]
        pipe = Pipeline(
            [
                ("count", CountVectorizer(vocabulary=vocabulary)),
                ("tfidf", TfidfTransformer()),
            ]
        ).fit(corpus)

        pipe.transform(corpus)
        features = pipe.get_feature_names_out()

        X1 = pipe["count"].transform(corpus).toarray()
        X2 = pipe["tfidf"].transform(X1).toarray()

        word_counts = pd.DataFrame(X1, columns=features)
        word_counts_sum = word_counts.sum(axis=0).sort_values(ascending=False)

        tf_idf = pd.DataFrame(X2, columns=features)
        tf_idf_sum = tf_idf.sum(axis=0).sort_values(ascending=False)

        return word_counts_sum, tf_idf_sum

    def empath(self, db: list[list[str]]):
        """Compute Empath categories for a list of tokenized documents."""
        text = " ".join([w for doc in db for w in doc])
        cat = self.lexicon.analyze(text, normalize=True)
        # cat = {k: v for k, v in cat.items() if v > 0.0}

        return cat
