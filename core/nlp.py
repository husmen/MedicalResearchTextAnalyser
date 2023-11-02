import nltk
import pandas as pd
import gensim

import spacy
# import medspacy

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.similarities import MatrixSimilarity

from empath import Empath

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


class NlpPipeline:
    def __init__(self):
        self.stemmer = SnowballStemmer("english")
        self.stop_words = set(stopwords.words("english"))
        self.lexicon = Empath()
        # self.medspacy_model = medspacy.load()
        self.gensim_model = Doc2Vec(vector_size=30, min_count=2, epochs=80)

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

    def stats(self, dataset: list[list[str]], vocabulary: list[str] = None):
        """Compute word counts and TF-IDF for a list of tokenized documents."""
        corpus = []
        for db in dataset:
            corpus.append(" ".join([" ".join(d) for d in db]))
        pipe = Pipeline(
            [
                ("count", CountVectorizer(vocabulary=vocabulary)),
                ("tfidf", TfidfTransformer()),
            ]
        ).fit(corpus)

        pipe.transform(corpus)
        features = pipe.get_feature_names_out()

        # x1 = pipe["count"].transform(corpus).toarray()

        X1 = pipe["count"].transform(corpus).toarray()
        X2 = pipe["tfidf"].transform(X1).toarray()

        word_counts = pd.DataFrame(X1, columns=features)
        # word_counts_sum = word_counts.sum(axis=0).sort_values(ascending=False)

        tf_idf = pd.DataFrame(X2, columns=features)
        # tf_idf_sum = tf_idf.sum(axis=0).sort_values(ascending=False)

        cos_sim = cosine_similarity(X2, X2)

        return word_counts, tf_idf

    def empath(self, db: list[list[str]]):
        """Compute Empath categories for a list of tokenized documents."""
        text = " ".join([w for doc in db for w in doc])
        cat = self.lexicon.analyze(text, normalize=True)
        # cat = {k: v for k, v in cat.items() if v > 0.0}

        return cat

    # def medspacy(self, db: list[str]):
    #     """Compute MedSpaCy entities for a list of documents."""
    #     text = " ".join([w for doc in db for w in doc])
    #     doc = self.medspacy_model(text)
    #     ents = [ent.text for ent in doc.ents]

    #     return ents

    def gensim(self, dataset: list[list[str]], vector_size: int = 100):
        """Compute Doc2Vec embeddings for a list of tokenized documents."""
        corpus = []
        for db in dataset:
            corpus.append(" ".join([" ".join(d) for d in db]))

        tagged_data = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]

        self.gensim_model.build_vocab(tagged_data)
        self.gensim_model.train(tagged_data, total_examples=self.gensim_model.corpus_count, epochs=self.gensim_model.epochs)
        self.gensim_model.save("d2v.model")        
        self.gensim_model = Doc2Vec.load("d2v.model")

        sim_mat = MatrixSimilarity(corpus)

        return sim_mat

        
