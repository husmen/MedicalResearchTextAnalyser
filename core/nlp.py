from itertools import combinations

import medspacy
import nltk
import numpy as np
import pandas as pd
import spacy
from empath import Empath

# import medspacy
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from quickumls import QuickUMLS

# from quickumls.spacy_component import SpacyQuickUMLS
# from core.utils import logger
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from tqdm import tqdm

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
        self.medspacy = medspacy.load()
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.spacy_nlp.max_length = 2000000
        # self.spacy_quickumls = SpacyQuickUMLS(self.spacy_nlp, "/data/UMLS/QuickUMLS/")
        # self.spacy_nlp.add_pipe(self.spacy_quickumls.name)
        self.matcher = QuickUMLS("/data/UMLS/QuickUMLS")

    def preprocess(self, db: list[str], stem: bool = False):
        """Preprocess a list of documents."""
        db_tkns = [self.tokenize(d) for d in db]

        if stem:
            db_tkns = [self.stem(d) for d in db_tkns]

        return db_tkns

    def tokenize(self, text: str):
        """Tokenize, and remove stop words in a document using NLTK."""

        words = word_tokenize(text, "english")
        words = [
            word.lower()
            for word in words
            if word.isalpha() and word.lower() not in self.stop_words
        ]

        return words

    def stem(self, words: list[str]):
        """Stem a list of words using NLTK."""
        words = [self.stemmer.stem(word) for word in words]

        return words

    def text_features(
        self, dataset: list[list[list[str]]], titles: list[list[str]] = None
    ):
        """Compute word counts and TF-IDF for a list of tokenized documents."""
        corpus = [" ".join(work) for db in dataset for work in db]
        titles_flat = (
            [title for db in titles for title in db] if titles is not None else None
        )
        pipe = Pipeline(
            [
                ("count", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
            ]
        )

        tfidf_vec = pipe.fit_transform(corpus)
        count_vec = pipe["count"].transform(corpus)
        features = pipe.get_feature_names_out()

        tfidf_df = pd.DataFrame(
            tfidf_vec.toarray(), index=titles_flat, columns=features
        )
        count_df = pd.DataFrame(
            count_vec.toarray(), index=titles_flat, columns=features
        )

        return tfidf_df, count_df

    def dict_features(
        self, dataset: list[list[dict[str, float]]], titles: list[list[str]] = None
    ):
        """Compute word counts and TF-IDF for a list of tokenized documents."""
        corpus = [work for db in dataset for work in db]
        titles_flat = (
            [title for db in titles for title in db] if titles is not None else None
        )
        pipe = Pipeline(
            [
                ("dict_count", DictVectorizer()),
                ("tfidf", TfidfTransformer()),
            ]
        )

        tfidf_vec = pipe.fit_transform(corpus)
        count_vec = pipe["dict_count"].transform(corpus)
        features = pipe.get_feature_names_out()

        tfidf_df = pd.DataFrame(
            tfidf_vec.toarray(), index=titles_flat, columns=features
        )
        count_df = pd.DataFrame(
            count_vec.toarray(), index=titles_flat, columns=features
        )

        return tfidf_df, count_df

    def empath(self, dataset: list[list[list[str]]]):
        """Compute Empath categories for a list of tokenized documents."""

        topics_per_doc: list[list[dict[str, float]]] = [
            [self.lexicon.analyze(" ".join(work), normalize=True) for work in db]
            for db in dataset
        ]
        topics_per_db: list[dict[str, float]] = [
            self.lexicon.analyze(
                " ".join([" ".join(work) for work in db]), normalize=True
            )
            for db in dataset
        ]

        # cat = {k: v for k, v in cat.items() if v > 0.0}

        return topics_per_doc, topics_per_db

    def topics_sim(self, dataset: list[list[dict[str, float]]]):
        """Compute cosine similarity for a list of tokenized documents."""

        sims = np.zeros((len(dataset), len(dataset)), dtype=float)

        for c in tqdm(combinations(sorted(range(len(dataset))), 2)):
            # logger.info(f"Computing cosine similarity for D{c[0]} and D{c[1]}")
            doc1 = dataset[c[0]]  # list of dicts
            doc2 = dataset[c[1]]

            doc1_vec = np.array([[v for k, v in d.items()] for d in doc1])
            doc2_vec = np.array([[v for k, v in d.items()] for d in doc2])

            similarity_matrix = cosine_similarity(doc1_vec, doc2_vec)
            sims[c[0], c[1]] = similarity_matrix.mean()
            sims[c[1], c[0]] = similarity_matrix.mean()

        df_header = [f"D{i+1}" for i in range(len(sims))]
        sims_df = pd.DataFrame(sims, index=df_header, columns=df_header)

        return sims_df

        # corpus = [" ".join(d) for d in dataset]
        # pipe = Pipeline(
        #     [
        #         ("count", CountVectorizer()),
        #         ("tfidf", TfidfTransformer()),
        #     ]
        # ).fit(corpus)

        # pipe.transform(corpus)
        # features = pipe.get_feature_names_out()

        # count_vec = pipe["count"].transform(corpus).toarray()
        # tf_idf_vec = pipe["tfidf"].transform(count_vec).toarray()

        # df = pd.DataFrame(tf_idf_vec, columns=features)

        # return df.dot(df.T)

    def tfidf_sim(self, tfidf: pd.DataFrame, titles: list[list[str]]):
        """Compute cosine similarity for a list of tokenized documents."""

        slice_sizes = [len(db) for db in titles]
        sims = np.zeros((len(titles), len(titles)), dtype=float)

        tfidf_arr = tfidf.to_numpy()

        for c in tqdm(combinations(sorted(range(len(titles))), 2)):
            # logger.info(f"Computing cosine similarity for D{c[0]} and D{c[1]}")
            doc1_idx_0 = sum(slice_sizes[: c[0]])
            doc1_idx_1 = sum(slice_sizes[: c[0] + 1])

            doc2_idx_0 = sum(slice_sizes[: c[1]])
            doc2_idx_1 = sum(slice_sizes[: c[1] + 1])

            doc1 = tfidf_arr[doc1_idx_0:doc1_idx_1]
            doc2 = tfidf_arr[doc2_idx_0:doc2_idx_1]

            similarity_matrix = cosine_similarity(doc1, doc2)
            sims[c[0], c[1]] = similarity_matrix.mean()
            sims[c[1], c[0]] = similarity_matrix.mean()

        df_header = [f"D{i+1}" for i in range(len(sims))]
        sims_df = pd.DataFrame(sims, index=df_header, columns=df_header)

        return sims_df

    def quickumls(self, db: list[str]):
        """Compute QuickUMLS entities for a list of documents."""
        # text = " ".join([w for doc in db for w in doc])
        # doc = self.spacy_nlp(text)

        text = " ".join(db)
        doc = self.matcher.match(text, best_match=True, ignore_syntax=False)

        return doc

    # def medspacy(self, db: list[str]):
    #     """Compute MedSpaCy entities for a list of documents."""
    #     text = " ".join([w for doc in db for w in doc])
    #     doc = self.medspacy_model(text)
    #     ents = [ent.text for ent in doc.ents]

    #     return ents

    # def gensim(self, dataset: list[list[str]], vector_size: int = 100):
    #     """Compute Doc2Vec embeddings for a list of tokenized documents."""
    #     corpus = []
    #     for db in dataset:
    #         corpus.append(" ".join([" ".join(d) for d in db]))

    #     tagged_data = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]

    #     self.gensim_model.build_vocab(tagged_data)
    #     self.gensim_model.train(tagged_data, total_examples=self.gensim_model.corpus_count, epochs=self.gensim_model.epochs)
    #     self.gensim_model.save("d2v.model")
    #     self.gensim_model = Doc2Vec.load("d2v.model")

    #     sim_mat = MatrixSimilarity(corpus)

    #     return sim_mat
