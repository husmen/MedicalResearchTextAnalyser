# try:
#     from pybliometrics.scopus.utils import config

#     config["Authentication"]["APIKey"] = "<API KEY>"
#     import pybliometrics
# except:
#     pass

import json
import logging

import httpx
import nltk
import pyalex
import pandas as pd

import matplotlib.pyplot as plt
# from itertools import chain
from pyalex import Works
from tqdm import tqdm

from empath import Empath
from wordcloud import WordCloud

# from tqdm.contrib.itertools import product
from pybliometrics.scopus import AbstractRetrieval, ScopusSearch

openalex_api_url = "https://api.openalex.org/works"
umls_api_key = "<API KEY>"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

pyalex.config.email = "<EMAIL>"

nltk.download("punkt")
nltk.download("stopwords")
# nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import math

stop_words = set(stopwords.words("english"))


def query_openalex(n=None, since_year=None):
    query = Works().search_filter(
        title="radiology|radiologist report",
        abstract="automatic|automation analysis|generation|classification|summary|summarization|parsing|understanding",
    )

    if since_year:
        query = query.filter(publication_year=f">{since_year}")

    query = query.sort(cited_by_count="desc", relevance_score="desc").select(
        [
            "id",
            "relevance_score",
            "publication_year",
            "title",
            "authorships",
            "cited_by_count",
            "concepts",
            "abstract_inverted_index",
        ]
    )
    if n:
        works = []
        for page in tqdm(query.paginate(method="page", page=n // 100, per_page=100)):
            works.extend(page)
            if len(works) >= n:
                break
    else:
        works = query.get()

    # works = [r for r in chain(*query.paginate(method="page", page=n//100, per_page=100))] if n else query.get()

    for w in works:
        w["authors"] = [a["author"]["display_name"] for a in w["authorships"]]
        w["abstract"] = w["abstract"]
        w["concepts"] = [
            {k: v for k, v in c.items() if k in ["display_name", "level", "score"]}
            for c in w["concepts"]
            if c["score"] > 0.2 and c["level"] > 0
        ]
        del w["authorships"]

    logger.info(f"Selected {len(works)} out of {query.count()} found works")

    return {"results": works}


# def query_openalex_2(keyword: str, fields: list[str], n) -> list[dict[str, str]]:
#     query = f"search=keyword&select={','.join(fields)}&q={keyword}&per-page={n}&page=1"
#     query_url = f"{openalex_api_url}?{query}"
#     logger.info(f"Querying {query_url}")
#     response = httpx.get(query_url)
#     return response.json()


def query_scopus(review_only=False):
    query_re = 'TITLE ( ( radiologist OR radiology ) AND report ) AND TITLE-ABS-KEY ( ( automatic OR automation ) AND ( analysis OR generation OR classification OR parsing OR summary OR summarization OR understanding ) ) AND ( LIMIT-TO ( DOCTYPE , "re" ) )'
    query_ar = 'TITLE ( ( radiologist OR radiology ) AND report ) AND TITLE-ABS-KEY ( ( automatic OR automation ) AND ( analysis OR generation OR classification OR parsing OR summary OR summarization OR understanding ) ) AND ( LIMIT-TO ( DOCTYPE , "ar" ) OR LIMIT-TO ( DOCTYPE , "cp" ) )'

    try:
        works = ScopusSearch(query_re if review_only else query_ar, refresh=True)
    except Exception as e:
        logger.error(e)
        df = pd.read_csv(f"data/scopus_{'re' if review_only else 'ar'}.csv")
        works = df.to_dict(orient="records")

    # change keys to lower case
    works = [{k.lower(): v for k, v in w.items()} for w in works]

    return {"results": works}


def tokenize_and_remove_stopwords(document):
    """Tokenize a document using NLTK and remove stop words."""
    words = word_tokenize(document)
    return [
        word.lower()
        for word in words
        if word.isalpha() and word.lower() not in stop_words
    ]


def compute_tf(doc_tokens):
    """Compute term frequency for a tokenized document."""
    tf = defaultdict(int)
    for word in doc_tokens:
        tf[word] += 1
    total_words = len(doc_tokens)
    for word, count in tf.items():
        tf[word] = count / total_words
    return tf


def compute_idf(documents):
    """Compute inverse document frequency for all documents."""
    tf_full = defaultdict(int)
    idf = defaultdict(int)
    total_docs = len(documents)
    for doc_tokens in documents:
        for word in set(doc_tokens):
            tf_full[word] += 1
    for word, count in tf_full.items():
        idf[word] = math.log(total_docs / count)
    return idf, tf_full


def compute_tf_idf(documents):
    """Compute TF-IDF for all documents."""
    tf_idfs = []
    idf, tf_full = compute_idf(documents)
    for doc_tokens in documents:
        tf = compute_tf(doc_tokens)
        tf_idf = {}
        for word, term_freq in tf.items():
            tf_idf[word] = term_freq * idf[word]
        tf_idfs.append(tf_idf)
    return tf_idfs, tf_full


def compute_empath(documents):
    lexicon = Empath()

    docs_flat = [" ".join(doc_tokens) for doc_tokens in documents]
    docs_flat = " ".join(docs_flat)
    cat = lexicon.analyze(docs_flat, normalize=True)
    cat = {k: v for k, v in cat.items() if v > 0.0}
    return docs_flat, cat


def run_nlp_pipeline(documents):
    """Run the full NLP pipeline."""
    documents_tokens = [tokenize_and_remove_stopwords(d) for d in documents]
    tf_idfs, tf_full = compute_tf_idf(documents_tokens)
    docs_flat, cat = compute_empath(documents_tokens)
    return documents_tokens, tf_idfs, tf_full, docs_flat, cat


def visualize_results(df):
    plt.figure(figsize=(15,10))
    country.max().sort_values(by="points",ascending=False)["points"].plot.bar()
    plt.xticks(rotation=50)
    plt.xlabel("Country of Origin")
    plt.ylabel("Highest point of Wines")
    plt.show()



if __name__ == "__main__":
    logger.info("fetching results")
    # d1 = query_openalex(n=500)
    # d2 = query_scopus(review_only=True)
    # d3 = query_openalex(n=500, since_year=2019)

    # for i, d in tqdm(enumerate([d1, d2, d3])):
    #     with open(f"data/d{i+1}.json", "w", encoding="utf8") as f:
    #         f.write(json.dumps(d))

    data = []

    for i in range(3):
        with open(f"data/d{i+1}.json", encoding="utf8") as f:
            data.append(json.loads(f.read()))

    logger.info("NLP pipeline")
    d_processed = []
    for i, d_raw in tqdm(enumerate(data)):
        nlp_out = run_nlp_pipeline([w["abstract"] for w in d_raw["results"]])
        d_processed.append(nlp_out)
        # print(nlp_out[0][0], nlp_out[0][1], nlp_out[0][2])

    for i, d_proc in enumerate(d_processed):
        wordcloud = WordCloud().generate(d_proc[3])
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

