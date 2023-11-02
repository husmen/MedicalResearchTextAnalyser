from core.utils import logger, user_email

import pyalex
import pandas as pd

from pyalex import Works
from tqdm import tqdm
import enum


# try:
#     from pybliometrics.scopus.utils import config

#     config["Authentication"]["APIKey"] = "<API KEY>"
#     import pybliometrics
# except:
#     pass

from pybliometrics.scopus import ScopusSearch

# from pybliometrics.scopus import AbstractRetrieval


class Engine(enum.Enum):
    """Enum class for literature search engine"""

    OPENALEX = "openalex"
    SCOPUS = "scopus"


class LiteratureDataset:
    """Class to load and process literature data from different sources"""

    def __init__(self, engine: Engine = Engine.SCOPUS, n=None, since_year=None):
        self.cols = [
            "Title",
            "Abstract",
            "Authors",
            "Year",
            "Cited by",
            "Document Type",
        ]
        self.engine = engine
        self.n = 500 if n is None else n
        self.since_year = 2020 if since_year is None else since_year

        self.df_ar: pd.DataFrame = None
        self.df_re: pd.DataFrame = None

        self.d1: pd.DataFrame = pd.DataFrame(columns=self.cols)
        self.d2: pd.DataFrame = pd.DataFrame(columns=self.cols)
        self.d3: pd.DataFrame = pd.DataFrame(columns=self.cols)
        self.d4: pd.DataFrame = pd.DataFrame(columns=self.cols)

        self.data: list[pd.DataFrame] = [self.d1, self.d2, self.d3, self.d4]
        self.data_description: list[str] = [
            "D1: Articles - Full",
            "D2: Reviews - Full",
            "D3: Articles - Recent",
            "D4: Reviews - Recent",
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_abstracts(self, idx):
        return self.data[idx]["Abstract"].tolist()

    def fetch_data(
        self,
    ):
        if self.engine is Engine.SCOPUS:
            works_ar, works_re = self._query_scopus()
        else:
            logger.warning("Review only mode is not supported by OpenAlex")
            logger.error("Only Scopus is supported at the moment")

            works_ar, works_re = self._query_openalex()

        # TODO: convert to pandas
        return works_ar, works_re

    def load_data(self):
        if self.engine is Engine.SCOPUS:
            self.df_ar = pd.read_csv("data/scopus_ar.csv")
            self.df_re = pd.read_csv("data/scopus_re.csv")

            self.df_ar = self._cleanup_entries(self.df_ar)
            self.df_re = self._cleanup_entries(self.df_re)

            self.d1 = self._get_top_cited(self.df_ar)
            self.d2 = self._get_top_cited(self.df_re)

            self.d3 = self._get_most_recet(self.df_ar)
            self.d4 = self._get_most_recet(self.df_re)

            self.data = [self.d1, self.d2, self.d3, self.d4]

        else:
            logger.error("Only Scopus is supported at the moment")

    def preview_data(self):
        print(len(self.data))
        for i, d in enumerate(self.data):
            print(self.data_description[i])
            d.info()
            print(d.head())
            print(d.describe())
            print()

    def _cleanup_entries(self, df_raw: pd.DataFrame):
        df_tmp = df_raw[df_raw["Abstract"] != "[No abstract available]"]
        df_tmp.loc[:, "Year"] = pd.to_numeric(df_tmp["Year"])
        df_tmp.loc[:, "Cited by"] = pd.to_numeric(df_tmp["Cited by"])

        return df_tmp

    def _get_top_cited(self, df_raw: pd.DataFrame):
        df_tmp = df_raw[self.cols].sort_values(by=["Cited by"], ascending=False)
        df_tmp = df_tmp[: self.n].sort_values(by=["Year"], ascending=False)

        return df_tmp

    def _get_most_recet(self, df_raw: pd.DataFrame):
        df_tmp = df_raw[self.cols][df_raw["Year"] >= self.since_year]
        df_tmp = self._get_top_cited(df_tmp)

        return df_tmp

    def _query_openalex(
        self,
    ):
        pyalex.config.email = user_email

        works = []
        query = Works().search_filter(
            title="radiology|radiologist report",
            abstract="automatic|automation analysis|generation|classification|summary|summarization|parsing|understanding",
        )

        # if review_only:
        #     logger.warning("Review only mode is not supported by OpenAlex")

        # if since_year:
        #     query = query.filter(publication_year=f">{since_year}")

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

        try:
            if self.n:
                works = []
                for page in tqdm(
                    query.paginate(method="page", page=self.n // 100, per_page=100)
                ):
                    works.extend(page)
                    if len(works) >= self.n:
                        break
            else:
                works = query.get()
        except Exception as e:
            logger.error(e)

        for w in works:
            w["authors"] = [a["author"]["display_name"] for a in w["authorships"]]
            w["abstract"] = w["abstract"]
            w["concepts"] = [
                {k: v for k, v in c.items() if k in ["display_name", "level", "score"]}
                for c in w["concepts"]
                if c["score"] > 0.2 and c["level"] > 0
            ]
            del w["authorships"]

        return works, works

    def _query_scopus(self):
        works_ar = []
        works_re = []
        query_re = 'TITLE ( ( radiology OR radiologist ) ) AND ( TITLE ( ( ( automatic OR automated ) AND report ) OR ( artificial AND intelligence AND report ) OR ( deep AND learning AND report ) OR ( natural AND language AND processing ) OR ( large AND language AND model ) ) OR TITLE-ABS-KEY (report AND ( ( automatic OR automated ) OR ( artificial AND intelligence AND report ) OR ( deep AND learning AND report ) OR ( natural AND language AND processing ) OR ( large AND language AND model ) OR ( information AND retrieval ) OR ( computational AND linguistics ) )) ) AND ( LIMIT-TO ( DOCTYPE , "re" ) ) AND ( LIMIT-TO ( LANGUAGE , "English" ) )'
        query_ar = 'TITLE ( ( radiology OR radiologist ) ) AND ( TITLE ( ( ( automatic OR automated ) AND report ) OR ( artificial AND intelligence AND report ) OR ( deep AND learning AND report ) OR ( natural AND language AND processing ) OR ( large AND language AND model ) ) OR TITLE-ABS-KEY (report AND ( ( automatic OR automated ) OR ( artificial AND intelligence AND report ) OR ( deep AND learning AND report ) OR ( natural AND language AND processing ) OR ( large AND language AND model ) OR ( information AND retrieval ) OR ( computational AND linguistics ) )) ) AND ( EXCLUDE ( DOCTYPE , "re" ) ) AND ( LIMIT-TO ( LANGUAGE , "English" ) )'
        try:
            works_ar = ScopusSearch(query_ar, refresh=True)
            works_re = ScopusSearch(query_re, refresh=True)

            # change keys to lower case
            # works = [{k.lower(): v for k, v in w.items()} for w in works]

            logger.error("API not fully implemented yet")
        except Exception as e:
            logger.error(e)

        return works_ar, works_re
