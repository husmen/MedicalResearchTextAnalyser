import os

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy.ndimage import gaussian_gradient_magnitude
from wordcloud import ImageColorGenerator, WordCloud

# alt.renderers.enable("altair_viewer", inline=True)
alt.data_transformers.enable("vegafusion")


class DataViz:
    def __init__(self):
        bg_color = np.array(
            Image.open(os.path.join(os.getcwd(), "core", "coloured-brain-mri.jpg"))
        )
        bg_color = bg_color[::2, ::2]
        bg_mask = bg_color.copy()
        bg_mask[bg_mask.sum(axis=2) == 0] = 255
        bg_edges = np.mean(
            [
                gaussian_gradient_magnitude(bg_color[:, :, i] / 255.0, 2)
                for i in range(3)
            ],
            axis=0,
        )
        bg_mask[bg_edges > 0.08] = 255

        self.wc = WordCloud(
            max_words=1000,
            mask=bg_mask,
            max_font_size=64,
            random_state=42,
            relative_scaling=0,
        )
        self.colors = ImageColorGenerator(bg_color)

    def word_clouds(self, data, titles):
        fig, ax = plt.subplots(1, len(titles), figsize=(10 * len(titles), 10))

        for i, db in enumerate(data):
            text = " ".join([word for work in db for word in work])
            self.wc.generate(text)
            self.wc.recolor(color_func=self.colors)
            ax[i].imshow(self.wc, interpolation="bilinear")
            ax[i].set_title(f"{titles[i]} | Word Cloud")
            ax[i].axis("off")

        fig.tight_layout()
        return fig

    def word_hist(self, data, titles, n=25):
        fig, ax = plt.subplots(1, len(titles), figsize=(10 * len(titles), 10))

        for i, db in enumerate(data):
            labels, values = zip(*db.items())
            sort_idx = np.argsort(values)[::-1]
            labels = np.array(labels)[sort_idx]
            values = np.array(values)[sort_idx]
            idx = np.arange(len(labels))

            ax[i].bar(idx[:n], values[:n])
            ax[i].set_title(f"{titles[i]} | topic Histogram")
            ax[i].set_xticks(idx[:n] + 0.1, labels[:n], rotation=90)

        fig.tight_layout()
        return fig

    def tfidf_chart(self, data: pd.DataFrame, n=25, title=""):
        # https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/03-TF-IDF-Scikit-Learn.html

        tfidf_df = data.stack().reset_index()
        tfidf_df = tfidf_df.rename(
            columns={
                0: "tfidf",
                "level_0": "document",
                "level_1": "term",
                "level_2": "term",
            }
        )
        tfidf_df = (
            tfidf_df.sort_values(by=["document", "tfidf"], ascending=[True, False])
            .groupby(["document"])
            .head(n=n)
        )

        # adding a little randomness to break ties in term ranking
        tfidf_rnd_df = tfidf_df.copy()
        tfidf_rnd_df["tfidf"] = (
            tfidf_rnd_df["tfidf"] + np.random.rand(tfidf_df.shape[0]) * 0.0001
        )

        # base for all visualizations, with rank calculation
        base = (
            alt.Chart(tfidf_rnd_df, title=f"{title} | TF-IDF")
            .encode(x="rank:O", y="document:N")
            .transform_window(
                rank="rank()",
                sort=[alt.SortField("tfidf", order="descending")],
                groupby=["document"],
            )
        )

        # heatmap specification
        heatmap = base.mark_rect().encode(color="tfidf:Q")

        # red circle over terms in above list
        # circle = base.mark_circle(size=100).encode(
        #     color=alt.condition(
        #         alt.FieldOneOfPredicate(field="term", oneOf=term_list),
        #         alt.value("red"),
        #         alt.value("#FFFFFF00"),
        #     )
        # )

        # text labels, white for darker heatmap colors
        text = base.mark_text(baseline="middle").encode(
            text="term:N",
            color=alt.condition(
                alt.datum.tfidf >= 0.23, alt.value("white"), alt.value("black")
            ),
        )

        # display the three superimposed visualizations
        chart = (heatmap + text).properties(width=600)
        return chart

    def tfidf_heatmap(self, data: pd.DataFrame):
        sns.heatmap(data, annot=True, fmt=".4f")
