import os

from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_gradient_magnitude


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
            max_font_size=40,
            random_state=42,
            relative_scaling=0,
        )
        self.colors = ImageColorGenerator(bg_color)

    def word_cloud(self, text, title):
        self.wc.generate(text)
        self.wc.recolor(color_func=self.colors)

        fig, ax = plt.subplots(figsize=(10, 10))
        plt.axis("off")
        ax.imshow(self.wc, interpolation="bilinear")
        ax.set_title(f"{title} | Word Cloud")

        return fig

    def word_hist(self, data, title, n=25):
        labels, values = zip(*data.items())
        sort_idx = np.argsort(values)[::-1]
        labels = np.array(labels)[sort_idx]
        values = np.array(values)[sort_idx]
        idx = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.bar(idx[:n], values[:n])
        ax.set_title(f"{title} | Word Histogram")
        ax.set_xticks(idx[:n] + 0.1, labels[:n], rotation=90)

        return fig
