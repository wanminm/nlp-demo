import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualizations import (
    create_word_frequency_chart,
    create_sentiment_gauge,
    create_cluster_scatter,
)

SAMPLE_TEXTS = [
    "快递太慢了等了五天",
    "物流很差包裹破损",
    "手机屏幕碎了质量差",
    "耳机音质很棒拍照清晰",
    "客服态度不好不回复",
]
SAMPLE_LABELS = [0, 0, 1, 2, 1]
SAMPLE_COORDS = [[1.0, 2.0], [1.5, 2.5], [-1.0, -2.0], [2.0, -1.0], [-1.5, -1.0]]


class TestCreateWordFrequencyChart:
    def test_returns_figure(self):
        from plotly.graph_objects import Figure
        fig = create_word_frequency_chart(SAMPLE_TEXTS, top_n=5)
        assert isinstance(fig, Figure)

    def test_respects_top_n(self):
        fig = create_word_frequency_chart(SAMPLE_TEXTS, top_n=3)
        assert len(fig.data[0].y) <= 3


class TestCreateSentimentGauge:
    def test_returns_figure(self):
        from plotly.graph_objects import Figure
        fig = create_sentiment_gauge(0.75)
        assert isinstance(fig, Figure)

    def test_has_gauge_data(self):
        fig = create_sentiment_gauge(0.2)
        assert fig.data[0].type == "indicator"


class TestCreateClusterScatter:
    def test_returns_figure(self):
        from plotly.graph_objects import Figure
        fig = create_cluster_scatter(SAMPLE_TEXTS, SAMPLE_LABELS, SAMPLE_COORDS)
        assert isinstance(fig, Figure)

    def test_hover_text_contains_original_review(self):
        fig = create_cluster_scatter(SAMPLE_TEXTS, SAMPLE_LABELS, SAMPLE_COORDS)
        hover = fig.data[0].hovertext
        assert hover is not None
        assert SAMPLE_TEXTS[0] in hover[0]
