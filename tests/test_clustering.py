import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.clustering import (
    build_tfidf_matrix,
    run_kmeans,
    reduce_with_pca,
    cluster_pipeline,
)

SAMPLE_TEXTS = [
    "快递太慢了等了五天",
    "物流很差包裹破损",
    "发货速度很快",
    "手机屏幕碎了质量差",
    "耳机音质很棒",
    "拍照效果非常清晰",
    "客服态度不好不回复",
    "售后处理速度很快",
    "性价比很高推荐购买",
    "降价了没有保价服务",
]


class TestBuildTfidfMatrix:
    def test_returns_matrix_and_vectorizer(self):
        matrix, vectorizer = build_tfidf_matrix(SAMPLE_TEXTS)
        assert matrix.shape[0] == len(SAMPLE_TEXTS)
        assert matrix.shape[1] > 0

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            build_tfidf_matrix([])


class TestRunKmeans:
    def test_returns_labels_for_each_text(self):
        matrix, _ = build_tfidf_matrix(SAMPLE_TEXTS)
        labels = run_kmeans(matrix, n_clusters=3)
        assert len(labels) == len(SAMPLE_TEXTS)
        assert set(labels).issubset({0, 1, 2})

    def test_respects_n_clusters(self):
        matrix, _ = build_tfidf_matrix(SAMPLE_TEXTS)
        for k in [2, 3, 5]:
            labels = run_kmeans(matrix, n_clusters=k)
            assert len(set(labels)) <= k


class TestReduceWithPca:
    def test_reduces_to_2d(self):
        matrix, _ = build_tfidf_matrix(SAMPLE_TEXTS)
        coords = reduce_with_pca(matrix)
        assert coords.shape == (len(SAMPLE_TEXTS), 2)

    def test_returns_numpy_array(self):
        matrix, _ = build_tfidf_matrix(SAMPLE_TEXTS)
        coords = reduce_with_pca(matrix)
        assert isinstance(coords, np.ndarray)


class TestClusterPipeline:
    def test_returns_complete_result(self):
        result = cluster_pipeline(SAMPLE_TEXTS, k=3)
        assert "labels" in result
        assert "coordinates" in result
        assert "texts" in result
        assert "k" in result
        assert len(result["labels"]) == len(SAMPLE_TEXTS)
        assert len(result["coordinates"]) == len(SAMPLE_TEXTS)
        assert result["k"] == 3

    def test_k_out_of_range_raises(self):
        with pytest.raises(ValueError):
            cluster_pipeline(SAMPLE_TEXTS, k=1)
        with pytest.raises(ValueError):
            cluster_pipeline(SAMPLE_TEXTS, k=6)
