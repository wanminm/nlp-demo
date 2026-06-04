"""端到端集成测试：验证完整数据流."""

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_sample_data, get_summary_stats
from utils.nlp_engine import (
    tokenize_with_pos,
    filter_content_words,
    extract_entities,
    analyze_sentiment,
    classify_business_category,
)
from utils.clustering import cluster_pipeline


class TestFullNLPPipeline:
    """验证单句分析完整链路."""

    TEST_TEXT = "昨天在上海买的iPhone屏幕碎了气死我了"

    def test_tokenize_returns_reasonable_output(self):
        result = tokenize_with_pos(self.TEST_TEXT)
        assert len(result) > 5

    def test_filter_retains_core_words(self):
        raw = tokenize_with_pos(self.TEST_TEXT)
        filtered = filter_content_words(raw)
        assert len(filtered) < len(raw)
        words = {w for w, _ in filtered}
        assert len(words) >= 2

    def test_ner_finds_entity_type(self):
        entities = extract_entities(self.TEST_TEXT)
        types_found = {e[1] for e in entities}
        assert len(types_found) >= 1

    def test_sentiment_in_range(self):
        score = analyze_sentiment(self.TEST_TEXT)
        assert 0.0 <= score <= 1.0

    def test_category_not_empty(self):
        cats = classify_business_category(self.TEST_TEXT)
        assert len(cats) >= 1
        assert all(isinstance(c, str) for c in cats)


class TestFullClusteringPipeline:
    """验证批量聚类完整链路."""

    def test_pipeline_with_sample_data(self):
        df = load_sample_data()
        texts = df["评论内容"].tolist()
        result = cluster_pipeline(texts, k=3)

        assert "labels" in result
        assert "coordinates" in result
        assert len(result["labels"]) == 20
        assert len(result["coordinates"]) == 20
        assert len(set(result["labels"])) <= 3

    def test_pipeline_k2_and_k5(self):
        df = load_sample_data()
        texts = df["评论内容"].tolist()

        for k in [2, 5]:
            result = cluster_pipeline(texts, k=k)
            assert len(set(result["labels"])) <= k


class TestDataFlowEndToEnd:
    """验证数据从加载到可视化坐标的完整流程."""

    def test_sample_data_to_cluster_coordinates(self):
        # 1. 加载
        df = load_sample_data()
        assert len(df) == 20

        # 2. 情感分析
        sentiments = [analyze_sentiment(t) for t in df["评论内容"]]
        assert len(sentiments) == 20
        assert all(0.0 <= s <= 1.0 for s in sentiments)

        # 3. 分类
        categories = [classify_business_category(t) for t in df["评论内容"]]
        assert len(categories) == 20

        # 4. 聚类到坐标
        result = cluster_pipeline(df["评论内容"].tolist(), k=3)
        assert len(result["coordinates"]) == 20
        assert all(len(c) == 2 for c in result["coordinates"])

        # 5. 统计
        df_copy = df.copy()
        df_copy["sentiment_score"] = sentiments
        stats = get_summary_stats(df_copy)
        assert stats["total"] == 20
        assert "positive_sentiment_rate" in stats
        assert "negative_alert_count" in stats
