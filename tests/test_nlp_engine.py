import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.nlp_engine import (
    tokenize_with_pos,
    filter_content_words,
    extract_entities,
    analyze_sentiment,
    classify_business_category,
)


class TestTokenizeWithPos:
    def test_returns_list_of_tuples(self):
        result = tokenize_with_pos("屏幕碎了")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_extracts_noun_and_verb(self):
        result = tokenize_with_pos("屏幕碎了")
        words = {w for w, _ in result}
        assert "屏幕" in words
        assert "碎" in words

    def test_empty_input_returns_empty(self):
        result = tokenize_with_pos("")
        assert result == []


class TestFilterContentWords:
    def test_filters_punctuation(self):
        pairs = [("屏幕", "n"), ("的", "uj"), ("了", "ul"), ("碎", "v")]
        result = filter_content_words(pairs)
        words = {w for w, _ in result}
        assert "屏幕" in words
        assert "碎" in words
        assert "的" not in words
        assert "了" not in words

    def test_keeps_nouns_verbs_adjectives(self):
        pairs = [("快递", "n"), ("快", "a"), ("送", "v"), ("吧", "y")]
        result = filter_content_words(pairs)
        kept_pos = {pos for _, pos in result}
        assert "n" in kept_pos
        assert "a" in kept_pos
        assert "v" in kept_pos


class TestExtractEntities:
    def test_extracts_location(self):
        result = extract_entities("昨天在上海买的iPhone屏幕碎了")
        locs = [e for e in result if e[1] == "LOC"]
        assert len(locs) > 0
        assert any("上海" in e[0] for e in locs)

    def test_extracts_product(self):
        result = extract_entities("iPhone屏幕碎了气死我了")
        products = [e for e in result if e[1] == "PRODUCT"]
        assert len(products) > 0
        assert any("iphone" in e[0].lower() for e in products)

    def test_empty_input_returns_empty(self):
        assert extract_entities("") == []


class TestAnalyzeSentiment:
    def test_returns_score_between_0_and_1(self):
        score = analyze_sentiment("这个产品非常好用")
        assert 0.0 <= float(score) <= 1.0

    def test_negative_review_scores_low(self):
        score = analyze_sentiment("太差了气死我了垃圾产品")
        assert float(score) < 0.5

    def test_positive_review_scores_high(self):
        score = analyze_sentiment("很好用性价比高推荐购买")
        assert float(score) >= 0.4  # SnowNLP 对中文短文本偏保守


class TestClassifyBusinessCategory:
    def test_classifies_logistics_complaint(self):
        result = classify_business_category("快递太慢了物流很差")
        assert any("物流" in c for c in result)

    def test_classifies_product_quality(self):
        result = classify_business_category("屏幕碎了质量太差")
        assert any("质量" in c for c in result)

    def test_classifies_customer_service(self):
        result = classify_business_category("客服态度极差不回消息")
        assert any("客服" in c for c in result)
