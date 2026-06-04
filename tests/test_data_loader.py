import pytest
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_csv, load_sample_data, load_file, get_summary_stats


class TestLoadCsv:
    def test_load_csv_returns_dataframe(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("评论内容,评分\n很好用,5\n太差了,1\n", encoding="utf-8")
        result = load_csv(str(csv_file))
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "评论内容" in result.columns
        assert "评分" in result.columns

    def test_load_csv_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            load_csv("nonexistent.csv")


class TestLoadSampleData:
    def test_returns_dataframe_with_20_rows(self):
        result = load_sample_data()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20
        assert "评论内容" in result.columns
        assert "评分" in result.columns

    def test_has_text_column(self):
        result = load_sample_data()
        assert all(isinstance(t, str) for t in result["评论内容"])
        assert all(len(t) > 0 for t in result["评论内容"])


class TestLoadFile:
    def test_load_csv_file(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("评论内容,评分\n很好,5\n", encoding="utf-8")
        result = load_file(str(csv_file))
        assert len(result) == 1

    def test_load_excel_file(self, tmp_path):
        xlsx_file = tmp_path / "test.xlsx"
        df = pd.DataFrame({"评论内容": ["很好"], "评分": [5]})
        df.to_excel(str(xlsx_file), index=False)
        result = load_file(str(xlsx_file))
        assert len(result) == 1


class TestGetSummaryStats:
    def test_returns_correct_stats(self):
        df = pd.DataFrame({
            "评论内容": ["a", "b", "c", "d"],
            "评分": [5, 1, 4, 2]
        })
        stats = get_summary_stats(df)
        assert stats["total"] == 4
