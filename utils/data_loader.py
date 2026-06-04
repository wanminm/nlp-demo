"""数据加载器：CSV/Excel 解析 + 内置样本数据提供."""

import os
import pandas as pd


def load_csv(file_path: str) -> pd.DataFrame:
    """加载 CSV 文件，要求包含'评论内容'列."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    df = pd.read_csv(file_path, encoding="utf-8")
    if "评论内容" not in df.columns:
        raise ValueError("CSV 文件必须包含'评论内容'列")
    return df


def load_excel(file_path: str) -> pd.DataFrame:
    """加载 Excel 文件."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    df = pd.read_excel(file_path, engine="openpyxl")
    if "评论内容" not in df.columns:
        raise ValueError("Excel 文件必须包含'评论内容'列")
    return df


def load_file(file_path: str) -> pd.DataFrame:
    """智能加载：根据后缀自动选择 CSV 或 Excel 解析."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return load_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        return load_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，支持 .csv / .xlsx / .xls")


def load_sample_data() -> pd.DataFrame:
    """加载内置样本数据集."""
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "sample_reviews.csv",
    )
    return load_csv(sample_path)


def get_summary_stats(df: pd.DataFrame) -> dict:
    """计算数据集的宏观统计指标."""
    total = len(df)
    stats = {"total": total}

    if "评分" in df.columns:
        rating = pd.to_numeric(df["评分"], errors="coerce")
        stats["avg_rating"] = round(float(rating.mean()), 2)
        stats["positive_rate"] = round(
            float((rating >= 4).sum() / total * 100), 1
        )
        stats["negative_count"] = int((rating <= 2).sum())

    if "sentiment_score" in df.columns:
        sentiment = pd.to_numeric(df["sentiment_score"], errors="coerce")
        stats["positive_sentiment_rate"] = round(
            float((sentiment >= 0.5).sum() / total * 100), 1
        )
        stats["negative_alert_count"] = int((sentiment < 0.4).sum())

    return stats
