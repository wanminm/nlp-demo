"""聚类引擎：TF-IDF 向量化 / KMeans 聚类 / PCA 降维."""

import jieba
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .nlp_engine import _STOP_WORDS


def _jieba_tokenize(text: str) -> str:
    """Jieba 分词后用空格拼接，供 TfidfVectorizer 使用."""
    words = jieba.cut(text)
    filtered = [w for w in words if w not in _STOP_WORDS and len(w.strip()) > 0]
    return " ".join(filtered)


def build_tfidf_matrix(
    texts: List[str],
) -> Tuple[np.ndarray, TfidfVectorizer]:
    """将文本列表转换为 TF-IDF 稀疏矩阵."""
    if not texts:
        raise ValueError("文本列表不能为空")
    tokenized = [_jieba_tokenize(t) for t in texts]
    vectorizer = TfidfVectorizer(max_features=500, min_df=1)
    matrix = vectorizer.fit_transform(tokenized)
    return matrix.toarray(), vectorizer


def run_kmeans(matrix: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """对特征矩阵执行 KMeans 聚类，返回每个样本的簇标签."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(matrix)


def reduce_with_pca(matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """使用 PCA 将高维矩阵降维到 2D（或指定维度）."""
    if matrix.shape[0] < 2:
        raise ValueError("样本数量至少为 2 才能进行 PCA")
    actual_components = min(n_components, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=actual_components, random_state=42)
    return pca.fit_transform(matrix)


def cluster_pipeline(texts: List[str], k: int = 3) -> Dict[str, Any]:
    """一键聚类流水线：文本 -> TF-IDF -> KMeans -> PCA -> 结果字典."""
    if k < 2 or k > 5:
        raise ValueError(f"K 值必须在 2~5 之间，当前值: {k}")
    if len(texts) < k:
        raise ValueError(f"文本数量 ({len(texts)}) 不能少于 K 值 ({k})")

    matrix, vectorizer = build_tfidf_matrix(texts)
    labels = run_kmeans(matrix, n_clusters=k)
    coords = reduce_with_pca(matrix)

    return {
        "labels": labels.tolist(),
        "coordinates": coords.tolist() if hasattr(coords, "tolist") else coords,
        "texts": texts,
        "k": k,
        "feature_names": list(vectorizer.get_feature_names()),
    }
