"""可视化引擎：Plotly 图表 + 词云生成."""

import jieba
import plotly.graph_objects as go
import plotly.express as px
from typing import List
from collections import Counter

from .nlp_engine import _STOP_WORDS


def create_word_frequency_chart(
    texts: List[str], top_n: int = 10
) -> go.Figure:
    """生成高频词横向条形图."""
    all_words: List[str] = []
    for text in texts:
        words = jieba.cut(text)
        all_words.extend(
            [w for w in words if w not in _STOP_WORDS and len(w.strip()) > 0]
        )

    counter = Counter(all_words)
    top_words = counter.most_common(top_n)

    words, counts = zip(*top_words) if top_words else ([], [])

    fig = go.Figure(
        go.Bar(
            x=list(counts),
            y=list(words),
            orientation="h",
            marker=dict(
                color=list(counts),
                colorscale="Viridis",
                showscale=False,
            ),
            text=list(counts),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="高频词 TOP {}".format(top_n),
        xaxis_title="词频",
        yaxis_title="",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def create_sentiment_gauge(score: float) -> go.Figure:
    """生成情感得分仪表盘."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score * 100,
            title={"text": "情感得分"},
            number={"suffix": "%"},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 33], "color": "#ff6b6b"},
                    {"range": [33, 66], "color": "#ffd93d"},
                    {"range": [66, 100], "color": "#6bcf7f"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 2},
                    "thickness": 0.75,
                    "value": 40,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_cluster_scatter(
    texts: List[str],
    labels: List[int],
    coordinates: List[List[float]],
) -> go.Figure:
    """生成聚类语义散点图，支持鼠标悬停显示原始评论."""
    x = [c[0] for c in coordinates]
    y = [c[1] for c in coordinates]

    fig = go.Figure()

    unique_labels = sorted(set(labels))
    colors = px.colors.qualitative.Plotly[: len(unique_labels)]

    for label, color in zip(unique_labels, colors):
        idx = [i for i, lbl in enumerate(labels) if lbl == label]
        fig.add_trace(
            go.Scatter(
                x=[x[i] for i in idx],
                y=[y[i] for i in idx],
                mode="markers",
                name=f"主题 {label + 1}",
                marker=dict(size=14, color=color, opacity=0.75),
                hovertext=[texts[i] for i in idx],
                hoverinfo="text",
                hoverlabel=dict(bgcolor="white", font_size=14),
            )
        )

    fig.update_layout(
        title="评论聚类语义地图 (PCA 降维至 2D)",
        xaxis_title="主成分 1",
        yaxis_title="主成分 2",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def create_pie_chart(categories: List[str]) -> go.Figure:
    """生成业务维度分类饼图."""
    counter = Counter(categories)
    fig = go.Figure(
        go.Pie(
            labels=list(counter.keys()),
            values=list(counter.values()),
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Set2),
        )
    )
    fig.update_layout(
        title="业务维度分布",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def create_metrics_html(total: int, positive_rate: float, negative_count: int) -> str:
    """生成指标卡片的 HTML 字符串（用于 st.markdown）."""
    return f"""
    <style>
    .metrics-container {{
        display: flex;
        gap: 20px;
        margin: 20px 0;
    }}
    .metric-card {{
        flex: 1;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .metric-card.blue {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
    .metric-card.green {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }}
    .metric-card.red {{ background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }}
    .metric-number {{ font-size: 36px; font-weight: bold; }}
    .metric-label {{ font-size: 14px; opacity: 0.9; margin-top: 4px; }}
    </style>
    <div class="metrics-container">
        <div class="metric-card blue">
            <div class="metric-number">{total}</div>
            <div class="metric-label">总分析评论数</div>
        </div>
        <div class="metric-card green">
            <div class="metric-number">{positive_rate}%</div>
            <div class="metric-label">正面好评率</div>
        </div>
        <div class="metric-card red">
            <div class="metric-number">{negative_count}</div>
            <div class="metric-label">负面预警数</div>
        </div>
    </div>
    """
