"""电商评论与舆情智能分析系统 — Streamlit Web 应用."""

import sys
import os
import tempfile
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_file, load_sample_data, get_summary_stats
from utils.nlp_engine import (
    tokenize_with_pos,
    filter_content_words,
    extract_entities,
    analyze_sentiment,
    classify_business_category,
)
from utils.clustering import cluster_pipeline
from utils.visualizations import (
    create_word_frequency_chart,
    create_sentiment_gauge,
    create_cluster_scatter,
    create_pie_chart,
    create_metrics_html,
)

# ---- 页面配置 ----
st.set_page_config(
    page_title="电商评论与舆情智能分析系统",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- 全局自定义 CSS ----
st.markdown(
    """
<style>
h1 { color: #2d3748; font-weight: 700; }
h2 { color: #4a5568; font-weight: 600; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }
h3 { color: #718096; font-weight: 500; }
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.stExpander {
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.stFileUploader {
    border-radius: 10px;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}
.footer {
    text-align: center;
    padding: 20px;
    color: #a0aec0;
    font-size: 13px;
    border-top: 1px solid #e2e8f0;
    margin-top: 40px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---- 侧边栏 ----
st.sidebar.title("🛒 电商舆情分析系统")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "📁 上传评论数据 (CSV / Excel)",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
    ) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    df = load_file(tmp_path)
    os.unlink(tmp_path)
    st.sidebar.success(f"✅ 已加载 {len(df)} 条评论")
else:
    df = load_sample_data()
    st.sidebar.info(f"📋 当前使用内置演示数据 ({len(df)} 条)")

st.session_state["df"] = df

# ---- 页面导航 ----
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "📑 导航菜单",
    ["📊 舆情数据看板", "🔍 单句交互分析", "🧠 批量聚类语义地图"],
)

# ============================================================
#  页面一：舆情数据看板
# ============================================================
if page == "📊 舆情数据看板":
    st.title("📊 舆情数据看板")
    st.markdown("实时监控电商评论的情感趋势与高频关注焦点")

    if "sentiment_scores" not in st.session_state:
        with st.spinner("正在分析评论情感..."):
            sentiments = []
            categories = []
            for text in df["评论内容"]:
                sentiments.append(analyze_sentiment(text))
                cats = classify_business_category(text)
                categories.append(cats[0] if cats else "其他")
            df["sentiment_score"] = sentiments
            df["business_category"] = categories
            st.session_state["sentiment_scores"] = True

    # 宏观指标卡
    stats = get_summary_stats(df)
    st.markdown(
        create_metrics_html(
            total=stats["total"],
            positive_rate=stats.get("positive_sentiment_rate", 0),
            negative_count=stats.get("negative_alert_count", 0),
        ),
        unsafe_allow_html=True,
    )

    # 图表行
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🔝 高频词 TOP 10")
        word_fig = create_word_frequency_chart(df["评论内容"].tolist(), top_n=10)
        st.plotly_chart(word_fig, use_container_width=True)

    with col_right:
        st.subheader("📈 业务维度分布")
        if "business_category" in df.columns:
            pie_fig = create_pie_chart(df["business_category"].tolist())
            st.plotly_chart(pie_fig, use_container_width=True)

    with st.expander("📋 查看原始数据"):
        st.dataframe(df)

# ============================================================
#  页面二：单句交互分析
# ============================================================
if page == "🔍 单句交互分析":
    st.title("🔍 单句交互分析")
    st.markdown("输入任意一条买家评论，实时获取分词、实体识别和分类结果")

    user_input = st.text_area(
        "✏️ 请输入评论内容：",
        placeholder="例如：昨天在上海买的 iPhone，屏幕竟然碎了，气死我了！",
        height=100,
    )

    if st.button("🚀 智能分析", key="analyze_btn"):
        if not user_input.strip():
            st.warning("请输入评论内容后再点击分析")
        else:
            st.markdown("---")

            # ---- 分词与词性标注 ----
            st.subheader("1️⃣ 分词与词性标注")
            word_pos_pairs = tokenize_with_pos(user_input)
            filtered_pairs = filter_content_words(word_pos_pairs)

            if filtered_pairs:
                pos_colors = {
                    "n": "#4CAF50", "nr": "#4CAF50", "ns": "#4CAF50",
                    "v": "#2196F3", "vn": "#2196F3",
                    "a": "#FF9800", "an": "#FF9800",
                    "eng": "#9C27B0",
                }

                html_tags = ""
                for word, pos in filtered_pairs:
                    color = pos_colors.get(pos, "#757575")
                    html_tags += (
                        f'<span style="display:inline-block;'
                        f'background:{color};color:white;padding:4px 8px;'
                        f'border-radius:6px;margin:3px;font-size:14px;">'
                        f'{word}<small style="opacity:0.8">({pos})</small>'
                        f'</span>'
                    )
                st.markdown(html_tags, unsafe_allow_html=True)
                st.caption("颜色说明：🟢 名词 🟠 形容词 🔵 动词 🟣 英文/品牌词")
            else:
                st.info("未提取到核心词汇")

            # ---- 命名实体识别 ----
            st.subheader("2️⃣ 命名实体识别 (NER)")
            entities = extract_entities(user_input)

            if entities:
                loc_entities = [e[0] for e in entities if e[1] == "LOC"]
                prod_entities = [e[0] for e in entities if e[1] == "PRODUCT"]

                col_ner1, col_ner2 = st.columns(2)
                with col_ner1:
                    st.markdown("📍 **地名/机构名 (LOC)**")
                    if loc_entities:
                        for loc in loc_entities:
                            st.success(f"📍 {loc}")
                    else:
                        st.caption("未检测到地名")

                with col_ner2:
                    st.markdown("📱 **产品/品牌名 (PRODUCT)**")
                    if prod_entities:
                        for prod in prod_entities:
                            st.info(f"📱 {prod}")
                    else:
                        st.caption("未检测到产品名")
            else:
                st.info("未检测到命名实体")

            # ---- 双重分类 ----
            st.subheader("3️⃣ 双重文本分类")

            col_cls1, col_cls2 = st.columns(2)

            with col_cls1:
                st.markdown("**😊 情感极性判定**")
                sentiment_score = analyze_sentiment(user_input)
                gauge_fig = create_sentiment_gauge(sentiment_score)
                st.plotly_chart(gauge_fig, use_container_width=True)

                if sentiment_score < 0.4:
                    st.error("🚨 **差评预警！** 该评论情感极低，建议商户立即跟进处理")
                elif sentiment_score >= 0.7:
                    st.success("✅ 该评论为正面好评")
                else:
                    st.warning("⚠️ 该评论情感偏中性")

            with col_cls2:
                st.markdown("**🏷️ 业务维度归类**")
                categories = classify_business_category(user_input)
                for cat in categories:
                    if "投诉" in cat:
                        st.error(f"⚠️ {cat}")
                    elif "反馈" in cat or "体验" in cat:
                        st.info(f"📝 {cat}")
                    else:
                        st.markdown(f"- {cat}")

# ============================================================
#  页面三：批量评论聚类语义地图
# ============================================================
if page == "🧠 批量聚类语义地图":
    st.title("🧠 批量评论聚类语义地图")
    st.markdown(
        "基于 TF-IDF + KMeans 的无监督文本聚类，利用 PCA 降维至二维空间进行可视化"
    )

    k_value = st.slider(
        "🎚️ 聚类主题数 K (将评论分为几个话题簇?)",
        min_value=2,
        max_value=5,
        value=3,
        step=1,
        help="拖动滑块调整聚类数量，系统将自动重新计算",
    )

    texts = df["评论内容"].tolist()

    if st.button("🚀 开始聚类分析", key="cluster_btn"):
        if len(texts) < k_value:
            st.error(f"评论文本数量 ({len(texts)}) 不能少于 K 值 ({k_value})")
        else:
            with st.spinner(f"正在运行 KMeans (K={k_value}) 聚类 + PCA 降维..."):
                result = cluster_pipeline(texts, k=k_value)

            st.success(
                f"✅ 聚类完成！已将 {len(texts)} 条评论划分为 {k_value} 个语义主题簇"
            )

            # Plotly 语义散点图
            st.subheader("🗺️ 聚类语义地图")
            st.caption("💡 将鼠标悬停在任意数据点上即可查看原始买家评论文本")

            scatter_fig = create_cluster_scatter(
                texts=result["texts"],
                labels=result["labels"],
                coordinates=result["coordinates"],
            )
            st.plotly_chart(scatter_fig, use_container_width=True)

            # 明细下钻
            st.subheader("📂 各聚类主题明细下钻")

            clusters = {}
            for i, label in enumerate(result["labels"]):
                clusters.setdefault(label, []).append(texts[i])

            for label in sorted(clusters.keys()):
                cluster_texts = clusters[label]
                with st.expander(
                    f"🔖 主题 {label + 1} — {len(cluster_texts)} 条评论",
                    expanded=False,
                ):
                    for j, t in enumerate(cluster_texts, 1):
                        st.markdown(f"**{j}.** {t}")

            # 聚类统计
            st.subheader("📊 各主题评论数量分布")
            cluster_counts = {
                f"主题 {label + 1}": len(cluster_texts)
                for label, cluster_texts in clusters.items()
            }
            count_df = pd.DataFrame(
                list(cluster_counts.items()),
                columns=["主题", "评论数"],
            )
            st.bar_chart(count_df.set_index("主题"))

# ---- 页脚 ----
st.markdown("---")
st.markdown(
    '<div class="footer">'
    "🛒 电商评论与舆情智能分析系统 | NLP Final Project | "
    "Built with Streamlit &amp; Jieba &amp; Scikit-learn"
    "</div>",
    unsafe_allow_html=True,
)
