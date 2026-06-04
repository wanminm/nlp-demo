"""NLP 引擎：分词、词性标注、命名实体识别、情感分析、业务分类."""

import jieba
import jieba.posseg as pseg
from typing import List, Tuple, Dict
from snownlp import SnowNLP


# ---- 中文停用词表 (精简版) ----
_STOP_WORDS = set(
    [
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
        "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
        "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
        "所", "为", "因为", "所以", "但是", "虽然", "如果", "可以", "还是",
        "又", "再", "能", "吧", "吗", "呢", "啊", "哦", "嗯", "哈", "呀",
        "什么", "怎么", "怎样", "哪", "哪里", "哪个", "多少", "几", "非常",
        "比较", "更加", "相当", "太", "最", "极", "有点", "一些", "一点",
        "这个", "那个", "这里", "那里", "已经", "曾经", "将", "正在",
        "并", "而", "但", "或", "且", "与", "及", "之", "以", "从",
        "更", "过", "还", "让", "被", "把", "向", "对", "对于", "关于",
        "只", "不止", "除了", "以及", "不仅", "不管", "无论",
        "时候", "时候的", "出来", "起来", "的话", "等等", "就是",
        "比如", "好像", "真的", "只是", "其实",
        " ", "\t", "\n", "\r", "　",
    ]
)

# 词性保留集合：名词、动词、形容词、英文、成语、简称
_KEEP_POS = {"n", "nr", "ns", "nt", "nz", "v", "vn", "a", "an", "eng", "i", "j"}


def tokenize_with_pos(text: str) -> List[Tuple[str, str]]:
    """对文本进行分词并返回 (词, 词性) 列表."""
    if not text or not text.strip():
        return []
    words = pseg.cut(text.strip())
    return [(w.word, w.flag) for w in words]


def filter_content_words(
    word_pos_pairs: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """过滤停用词和非核心词性，保留名词/动词/形容词等."""
    return [
        (w, pos)
        for w, pos in word_pos_pairs
        if w not in _STOP_WORDS
        and len(w.strip()) > 0
        and pos in _KEEP_POS
    ]


# ---- 命名实体识别 ----
# 地名后缀词表
_LOCATION_SUFFIX = {"省", "市", "区", "县", "镇", "州", "路", "街", "广场", "大厦"}
# 常见电子消费品品牌/产品词
_PRODUCT_KEYWORDS = {
    "iphone", "ipad", "macbook", "mac", "apple", "华为", "小米", "oppo",
    "vivo", "三星", "samsung", "sony", "索尼", "联想", "lenovo", "dell",
    "戴尔", "hp", "惠普", "华硕", "asus", "acer", "宏碁", "surface",
    "kindle", "airpods", "pixel", "oneplus", "一加", "realme", "荣耀",
    "honor", "watch", "band", "buds", "phone", "手机", "电脑", "平板",
    "耳机", "手表", "充电器", "数据线", "键盘", "鼠标", "显示器",
}


def extract_entities(text: str) -> List[Tuple[str, str]]:
    """从文本中提取命名实体：LOC（地名）和 PRODUCT（产品/品牌）."""
    if not text or not text.strip():
        return []

    entities: List[Tuple[str, str]] = []
    words_with_pos = tokenize_with_pos(text)

    for word, pos in words_with_pos:
        # 利用 ns(地名) 词性标签
        if pos == "ns":
            entities.append((word, "LOC"))
        # 检查地名后缀
        elif any(word.endswith(suffix) for suffix in _LOCATION_SUFFIX):
            entities.append((word, "LOC"))
        # 检查已知品牌/产品词
        elif word.lower() in _PRODUCT_KEYWORDS:
            entities.append((word, "PRODUCT"))
        # 检查英文词作为潜在品牌名
        elif pos == "eng" and len(word) > 1:
            entities.append((word, "PRODUCT"))

    return entities


# ---- 情感分析 ----
def analyze_sentiment(text: str) -> float:
    """使用 SnowNLP 返回 0~1 之间的情感得分。接近 1 = 正面，接近 0 = 负面."""
    if not text or not text.strip():
        return 0.5
    s = SnowNLP(text.strip())
    return round(float(s.sentiments), 4)


# ---- 业务维度关键词规则 ----
_CATEGORY_RULES: Dict[str, List[str]] = {
    "物流配送类投诉": [
        "快递", "物流", "配送", "发货", "收货", "包裹", "运输",
        "速度慢", "送货", "等了", "送来", "寄", "收件",
    ],
    "产品质量投诉": [
        "质量", "坏了", "碎", "划痕", "二手", "瑕疵", "破损",
        "屏幕", "电池", "充电", "发热", "卡顿", "死机", "故障",
        "色差", "变形", "掉色", "脱线",
    ],
    "客服售后服务反馈": [
        "客服", "售后", "态度", "回复", "退款", "退货", "换货",
        "保价", "投诉", "处理", "解决", "补偿", "赔偿",
    ],
    "性价比与使用体验": [
        "性价比", "便宜", "贵", "降价", "优惠", "赠品", "活动",
        "推荐", "值得", "好用", "实用", "方便",
    ],
}


def classify_business_category(text: str) -> List[str]:
    """根据关键词规则将评论归类到业务维度。返回匹配的类别列表（多标签）."""
    if not text or not text.strip():
        return ["未分类"]

    matched: List[str] = []
    for category, keywords in _CATEGORY_RULES.items():
        score = sum(1 for kw in keywords if kw in text)
        if score >= 1:
            matched.append(category)

    if not matched:
        matched.append("其他反馈")

    return matched
