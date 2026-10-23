"""语言检测与 embedding 模型选择测试。"""

from book_rag.rag.language import choose_embedding_model, detect_language, normalize_language

ZH_TEXT = (
    "清晨的雾气笼罩着山谷，旅人背起行囊踏上了寻找答案的道路。"
    "森林深处没有路标，只有风声与鸟鸣指引着方向。"
    "他在溪边停下脚步，思考着接下来应该往东走还是继续向北。"
)

EN_TEXT = (
    "The morning fog covered the valley as the traveler shouldered his pack "
    "and set out on the long road in search of answers. Deep in the forest "
    "there were no signposts, only the sound of wind and birdsong to guide him."
)


def test_detect_language_chinese():
    """中文文本应检测为 zh（langdetect 返回 zh-cn / zh-tw，经归一化后为 zh）。"""
    assert normalize_language(detect_language(ZH_TEXT)) == "zh"


def test_detect_language_english():
    assert detect_language(EN_TEXT) == "en"


def test_detect_language_empty_returns_unknown():
    """空文本 / 纯空白文本检测失败时返回 unknown。"""
    assert detect_language("") == "unknown"
    assert detect_language("   \n  ") == "unknown"


def test_detect_language_short_chinese_word():
    """短中文词不应被 langdetect 误判为韩语，应直接判为 zh。"""
    assert detect_language("招聘") == "zh"


def test_detect_language_short_chinese_question():
    """带标点的短中文问句不应被误判为韩语。"""
    assert normalize_language(detect_language("咖啡怎么冲泡才香？")) == "zh"


def test_detect_language_japanese_kana():
    """含假名的文本应判定为日语 ja（优先于汉字判断）。"""
    assert detect_language("これは日本語のテストです") == "ja"


def test_detect_language_korean_hangul():
    """含谚文的文本应判定为韩语 ko。"""
    assert detect_language("안녕하세요 반갑습니다") == "ko"


def test_normalize_language():
    """带区域后缀的语言码应归一化为主语言码。"""
    assert normalize_language("zh-cn") == "zh"
    assert normalize_language("zh-tw") == "zh"
    assert normalize_language("en-us") == "en"
    assert normalize_language("en") == "en"
    assert normalize_language("ZH-CN") == "zh"


def test_choose_embedding_model_is_always_bge_m3():
    """S3 起向量模型统一为 bge-m3：任何语言都不再路由到专用模型。"""
    for language in ("zh", "zh-cn", "en", "ja", "ko", "unknown", ""):
        assert choose_embedding_model(language) == "BAAI/bge-m3"
