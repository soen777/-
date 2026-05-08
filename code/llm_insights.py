"""
LLM 驱动的电商分析洞察生成模块。
调用大模型 API 为每个分析环节自动生成业务解读和运营建议。

默认使用 DeepSeek API（国内可直接访问，价格便宜）
也支持任何 OpenAI 兼容接口（通义千问、智谱GLM、Moonshot等）
"""

import os
import hashlib
import streamlit as st

# ── 缓存 ──────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "result", "insights")


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(insight_type, data_hash):
    _ensure_cache_dir()
    return os.path.join(CACHE_DIR, f"{insight_type}_{data_hash}.txt")


def _hash(data_str):
    return hashlib.md5(data_str.encode()).hexdigest()[:12]


def _read_cache(insight_type, data_hash):
    path = _cache_path(insight_type, data_hash)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def _write_cache(insight_type, data_hash, content):
    with open(_cache_path(insight_type, data_hash), "w", encoding="utf-8") as f:
        f.write(content)


# ── API 调用 ───────────────────────────────────────────

SYSTEM_PROMPT = """你是一位资深的电商数据分析师，擅长从数据中提炼业务洞察。
你的回复要求：
1. 基于数据说话，引用具体数字
2. 给出可落地的运营建议，而非空泛结论
3. 简洁有力，3-5个要点即可
4. 使用中文，Markdown 格式"""


def _call_llm(client, model, user_prompt):
    """调用 LLM 并返回文本结果。"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=800,
    )
    return response.choices[0].message.content


# ── 各模块 Prompt 模板 ────────────────────────────────

def _funnel_prompt(report_df):
    lines = ["当前电商转化漏斗数据："]
    for _, row in report_df.iterrows():
        lines.append(f"- {row.to_dict()}")
    lines.append("""
请分析这个转化漏斗，按以下结构输出：

**核心发现**
- 2-3个关键发现，引用具体转化率数字

**最大瓶颈**
- 哪个环节流失最严重，可能的原因是什么

**优化建议**
- 2-3条具体可落地的优化措施
""")
    return "\n".join(lines)


def _rfm_prompt(rfm_df):
    counts = rfm_df["user_type_name"].value_counts().to_dict()
    total = sum(counts.values())
    stats = {}
    for utype in rfm_df["user_type_name"].unique():
        g = rfm_df[rfm_df["user_type_name"] == utype]
        stats[utype] = {
            "人数": len(g),
            "占比": f"{len(g) / total * 100:.1f}%",
            "平均R值(天)": round(g["R"].mean(), 1),
            "平均F值(次)": round(g["F"].mean(), 1),
        }

    import json
    return f"""RFM + KMeans 用户分层结果：

总用户数：{total}
各层分布：{json.dumps(counts, ensure_ascii=False)}

各层详细统计：
{json.dumps(stats, ensure_ascii=False, indent=2)}

说明：R = 最近一次购买距今天数（越小越好），F = 累计购买次数（越大越好）

请按以下结构输出：

**用户结构诊断**
- 当前用户结构健康度评价

**分层运营策略**
- 针对每类用户的具体运营动作

**优先级建议**
- 哪类用户最值得优先投入资源
"""


def _association_prompt(rules_df):
    lines = [f"FP-Growth 商品关联规则挖掘结果：共 {len(rules_df)} 条规则\n"]
    lines.append("提升度最高的 5 条规则：")
    for _, row in rules_df.head(5).iterrows():
        lines.append(
            f"- {row.get('antecedents', '')} → {row.get('consequents', '')}"
            f"  | 置信度: {row.get('confidence', 0):.2%}"
            f"  | 提升度: {row.get('lift', 0):.1f}"
        )
    lines.append("""
说明：提升度 > 1 表示关联强于随机，> 3 即为强关联

请按以下结构输出：

**关联发现**
- 发现了哪些有意义的商品关联关系

**交叉销售建议**
- 2-3个具体的捆绑销售或推荐策略

**落地方式**
- 在哪些场景可以应用这些规则（如购物车推荐、详情页等）
""")
    return "\n".join(lines)


def _sales_prompt(sales_df):
    lines = ["日销量预测数据概要："]
    if "sales" in sales_df.columns:
        s = sales_df["sales"]
        lines.append(f"- 平均日销量: {s.mean():.0f}")
        lines.append(f"- 最高日销量: {s.max():.0f}")
        lines.append(f"- 最低日销量: {s.min():.0f}")
        if len(s) >= 7:
            recent = s.tail(7).tolist()
            lines.append(f"- 最近7日趋势: {[int(x) for x in recent]}")
    lines.append("- 模型: 线性回归（含周末效应特征）")
    lines.append("""
请按以下结构输出：

**销量趋势**
- 从数据中看到的规律或模式

**运营建议**
- 2-3条基于预测的经营建议

**注意事项**
- 值得关注的风险点或不确定性
""")
    return "\n".join(lines)


# ── 公开接口 ──────────────────────────────────────────

def generate_insights(insight_type, df, api_key, model, base_url, force=False):
    """
    生成 AI 洞察（自动缓存）。

    参数:
        insight_type: "funnel" | "rfm" | "association" | "sales"
        df: pandas DataFrame（对应分析结果数据）
        api_key: API 密钥
        model: 模型名称
        base_url: API 地址
        force: 强制刷新，忽略缓存

    返回:
        str: Markdown 格式的业务洞察文本
    """
    data_str = df.to_csv(index=False)
    data_hash = _hash(data_str)

    if not force:
        cached = _read_cache(insight_type, data_hash)
        if cached:
            return cached

    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return "❌ 请先安装 openai 包：`pip install openai`"

    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt_funcs = {
        "funnel": _funnel_prompt,
        "rfm": _rfm_prompt,
        "association": _association_prompt,
        "sales": _sales_prompt,
    }

    prompt = prompt_funcs[insight_type](df)

    try:
        result = _call_llm(client, model, prompt)
        _write_cache(insight_type, data_hash, result)
        return result
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            return "❌ API Key 无效，请检查后重试。"
        elif "429" in error_msg:
            return "❌ API 调用频率超限，请稍后重试。"
        elif "timeout" in error_msg.lower():
            return "❌ API 请求超时，请检查网络后重试。"
        else:
            return f"❌ 生成洞察时出错：{error_msg}"
