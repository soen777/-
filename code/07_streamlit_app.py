import streamlit as st
import pandas as pd
from llm_insights import generate_insights

st.set_page_config(page_title="电商数据分析看板", layout="wide")
st.title("电商全链路数据分析项目")
st.caption("淘宝用户行为分析 · 漏斗转化 · 用户分层 · 关联规则 · 销量预测")

# ── 侧边栏：API 配置 ──────────────────────────────────
with st.sidebar:
    st.header("🤖 AI 洞察配置")

    api_key = st.text_input(
        "API Key",
        type="password",
        value=st.session_state.get("api_key", ""),
        help="DeepSeek 可在 platform.deepseek.com 免费获取",
        placeholder="sk-xxxxxxxxxxxxxxxx",
    )
    if api_key:
        st.session_state["api_key"] = api_key

    base_url = st.text_input(
        "API 地址",
        value=st.session_state.get("base_url", "https://api.deepseek.com"),
        help="DeepSeek 默认地址，也支持其他 OpenAI 兼容接口",
    )
    st.session_state["base_url"] = base_url

    model = st.selectbox(
        "模型",
        ["deepseek-chat", "deepseek-reasoner", "gpt-4o-mini", "qwen-plus", "glm-4-flash"],
        index=0,
        help="deepseek-chat 性价比最高，推荐使用",
    )

    st.divider()

    force_refresh = st.checkbox("强制刷新所有洞察", value=False, help="勾选后将重新调用 API 生成洞察")

    st.divider()
    st.caption("💡 没有 API Key？")
    st.caption("访问 [platform.deepseek.com](https://platform.deepseek.com) 注册即送免费额度")

# ── 加载数据 ──────────────────────────────────────────
report = pd.read_csv("../result/basic_report.csv")
rfm = pd.read_csv("../result/rfm_user.csv")
rules = pd.read_csv("../result/fpgrowth_rules.csv")
sales = pd.read_csv("../result/sales_pred.csv")

# ── 全部生成按钮 ──────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col2:
    generate_all = st.button("🤖 一键生成全部洞察", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════
# 1. 漏斗分析
# ═══════════════════════════════════════════════════════
st.header("1. 电商核心转化漏斗")
st.dataframe(report, use_container_width=True)

with st.expander("🤖 AI 洞察：漏斗分析解读", expanded=False):
    btn_col, _ = st.columns([1, 5])
    with btn_col:
        trigger = st.button("生成洞察", key="btn_funnel") or generate_all

    if trigger:
        with st.spinner("AI 正在分析漏斗数据..."):
            insight = generate_insights(
                "funnel", report,
                api_key=api_key, model=model, base_url=base_url,
                force=force_refresh,
            )
            if insight is None:
                st.warning("请在侧边栏填写 API Key 后重试")
            else:
                st.session_state["insight_funnel"] = insight

    if "insight_funnel" in st.session_state:
        st.markdown(st.session_state["insight_funnel"])

# ═══════════════════════════════════════════════════════
# 2. 用户分层
# ═══════════════════════════════════════════════════════
st.header("2. RFM + K-Means 用户分层")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("用户分布")
    dist = rfm["user_type_name"].value_counts()
    st.bar_chart(dist)
with col_b:
    st.subheader("样本数据")
    st.dataframe(rfm.head(10), use_container_width=True)

with st.expander("🤖 AI 洞察：用户分层运营策略", expanded=False):
    btn_col, _ = st.columns([1, 5])
    with btn_col:
        trigger = st.button("生成洞察", key="btn_rfm") or generate_all

    if trigger:
        with st.spinner("AI 正在分析用户分层..."):
            insight = generate_insights(
                "rfm", rfm,
                api_key=api_key, model=model, base_url=base_url,
                force=force_refresh,
            )
            if insight is None:
                st.warning("请在侧边栏填写 API Key 后重试")
            else:
                st.session_state["insight_rfm"] = insight

    if "insight_rfm" in st.session_state:
        st.markdown(st.session_state["insight_rfm"])

# ═══════════════════════════════════════════════════════
# 3. 商品关联规则
# ═══════════════════════════════════════════════════════
st.header("3. 商品关联规则（买 A 必买 B）")
st.dataframe(rules.head(10), use_container_width=True)
st.caption(f"共 {len(rules)} 条规则，按提升度排序。提升度 > 1 表示关联强于随机，> 3 为强关联。")

with st.expander("🤖 AI 洞察：交叉销售建议", expanded=False):
    btn_col, _ = st.columns([1, 5])
    with btn_col:
        trigger = st.button("生成洞察", key="btn_assoc") or generate_all

    if trigger:
        with st.spinner("AI 正在分析关联规则..."):
            insight = generate_insights(
                "association", rules,
                api_key=api_key, model=model, base_url=base_url,
                force=force_refresh,
            )
            if insight is None:
                st.warning("请在侧边栏填写 API Key 后重试")
            else:
                st.session_state["insight_assoc"] = insight

    if "insight_assoc" in st.session_state:
        st.markdown(st.session_state["insight_assoc"])

# ═══════════════════════════════════════════════════════
# 4. 销量预测
# ═══════════════════════════════════════════════════════
st.header("4. 日销量预测（线性回归）")

if "date" in sales.columns:
    chart_data = sales.set_index("date")["sales"]
else:
    chart_data = sales["sales"]

st.line_chart(chart_data)

with st.expander("🤖 AI 洞察：销量趋势与运营建议", expanded=False):
    btn_col, _ = st.columns([1, 5])
    with btn_col:
        trigger = st.button("生成洞察", key="btn_sales") or generate_all

    if trigger:
        with st.spinner("AI 正在分析销量趋势..."):
            insight = generate_insights(
                "sales", sales,
                api_key=api_key, model=model, base_url=base_url,
                force=force_refresh,
            )
            if insight is None:
                st.warning("请在侧边栏填写 API Key 后重试")
            else:
                st.session_state["insight_sales"] = insight

    if "insight_sales" in st.session_state:
        st.markdown(st.session_state["insight_sales"])

# ── 页脚 ──────────────────────────────────────────────
st.divider()
st.success("✅ 项目运行完成 — 电商全链路数据分析")
