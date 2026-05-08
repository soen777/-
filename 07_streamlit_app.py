import streamlit as st
import pandas as pd

st.set_page_config(page_title="电商数据分析看板", layout="wide")
st.title("📊 电商全链路数据分析项目（大一实习版）")

# 1. 基础数据概览
st.subheader("1. 电商核心转化漏斗")
report = pd.read_csv("result/basic_report.csv")
st.dataframe(report)

# 2. 用户分层
st.subheader("2. RFM+K-Means用户分层")
rfm = pd.read_csv("result/rfm_user.csv")
st.dataframe(rfm.head(10))
st.write("用户分布：", rfm['user_type_name'].value_counts())

# 3. 商品关联推荐
st.subheader("3. 商品关联推荐（买A必买B）")
rules = pd.read_csv("result/fpgrowth_rules.csv")
st.dataframe(rules.head(10))

# 4. 销量预测
st.subheader("4. 日销量预测（线性回归）")
sales = pd.read_csv("result/sales_pred.csv")
st.line_chart(sales.set_index('date')['sales'])
st.success("✅ 项目运行完成！")
