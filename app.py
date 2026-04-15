
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model", "model.pkl")
columns_path = os.path.join(BASE_DIR, "model", "columns.pkl")
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from streamlit_option_menu import option_menu

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Risk Analyzer", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
}

h1, h2, h3 {
    color: #2c3e50;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD ----------------
df = pd.read_csv("data/project_data.csv")
model = joblib.load(model_path)
columns = joblib.load(columns_path)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    selected = option_menu(
        "Risk Analyzer",
        ["Dashboard", "Predict", "Insights"],
        icons=["bar-chart", "search", "graph-up"],
        menu_icon="rocket",
        default_index=0,
    )

# ---------------- DASHBOARD ----------------
if selected == "Dashboard":
    st.title("📊 Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"""
    <div class="card">
    <h4>Total Projects</h4>
    <h2>{len(df)}</h2>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="card">
    <h4>High Risk</h4>
    <h2>{(df['risk_level']=='High').sum()}</h2>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="card">
    <h4>Low Risk</h4>
    <h2>{(df['risk_level']=='Low').sum()}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Risk Distribution")

    fig = px.bar(
        df['risk_level'].value_counts().reset_index(),
        x='index',
        y='risk_level',
        color='index',
        title="Projects by Risk Level"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- PREDICT ----------------
elif selected == "Predict":
    st.title("🔍 Predict Risk")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        budget = st.number_input("Budget", 10000)
        duration = st.number_input("Duration", 1)
        team_size = st.number_input("Team Size", 1)

    with col2:
        complexity = st.selectbox("Complexity", ["Low","Medium","High"])
        stakeholder = st.selectbox("Stakeholder", ["Poor","Average","Good"])
        past_risk = st.number_input("Past Risk", 0)

    if st.button("Predict Risk"):
        input_data = pd.DataFrame([{
            "budget": budget,
            "duration": duration,
            "team_size": team_size,
            "complexity": complexity,
            "stakeholder_engagement": stakeholder,
            "past_risk_incidents": past_risk
        }])

        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=columns, fill_value=0)

        pred = model.predict(input_data)[0]

        result = {0:"Low",1:"Medium",2:"High"}[pred]

        if result == "High":
            st.error("⚠️ High Risk Project")
        elif result == "Medium":
            st.warning("⚠️ Medium Risk")
        else:
            st.success("✅ Low Risk")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- INSIGHTS ----------------
elif selected == "Insights":
    st.title("📈 Insights")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="duration", title="Project Duration")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            df,
            x="budget",
            y="past_risk_incidents",
            color="risk_level",
            title="Budget vs Risk"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df)