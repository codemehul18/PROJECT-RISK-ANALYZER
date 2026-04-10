import matplotlib.pyplot as plt
import pandas as pd

import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing
model = joblib.load("model/risk_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

st.subheader("📊 Feature Importance")

try:
    importances = model.feature_importances_

    # Use dummy feature names (safe fallback)
    feature_names = [f"Feature {i}" for i in range(len(importances))]

    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.barh(df_importance["Feature"], df_importance["Importance"])
    ax.invert_yaxis()

    st.pyplot(fig)

except Exception as e:
    st.error(f"Error: {e}")

st.title("📊 Project Risk Prediction System")

st.write("Enter project details to predict risk level")

# Inputs
budget = st.number_input("Budget", min_value=10000, max_value=1000000, value=50000)
duration = st.number_input("Duration (months)", min_value=1, max_value=36, value=12)
team_size = st.number_input("Team Size", min_value=1, max_value=50, value=5)

complexity = st.selectbox("Complexity", ["Low", "Medium", "High"])
stakeholder_engagement = st.selectbox("Stakeholder Engagement", ["Poor", "Average", "Good"])
past_risk_incidents = st.number_input("Past Risk Incidents", min_value=0, max_value=10, value=1)

# Convert to dataframe
input_data = pd.DataFrame({
    "budget": [budget],
    "duration": [duration],
    "team_size": [team_size],
    "complexity": [complexity],
    "stakeholder_engagement": [stakeholder_engagement],
    "past_risk_incidents": [past_risk_incidents]
})

# Encode categorical
input_data = pd.get_dummies(input_data)

# Align columns with training data
model_features = scaler.feature_names_in_

input_data = input_data.reindex(columns=model_features, fill_value=0)
# Scale
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Risk"):
    prediction = model.predict(input_scaled)

    risk_map = {0: "Low", 1: "Medium", 2: "High"}
    result = risk_map[prediction[0]]

    st.success(f"Predicted Risk Level: {result}")