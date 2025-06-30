import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model
with open('survival_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load data dictionary
data_dict = pd.read_csv('data_dictionary.csv')

st.title("Post-HCT Survival Prediction App")
st.markdown("Predict event-free survival (EFS) and get personalized suggestions to improve outcomes.")

input_data = {}
st.header("📝 Patient Input")

for _, row in data_dict.iterrows():
    var = row['variable']
    desc = row['description']
    dtype = row['type']
    allowed = row['values']

    help_text = desc if pd.notna(desc) else "Provide input"

    if dtype == "Numerical":
        input_data[var] = st.number_input(var, help=help_text, step=1.0)
    elif dtype == "Categorical":
        if pd.notna(allowed):
            try:
                choices = eval(allowed)
                choices = [str(c) for c in choices if str(c) != 'nan']
                if len(choices) == 2 and set(choices) <= {"Yes", "No", "True", "False"}:
                    input_data[var] = st.radio(var, choices, help=help_text)
                else:
                    input_data[var] = st.selectbox(var, choices, help=help_text)
            except:
                input_data[var] = st.text_input(var, help=help_text)
        else:
            input_data[var] = st.text_input(var, help=help_text)
    else:
        input_data[var] = st.text_input(var, help=help_text)

if st.button("🔍 Predict Survival"):
    input_df = pd.DataFrame([input_data])

    try:
        prediction = model.predict_proba(input_df)[0][1] * 100  # Probability for positive class
        rounded = round(prediction, 2)

        if prediction < 25:
            st.error(f"Survival Score: {rounded}% ❌ (High Risk)")
        elif prediction < 50:
            st.warning(f"Survival Score: {rounded}% ⚠️ (Moderate Risk)")
        elif prediction < 85:
            st.info(f"Survival Score: {rounded}% 🟡 (Manageable Risk)")
        else:
            st.success(f"Survival Score: {rounded}% ✅ (Good Prognosis)")

        # --- AI Suggestions ---
        st.subheader("🤖 AI Suggestions to Improve Survival")
        suggestions = []

        if input_data.get("age", 0) > 60:
            suggestions.append("- Consider reduced-intensity conditioning for age > 60.")
        if input_data.get("hla_match_c_high") not in ["Matched", "8/8"]:
            suggestions.append("- Try improving HLA match with alternative donor search.")
        if input_data.get("cmv_status") == "Positive":
            suggestions.append("- CMV prophylaxis and monitoring recommended.")
        if input_data.get("dri_score") == "High":
            suggestions.append("- High disease risk. Discuss clinical trial options.")

        if not suggestions:
            st.markdown("✅ No high-risk features detected. Continue current protocol.")
        else:
            for s in suggestions:
                st.markdown(s)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
