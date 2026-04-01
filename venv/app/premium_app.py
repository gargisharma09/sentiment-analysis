import streamlit as st
import requests
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="AI Sentiment Dashboard",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# PREMIUM CSS
# -----------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.title {
    font-size: 50px;
    font-weight: bold;
    color: #00FFD1;
    text-align: center;
}

.card {
    padding: 25px;
    border-radius: 15px;
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 30px rgba(0,0,0,0.3);
    text-align: center;
}

.result-positive {
    background-color: #00ff88;
}

.result-negative {
    background-color: #ff4b4b;
}

.result-neutral {
    background-color: #ffaa00;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.markdown('<div class="title">💬 AI Sentiment Intelligence</div>', unsafe_allow_html=True)
st.write("Analyze social media comments with advanced AI")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Dashboard Controls")
show_chart = st.sidebar.checkbox("Show Confidence Chart", True)
show_history = st.sidebar.checkbox("Show History", True)

# -----------------------------
# INPUT
# -----------------------------
user_input = st.text_area("✍️ Enter your comment", height=120)

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# ANALYZE BUTTON
# -----------------------------
if st.button("🚀 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter text")
    else:
        response = requests.post(API_URL, json={"text": user_input})

        if response.status_code == 200:
            result = response.json()

            sentiment = result["prediction"]
            confidence = result["confidence"]
            slang_info = result["slang_interpretation"]
            scores = result["all_scores"]

            # Color logic
            if "Positive" in sentiment:
                color_class = "result-positive"
            elif "Negative" in sentiment:
                color_class = "result-negative"
            else:
                color_class = "result-neutral"

            col1, col2 = st.columns(2)

            # -----------------------------
            # RESULT CARD
            # -----------------------------
            with col1:
                st.markdown(f"""
                <div class="card {color_class}">
                    <h2>{sentiment}</h2>
                    <p>Confidence: {confidence}</p>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("🧠 Slang Interpretation")
                st.write(slang_info)

            # -----------------------------
            # CHART
            # -----------------------------
            with col2:
                if show_chart:
                    chart_data = pd.DataFrame({
                        "Sentiment": ["Negative", "Neutral", "Positive", "Sarcasm"],
                        "Score": scores
                    })
                    st.bar_chart(chart_data.set_index("Sentiment"))

            # Save history
            st.session_state.history.append({
                "Text": user_input,
                "Sentiment": sentiment,
                "Confidence": confidence
            })

        else:
            st.error("API Error")

# -----------------------------
# HISTORY
# -----------------------------
if show_history and len(st.session_state.history) > 0:
    st.subheader("📜 Prediction History")
    history_df = pd.DataFrame(st.session_state.history[::-1])
    st.dataframe(history_df)