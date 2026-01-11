import streamlit as st
import pickle

# Page config
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Custom CSS styling
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.main-card {
    background-color: white;
    padding: 30px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
.footer {
    position: fixed;
    bottom: 10px;
    right: 20px;
    font-size: 14px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# Main UI
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.title("📰 AI-Based Fake News Detection System")
st.write("Enter a news article below to check whether it is **REAL** or **FAKE**.")

news_text = st.text_area("📝 News Text", height=180)

if st.button("🔍 Check News"):
    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:
        data = vectorizer.transform([news_text])
        prediction = model.predict(data)[0]

        if prediction == "FAKE":
            st.error("❌ This news is predicted as **FAKE**")
        else:
            st.success("✅ This news is predicted as **REAL**")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <b>Mohitha Nandini</b> | 23B01A4582
</div>
""", unsafe_allow_html=True)
