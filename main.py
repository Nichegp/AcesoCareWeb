import streamlit as st
from streamlit.elements.lib.layout_utils import Height
import Heart_disease_predictor
import DiseasePredictor
import parkinsons
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Aceso Care",
    page_icon="🏥",
    layout="wide"
)

# --------------------------------------------------
# Global CSS – remove wasted space
# --------------------------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 0.5rem;
    padding-bottom: 0rem;
}
button[data-baseweb="tab"] > div {
    font-size: 20px;
    font-weight:200;
}
h1 {
    margin-bottom: 0.1rem;
}

h3 {
    margin-top: 0rem;
    margin-bottom: 0.2rem;
}

[data-baseweb="tab-list"] {
    margin-top: 0rem;
    font-size:40px
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
img {
    max-height: 300px;
    object-fit: cover;
}
</style>
""", unsafe_allow_html=True)

st.image("./data/Banner.png", use_container_width=True)
# --------------------------------------------------
# Header / Hero Section
# --------------------------------------------------
# --------------------------------------------------
# Tabs Section
# --------------------------------------------------
symptomPredictor, heart, parkinson = st.tabs([
    "🩺 Consult your AI Doc",
    "❤️ AI Cardiologist",
    "🧠 AI Parkinson’s Detector"
])

with symptomPredictor:
    DiseasePredictor.render()

with heart:
    Heart_disease_predictor.render()

with parkinson:
    parkinsons.render()

# Footer
st.markdown("""
<div style='text-align: center; padding: 0rem; color: green; font-size: 1.2rem;'>
Aceso Care is not designed to replace doctors. It is to ensure patients reach out to doctors at the right time and well informed.
</div>
""", unsafe_allow_html=True)