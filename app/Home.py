import streamlit as st

st.set_page_config(
    page_title="Natural Language Processing with Disaster Tweets",
    page_icon="👋",
)

st.title("Home Page")
st.image("app/artifactory/Home.jpeg", caption='', use_column_width=True)

st.header("The Problem")
st.write("""The analysis of social media data during natural disasters can be challenging due to the sheer volume of data generated and the need to quickly identify relevant information. Additionally, tweets are often short, informal, and contain non-standard language, making them difficult to analyse using traditional NLP techniques. As a result, there is a need for more advanced NLP techniques that can accurately classify disaster-related tweets and extract relevant information in real-time.

The dataset provided for this challenge consists of a collection of tweets that have been labelled as either “disaster” or “not disaster”. The goal is to build a model that can learn to distinguish between the two classes based on the text content of the tweets. The challenge is designed to test participants’ skills in natural language processing (NLP) and machine learning. It requires them to preprocess the text data, perform feature engineering, and build a model that can accurately classify tweets.
.""")

st.header("Want to know more?")
st.markdown("* [Omdena Page](https://omdena.com/chapter-challenges/natural-language-processing-with-disaster-tweets/)")

st.sidebar.success("Select a page above.")
