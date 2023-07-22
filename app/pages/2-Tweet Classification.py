import pickle
import numpy as np
import pandas as pd
import streamlit as st

def header():
    st.header("Twitter Disaster Tweets Classification")

@st.cache_resource
def load_pkl():
    return pickle.load(open('app/artifactory/logreg.pkl', 'rb'))

def show_search_query():
    query = st.text_input('Enter the Tweet')

    if query:
        predict(query)
def predict(raw_data:str):

    model = load_pkl()
    #test_data = ["tornado incoming take shelter and provisions"]
    input_data = [raw_data]
    # Make predictions
    predictions = model.predict(input_data)

    # Print the predicted labels
    for prediction in predictions:
        if prediction == 0:
            st.markdown("Not a disaster")
        else:
            st.markdown("Related to a disaster")


def main():
    header()
    show_search_query()


main()