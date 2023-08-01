import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from utils import query_agent

st.title("Let's do some analysis on your file")
st.header("Please upload the document here:")

# Capture the file

data = st.file_uploader("Upload CSV file", type="csv")

query = st.text_area("Enter your query")
button = st.button("Generate Response")

if button:
    # Get Response
    answer = query_agent(data,query)
    st.write(answer)