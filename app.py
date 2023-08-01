import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Set the matplotlib backend to a non-interactive one
matplotlib.use("agg")

def query_agent(df, query):
    # Use the environment variable for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")

    llm = OpenAI(openai_api_key=openai_api_key)

    # Create a Pandas DataFrame Agent.
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)

    # Python REPL: A python shell used to evaluating and executing Python commands
    # It takes python code as an input and outputs the results. The input python code can be generated from another tool in the langchain
    return agent.run(query)

def plot_bar_chart(df, x_column, y_column):
    fig, ax = plt.subplots()
    ax.bar(df[x_column], df[y_column])
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f"Bar Chart: {x_column} vs {y_column}")
    st.pyplot(fig)

def plot_line_chart(df, x_column, y_column):
    fig, ax = plt.subplots()
    ax.plot(df[x_column], df[y_column])
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f"Line Chart: {x_column} vs {y_column}")
    st.pyplot(fig)

def plot_scatter_plot(df, x_column, y_column):
    fig, ax = plt.subplots()
    ax.scatter(df[x_column], df[y_column])
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f"Scatter Plot: {x_column} vs {y_column}")
    st.pyplot(fig)

def plot_heatmap(df):
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    ax.set_title("Heatmap")
    st.pyplot(fig)

def main():
    st.title("Data Analysis with Streamlit")
    st.header("Please upload the document here:")
    
    # Initialize df to None
    df = None

    # Capture the file
    data = st.file_uploader("Upload CSV file", type="csv")
    
    query = st.text_area("Enter your query")
    button = st.button("Generate Response")
    
    if data is not None:
        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(data)

    if button:
        if df is not None:
            # Get Response using query_agent function
            answer = query_agent(df, query)
            st.write("Response: ", answer)
    
    # Data Visualization: Chart Options
    st.subheader("Data Visualization: Chart Options")
    chart_types = ["Bar Chart", "Line Chart", "Scatter Plot", "Heatmap"]
    selected_chart = st.selectbox("Select Chart Type", chart_types)

    if df is not None:
        x_column = st.selectbox("Select X-axis column", df.columns)
        y_column = st.selectbox("Select Y-axis column", df.columns)

        if x_column != y_column:
            if selected_chart == "Bar Chart":
                plot_bar_chart(df, x_column, y_column)
            elif selected_chart == "Line Chart":
                plot_line_chart(df, x_column, y_column)
            elif selected_chart == "Scatter Plot":
                plot_scatter_plot(df, x_column, y_column)
            elif selected_chart == "Heatmap":
                plot_heatmap(df)
        else:
            st.write("Please select different columns for X-axis and Y-axis.")
    else:
        st.warning("Please upload a CSV file.")

if __name__ == "__main__":
    main()
