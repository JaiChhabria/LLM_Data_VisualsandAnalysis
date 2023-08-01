from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
from langchain.llms import OpenAI


def query_agent(data, query):

    #Parse the CSV file and create a Pandas DataFrame from its contents.
    df = pd.read_csv(data)

    llm = OpenAI()

    # Create a Pandas DataFrame Agent.
    agent = create_pandas_dataframe_agent(llm,df,verbose=True)

    #Python REPL: A python shell used to evaluating and excuting Python commands
    #It takes python code as an input and outputs the results. The input python code can be generated from another tool in the langchain
    return agent.run(query)
    