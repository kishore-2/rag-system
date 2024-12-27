import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


def load_data(file_path):
    """
    Load a dataset into a Pandas DataFrame.
    :param file_path: Path to the dataset file.
    :return: Pandas DataFrame
    """
    try:
        df = pd.read_excel(file_path)
        print("Data successfully loaded!")
        print(df.info())
        print(df.describe())
        return df
    except Exception as e:
        raise FileNotFoundError(f"Error loading data from {file_path}: {str(e)}")


def main():
    """
    Main function to set up and interact with the Pandas AI Agent.
    """
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is missing! Please add it to a .env file.")

    # Initialize Groq Client and LLM
    client = Groq(api_key=api_key)
    llm = ChatGroq(model_name="llama3-70b-8192")

    # Path to the dataset (update the path as needed)
    dataset_path = "F:\Simple_linear_regression_prediction.xlsx"

    # Load the dataset
    df = load_data(dataset_path)

    # Create a Pandas AI Agent
    agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

    # Interact with the agent
    print("\n--- Example Queries ---")
    queries = [
        "Give me insights about the data",
        "Provide information from the fifth row",
        "What conclusions can you draw about the data?",
    ]
    for query in queries:
        print(f"\nQuery: {query}")
        result = agent.invoke(query)
        print("Response:", result["output"])


if __name__ == "__main__":
    main()
