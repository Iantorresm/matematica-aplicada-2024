import pandas as pd
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

def csv_to_dataframe(file_path):
    """
    Reads a .csv file and converts it into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the .csv file.

    Returns:
    pandas.DataFrame: The DataFrame created from the .csv file, or None if an error occurs.
    """

    try:
        df = pd.read_csv(file_path)
        print("Archivo .csv cargado correctamente.")
        return df
    except FileNotFoundError:
        print(f"El archivo en la ruta '{file_path}' no se encontró.")
    except pd.errors.EmptyDataError:
        print("El archivo .csv está vacío.")
    except pd.errors.ParserError:
        print("Error al analizar el archivo .csv.")
    return None


def expand_contractions(df, column_name):
    """
    Expands contractions in the text within a specific column of a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to process.
    column_name (str): The name of the column to process.

    Returns:
    pandas.DataFrame: The DataFrame with contractions expanded in the specified column.
    """

    # Verify if the column exists in the DataFrame
    if column_name not in df.columns:
        print(f"La columna '{column_name}' no existe en el DataFrame.")
        return df

    contractions_dict = {
        r"\bcan't\b": "can not", r"\bcan t\b": "can not",
        r"\bcannot\b": "can not",
        r"\bdon't\b": "do not", r"\bdon t\b": "do not",
        r"\bdoesn't\b": "does not", r"\bdoesn t\b": "does not",
        r"\bdidn't\b": "did not", r"\bdidn t\b": "did not",
        r"\bwon't\b": "will not", r"\bwon t\b": "will not",
        r"\bwouldn't\b": "would not", r"\bwouldn t\b": "would not",
        r"\bshouldn't\b": "should not", r"\bshouldn t\b": "should not",
        r"\bcouldn't\b": "could not", r"\bcouldn t\b": "could not",
        r"\baren't\b": "are not", r"\baren t\b": "are not",
        r"\bisn't\b": "is not", r"\bisn t\b": "is not",
        r"\bwasn't\b": "was not", r"\bwasn t\b": "was not",
        r"\bweren't\b": "were not", r"\bweren t\b": "were not",
        r"\bhasn't\b": "has not", r"\bhasn t\b": "has not",
        r"\bhaven't\b": "have not", r"\bhaven t\b": "have not",
        r"\bhadn't\b": "had not", r"\bhadn t\b": "had not",
        r"\bI'll\b": "I will", r"\bI ll\b": "I will",
        r"\bI'm\b": "I am", r"\bI m\b": "I am",
        r"\bI've\b": "I have", r"\bI ve\b": "I have",
        r"\bwe're\b": "we are", r"\bwe re\b": "we are",
        r"\bthey're\b": "they are", r"\bthey re\b": "they are",
        r"\byou're\b": "you are", r"\byou re\b": "you are",
        r"\bthey've\b": "they have", r"\bthey ve\b": "they have",
        r"\bit's\b": "it is", r"\bit s\b": "it is",
        r"\bthat's\b": "that is", r"\bthat s\b": "that is",
        r"\bthere's\b": "there is", r"\bthere s\b": "there is",
        r"\bhere's\b": "here is", r"\bhere s\b": "here is"
    }
    
    # Function to expand constractions in a sentence
    def expand_text(text):
        if isinstance(text, str):
            for contraction, expanded in contractions_dict.items():
                text = re.sub(contraction, expanded, text, flags=re.IGNORECASE)
        return text

    # The function is applied to the DataFrame in the specified column
    df[column_name] = df[column_name].apply(expand_text)
    return df


def preprocess_dataframe(df):
    """
    Preprocesses a DataFrame by removing special characters and single-letter words.

    Parameters:
    df (pandas.DataFrame): The DataFrame to preprocess.

    Returns:
    pandas.DataFrame: The preprocessed DataFrame.
    """

    def clean_text(text):
        if isinstance(text, str):
            text = re.sub(r'[^A-Za-z0-9\s]', '', text)   # Delete special characters 
            text = re.sub(r'\b\w\b', '', text)           # Delete single letter words
            text = re.sub(r'\s+', ' ', text).strip()     # Remove extra spaces 
        return text

    return df.map(clean_text)


def add_sentiment_scores(df, text_column):
    """
    Adds positive and negative score columns to a DataFrame using VADER from NLTK.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the texts to analyze.
    text_column (str): The name of the column containing the text.

    Returns:
    pandas.DataFrame: The DataFrame with the added positive and negative score columns.
    """

    sia = SentimentIntensityAnalyzer()
    
    # Get positive and negative sentiment scores
    def get_sentiment_scores(text):
        scores = sia.polarity_scores(text)
        round_scores = {key: round(value, 3) for key, value in scores.items()}
        return pd.Series([round_scores['pos'], round_scores['neg']])
    
    # Apply function and add columns
    df[['Puntaje positivo', 'Puntaje negativo']] = df[text_column].apply(get_sentiment_scores)
    
    return df




# Main execution
# csv to dataframe conversion 
df = csv_to_dataframe("test_data.csv")


if df is not None:  #Ensures df isn´t 'None' before proceeding 
    #Preprocesing
    df = expand_contractions(df, "sentence")
    df = preprocess_dataframe(df)

    #Add columns for positive and negative sentiment scores to the DataFrame using VADER
    df = add_sentiment_scores(df, "sentence")

# Generate a .csv file that includes the newly added columns
df.to_csv('test_data.csv', index=False)
