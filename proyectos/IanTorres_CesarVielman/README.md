# Sentiment Analysis with Fuzzy Logic

This project is based on the paper *"[Fuzzy Rule-based Unsupervised Sentiment Analysis from Social Media Posts](https://www.researchgate.net/profile/Srishti-Vashishtha-2/publication/334622166_Fuzzy_Rule_based_Unsupervised_Sentiment_Analysis_from_Social_Media_Posts/links/5ece42174585152945149e5b/Fuzzy-Rule-based-Unsupervised-Sentiment-Analysis-from-Social-Media-Posts.pdf)"* by Srishti Vashishtha and Seba Susan. It implements sentiment analysis using fuzzy logic principles. The pipeline includes steps for fuzzification, inference using Mamdani rules, and defuzzification to yield sentiment scores.


## Project Structure

- **sentimentAnalysis_skfuzzy.py**: Main script that handles the sentiment analysis process, including fuzzification, inference, and defuzzification.
- **dataFrame.py**: Contains functions for loading a CSV file into a DataFrame and performing basic sentiment analysis.
- **Makefile**: Simplifies the installation of dependencies and running of the project.

## Requirements

The following libraries are required to run this project:

- `numpy`
- `scikit-fuzzy`
- `pandas`
- `nltk`

## Installation

You can install all required libraries using `pip` by running the following commands:

```bash
pip install numpy
pip install scikit-fuzzy
pip install pandas
pip install nltk
```

Or install all dependencies by running:

```bash
pip install -r requirements.txt
```