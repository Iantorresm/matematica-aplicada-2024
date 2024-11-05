'''
This file manages the process: 
Input -> Fuzzification -> Inference (Mamdani Rules) ->defuzzification->Output
'''
import numpy as np
import skfuzzy as fuzz
from dataFrame import df
import time

# Define the range for x_op
x_op = np.linspace(0, 1, 100)

#Find max and min values for op_neg and op_pos
def get_min_max(df, column_name):
    if column_name in df.columns:
        min_val = df[column_name].min()
        max_val = df[column_name].max()
        return min_val, max_val
    else:
        print(f"La columna '{column_name}' no existe en el DataFrame.")
        return None, None

# Define output membership functions
op_neg = fuzz.trimf(x_op, [0, 0, 0.5])
op_neu = fuzz.trimf(x_op, [0, 0.5, 1])
op_pos = fuzz.trimf(x_op, [0.5, 1, 1])

'''
# max, min and med values for each 'scores' column 
min_p, max_p = get_min_max(df, "Puntaje positivo")
min_n, max_n = get_min_max(df, "Puntaje negativo")
mid_p = (max_p+min_p)/2
mid_n = (max_n+min_n)/2

op_neg = fuzz.trimf(x_op, [min_n, min_n, mid_n])
op_neu = fuzz.trimf(x_op, [0, 0.5, 1])
op_pos = fuzz.trimf(x_op, [mid_p, max_p, max_p])
'''

# Function to calculate membership values for TweetPos and TweetNeg
def calculate_membership_values(tweet_pos, tweet_neg):
    # Calculate membership values for positive scores
    pos_low  = fuzz.interp_membership(x_op, op_neg, tweet_pos)
    pos_med  = fuzz.interp_membership(x_op, op_neu, tweet_pos)
    pos_high = fuzz.interp_membership(x_op, op_pos, tweet_pos)
    
    # Calculate membership values for negative scores
    neg_low  = fuzz.interp_membership(x_op, op_neg, tweet_neg)
    neg_med  = fuzz.interp_membership(x_op, op_neu, tweet_neg)
    neg_high = fuzz.interp_membership(x_op, op_pos, tweet_neg)
    
    return pos_low, pos_med, pos_high, neg_low, neg_med, neg_high

# Calculate firing strengths for each rule using fuzzy AND (min operator)
def calculate_rules_fs(tweet_pos, tweet_neg):
    pos_low, pos_med, pos_high, neg_low, neg_med, neg_high = calculate_membership_values(tweet_pos, tweet_neg)
    R_1 = np.fmin(pos_low, neg_low)
    R_2 = np.fmin(pos_med, neg_low)
    R_3 = np.fmin(pos_high, neg_low)
    R_4 = np.fmin(pos_low, neg_med)
    R_5 = np.fmin(pos_med, neg_med)
    R_6 = np.fmin(pos_high, neg_med)
    R_7 = np.fmin(pos_low, neg_high)
    R_8 = np.fmin(pos_med, neg_high)
    R_9 = np.fmin(pos_high, neg_high)
     # Store all rules in a list
    rules = [R_1, R_2, R_3, R_4, R_5, R_6, R_7, R_8, R_9]
    return rules

def aggregation_of_rule_outputs(rules):
    # Ensure rules is an array of length 9
    if len(rules) != 9:
        raise ValueError("The rules array must contain exactly 9 elements.")

    # Unpack rules for clarity
    R1, R2, R3, R4, R5, R6, R7, R8, R9 = rules

    # Calculate weights using maximum operator
    w_neg = np.fmax(np.fmax(R4, R7), R8)  # Maximum between R4, R7 and R8
    w_neu = np.fmax(np.fmax(R1, R5), R9)  # Maximum between R1, R5 and R9
    w_pos = np.fmax(np.fmax(R2, R3), R6)  # Maximum between R2, R3 and R6

    # Aggregation of Rules outputs
    # Calculate activation levels using fuzzy AND (minimum operator)
    op_activation_low = np.fmin(w_neg, op_neg)
    op_activation_med = np.fmin(w_neu, op_neu)
    op_activation_hi = np.fmin(w_pos, op_pos)
    # Aggregate results using fuzzy OR (maximum operator)
    aggregated = np.fmax(np.fmax(op_activation_low, op_activation_med), op_activation_hi)

    return aggregated

# Polarity output
def determine_polarity (coa):
    if (0 < coa < 0.33 ):
        polarity = 'Negativo'
    elif (0.33 <= coa <= 0.67):
        polarity = 'Neutral'
    elif (0.67 < coa < 1):
        polarity = 'Positivo'
    else:
        polarity = 'Error'
    return polarity

def sentimentAnalysis (row):
    #Extract respective values from dataframe
    tweet_pos, tweet_neg = row['Puntaje positivo'], row['Puntaje negativo']
    rules = calculate_rules_fs(tweet_pos, tweet_neg)
    aggregated_result = aggregation_of_rule_outputs(rules)
    #defuzzification
    coa = fuzz.defuzz(x_op, aggregated_result, 'centroid')
    polarity = determine_polarity(coa)
    return polarity


#Benchmarks and sentiment analysis, dataframe update (adds 'Resultado de Inferencia' and 'Tiempo de Ejecucion' columns )
def benchmark_analysis(df):
    results = []
    times = []
    
    for index, row in df.iterrows():
        start_time = time.time()
 
        inference_output = sentimentAnalysis(row)
        end_time = time.time()
        
        # Store 'inference_output' and execution time
        results.append(inference_output)
        times.append(end_time - start_time)
    
    # Add the two new columns to the DataFrame
    df['Resultado de Inferencia'] = results
    df['Tiempo de Ejecución'] = times

    # Calculate and report the average time
    avg_time = sum(times) / len(times)
    print('')
    print(f'Tiempo total de ejecución:          {round(sum(times), 3)} segundos')
    print(f"Tiempo de ejecución promedio total: {round(avg_time, 7)}   segundos")

    return df

def count_sentiment_values(df, column_name):
    # Initialize counters
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    # Loop through the specified column and count each sentiment
    for value in df[column_name]:
        if value == 'Positivo':
            positive_count += 1
        elif value == 'Negativo':
            negative_count += 1
        elif value == 'Neutral':
            neutral_count += 1

    # Print the results
    print(f"Total de tweets positivos:  {positive_count}")
    print(f"Total de tweets negativos:  {negative_count}")
    print(f"Total de tweets neutrales:  {neutral_count}")

def modify_labels(df):
    # Rename the first and second columns
    df = df.rename(columns={df.columns[0]: 'Oración original', df.columns[1]: 'Label original'})
    return df

# Perform benchmark analysis on the DataFrame and store the result back in df
df = benchmark_analysis(df)

# Modify the column labels to 'Oración original' and 'Label original'
df = modify_labels(df)

# Save the modified DataFrame to a CSV file named "resultado_final.csv" without including the index column
df.to_csv("resultado_final.csv", index=False)

#Show polarity count
count_sentiment_values(df, 'Resultado de Inferencia')