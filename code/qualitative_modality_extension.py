import pandas as pd
import os
import re

# Function to safely convert columns to integers
def convert_to_int(df, column_name):
    # Convert to string first to handle mixed types
    df[column_name] = df[column_name].astype(str)
    
    # Use to_numeric to convert to integer, errors='coerce' will set invalid parsing as NaN
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')

    # Fill NaN values with a default integer, e.g., 0 or any other suitable default
    df[column_name] = df[column_name].fillna(-1).astype(int)

    return df

#------------------------------------------------    
run_value = 0
dataset_name = 'citeseer'
sampling = 'ego'
#cases_to_consider = ['text', 'text+image', 'image']
cases_to_consider = ['text', 'text+motif', 'motif']
#------------------------------------------------
results_location = f'results/{dataset_name}/graph_images/sample_size_50/run_{run_value}'

filenames= []
for i in range(len(cases_to_consider)):
    filenames.append(f'{sampling}_run_{run_value}_{sampling}_{cases_to_consider[i]}_encoder_results.csv')


# read each file and extract the appropriate columns to a new file.
all_dfs = []  # List to store all the dataframes

#Read each file, extract and rename the appropriate columns, then append to the list
for i, filename in enumerate(filenames):
    df = pd.read_csv(os.path.join(results_location, filename))

    if i == 0:
        selected_columns = ['graph_id', 'node_with_question_mark', 'ground_truth', 'response', 'parsed_response']
    else:
        selected_columns = ['response', 'parsed_response']

    # Extract selected columns
    df = df[selected_columns]

    # Rename columns for each file
    df.columns = [f'{cases_to_consider[i]}_{col}' for col in df.columns]

    all_dfs.append(df)

# Concatenate all dataframes horizontally
result_df = pd.concat(all_dfs, axis=1)
#sanity checking 
digit_pattern = re.compile(r'\d+')
result_df[f'{cases_to_consider[0]}_parsed_response'] = result_df[f'{cases_to_consider[0]}_parsed_response'].apply(lambda x: ''.join(digit_pattern.findall(str(x))))
result_df[f'{cases_to_consider[1]}_parsed_response'] = result_df[f'{cases_to_consider[1]}_parsed_response'].apply(lambda x: ''.join(digit_pattern.findall(str(x))))
result_df[f'{cases_to_consider[2]}_parsed_response'] = result_df[f'{cases_to_consider[2]}_parsed_response'].apply(lambda x: ''.join(digit_pattern.findall(str(x))))

# Convert the columns to integers
result_df = convert_to_int(result_df, f'{cases_to_consider[0]}_ground_truth')
result_df = convert_to_int(result_df, f'{cases_to_consider[0]}_parsed_response')
result_df = convert_to_int(result_df, f'{cases_to_consider[1]}_parsed_response')
result_df = convert_to_int(result_df, f'{cases_to_consider[2]}_parsed_response')

#extract the relevant cases
result_d1 = result_df.loc[(result_df[f'{cases_to_consider[0]}_ground_truth'] != result_df[f'{cases_to_consider[0]}_parsed_response']) & (result_df[f'{cases_to_consider[0]}_ground_truth'] == result_df[f'{cases_to_consider[1]}_parsed_response'])]
#result_d1 = result_df.loc[(result_df['text_ground_truth'] != result_df['text_parsed_response'])]

print("relevant no of changes",result_d1.shape)


# Write the concatenated dataframe to a new CSV file
result_d1.to_csv(results_location+'/qual.csv', index=False)