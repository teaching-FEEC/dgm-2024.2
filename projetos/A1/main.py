import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import missingno as msno


def expand_hemogram_data(df):
    # Initialize new DataFrame with unique patient ID and exam date
    expanded_df = df[['data do exame', 'idade', 'ID do paciente', 'sexo']].drop_duplicates().reset_index(drop=True)

    # List of columns to add based on different types of exams
    exam_columns = df['tipo de exame'].unique()

    # Initialize these columns with NaN
    for col in exam_columns:
        expanded_df[col] = np.nan

    # Fill in the values based on the 'tipo de exame'
    for index, row in df.iterrows():
        patient_id = row['ID do paciente']
        exam_date = row['data do exame']
        exam_type = row['tipo de exame']
        exam_result = row['resultado do exame']

        # Check if the exam_type is in the exam_columns list
        if exam_type not in exam_columns:
            continue

        # Find the matching row(s) in expanded_df
        match_index = expanded_df[
            (expanded_df['ID do paciente'] == patient_id) & (expanded_df['data do exame'] == exam_date)].index

        # Assign the exam result to the appropriate column for each matching index
        for idx in match_index:
            expanded_df.at[idx, exam_type] = exam_result

    return expanded_df

# Define file paths and column names
file_paths = ['data1.csv', 'data2.csv', 'data3.csv']  # Update with your file paths
column_names = ['data do exame', 'idade', 'ID do paciente', 'sexo', 'tipo de exame', 'resultado do exame']

# Initialize an empty list to store dataframes
dataframes = []

# Loop through file paths to load data
for file_path in file_paths:
    data = pd.read_csv(file_path, names=column_names)

    # Data cleaning steps
    data['data do exame'] = pd.to_datetime(data['data do exame'], errors='coerce')
    data['resultado do exame'] = data['resultado do exame'].str.strip()
    data['resultado do exame'] = data['resultado do exame'].str.replace(',', '.')
    data['resultado do exame'] = pd.to_numeric(data['resultado do exame'], errors='coerce')

    # Drop rows with missing values
    data = data.dropna()

    # Append cleaned dataframe to the list
    dataframes.append(expand_hemogram_data(data))

combined_dataframe = pd.concat(dataframes)
print("Figura do Banco de Dados enteiro")
msno.bar(combined_dataframe, color="RoyalBlue")
plt.show()

ranked_ids_combined_dataframe = combined_dataframe['ID do paciente'].value_counts()
print(ranked_ids_combined_dataframe)


def split_dataframe(df):
    # Identifying patient IDs that occur only once
    unique_patients = df['ID do paciente'].value_counts()[df['ID do paciente'].value_counts() == 1].index

    # DataFrame with rows where 'ID do paciente' occurs only once
    df_unique = df[df['ID do paciente'].isin(unique_patients)]

    # DataFrame with the rest of the rows
    df_non_unique = df[~df['ID do paciente'].isin(unique_patients)]

    return df_unique, df_non_unique


# Applying the function to the sample DataFrame
df_unique, df_non_unique = split_dataframe(combined_dataframe)

print("Figura do Banco de Dados Considerando só os pascientes que tem um unico registro")
msno.bar(df_unique, color="RoyalBlue")
plt.show()
ranked_ids_df_unique = df_unique['ID do paciente'].value_counts()
print(ranked_ids_df_unique)

print("Figura do Banco de Dados Considerando só os pascientes que tem multiples registro")
msno.bar(df_non_unique, color="RoyalBlue")
plt.show()
ranked_ids_df_non_unique = df_non_unique['ID do paciente'].value_counts()
print(ranked_ids_df_non_unique)