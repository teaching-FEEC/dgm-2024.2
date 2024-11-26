import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sampling import generate_samples

def process_pv_data(pv_data_path, output_path='data_profiles_labeled.csv'):
    """
    This function processes the PV data by extracting daily profiles, 
    selecting even-indexed hourly data, and adding a 'month' column.

    Parameters:
    - pv_data_path: str, path to the input CSV file containing the PV data.
    - output_path: str, path to save the final processed dataset (default is 'data_profiles_labeled.csv').
    """
    # Load the PV data from the CSV file
    pv_data = pd.read_csv(pv_data_path)

    # Convert the 'Date' column to datetime and sort the data
    pv_data['Date'] = pd.to_datetime(pv_data['Date'])
    pv_data.sort_values(by='Date', inplace=True)

    # Extract the hour from the Date column and add it as a new column
    pv_data['Hour'] = pv_data['Date'].dt.hour

    # Divide the 'P_Gen' values by 1000 to reduce dimensionality
    pv_data['P_Gen'] = pv_data['P_Gen'] / 1000

    # Group the data by each day (Date)
    daily_data = pv_data.groupby(pv_data['Date'].dt.date)['P_Gen'].apply(list).reset_index()

    # Select the 24 even indexed points (1,3,5,7,...,47) from each daily profile
    even_indexed_pv_data = daily_data['P_Gen'].apply(lambda x: [x[i] for i in range(0, 48, 2)])

    # Create the second dataset (daily_profiles_even.csv)
    daily_profiles_even_generated = pd.DataFrame(even_indexed_pv_data.tolist(), columns=[f'Hour_{i+1}' for i in range(24)])

    # Add the 'month' column to create the final dataset
    # Extract the month from the Date column and add it as a new column
    daily_data['month'] = pd.to_datetime(daily_data['Date']).dt.month

    # Combine the generated even-indexed hourly data with the 'month' column
    final_dataset = pd.concat([daily_profiles_even_generated, daily_data['month']], axis=1)

    # Save the final dataset to a CSV file
    final_dataset.to_csv(output_path, index=False)

    print(f"Data processing complete. The final dataset has been saved as '{output_path}'.")

def prepare_data(df_cleaned):
    data = df_cleaned.iloc[:, :-1].values
    months = df_cleaned.iloc[:, -1].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    data_tensor = torch.tensor(scaled_data, dtype=torch.float32)
    month_tensor = torch.tensor(months, dtype=torch.float32)

    return data_tensor, month_tensor, scaler

def generate_synthetic_data(maf_model, scaler, num_samples=50):
    all_samples = []
    for month in range(1, 13):
        condition = torch.tensor([[float(month)]])
        new_samples = generate_samples(maf_model, scaler, condition, num_samples=num_samples)

        reshaped_samples = new_samples.reshape(-1, new_samples.shape[-1])
        month_column = torch.full((reshaped_samples.shape[0], 1), float(month))
        reshaped_samples_with_month = torch.cat((reshaped_samples, month_column), dim=1)

        all_samples.append(reshaped_samples_with_month.numpy())

    all_samples = torch.vstack([torch.tensor(month_samples) for month_samples in all_samples])
    
    return all_samples
