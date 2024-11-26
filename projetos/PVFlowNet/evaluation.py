import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from fastdtw import fastdtw
from sampling import plot_samples_by_month

def load_data(file_path, delimiter=','):
    """Load the CSV dataset."""
    return pd.read_csv(file_path, delimiter=delimiter)

def calculate_representative_curves(data):
    """Calculate the average daily profile for each month."""
    representative_curves = data.groupby('month').mean()
    representative_curves.reset_index(inplace=True)
    return representative_curves

def calculate_metrics(real_curves, synthetic_curves):
    """Calculate metrics between real and synthetic curves."""
    metrics = []
    months = real_curves['month']
    
    for month in months:
        real_values = real_curves[real_curves['month'] == month].iloc[:, 1:].values.flatten()
        synthetic_values = synthetic_curves[synthetic_curves['month'] == month].iloc[:, 1:].values.flatten()
        
        # MAE
        mae = mean_absolute_error(real_values, synthetic_values)
        
        # RMSE
        rmse = np.sqrt(np.mean((real_values - synthetic_values) ** 2))
        
        # Pearson Correlation Coefficient
        pcc = np.corrcoef(real_values, synthetic_values)[0, 1]
        
        # DTW (Dynamic Time Warping)
        distance, _ = fastdtw(real_values, synthetic_values)
        
        # Append metrics for the month
        metrics.append({
            "month": month,
            "MAE": mae,
            "RMSE": rmse,
            "PCC": pcc,
            "DTW": distance
        })
    
    return metrics

def main():
    # Load the real dataset
    file_path = 'data_profiles_labeled.csv'  # Replace with the correct path
    data = load_data(file_path, delimiter=',')
    
    # Calculate the representative curves for the real dataset
    representative_curves = calculate_representative_curves(data)
    print("Representative Curves (Real Data):")
    print(representative_curves)
    
    # Load the synthetic dataset
    synthetic_file_path = 'synthetic_data_profiles.csv'  # Replace with the correct path
    synthetic_data = load_data(synthetic_file_path, delimiter=',')
    
    # Calculate the representative curves for the synthetic dataset
    synthetic_representative_curves = calculate_representative_curves(synthetic_data)
    print("Representative Curves (Synthetic Data):")
    print(synthetic_representative_curves)
    
    # Calculate metrics
    metrics = calculate_metrics(representative_curves, synthetic_representative_curves)
    
    # Convert the results to a DataFrame 
    metrics_df = pd.DataFrame(metrics)
    print("Metrics:")
    print(metrics_df)

    plot_samples_by_month(synthetic_data, data, month=7)

if __name__ == "__main__":
    main()
