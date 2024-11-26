import pandas as pd
from maf_model import MAFModel
from train import train_maf_with_loss_tracking
from data_preparation import process_pv_data, prepare_data, generate_synthetic_data

# Load the data and process it
pv_data_path = 'PV_NY.csv'  
process_pv_data(pv_data_path)

df_cleaned = pd.read_csv('data_profiles_labeled.csv', delimiter=',')

# Prepare the data
data_tensor, month_tensor, scaler = prepare_data(df_cleaned)

# Define the parameters for the MAF model
num_inputs = data_tensor.shape[1]
num_hidden = 64
num_layers = 4
condition_size = 1

# Instantiate the MAF model
maf_model = MAFModel(num_inputs=num_inputs, num_hidden=num_hidden, num_layers=num_layers, condition_size=condition_size)

# Train the model
maf_model, losses = train_maf_with_loss_tracking(maf_model, data_tensor, month_tensor)

# Generate synthetic data
synthetic_data = generate_synthetic_data(maf_model, scaler, num_samples=500)

# Convert to DataFrame (24 columns for profiles + 1 column for month)
df = pd.DataFrame(synthetic_data)

# Define the header row
header = ['Hour_1', 'Hour_2', 'Hour_3', 'Hour_4', 'Hour_5', 'Hour_6', 'Hour_7', 'Hour_8', 
          'Hour_9', 'Hour_10', 'Hour_11', 'Hour_12', 'Hour_13', 'Hour_14', 'Hour_15', 'Hour_16', 
          'Hour_17', 'Hour_18', 'Hour_19', 'Hour_20', 'Hour_21', 'Hour_22', 'Hour_23', 'Hour_24', 'month']

# Assign the header
df.columns = header

# Save to CSV with the header
csv_file_path = 'synthetic_data_profiles.csv'
df.to_csv(csv_file_path, index=False)

print(f'Synthetic data profiles saved to {csv_file_path}')
