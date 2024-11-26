import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def generate_samples(model, scaler, condition, num_samples=5, threshold=1):
    model.eval()
    with torch.no_grad():
        valid_samples = []
        while len(valid_samples) < num_samples:
            samples = model.sample(num_samples, condition=condition)

            samples_2d = samples.view(-1, 24).cpu().numpy()
            original_scale_samples = scaler.inverse_transform(samples_2d)
            original_scale_samples = F.relu(torch.tensor(original_scale_samples))

            for sample in original_scale_samples:
                if torch.max(sample) <= threshold:
                    valid_samples.append(sample)
                if len(valid_samples) >= num_samples:
                    break

    return torch.stack(valid_samples)

import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_samples_by_month(synthetic_data, real_data, month, num_samples=50):
    # Create a figure with two subplots (left and right)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- Synthetic Data Plot ---
    # Filter samples by the specified month
    filtered_synthetic = synthetic_data[synthetic_data['month'] == month]
    
    if torch.is_tensor(filtered_synthetic):
        filtered_synthetic = filtered_synthetic.numpy()
    
    # Extract hourly data and calculate average, max, and min profiles
    synthetic_hourly_data = filtered_synthetic.iloc[:, :-1].values  # Exclude the 'month' column
    average_sample = np.mean(synthetic_hourly_data, axis=0)
    max_profile = np.max(synthetic_hourly_data, axis=0)
    min_profile = np.min(synthetic_hourly_data, axis=0)
    
    # Randomly select indices for the samples to plot
    num_samples_to_plot = min(num_samples, len(filtered_synthetic))
    selected_indices = np.random.choice(len(filtered_synthetic), size=num_samples_to_plot, replace=False)
    
    # Plot the synthetic data
    for i in selected_indices:
        axes[0].plot(synthetic_hourly_data[i], color='gray', alpha=0.6)
    
    axes[0].plot(average_sample, label='Perfil Médio', color='green', linewidth=2)
    axes[0].plot(max_profile, label='Perfil Máximo', color='blue', linestyle='dashed', linewidth=2)
    axes[0].plot(min_profile, label='Perfil Mínimo', color='red', linestyle='dashed', linewidth=2)
    
    axes[0].set_title(f"Perfis Sintéticos para o Mês {month}")
    axes[0].set_xlabel("Hora")
    axes[0].set_ylabel("Geração PV")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True)
    axes[0].legend()

    # --- Real Data Plot ---
    # Filter samples by the specified month
    filtered_real = real_data[real_data['month'] == month]
    
    if torch.is_tensor(filtered_real):
        filtered_real = filtered_real.numpy()
    
    # Extract hourly data and calculate average, max, and min profiles
    real_hourly_data = filtered_real.iloc[:, :-1].values  # Exclude the 'month' column
    average_sample = np.mean(real_hourly_data, axis=0)
    max_profile = np.max(real_hourly_data, axis=0)
    min_profile = np.min(real_hourly_data, axis=0)
    
    # Randomly select indices for the samples to plot
    num_samples_to_plot = min(num_samples, len(filtered_real))
    selected_indices = np.random.choice(len(filtered_real), size=num_samples_to_plot, replace=False)
    
    # Plot the real data
    for i in selected_indices:
        axes[1].plot(real_hourly_data[i], color='gray', alpha=0.6)
    
    axes[1].plot(average_sample, label='Perfil Médio', color='green', linewidth=2)
    axes[1].plot(max_profile, label='Perfil Máximo', color='blue', linestyle='dashed', linewidth=2)
    axes[1].plot(min_profile, label='Perfil Mínimo', color='red', linestyle='dashed', linewidth=2)
    
    axes[1].set_title(f"Perfis Reais para o Mês {month}")
    axes[1].set_xlabel("Hora")
    axes[1].set_ylabel("Geração PV")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True)
    axes[1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()
