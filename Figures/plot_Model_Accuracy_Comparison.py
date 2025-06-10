import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
# Load the dataset (assuming it's in Excel format)
# file_path = "plot_data/Model_Accuracy_Comparison.xlsx"  # Replace with the actual file path
# data = pd.read_excel(file_path, header=None)


data = pd.read_excel("plot_data/Model_Accuracy_Comparison.xlsx",header=None)


# Assume the dataset has 20 rows and 4 columns with "accuracy ± std" format in cells
# Parse accuracy and standard deviation for each model
parsed_data = {
    'Datasets': [f"{i + 1}" for i in range(20)]
}

# Extract accuracy and std from each cell, assuming format "accuracy ± std"
for i in range(4):
    parsed_data[f'Model_{i + 1}_Mean'] = data.iloc[:, i].str.split('±').str[0].astype(float)
    parsed_data[f'Model_{i + 1}_Std'] = data.iloc[:, i].str.split('±').str[1].astype(float)

# Convert parsed data to a DataFrame for easy access
parsed_df = pd.DataFrame(parsed_data)

# Extract datasets and separate model columns
datasets = parsed_df['Datasets']
models = ["GBkNN", "ACC-GBkNN", "ADP-GBkNN", "SGBkNN"]
colors = ['orange', 'green', 'red', 'blue']  # Colors for the lines with SGBkNN in blue
shades = ['lightcoral', 'lightgreen', 'lightpink', 'lightblue']  # Colors for the shades

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(14, 18))  # Removed sharex=True for independent x-axis labels



# Plot each of the first three models compared to the last model (SGBkNN)
for i in range(3):  # Three subplots for each comparison with SGBkNN
    last_model_mean = parsed_df[f'Model_4_Mean']
    last_model_std = parsed_df[f'Model_4_Std']
    model_mean = parsed_df[f'Model_{i + 1}_Mean']
    model_std = parsed_df[f'Model_{i + 1}_Std']

    ax = axes[i]

    # Plot SGBkNN (always the last model)
    ax.plot(datasets, last_model_mean, label=models[3], color=colors[3], marker='o', linewidth=2)
    ax.fill_between(datasets, last_model_mean - last_model_std, last_model_mean + last_model_std, color=shades[3], alpha=0.3)

    # Plot the current model
    ax.plot(datasets, model_mean, label=models[i], color=colors[i], marker='o', linewidth=2)
    ax.fill_between(datasets, model_mean - model_std, model_mean + model_std, color=shades[i], alpha=0.3)

    # Calculate dynamic y-axis limits for each subplot based on min/max values with some padding
    min_val = min((model_mean - model_std).min(), (last_model_mean - last_model_std).min())
    max_val = max((model_mean + model_std).max(), (last_model_mean + last_model_std).max())
    ax.set_ylim(min_val - 0.05, max_val + 0.05)  # Adding some padding

    # Labeling each subplot
    ax.set_title(f"Comparison of SGBkNN and {models[i]}", fontsize=35)
    ax.set_ylabel("Accuracy", fontsize=35)
    ax.legend(fontsize=25, loc='lower right')  # Place legend in the upper right corner
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_xticks(np.arange(0, 21))  # Set x-ticks from 1 to 20

# Set x-axis label for the last subplot only and add a main title
axes[-1].set_xlabel("Datasets", fontsize=35)

#plt.suptitle("Accuracy and Standard Deviation Comparison Across 20 Datasets", fontsize=title_fontsize)
plt.subplots_adjust(hspace=0.8)  # Adjust spacing between subplots
plt.tight_layout(rect=[0, 0.01, 1, 0.99])
plt.savefig("Model_Accuracy_Comparison.png", dpi=300)
plt.show()
