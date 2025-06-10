import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
# Load the dataset (assuming it's in Excel format)
file_path = "plot_data/all_algorithm_run_time_results.xlsx"  # Replace with the actual file path

data = pd.read_excel(file_path)

# Extract the unique model names and datasets
models = data['Algorithm'].unique()
datasets = data['数据集'].unique()

# Prepare data for plotting
data['Train Time (ms, log)'] = np.log(data['Train Time (s)'] * 1000)  # Convert to milliseconds and apply log
data['Test Time (ms, log)'] = np.log(data['Test Time (s)'] * 1000)  # Convert to milliseconds and apply log

# Define colors for the lines
colors = ['#FFB347', '#77DD77', '#FF6961', '#3A5F9B']  # Colors specified for each model
marks = ['v', 's', 'D', '^']

# Create a dictionary to store training and testing times for each model
train_times = {model: data[data['Algorithm'] == model]['Train Time (ms, log)'].values for model in models}
test_times = {model: data[data['Algorithm'] == model]['Test Time (ms, log)'].values for model in models}

# Determine the y-axis limits based on the combined min and max of training and testing times
min_y = min(data['Train Time (ms, log)'].min(), data['Test Time (ms, log)'].min())
max_y = max(data['Train Time (ms, log)'].max(), data['Test Time (ms, log)'].max())

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# Plot training time
for i, model in enumerate(models):
    axes[0].plot(range(1, len(train_times[model]) + 1), train_times[model], marker=marks[i],markersize=8, color=colors[i], label=model)

axes[0].set_xlabel("Datasets", fontsize=35)
axes[0].set_ylabel("Training time (log ms)", fontsize=35)
axes[0].set_xticks(range(1, 21))  # Set x-ticks from 1 to 20
axes[0].tick_params(axis='x', labelsize=25)
axes[0].tick_params(axis='y', labelsize=25)
axes[0].set_ylim(min_y, max_y + 1)  # Set the same y-axis range for both subplots
axes[0].legend(loc='upper left', fontsize=25)  # Set legend position to upper left and font size to 25
axes[0].grid(True)
axes[0].text(0.5, -0.25, "(a) Training Time on Different Datasets", transform=axes[0].transAxes, fontsize=35, ha='center', va='top')  # Title at the bottom

# Plot testing time
for i, model in enumerate(models):
    axes[1].plot(range(1, len(test_times[model]) + 1), test_times[model], marker=marks[i],markersize=8, color=colors[i], label=model)

axes[1].set_xlabel("Datasets", fontsize=35)
axes[1].set_ylabel("Testing time (log ms)", fontsize=35)
axes[1].set_xticks(range(1, 21))  # Set x-ticks from 1 to 20
axes[1].tick_params(axis='x', labelsize=25)
axes[1].tick_params(axis='y', labelsize=25)
axes[1].set_ylim(min_y, max_y + 1)  # Set the same y-axis range for both subplots
axes[1].legend(loc='upper left', fontsize=25)  # Set legend position to upper left and font size to 25
axes[1].grid(True)
axes[1].text(0.5, -0.25, "(b) Testing Time on Different Datasets", transform=axes[1].transAxes, fontsize=35, ha='center', va='top')  # Title at the bottom

plt.tight_layout()
plt.subplots_adjust(hspace=0.5)  # Adjust spacing for the text labels
plt.savefig("Model_Time_Comparison.png", dpi=300)  # Save the plot with 300 DPI
#plt.show()
