

import pandas as pd
import re

# Load the data
file_path = '/Users/shenhai/Desktop/paper 4/返修实验/src/Tables(性能及统计实验)/no_noisy_9models_on_20datasets.xlsx'
data = pd.read_excel(file_path, header=None)  # Assuming no header in the original file
print("Original Data Preview:\n", data.head())

# Define a function to extract content between '平均精确率' and '平均召回率'
def extract_acc_data(value):
    # Convert the value to a string to handle non-string types
    value_str = str(value)
    # Use regular expression to capture the value after '平均准确率:'
    match = re.search(r"平均F1分数:\s*([\d.]+\s*±\s*[\d.]+)", value_str)
    return match.group(1).strip() if match else None




def extract_max_min_acc_data(value):
    # Convert the value to a string to handle non-string types
    value_str = str(value)
    # 使用正则表达式捕获"最大准确率:"后面的值
    match = re.search(r"最大准确率:\s*([\d.]+)", value_str)
    return match.group(1).strip() if match else None






# Apply extraction function to each cell and store results in a new DataFrame
# extracted_data = data.applymap(extract_acc_data)  # 提取性能值
extracted_data = data.applymap(extract_max_min_acc_data)  # 提取最大最小值

# 计算每列±号前后的均值
def calculate_column_means(df):
    means_before = []
    means_after = []
    
    for col in df.columns:
        column_values = df[col].dropna()  # 去除空值
        before_values = []
        after_values = []
        
        for value in column_values:
            if isinstance(value, str) and '±' in value:
                parts = value.split('±')
                if len(parts) == 2:
                    try:
                        before_values.append(float(parts[0].strip()))
                        after_values.append(float(parts[1].strip()))
                    except ValueError:
                        continue
        
        mean_before = sum(before_values) / len(before_values) if before_values else None
        mean_after = sum(after_values) / len(after_values) if after_values else None
        
        means_before.append(mean_before)
        means_after.append(mean_after)
    
    # 创建均值行
    means_row = [f"{before:.4f}±{after:.4f}" if before is not None and after is not None else None 
                 for before, after in zip(means_before, means_after)]
    
    # 添加均值行到DataFrame
    df.loc[len(df)] = means_row
    df.index = list(df.index[:-1]) + ['均值']
    
    return df

# 计算并添加均值行
extracted_data = calculate_column_means(extracted_data)

# Save the extracted data to a new Excel file
output_file = '/Users/shenhai/Desktop/paper 4/返修实验/src/Tables(性能及统计实验)/extracted_5_GBmodels_max_accuracy1.xlsx'
extracted_data.to_excel(output_file, index=True, header=False)
# ... existing code ...