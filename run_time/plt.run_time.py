import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook

plt.rcParams['font.family'] = 'Times New Roman'
# 检查文件夹是否存在
folder_path = '/Users/shenhai/Desktop/paper 4/返修实验/src/运行时间对比/time_results'
if not os.path.exists(folder_path):
    print(f"错误：{folder_path} 文件夹不存在，请确保文件夹路径正确")
    exit(1)

# 检查文件夹中是否有Excel文件
all_files = [f for f in os.listdir(folder_path) if f.endswith('_time_summary.xlsx')]
if not all_files:
    print(f"错误：{folder_path} 文件夹中没有找到任何 Excel 文件")
    exit(1)

# 用于存储所有数据集的结果
all_results = []

# 读取每个文件的数据
for file in all_files:
    file_path = os.path.join(folder_path, file)

    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        print(f"读取成功：{file_path}")
    except Exception as e:
        print(f"读取失败：{file_path}")
        print(f"错误信息：{e}")
        continue

    # 获取数据集名称（文件名的第一个词）
    dataset_name = file.split('_')[0]
    
    # 解析均值和标准差
    for _, row in df.iterrows():
        algorithm = row['Algorithm']
        
        train_mean = float(row['Training Time (s)'].split('±')[0])
        train_std = float(row['Training Time (s)'].split('±')[1])
        # Convert to milliseconds and apply natural log
        train_mean = np.log(train_mean * 1000) if train_mean > 0 else np.log(1e-3)  # Use 0.001ms as minimum
        train_std = (np.log(train_std * 1000))/10 if train_std > 0 else np.log(1e-3)
        
        # 解析测试时间
        test_mean = float(row['Testing Time (s)'].split('±')[0])
        test_std = float(row['Testing Time (s)'].split('±')[1])
        # Convert to milliseconds and apply natural log
        test_mean = np.log(test_mean * 1000) if test_mean > 0 else np.log(1e-3)
        test_std = (np.log(test_std * 1000))/10 if test_std > 0 else np.log(1e-3)

        
        all_results.append({
            'Dataset': dataset_name,
            'Algorithm': algorithm,
            'Train_Mean': train_mean,
            'Train_Std': train_std,
            'Test_Mean': test_mean,
            'Test_Std': test_std
        })


dataset_order = [
    'Echocardiogram', 'Iris', 'Parkinsons', 'Seeds', 'Glass', 'Audiology', 'Ecoli',
    'Wdbc', 'Balance-scale', 'Breast', 'Pima-diabetes',
    'Fourclass', 'Biodeg', 'Banknote', 'Cardio', 'Thyroid', 'Rice',
    'Waveform', 'Page-blocks', 'Eletrical'
]
# 创建数据集序号映射
dataset_numbers = [str(i+1) for i in range(len(dataset_order))]


# 转换为DataFrame
results_df = pd.DataFrame(all_results)

# 创建图形和子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))

# 设置颜色和标记
colors = [ '#4DBBD5', '#00A087', '#3C5488','#3A5F9B','r']
markers = ['o', '^', 'x', 's', 'D']  # 不同模型使用不同的符号

# 获取所有算法名称和数据集名称
dataset_names = sorted(results_df['Dataset'].unique())
algorithm_order = ['GBkNN', 'ACC-GBkNN', 'ADP-GBkNN', 'GBkNN++', 'SGBkNN']
algorithms = pd.Categorical(results_df['Algorithm'], categories=algorithm_order, ordered=True)
results_df['Algorithm'] = algorithms




# for i, algo in enumerate(algorithm_order):
#     algo_data = results_df[results_df['Algorithm'] == algo]
#     # 确保数据集顺序与预定义顺序一致
#     algo_data = algo_data.set_index('Dataset').reindex(dataset_order).reset_index()
#     x_pos = range(len(dataset_order))
#     ax1.errorbar(x_pos, algo_data['Train_Mean'],yerr=algo_data['Test_Std'],
#                 label=algo, color=colors[i], marker=markers[i], capsize=3)

# # ax1.set_title('Train time', fontsize=22)
# ax1.set_xlabel('Datasets', fontsize=22)
# ax1.set_ylabel('Train time(log ms)', fontsize=22)
# ax1.grid(True, linestyle='--', alpha=0.7)
# ax1.legend(fontsize=18)
# # 设置横轴刻度和标签，使用序号
# ax1.set_xticks(range(len(dataset_order)))
# ax1.set_xticklabels(dataset_numbers,  ha='right', fontsize = 18)
# ax1.tick_params(axis='y', labelsize=18)  # 设置y轴刻度标签的字号


# # 绘制测试时间
# for i, algo in enumerate(algorithm_order):
#     algo_data = results_df[results_df['Algorithm'] == algo]
#     # 确保数据集顺序与预定义顺序一致
#     algo_data = algo_data.set_index('Dataset').reindex(dataset_order).reset_index()
#     x_pos = range(len(dataset_order))
#     ax2.errorbar(x_pos, algo_data['Test_Mean'],yerr=algo_data['Test_Std'],
#                 label=algo, color=colors[i], marker=markers[i], capsize=3)


# # ax2.set_title('Test time')
# ax2.set_xlabel('Datasets', fontsize=22)
# ax2.set_ylabel('Test time(log ms)', fontsize=22)
# ax2.grid(True, linestyle='--', alpha=0.7)
# ax2.legend(fontsize=18)
# # 设置横轴刻度和标签
# ax2.set_xticks(range(len(dataset_order)))
# ax2.set_xticklabels(dataset_numbers, ha='right', fontsize=18)
# ax2.tick_params(axis='y', labelsize=18)  # 设置y轴刻度标签的字号


# # 调整布局以防止标签重叠
# plt.tight_layout()

# # 保存图形
# plt.savefig('/Users/shenhai/Desktop/paper 4/返修实验/src/运行时间对比/time_comparison.png', dpi=300, bbox_inches='tight')
# plt.close()



# 创建训练时间图形
plt.figure(figsize=(12, 9))
ax1 = plt.gca()

# 绘制训练时间
for i, algo in enumerate(algorithm_order):
    algo_data = results_df[results_df['Algorithm'] == algo]
    # 确保数据集顺序与预定义顺序一致
    algo_data = algo_data.set_index('Dataset').reindex(dataset_order).reset_index()
    x_pos = range(len(dataset_order))
    ax1.errorbar(x_pos, algo_data['Train_Mean'],yerr=algo_data['Test_Std'],
                label=algo, color=colors[i], marker=markers[i], capsize=3)

ax1.set_xlabel('Datasets', fontsize=30)
ax1.set_ylabel('Train time (log ms)', fontsize=30)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(fontsize=24)
ax1.set_xticks(range(len(dataset_order)))
ax1.set_xticklabels(dataset_numbers, ha='right', fontsize=24)
ax1.tick_params(axis='y', labelsize=24)

# 调整布局并保存训练时间图形
plt.tight_layout()
plt.savefig('/Users/shenhai/Desktop/paper 4/返修实验/src/运行时间对比/train_time_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 创建测试时间图形
plt.figure(figsize=(12, 9))
ax2 = plt.gca()

# 绘制测试时间
for i, algo in enumerate(algorithm_order):
    algo_data = results_df[results_df['Algorithm'] == algo]
    # 确保数据集顺序与预定义顺序一致
    algo_data = algo_data.set_index('Dataset').reindex(dataset_order).reset_index()
    x_pos = range(len(dataset_order))
    ax2.errorbar(x_pos, algo_data['Test_Mean'],yerr=algo_data['Test_Std'],
                label=algo, color=colors[i], marker=markers[i], capsize=3)

ax2.set_xlabel('Datasets', fontsize=30)
ax2.set_ylabel('Test time (log ms)', fontsize=30)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(fontsize=24)
ax2.set_xticks(range(len(dataset_order)))
ax2.set_xticklabels(dataset_numbers, ha='right', fontsize=24)
ax2.tick_params(axis='y', labelsize=24)

# 调整布局并保存测试时间图形
plt.tight_layout()
plt.savefig('/Users/shenhai/Desktop/paper 4/返修实验/src/运行时间对比/test_time_comparison.png', dpi=300, bbox_inches='tight')
plt.close()