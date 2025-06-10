import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'

# 读取文件夹中的所有xlsx文件
folder_path = 'Mixed_noise_results_on_20_datasets'
xlsx_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') and not f.startswith('~$')]

# 定义噪声率列表和算法列表
noise_ratios = [0.1, 0.2, 0.3, 0.4]
algorithms = ['GB_origin', 'GB_accelerate', 'GB_adaptive', 'GBkNN++', 'Our_model']

# 指定数据集顺序及编号
dataset_order = [
    'Echocardiogram', 'Iris', 'Parkinsons', 'Seeds', 'Glass', 'Audiology', 'Ecoli',
    'Wdbc', 'Balance-scale', 'Breast', 'Pima-diabetes',
    'Fourclass', 'Biodeg', 'Banknote', 'Cardio', 'Thyroid', 'Rice',
    'Waveform', 'Page-blocks', 'Eletrical'
]
dataset_indices = {name: idx + 1 for idx, name in enumerate(dataset_order)}

# 创建一个大图和4x1的子图布局
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1])
axes = [fig.add_subplot(gs[i]) for i in range(4)]

# 为每个噪声率创建一个子图
for idx, noise_ratio in enumerate(noise_ratios):
    # 存储所有数据集的数据
    datasets = []
    accuracies = {algo: [] for algo in algorithms}
    stds = {algo: [] for algo in algorithms}
    
    # 处理每个xlsx文件
    for file in xlsx_files:
        df = pd.read_excel(os.path.join(folder_path, file))
        dataset_name = file.split('_')[0]
        noise_data = df[df['Feature Noise Ratio'] == noise_ratio]
        
        if not noise_data.empty:
            datasets.append(dataset_name)
            for algo in algorithms:
                algo_data = noise_data[noise_data['Algorithm'] == algo]
                if not algo_data.empty:
                    accuracies[algo].append(algo_data['Average Accuracy'].values[0])
                    if 'Accuracy Std' in algo_data.columns:
                        std_value = algo_data['Accuracy Std'].values[0]
                    else:
                        print(f"警告：在文件 {file} 中没有找到'Accuracy Std'列")
                        std_value = 0
                    stds[algo].append(std_value)
                else:
                    accuracies[algo].append(0)
                    stds[algo].append(0)

    # 排序索引（仅保留存在的子集）
    datasets = [d for d in dataset_order if d in datasets]
    sorted_datasets = [d for d in dataset_order if d in datasets]
    xtick_labels = [str(dataset_indices[d]) for d in sorted_datasets]

    # 重排数据
    for algo in algorithms:
        accuracies[algo] = [accuracies[algo][datasets.index(d)] for d in sorted_datasets]
        stds[algo] = [stds[algo][datasets.index(d)] for d in sorted_datasets]

    # 设置柱状图的位置
    x = np.arange(len(sorted_datasets))
    width = 0.15
    colors = ['#4DBBD5', '#3C5488', '#00A087', '#3A5F9B', 'r']
    # colors = ['#1f77b4',  # 蓝 - 稳重主色
    #  '#ff7f0e',  # 橙 - 明亮对比
    #  '#2ca02c',  # 绿色 - 适中点缀
    #  '#d62728',  # 红 - 强调色（保留）
    #  '#9467bd']  # 紫 - 中性平衡

    # 绘制柱状图
    bars = []
    for i, algo in enumerate(algorithms):
        bar = axes[idx].bar(x + i * width - 2 * width,
                            accuracies[algo],
                            width,
                            label=algo,
                            yerr=stds[algo],
                            capsize=3,
                            color=colors[i])
        bars.append(bar)

    # 设置子图属性
    axes[idx].set_xlabel('Datasets', fontsize=22)
    axes[idx].set_ylabel('Accuracy', fontsize=22)
    axes[idx].set_title(f'Hybrid noise rate {int(noise_ratio * 100)}%', fontsize=22)
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels(xtick_labels, rotation=0, fontsize=20)
    axes[idx].tick_params(axis='y', labelsize=20)
    axes[idx].grid(True, linestyle='--', alpha=0.7)

# 调整整体布局
plt.tight_layout()

# 设置图例
algorithm_display_names = ['GBkNN', 'ACC-GBkNN', 'ADP-GBkNN', 'GBkNN++', 'SGBkNN']
fig.legend(bars, algorithm_display_names,
           loc='center',
           bbox_to_anchor=(0.5, 0.02),
           ncol=len(algorithms),
           fontsize=20)

plt.subplots_adjust(bottom=0.1)

# 保存图像
plt.savefig('mixed_noise_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print('图片已生成完成！')
