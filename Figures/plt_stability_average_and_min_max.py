import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'

# 加载三个Excel文件中的数据
avg_data = pd.read_excel("plot_data/Model_Avg_Accuracy.xlsx", header=None)
max_data = pd.read_excel("plot_data/Model_Max_Accuracy.xlsx", header=None)
min_data = pd.read_excel("plot_data/Model_Min_Accuracy.xlsx", header=None)

# 准备数据结构
parsed_data = {
    'Datasets': [f"{i + 1}" for i in range(20)]
}

# 模型名称列表
models = ["GBkNN", "ACC-GBkNN", "ADP-GBkNN", "SGBkNN", "Model5", "Model6", "Model7", "Model8", "Model9"]
# 颜色列表 - 扩展为9个模型
colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148']

num_models = len(models)

# 解析数据
for i in range(num_models):
    if i < len(avg_data.columns):
        parsed_data[f'Model_{i+1}_Avg'] = avg_data.iloc[:, i].astype(float)
        parsed_data[f'Model_{i+1}_Max'] = max_data.iloc[:, i].astype(float)
        parsed_data[f'Model_{i+1}_Min'] = min_data.iloc[:, i].astype(float)

parsed_df = pd.DataFrame(parsed_data)

# 设置图表参数
datasets = np.arange(1, 21)
width = 0.08  # 模型之间横向偏移，由于模型增多，减小宽度

# 创建图表
fig, ax = plt.subplots(figsize=(24, 10))

for j in range(num_models):
    if f'Model_{j+1}_Avg' in parsed_df.columns:
        avgs = parsed_df[f'Model_{j+1}_Avg']
        maxs = parsed_df[f'Model_{j+1}_Max']
        mins = parsed_df[f'Model_{j+1}_Min']
        
        # 计算每个点的位置
        x_positions = datasets + (j - num_models/2 + 0.5) * width
        
        # 计算上下误差棒的长度
        yerr_up = maxs - avgs  # 上误差棒：最大值-平均值
        yerr_down = avgs - mins  # 下误差棒：平均值-最小值
        
        # 确保误差棒不超出合理范围
        yerr_up = np.minimum(yerr_up, 1 - avgs)  # 上误差棒不超过1
        yerr_down = np.minimum(yerr_down, avgs)  # 下误差棒不低于0
        
        # 组合上下误差棒
        yerr = np.vstack((yerr_down, yerr_up))
        
        # 绘制误差棒
        ax.errorbar(
            x_positions, avgs, yerr=yerr,
            fmt='s', capsize=5, label=models[j],
            color=colors[j], markersize=6, linewidth=2
        )

# 设置图表属性
ax.set_xticks(datasets)
ax.set_xticklabels(datasets, fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlabel("数据集", fontsize=22)
ax.set_ylabel("准确率", fontsize=22)
ax.set_ylim(0.35, 1.05)
ax.legend(fontsize=18, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.15))

# 添加网格线以提高可读性
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig("Model_Stability_Comparison_Nine_Models.png", dpi=300, bbox_inches='tight')
plt.show()