import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 模型和数据集
models = ['A', 'B', 'C', 'D', 'E']
datasets = ['DS1', 'DS2', 'DS3', 'DS4', 'DS5', 'DS6']
colors = ['blue', 'green', 'red', 'purple', 'orange']
num_datasets = len(datasets)
num_models = len(models)

# 模拟数据（加入第五个模型的数据）
means = np.array([
    [0.9667, 0.9167, 0.9000, 0.9667, 0.9833],
    [0.8494, 0.8574, 0.8648, 0.8294, 0.8433],
    [0.8592, 0.8598, 0.8564, 0.8533, 0.8570],
    [0.8439, 0.8294, 0.8577, 0.8391, 0.8345],
    [0.8396, 0.8404, 0.8303, 0.8296, 0.8382],
    [0.8413, 0.8422, 0.8687, 0.8224, 0.8456]
])

stds = np.array([
    [0.0667, 0.1118, 0.1333, 0.0667, 0.0500],
    [0.0103, 0.0072, 0.0089, 0.0112, 0.0087],
    [0.0074, 0.0098, 0.0059, 0.0091, 0.0101],
    [0.0135, 0.0081, 0.0120, 0.0119, 0.0110],
    [0.0128, 0.0102, 0.0112, 0.0111, 0.0104],
    [0.0100, 0.0141, 0.0093, 0.0075, 0.0098]
])

mins = np.array([
    [0.8312, 0.8527, 0.8463, 0.8514, 0.8488],
    [0.8358, 0.8444, 0.8526, 0.8129, 0.8296],
    [0.8440, 0.8445, 0.8421, 0.8382, 0.8429],
    [0.8261, 0.8136, 0.8437, 0.8261, 0.8201],
    [0.8250, 0.8237, 0.8146, 0.8123, 0.8242],
    [0.8255, 0.8240, 0.8545, 0.8077, 0.8311]
])

maxs = np.array([
    [0.8638, 0.8820, 0.8764, 0.8767, 0.8743],
    [0.8631, 0.8710, 0.8783, 0.8462, 0.8579],
    [0.8738, 0.8746, 0.8707, 0.8683, 0.8714],
    [0.8617, 0.8453, 0.8718, 0.8521, 0.8486],
    [0.8543, 0.8571, 0.8461, 0.8469, 0.8520],
    [0.8572, 0.8604, 0.8829, 0.8371, 0.8600]
])

# 绘图
fig, ax = plt.subplots(figsize=(14, 6))
width = 0.15  # 模型水平偏移控制

for i in range(num_datasets):  # 遍历数据集
    for j in range(num_models):  # 遍历模型
        x = i + (j - (num_models - 1)/2) * width  # 模型在当前数据集位置

        mean = means[i][j]
        std = stds[i][j]
        min_val = mins[i][j]
        max_val = maxs[i][j]

        color = colors[j]

        # std方形
        rect = patches.Rectangle(
            (x - 0.05, mean - std),
            width=0.1,
            height=2 * std,
            color=color,
            alpha=0.3
        )
        ax.add_patch(rect)

        # min–max 误差棒 + 均值点
        lower = mean - min_val
        upper = max_val - mean
        ax.errorbar(x, mean, yerr=[[lower], [upper]], fmt='o', capsize=6, color=color)

# 坐标轴设置
ax.set_xticks(range(num_datasets))
ax.set_xticklabels(datasets)
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Across Datasets')
ax.set_ylim(0.5, 1)
# ax.grid(True)

# 图例：每个模型一个颜色
legend_elements = [
    patches.Patch(color=colors[i], alpha=0.4, label=f'Model {models[i]}')
    for i in range(num_models)
]
ax.legend(handles=legend_elements, title="Models")

plt.tight_layout()
plt.show()
