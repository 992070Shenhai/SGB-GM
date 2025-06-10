
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 构造原始数据（示例为第1个数据集）
# data 1
# data = {
#     'Model': ['GB-ORI', 'ACC-GB', 'ADP-GB', 'GBG++', 'SGB'],
#     'N_GBs': [69, 15, 149, 25, 23],       # 越小越好（成本型）
#     'CR': [0.4630, 0.5167, 1.0000, 0.94, 0.9395],  # 越大越好（效益型）
#     'SR': [1.0, 1.0, 1.0, 0.04, 1.0]    # 越大越好（效益型）
# }


# data 3
# data = {
#     'Model': ['GB-ORI', 'ACC-GB', 'ADP-GB', 'GBG++', 'SGB'],
#     'N_GBs': [122, 56, 29, 38, 63],            # 越小越好（成本型）
#     'CR': [0.6256, 0.5794, 0.5897, 0.7795, 0.7897],  # 越大越好（效益型）
#     'SR': [1.0, 0.9643, 1.0, 0.1578, 1.0]       # 越大越好（效益型）
# }

# # data 4
# data = {
#     'Model': ['GB-ORI', 'ACC-GB', 'ADP-GB', 'GBG++', 'SGB'],
#     'N_GBs': [55, 32, 95, 34, 52],                 # 越小越好（成本型）
#     'CR': [0.2619, 0.5047, 0.6476, 0.8952, 0.8476], # 越大越好（效益型）
#     'SR': [1.0, 1.0, 1.0, 0.1471, 1.0]             # 越大越好（效益型）
# }

# dada 5
# data = {
#     'Model': ['GB-ORI', 'ACC-GB', 'ADP-GB', 'GBG++', 'SGB'],
#     'N_GBs': [63, 105, 208, 55, 50],                # 成本型，越小越好
#     'CR': [0.2957, 0.6667, 0.9812, 0.7009, 0.4695],  # 效益型，越大越好
#     'SR': [0.9333, 1.0, 1.0, 0.0545, 1.0]            # 效益型，越大越好
# }

# dada 6
# data = {
#     'Model': ['GB-ORI', 'ACC-GB', 'ADP-GB', 'GBG++', 'SGB'],
#     'N_GBs': [113, 57, 171, 51, 90],                # 成本型，越小越好
#     'CR': [0.5765, 0.5714, 0.9744, 0.7832, 0.5816],  # 效益型，越大越好
#     'SR': [0.6500, 0.7895, 1.0000, 0.0392, 1.0000]   # 效益型，越大越好
# }
# dada 7
# data = {
#     'Model': ['GB-ORI', 'ACC-GB', 'ADP-GB', 'GBG++', 'SGB'],
#     'N_GBs': [59, 103, 317, 93, 109],                 # 成本型，越小越好
#     'CR': [0.1755, 0.5803, 0.9583, 0.8393, 0.6160],   # 效益型，越大越好
#     'SR': [1.0000, 0.9612, 1.0000, 0.0107, 1.0000]     # 效益型，越大越好
# }
# dada 8
# data = {
#     'Model': ['GB-ORI', 'ACC-GB', 'ADP-GB', 'GBG++', 'SGB'],
#     'N_GBs': [325, 47, 492, 100, 116],               # 成本型，越小越好
#     'CR': [0.5711, 0.5342, 0.9068, 0.9016, 0.8805],  # 效益型，越大越好
#     'SR': [1.0000, 1.0000, 1.0000, 0.0500, 1.0000]   # 效益型，越大越好
# }
# data 9
# data = {
#     'Model': ['GB-ORI', 'ACC-GB', 'ADP-GB', 'GBG++', 'SGB'],
#     'N_GBs': [99, 195, 528, 64, 168],                # 成本型，越小越好
#     'CR': [0.1584, 0.6896, 0.9552, 0.8144, 0.6176],  # 效益型，越大越好
#     'SR': [1.0000, 0.7128, 1.0000, 0.0000, 1.0000]   # 效益型，越大越好
# }

# data 10
# data = {
#     'Model': ['GB-ORI', 'ACC-GB', 'ADP-GB', 'GBG++', 'SGB'],
#     'N_GBs': [277, 17, 299, 61, 120],                   # 成本型，越小越好
#     'CR': [0.6169, 0.5857, 0.8218, 0.6144, 0.7639],     # 效益型，越大越好
#     'SR': [1.0000, 0.1176, 1.0000, 0.0000, 1.0000]       # 效益型，越大越好
# }

# data 11
# data = {
#     'Model': ['GB-ORI', 'ACC-GB', 'ADP-GB', 'GBG++', 'SGB'],
#     'N_GBs': [567, 364, 744, 201, 169],                   # 成本型，越小越好
#     'CR': [0.7382, 0.7083, 0.9895, 0.7435, 0.4726],       # 效益型，越大越好
#     'SR': [0.9708, 0.9615, 1.0000, 0.0348, 1.0000]         # 效益型，越大越好
# }
# data 12
# data = {
#     'Model': ['GB-ORI', 'ACC-GB', 'ADP-GB', 'GBG++', 'SGB'],
#     'N_GBs': [435, 36, 31, 114, 130],                    # 成本型，越小越好
#     'CR': [0.5046, 0.5162, 0.5058, 0.9675, 0.9988],      # 效益型，越大越好
#     'SR': [1.0000, 1.0000, 1.0000, 0.0940, 1.0000]       # 效益型，越大越好
# }
#data 13
# data = {
#     'Model': ['GB-ORI', 'ACC-GB', 'ADP-GB', 'GBG++', 'SGB'],
#     'N_GBs': [706, 356, 1029, 282, 307],             # 成本型，越小越好
#     'CR': [0.6711, 0.6254, 0.9866, 0.7754, 0.5171],  # 效益型，越大越好
#     'SR': [0.9381, 0.9326, 1.0000, 0.0000, 1.0000]   # 效益型，越大越好
# }
#data 14
# data = {
#     'Model': ['GB-ORI', 'ACC-GB', 'ADP-GB', 'GBG++', 'SGB'],
#     'N_GBs': [734, 21, 81, 95, 106],             # 成本型，越小越好
#     'CR': [0.5445, 0.5348, 0.5274, 0.9657, 0.9940],  # 效益型，越大越好
#     'SR': [1.0000, 1.0000, 1.0000, 0.0631, 1.0000]   # 效益型，越大越好
# }
#data 15
# data = {
#     'Model': ['GB-ORI', 'ACC-GB', 'ADP-GB', 'GBG++', 'SGB'],
#     'N_GBs': [1104, 40, 978, 101, 338],            # 成本型，越小越好
#     'CR': [0.6059, 0.5636, 0.7184, 0.9732, 0.9637],  # 效益型，越大越好
#     'SR': [1.0000, 0.4000, 1.0000, 0.0495, 1.0000]   # 效益型，越大越好
# }
# data 16
# data = {
#     'Model': ['GB-ORI', 'ACC-GB', 'ADP-GB', 'GBG++', 'SGB'],
#     'N_GBs': [1995, 1, 899, 150, 317],               # 成本型，越小越好
#     'CR': [0.5456, 0.5456, 0.5533, 0.9491, 0.9759],  # 效益型，越大越好
#     'SR': [1.0000, 1.0000, 1.0000, 0.0133, 1.0000]   # 效益型，越大越好
# }
# data17
# data = {
#     'Model': ['GB-ORI', 'ACC-GB', 'ADP-GB', 'GBG++', 'SGB'],
#     'N_GBs': [2090, 553, 1395, 697, 1016],        # 成本型，越小越好
#     'CR': [0.5485, 0.5475, 0.6582, 0.8276, 0.7942],  # 效益型，越大越好
#     'SR': [1.0000, 1.0000, 1.0000, 0.0717, 1.0000]   # 效益型，越大越好
# }
# data 18
# data = {
#     'Model': ['GB-ORI', 'ACC-GB', 'ADP-GB', 'GBG++', 'SGB'],
#     'N_GBs': [1783, 1697, 5000, 1110, 269],        # 成本型，越小越好
#     'CR': [0.3566, 0.5802, 1.0000, 0.9008, 0.1054],  # 效益型，越大越好
#     'SR': [0.8054, 0.6229, 1.0000, 0.0000, 1.0000]   # 效益型，越大越好
# }

#data 19
# data = {
#     'Model': ['GB-ORI', 'ACC-GB', 'ADP-GB', 'GBG++', 'SGB'],
#     'N_GBs': [4791, 283, 3668, 424, 550],          # 成本型，越小越好
#     'CR': [0.8883, 0.5685, 0.9293, 0.9312, 0.9400],  # 效益型，越大越好
#     'SR': [1.0000, 0.9293, 1.0000, 0.0731, 1.0000]   # 效益型，越大越好
# }
# data 20
data = {
    'Model': ['GB-ORI', 'ACC-GB', 'ADP-GB', 'GBG++', 'SGB'],
    'N_GBs': [6207, 4017, 9982, 2604, 2013],          # 成本型，越小越好
    'CR': [0.6207, 0.5833, 0.9986, 0.7640, 0.3913],  # 效益型，越大越好
    'SR': [0.9113, 0.9177, 1.0000, 0.0000, 1.0000]   # 效益型，越大越好
}













df = pd.DataFrame(data)

# 提取决策矩阵
decision_matrix = df[['N_GBs', 'CR', 'SR']].values.astype(float)

# 标准化处理（最小-最大规范化）
scaler = MinMaxScaler()
normalized_matrix = scaler.fit_transform(decision_matrix)

# 成本型指标 N_GBs 取反（变为效益型）
normalized_matrix[:, 0] = 1 - normalized_matrix[:, 0]

# 各指标权重（等权重）
weights = np.array([1/3, 1/3, 1/3])

# 加权标准化矩阵
weighted_matrix = normalized_matrix * weights

# 计算理想解和负理想解
ideal_solution = weighted_matrix.max(axis=0)
negative_ideal_solution = weighted_matrix.min(axis=0)

# 计算距离
dist_to_ideal = np.linalg.norm(weighted_matrix - ideal_solution, axis=1)
dist_to_negative_ideal = np.linalg.norm(weighted_matrix - negative_ideal_solution, axis=1)

# 计算 TOPSIS 得分
topsis_scores = dist_to_negative_ideal / (dist_to_ideal + dist_to_negative_ideal)

# 添加得分和排名
df['TOPSIS Score'] = topsis_scores.round(7)
df['Rank'] = df['TOPSIS Score'].rank(ascending=False).astype(int)

df.sort_values(by='TOPSIS Score', ascending=False)
print(df)
