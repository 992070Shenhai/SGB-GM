
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import pairwise_distances, accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from collections import Counter
import matplotlib.pyplot as plt
import time



# 计算欧几里得距离

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

# 计算隶属度的高斯函数，基于给定的 (d, 0.5) 决定 sigma
def calculate_sigma(d):
    return d / np.sqrt(-np.log(0.5))

def gaussian_membership(x_k, x, d):
    """
    计算基于高斯函数的隶属度
    :param x_k: 中心点
    :param x: 样本点
    :param d: 在距离 d 时隶属度为 0.5 的点
    :return: 隶属度值
    """
    sigma = calculate_sigma(d)
    distance = euclidean_distance(x, x_k)
    membership = np.exp(-((distance / sigma) ** 2))
    return membership

# 计算数据集的近邻半径
def find_neighborhood_radius(X, y):
    n_samples = X.shape[0]
    eta_list = []

    # 计算所有样本对之间的欧几里得距离
    distances = pairwise_distances(X, metric='euclidean')

    for i in range(n_samples):
        label_x = y[i]
        same_label_indices = np.where(y == label_x)[0]
        different_label_indices = np.where(y != label_x)[0]

        same_label_indices = same_label_indices[same_label_indices != i]

        if len(same_label_indices) > 0:
            nh_distance = np.min(distances[i, same_label_indices])
        else:
            nh_distance = np.inf

        nm_distance = np.min(distances[i, different_label_indices])
        eta = nm_distance - 0.01 * nh_distance
        eta_list.append(eta)

    return eta_list



def find_optimal_alpha(membership, lam):
    # Sort the membership values
    membership_sorted = np.sort(membership)

    # Transform membership values greater than 0.5 to 1 - value
    transformed_membership = np.where(membership_sorted > 0.5, 1 - membership_sorted, membership_sorted)

    # Vectorized loss function
    def compute_loss(alpha):
        loss1 = np.sum(
            transformed_membership[transformed_membership <= alpha])  # 对 transformed_membership <= alpha 的部分求和
        loss2 = np.sum(
            0.5 - transformed_membership[transformed_membership > alpha])  # 对 transformed_membership > alpha 的部分求和
        return lam * loss1 + loss2

    # Search for optimal alpha among unique transformed membership values
    alpha_candidates = np.unique(transformed_membership)
    losses = np.array([compute_loss(alpha) for alpha in alpha_candidates])

    # Find the alpha corresponding to the minimum loss
    min_loss_idx = np.argmin(losses)
    optimal_alpha = alpha_candidates[min_loss_idx]

    return optimal_alpha


# 构造粒球
def construct_granules_by_class(X, y, eta_list):
    """
    根据标签构建粒球，并按标签对粒球进行分类。
    :param X: 特征矩阵
    :param y: 标签数组
    :param eta_list: 每个样本的近邻半径列表
    :param d: 距离 d 时隶属度为 0.5 的点
    :return: 按标签分类的粒球字典
    """
    optimal_alpha_list = []
    unique_classes = np.unique(y)
    granules_by_class = {label: [] for label in unique_classes}  # 初始化按标签分类的粒球字典

    for i in range(len(X)):
        x_k = X[i]
        label = y[i]  # 样本的标签
        d = eta_list[i]

        # 计算每个样本对中心样本 x_k 的隶属度，使用高斯隶属度函数
        memberships = np.array([gaussian_membership(x_k, X[j], d) for j in range(len(X))])

        # 找到最优的 alpha
        optimal_alpha = find_optimal_alpha(memberships, lam=1)
        optimal_alpha_list.append(optimal_alpha)

        # 筛选出隶属度大于 alpha 的样本
        inside_granule_indices = np.where(memberships >= 1 - optimal_alpha)[0]

        # 仅保留隶属度大于 alpha 的样本
        points = [X[j] for j in inside_granule_indices]

        # 如果粒球内没有样本点，则跳过创建粒球
        if len(points) == 0:
            continue  # 跳过这次循环，不添加粒球

        # 重新计算半径为粒球内样本点与中心点的最远距离
        new_distances = pairwise_distances(points, [x_k], metric='euclidean').flatten()
        new_radius = np.max(new_distances)

        # 构造粒球，包含中心、半径、样本及其隶属度
        granule = {
            "center": x_k,
            "radius": new_radius,  # 使用更新后的半径
            "points": points,  # 每个点包括该点
            "label": label,
            "optimal_alpha": optimal_alpha  # 记录该粒球的最优 alpha
        }

        # 将粒球加入对应标签的列表
        granules_by_class[label].append(granule)

    return granules_by_class




#
# # 合并同质粒球
# def process_homogeneous_granules(granules_by_class):
#     for label, granules in granules_by_class.items():
#         i = 0
#         while i < len(granules):
#             granule_i = granules[i]
#             center_i = granule_i['center']
#             radius_i = granule_i['radius']
#             j = i + 1
#
#             while j < len(granules):
#                 granule_j = granules[j]
#                 center_j = granule_j['center']
#                 radius_j = granule_j['radius']
#
#                 # 计算粒球之间的距离
#                 distance = np.linalg.norm(center_i - center_j)
#
#                 # 判断是否满足合并条件
#                 if distance <= abs(radius_i - radius_j):
#                     # 删除半径较小的粒球
#                     if radius_i < radius_j:
#                         granules.pop(i)
#                         j = i + 1
#                     else:
#                         granules.pop(j)
#                 else:
#                     j += 1
#
#             i += 1
#
#     return granules_by_class

def merge_homogeneous_granules(granules_by_class):
    """
    合并同质粒球，将距离较近的粒球合并为更大的粒球。
    :param granules_by_class: 按类别存储的粒球字典
    :return: 更新后的粒球字典
    """
    for label, granules in granules_by_class.items():
        i = 0
        while i < len(granules):
            granule_i = granules[i]
            center_i = granule_i['center']
            radius_i = granule_i['radius']
            points_i = granule_i['points']

            j = i + 1

            while j < len(granules):
                granule_j = granules[j]
                center_j = granule_j['center']
                radius_j = granule_j['radius']
                points_j = granule_j['points']

                # 计算粒球之间的距离
                distance = np.linalg.norm(center_i - center_j)

                # 判断是否满足合并条件
                if distance < min(radius_i, radius_j):
                    # 合并粒球
                    new_center = (center_i + center_j) / 2
                    new_radius = min(radius_i, radius_j)

                    # 将两个粒球的样本点合并
                    merged_points = points_i + points_j
                    merged_points = np.vstack(merged_points)

                    # 计算每个点到新中心的距离
                    distances = np.linalg.norm(merged_points - new_center, axis=1)

                    # 只保留距离小于等于 new_radius 的点
                    filtered_points = merged_points[distances <= new_radius]

                    # 检查合并后的粒球点集是否为空
                    if filtered_points.size == 0:
                        # print(f"粒球 {i} 和 {j} 合并后点集为空，删除这两个粒球")
                        # 删除粒球 i 和 j
                        granules.pop(j)
                        granules.pop(i)
                        i -= 1  # 更新 i，避免跳过下一个粒球
                        break  # 重新开始内层循环，跳过此轮合并
                    else:
                        # 创建新的粒球
                        new_granule = {
                            "center": new_center,
                            "radius": new_radius,
                            "points": filtered_points,
                            "label": label
                        }

                        # 用新的粒球替换原来的两个粒球
                        granules[i] = new_granule
                        granules.pop(j)  # 删除粒球 j
                else:
                    j += 1

            i += 1

    return granules_by_class


#
# # 处理异质粒球
# def process_heterogeneous_granules_ori(granules_by_class):
#     labels = list(granules_by_class.keys())
#     num_labels = len(labels)
#
#     for idx_i in range(num_labels):
#         label_i = labels[idx_i]
#         granules_i = granules_by_class[label_i]
#
#         for idx_j in range(idx_i + 1, num_labels):
#             label_j = labels[idx_j]
#             granules_j = granules_by_class[label_j]
#
#             to_remove_i = set()
#             to_remove_j = set()
#
#             for granule_idx_i, granule_i in enumerate(granules_i):
#                 center_i = granule_i['center']
#                 radius_i = granule_i['radius']
#
#                 for granule_idx_j, granule_j in enumerate(granules_j):
#                     center_j = granule_j['center']
#                     radius_j = granule_j['radius']
#
#                     # 计算粒球之间的距离
#                     distance = np.linalg.norm(center_i - center_j)
#
#                     if abs(radius_i - radius_j) < distance < radius_i + radius_j:
#                         # 判断哪个粒球的半径更大
#                         if radius_i > radius_j:
#                             # 修改大粒球的半径
#                             new_radius_i = distance - radius_j
#                             radius_i = new_radius_i
#                         else:
#                             # 修改小粒球的半径
#                             new_radius_j = distance - radius_i
#                             radius_j = new_radius_j
#
#                         # 更新粒球的半径
#                         granule_i['radius'] = radius_i
#                         granule_j['radius'] = radius_j
#
#                     elif distance < abs(radius_i - radius_j):
#                         # 粒球嵌套的情况，删除嵌套的两个粒球
#                         to_remove_i.add(granule_idx_i)
#                         to_remove_j.add(granule_idx_j)
#
#             # 删除嵌套的粒球
#             granules_i = [g for idx, g in enumerate(granules_i) if idx not in to_remove_i]
#             granules_j = [g for idx, g in enumerate(granules_j) if idx not in to_remove_j]
#
#             granules_by_class[label_i] = granules_i
#             granules_by_class[label_j] = granules_j
#
#     return granules_by_class

def process_heterogeneous_granules(granules_by_class):
    labels = list(granules_by_class.keys())
    num_labels = len(labels)

    for idx_i in range(num_labels):
        label_i = labels[idx_i]
        granules_i = granules_by_class[label_i]

        for idx_j in range(idx_i + 1, num_labels):
            label_j = labels[idx_j]
            granules_j = granules_by_class[label_j]

            to_remove_i = set()
            to_remove_j = set()

            for granule_idx_i, granule_i in enumerate(granules_i):
                center_i = np.array(granule_i['center'])
                radius_i = granule_i['radius']
                points_i = np.array(granule_i['points'])

                for granule_idx_j, granule_j in enumerate(granules_j):
                    center_j = np.array(granule_j['center'])
                    radius_j = granule_j['radius']
                    points_j = np.array(granule_j['points'])

                    # 计算粒球之间的距离
                    distance = np.linalg.norm(center_i - center_j)

                    if abs(radius_i - radius_j) < distance < radius_i + radius_j:
                        # 判断哪个粒球的半径更大
                        if radius_i > radius_j:
                            # 修改大粒球的半径
                            new_radius_i = distance - radius_j
                            radius_i = new_radius_i
                        else:
                            # 修改小粒球的半径
                            new_radius_j = distance - radius_i
                            radius_j = new_radius_j

                        # 更新粒球的点，保留在新半径范围内的点
                        distances_i = np.linalg.norm(points_i - center_i, axis=1)
                        distances_j = np.linalg.norm(points_j - center_j, axis=1)

                        granule_i['points'] = points_i[distances_i <= radius_i].tolist()
                        granule_j['points'] = points_j[distances_j <= radius_j].tolist()

                        # 更新粒球的半径
                        granule_i['radius'] = radius_i
                        granule_j['radius'] = radius_j

                    elif distance < abs(radius_i - radius_j):
                        # 粒球嵌套的情况，删除嵌套的两个粒球
                        to_remove_i.add(granule_idx_i)
                        to_remove_j.add(granule_idx_j)

            # 删除嵌套的粒球
            granules_i = [g for idx, g in enumerate(granules_i) if idx not in to_remove_i]
            granules_j = [g for idx, g in enumerate(granules_j) if idx not in to_remove_j]

            granules_by_class[label_i] = granules_i
            granules_by_class[label_j] = granules_j

    return granules_by_class


# 更新粒球
# 修改代码以打印总的粒球数量
def update_granules(granules_by_class):
    """
    更新粒球，合并同质粒球并处理异质粒球，同时根据条件删除某些粒球。
    :param granules_by_class: 按类别存储的粒球字典
    :return: 更新后的粒球字典
    """
    final_granules_by_class = {}

    # 进行同质和异质处理，假设 merge_homogeneous_granules 和 process_heterogeneous_granules 已定义
    updated_granules_by_class = merge_homogeneous_granules(granules_by_class)


    updated_granules_by_class = process_heterogeneous_granules(updated_granules_by_class)

    for label, granules in updated_granules_by_class.items():
        updated_granules = []
        radii = [g['radius'] for g in granules]
        mean_radius = np.mean(radii)
        median_radius = np.median(radii)
        threshold_radius = 2 * max(mean_radius, median_radius)

        for granule in granules:
            if granule['radius'] == 0:
                continue
            if len(granule['points']) <= 1:
                continue
            updated_granules.append(granule)

        final_granules_by_class[label] = updated_granules

    # 打印每个类别下的粒球数量
    total_granules = sum(len(granules) for granules in final_granules_by_class.values())
    for label, granules in final_granules_by_class.items():
        print(f"类别 {label} 下的粒球数量: {len(granules)}")

    # 打印总的粒球数量
    print(f"总的粒球数量: {total_granules}")

    return final_granules_by_class


def plot_granules_with_samples(granules):
    """绘制粒球和粒球内的样本"""
    unique_labels = set(granules.keys())
    colors = ['#3C5488', '#ff7f0e']
    label_to_color = {label: colors[i % len(colors)] for i, label in enumerate(sorted(unique_labels))}

    plt.figure(figsize=(7, 6))
    for label, granules_list in granules.items():
        class_color = label_to_color[label]


        for granule in granules_list:
            center = granule['center']
            radius = granule['radius']
            samples = np.array(granule['points'])

            if radius == 0 or len(samples) == 0:
                continue

            # 绘制粒球样本点
            plt.scatter(samples[:, 0], samples[:, 1], c=class_color, alpha=0.5, s=20)

            # 绘制粒球边界
            theta = np.linspace(0, 2 * np.pi, 100)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            plt.plot(x, y, color=class_color, linestyle='--', linewidth=0.8)
            plt.tick_params(axis='both', which='major', labelsize=25)
            plt.ylim(-1.5, 1.5)
            # 绘制粒球中心
            plt.scatter(center[0], center[1], c='black', marker='x', s=30)
    plt.tight_layout()
    plt.title("Granules and Samples")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.legend()
    plt.show()



def add_noise_to_label(data, noise_ratio):
    """
    为数据集添加噪声，随机替换一定比例的标签，适合多类情形
    :param data: 输入的数据集，第一列为标签
    :param noise_ratio: 噪声比例 (0-1)，用于确定添加噪声的样本比例
    :return: 添加噪声后的数据
    """
    # 获取样本数量和标签类别
    sample_number = data.shape[0]
    labels = data[:, 0].astype(int)  # 假设标签在第一列
    unique_labels = np.unique(labels)  # 获取所有唯一的标签类别

    # 计算要替换的噪声样本数量
    noise_sample_size = int(noise_ratio * sample_number)

    # 随机选择要添加噪声的样本索引
    noise_indices = np.random.choice(sample_number, size=noise_sample_size, replace=False)

    # 为这些样本随机分配新的标签，确保新标签不同于原始标签
    new_labels = labels.copy()  # 复制原始标签

    for idx in noise_indices:
        current_label = labels[idx]
        # 选择与当前标签不同的其他类别
        possible_labels = unique_labels[unique_labels != current_label]
        # 从这些类别中随机选择一个作为新的标签
        new_label = np.random.choice(possible_labels)
        new_labels[idx] = new_label

    # 将新标签与数据的特征部分拼接，返回添加噪声后的数据集
    data_with_noise = np.hstack((new_labels.reshape(sample_number, 1), data[:, 1:]))

    return data_with_noise



def plot_granules_no_samples(granules):
    """绘制粒球和粒球内的样本"""
    unique_labels = set(granules.keys())
    colors = [ '#3C5488', '#ff7f0e']
    label_to_color = {label: colors[i % len(colors)] for i, label in enumerate(sorted(unique_labels))}

    plt.figure(figsize=(7, 6))
    for label, granules_list in granules.items():
        class_color = label_to_color[label]


        for granule in granules_list:
            center = granule['center']
            radius = granule['radius']
            # samples = np.array(granule['points'])
            #
            # if radius == 0 or len(samples) == 0:
            #     continue
            #
            # # 绘制粒球样本点
            # plt.scatter(samples[:, 0], samples[:, 1], c=class_color, alpha=0.5, s=20)

            # 绘制粒球边界
            theta = np.linspace(0, 2 * np.pi, 100)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)

            plt.tick_params(axis='both', which='major', labelsize=25)
            plt.plot(x, y, color=class_color, linestyle='--', linewidth=0.8)
            plt.ylim(-1.5, 1.5)

            # 绘制粒球中心
            plt.scatter(center[0], center[1], c='black', marker='x', s=20)
    plt.tight_layout()
    #plt.title("Granules and Samples")
    # plt.xlabel("Feature 1")
    # plt.ylabel("Feature 2")
    #plt.grid(True)
    #plt.legend()
    # plt.show()


def plot_granules_with_all_samples(X, y, granules):
    """
    绘制整个数据集的样本点和粒球
    :param X: 数据集的特征矩阵
    :param y: 数据集的标签数组
    :param granules: 构造的粒球结构
    """
    unique_labels = set(granules.keys())
    colors = ['#3C5488', '#ff7f0e']
    label_to_color = {label: colors[i % len(colors)] for i, label in enumerate(sorted(unique_labels))}

    plt.figure(figsize=(7, 6))
   
    # 绘制整个数据集的样本点
    for label in np.unique(y):
        class_color = label_to_color[label]
        samples = X[y == label]
        plt.scatter(samples[:, 0], samples[:, 1], c=class_color, alpha=0.8, s=10, label=f"Class {label}")

    # 绘制粒球
    for label, granules_list in granules.items():
        class_color = label_to_color[label]

        for granule in granules_list:
            center = granule['center']
            radius = granule['radius']

            if radius == 0:
                continue

            # 绘制粒球边界
            theta = np.linspace(0, 2 * np.pi, 100)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            plt.plot(x, y, color=class_color, linestyle='-', linewidth=0.8)

            # 绘制粒球中心
            plt.scatter(center[0], center[1], c='black', marker='x', s=30)

    # 设置坐标轴范围和刻度
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.xticks(np.arange(-1, 1.2, 0.5))
    plt.yticks(np.arange(-1, 1.2, 0.5))
    
    # 设置刻度字体大小
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.tight_layout()
    # plt.legend()
    # plt.show()



#
# # 主程序
# # 加载数据集并进行预处理
# data = load_breast_cancer()
# X, y = data.data, data.target
#
# 加载fourclass数据集

data = pd.read_csv('../dataset/fourclass.csv',header= None)  # 从文件中读取数据
data = np.array(data)  # 转换为 numpy 数组



X = data[:, 1:]  # 特征数据
y = data[:, 0]  # 标签列 (第一列)
print(data.shape)


scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)


plt.rcParams['font.family'] = 'Times New Roman'


# 计算每个样本的近邻半径（eta）
eta_list = find_neighborhood_radius(X, y)

# 构造每个类别的粒球
granules_by_class = construct_granules_by_class(X, y, eta_list)
plot_granules_with_all_samples(X,y,granules_by_class)
plt.savefig("original1_new.png", dpi=300)
plt.show()

# 更新粒球
granules_by_class_updated = update_granules(granules_by_class)
plot_granules_with_all_samples(X,y,granules_by_class_updated)
plt.savefig("optimized_GBs_new.png", dpi=300)
plt.show()

plot_granules_no_samples(granules_by_class_updated)
plt.savefig("no_samples_new.png", dpi=300)
plt.show()






