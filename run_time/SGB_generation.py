
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
import warnings
import time
from sklearn.metrics import pairwise_distances, accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
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
            nho_distance = np.min(distances[i, same_label_indices])
        else:
            nho_distance = np.inf

        nhe_distance = np.min(distances[i, different_label_indices])
        eta = nhe_distance - 0.01 * nho_distance
        eta_list.append(eta)
    return eta_list


# 计算最优alpha
# def find_optimal_alpha(membership, lam):
#     # Sort the membership values
#     membership_sorted = np.sort(membership)
#
#
#     def compute_loss(alpha):
#         loss1 = np.sum(membership[membership <= alpha])  # 对 membership <= alpha 的部分求和
#         loss2 = np.sum(0.5 - membership[membership > alpha])  # 对 membership > alpha 的部分求和
#         return lam * loss1 + loss2
#
#     # Search for optimal alpha among unique membership values
#     alpha_candidates = np.unique(membership_sorted)
#     losses = np.array([compute_loss(alpha) for alpha in alpha_candidates])
#
#     # Find the alpha corresponding to the minimum loss
#     min_loss_idx = np.argmin(losses)
#     optimal_alpha = alpha_candidates[min_loss_idx]
#
#     return optimal_alpha


def find_optimal_alpha(membership, lam):
    # Adjust membership values: for values greater than 0.5, transform them to 1 - membership
    membership_adjusted = np.where(membership > 0.5, 1 - membership, membership)

    # Sort the adjusted membership values
    membership_sorted = np.sort(membership_adjusted)

    # Vectorized loss function
    def compute_loss(alpha):
        loss1 = np.sum(membership_adjusted[membership_adjusted <= alpha])  # 对 membership <= alpha 的部分求和
        loss2 = np.sum(0.5 - membership_adjusted[membership_adjusted > alpha])  # 对 membership > alpha 的部分求和
        return lam * loss1 + loss2

    # Search for optimal alpha among unique membership values
    alpha_candidates = np.unique(membership_sorted)
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
        # print("memberships", np.round(memberships, 3))


        # # 找到最优的 alpha
        # optimal_alpha = find_optimal_alpha(memberships, lam=1)
        # optimal_alpha_list.append(optimal_alpha)


        # 筛选出隶属度大于 alpha 的样本
        inside_granule_indices = np.where(memberships >= 0.75)[0]

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
            "optimal_alpha": 0.75  # 记录该粒球的最优 alpha
        }
        # 将粒球加入对应标签的列表
        granules_by_class[label].append(granule)

    return granules_by_class



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

                    # if abs(radius_i - radius_j) < distance < radius_i + radius_j:
                    if  distance < radius_i + radius_j:
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

    # 进行同质和异质处理
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





# KNN分类函数，使用最近邻的粒球标签
def knn_classify(test_samples, granules_by_class):
    predictions = []

    for test_sample in test_samples:
        nearest_distance = float('inf')
        nearest_label = None
        nearest_granules = []  # 存储距离相同的粒球

        # 计算测试样本到每个粒球的距离
        for label, granules in granules_by_class.items():
            for granule in granules:
                center = granule['center']
                radius = granule['radius']
                points = granule['points']
                distance = np.linalg.norm(test_sample - center) - radius

                # 如果找到更近的粒球，更新最近距离和标签
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_label = label
                    nearest_granules = [(label, granule)]
                # 如果距离相同，添加到候选列表
                elif distance == nearest_distance:
                    nearest_granules.append((label, granule))

        # 如果有多个距离相同的粒球，使用points/radius比值决定
        if len(nearest_granules) > 1:
            max_density = -1
            for label, granule in nearest_granules:
                # 计算粒球密度指标
                density = len(granule['points']) / granule['radius'] if granule['radius'] > 0 else 0
                if density > max_density:
                    max_density = density
                    nearest_label = label

        predictions.append(nearest_label)

    return predictions



# # KNN分类函数，使用最近邻的粒球标签
# def knn_classify(test_samples, granules_by_class):
#     predictions = []

#     for test_sample in test_samples:
#         nearest_distance = float('inf')
#         nearest_label = None

#         # 计算测试样本到每个粒球的距离
#         for label, granules in granules_by_class.items():
#             for granule in granules:
#                 center = granule['center']
#                 radius = granule['radius']
#                 distance = np.linalg.norm(test_sample - center) - radius

#                 # 找到最近的粒球
#                 if distance < nearest_distance:
#                     nearest_distance = distance
#                     nearest_label = label

#         predictions.append(nearest_label)

#     return predictions






def check_coverage(granules_by_class, original_data):
    """
    检查更新后的粒球集合是否覆盖了原始数据集。
    :param granules_by_class: 更新后的粒球字典
    :param original_data: 原始数据集（通常为训练集）
    :return: 布尔值，表示粒球集合是否覆盖原始数据集
    """
    # 原始数据集的样本集合
    original_samples_set = set(tuple(sample) for sample in original_data)

    # 更新后的粒球集合的样本集合
    covered_samples_set = set()

    # 遍历所有粒球，收集其包含的样本点
    for label, granules in granules_by_class.items():
        for granule in granules:
            for point in granule['points']:
                covered_samples_set.add(tuple(point))  # 将点转换为不可变的元组以便添加到集合中

    # 检查是否覆盖原始数据集
    is_covered = original_samples_set.issubset(covered_samples_set)

    print(f"原始数据集中的样本点数: {len(original_samples_set)}")
    print(f"更新后粒球覆盖的样本点数: {len(covered_samples_set)}")
    print(f"粒球集合是否覆盖原始数据集: {is_covered}")

    return len(covered_samples_set), is_covered

def new_granule_knn_algorithm(train, test):
    print("******************************")
    """
    使用粒球生成和KNN分类进行分类的算法封装
    :param train: 训练集数据 (包含标签和特征)
    :param test: 测试集数据 (包含标签和特征)
    :return: 准确率, 精确率, 召回率, F1分数, 训练耗时, 测试耗时
    """

    # 记录训练开始时间
    train_start_time = time.time()

    # 计算每个样本的近邻半径（eta）
    eta_list = find_neighborhood_radius(train[:, 1:], train[:, 0])

    # 构造每个类别的粒球
    granules_by_class = construct_granules_by_class(train[:, 1:], train[:, 0], eta_list)

    # 更新粒球
    granules_by_class_updated = update_granules(granules_by_class)

    # 检查覆盖率（可选）
    is_covered = check_coverage(granules_by_class_updated, train)

    # 记录训练结束时间
    train_end_time = time.time()
    train_time = train_end_time - train_start_time

    # 记录测试开始时间
    test_start_time = time.time()

    # 使用KNN分类测试样本
    y_pred = knn_classify(test[:, 1:], granules_by_class_updated)

    # 记录测试结束时间
    test_end_time = time.time()
    test_time = test_end_time - test_start_time

    # 计算分类评估指标
    accuracy = accuracy_score(test[:, 0], y_pred)
    precision = precision_score(test[:, 0], y_pred, average='macro')
    recall = recall_score(test[:, 0], y_pred, average='macro')
    f1 = f1_score(test[:, 0], y_pred, average='macro')

    return accuracy, precision, recall, f1, train_time, test_time



#
# # 主程序
# # 加载数据集并进行预处理
# # data = load_breast_cancer()
# # X, y = data.data, data.target
#
# # 加载fourclass数据集
#
# data = pd.read_excel('dataset/BC_DDSM.xlsx',header =0)  # 从文件中读取数据
# data = np.array(data)  # 转换为 numpy 数组
# X = data[:, 1:]  # 特征数据
# y = data[:, 0]  # 标签列 (第一列)
# print(data.shape)
#
# # # 标准化
# # scaler =  StandardScaler()
# scaler = MinMaxScaler()
# data_array_normalized = scaler.fit_transform(data[:, 1:])
# data = np.hstack((data[:, [0]], data_array_normalized))
#
#
#
# #
# # # 分离特征和标签
# X_initial = data[:, 1:]
# y_initial = data[:, 0]
# #
# # 特征选择，选择最重要的50个特征
# selector = SelectKBest(score_func=f_classif, k=30)
# X_selected = selector.fit_transform(X_initial, y_initial)
# data = np.hstack((y_initial.reshape(-1, 1), X_selected))
# print("特征选择后数据量：", data.shape)
#
#
# # 分割训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
#
# # 计算每个样本的近邻半径（eta）
# eta_list = find_neighborhood_radius(X_train, y_train)
#
# # 构造每个类别的粒球
# granules_by_class = construct_granules_by_class(X_train, y_train, eta_list)
#
# # 更新粒球
# granules_by_class_updated = update_granules(granules_by_class)
#
# # 使用KNN分类测试样本
# y_pred = knn_classify(X_test, granules_by_class_updated)
#
# # 评估模型性能
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
#
# # 打印性能评估结果
# print("Model Performance:")
# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1 Score: {f1:.4f}")
#
#

#
#
#
# #
# #
# # #  # A 实验 比较各个粒球生成算法的粒球个数与覆盖率
# def gb_new_summary(train):
#     """
#     GB_ORIGIN 算法的粒球划分与覆盖检查。
#     :param train: 训练数据集，包含标签列
#     :return: 产生的粒球数量以及覆盖的样本数量
#     """
#     data_num = train.shape[0]
#
#     # 计算每个样本的近邻半径（eta）
#     eta_list = find_neighborhood_radius(train[:, 1:], train[:, 0])
#
#     # 构造每个类别的粒球
#     granules_by_class = construct_granules_by_class(train[:, 1:], train[:, 0], eta_list)
#
#     # 更新粒球
#     granules_by_class_updated = update_granules(granules_by_class)
#
#     # 计算总的粒球个数
#     num_gb = sum(len(granules) for granules in granules_by_class_updated.values())
#
#     # 计算覆盖样本数
#     covered_sample_count, is_covered = check_coverage(granules_by_class_updated, train)
#
#     # 计算覆盖率
#     cover_radio = covered_sample_count / data_num
#
#     return num_gb,covered_sample_count, cover_radio
#
#
#
# warnings.filterwarnings("ignore")
# # 定义数据集文件路径列表
#
# # 定义数据集文件路径列表
# datasets = [
#     # 按 数据集样本个数 排序
#     # 'dataset/echocardiogram.csv',
#     # 'dataset/Iris.csv',
#     # 'dataset/parkinsons.csv',
#     # 'dataset/seeds.csv',
#     # 'dataset/Glass.csv',
#     # 'dataset/audiology.csv',
#     # 'dataset/wdbc.csv',
#     # 'dataset/Balance Scale.csv',
#     # 'dataset/Credit Approval.csv',
#     # 'dataset/Pima Indians Diabetes Database.csv',
#
#     # 'dataset/diabetes.csv',
#     # 'dataset/fourclass1.csv',
#     # 'dataset/biodeg.csv',
#     # 'dataset/data_banknote_authentication.csv',
#     # 'dataset/cardio.csv',
#     # 'dataset/thyroid.csv',
#     # 'dataset/rice.csv',
#     # 'dataset/waveform.csv',
#     # 'dataset/page-blocks.csv',
#     # 'dataset/eletrical.csv',
# ]
#
#
# num_gb_list = []
# covered_samples = []
# cover_radio_list = []
#
# # 逐个数据集进行处理
# for dataset_path in datasets:
#     print(f"\n处理数据集: {dataset_path}")
#
#     # 读取数据集
#     data = pd.read_csv(dataset_path, header=None)
#     data = np.array(data)  # 转换为 numpy 数组
#     # print(data.shape)
#
#
#     # 去重：保留唯一行
#     data = np.unique(data, axis=0)
#
#     # 去重：只保留每组的唯一特征组合
#     data_temp = []
#     data_list = data.tolist()
#     data = []
#     for data_single in data_list:
#         if data_single[1:] not in data_temp:
#             data_temp.append(data_single[1:])
#             data.append(data_single)
#     data = np.array(data)
#
#     # 使用 gb_origin_summary 计算覆盖样本数和覆盖率
#     num_gb ,covered_sample_count, cover_radio = gb_new_summary(data)
#     print(f"覆盖率: {cover_radio}")
#
#    # 将结果添加到列表中
#     num_gb_list.append(num_gb)
#     cover_radio_list.append(cover_radio)
#     covered_samples.append(covered_samples)
#
# # 遍历所有数据集后打印粒球个数和覆盖率
# # print("\n粒球个数列表:", num_gb_list)
# # print("\n覆盖的样本数列表:", num_gb_list)
# # print("\n覆盖率列表:", cover_radio_list)
#
#
