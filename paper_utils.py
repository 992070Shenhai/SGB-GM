import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
from sklearn.datasets import make_blobs
from math import sqrt
from _shadowed_c_means_clustering import ShadowedCMeans
from _shadowed_c_means_clustering import *
from _shadowed_c_means_up_clustering import *
from _shadowed_weighted_c_means_clustering import *
from _shadowed_region_c_means_clustering import ShadowedCMeansRegion
from clustering_algorithm_utils import *
import os


## 对数据集的各个类别下的子数据集聚类
def perform_subset_clustering(X, y):
    unique_labels = np.unique(y)
    eachclass_clustering_results = []  # List to store clustering results for each data subset
    all_clusters = []  # List to store all clusters from all data subsets
    model = ShadowedCMeans()  # Create an instance of ShadowedCMeans
    # Uncomment any alternative model to experiment with different clustering methods:
    # model = ShadowedCMeansUp(optimization_type="none", verbose=True)
    # model = ShadowedWeightedCMeans(verbose=True)
    # model = ShadowedCMeansRegion(verbose=True)

    for label in unique_labels:
        data_subset = X[y == label]  # Get data corresponding to the current label (class)
        #print(f"data_subset shpae is {data_subset.shape}")
        n = len(data_subset)  # Get the number of samples in this subset

        if n < 2:
            print(f"类别 {label} 的样本数不足以进行聚类。")
            continue

        n_clusters = int(sqrt(n))  # Set the number of clusters to the square root of the number of samples
        print(f"对类别 {label} 进行 {n_clusters} 个聚类...")
        model.n_clusters = n_clusters  # Update the model with the number of clusters

        # Perform clustering and get the results
        output, instances_status, centroids = model.fit(data_subset)  # `fit` returns the clustering output, instance status, and centroids
        #print(f"instances_status shape is {instances_status.shape}")
        # Store the clustering result for this subset
        class_results = []  # Store individual class results
        for cluster_id in range(n_clusters):
            # Get core samples (instances_status == 2)
            core_samples = data_subset[instances_status[cluster_id] == 2]
            num_core_samples = len(core_samples)

            # Get core + shadow samples (instances_status == 1 or 2)
            core_and_shadow_samples = data_subset[np.logical_or(instances_status[cluster_id] == 1, instances_status[cluster_id] == 2)]
            num_core_and_shadow_samples = len(core_and_shadow_samples)

            # Store cluster center
            cluster_center = centroids[cluster_id]

            # Store results for this cluster
            cluster_result = {
                'label': label,
                'cluster_id': cluster_id,
                #'instances_status': instances_status[cluster_id],  # status for each instance
                'core_samples': core_samples,
                'num_core_samples': num_core_samples,
                'core_and_shadow_samples': core_and_shadow_samples,
                'num_core_and_shadow_samples': num_core_and_shadow_samples,
                'cluster_center': cluster_center
            }
            class_results.append(cluster_result)
            all_clusters.append(cluster_result)

            # Print cluster details
            # print(f"类别: {label}, 簇ID: {cluster_id}")
            # print(f"核心区域样本: {core_samples}")
            # print(f"核心区域样本数: {num_core_samples}")
            # print(f"核心区域样本索引: {np.where(instances_status[cluster_id] == 2)}")
            # print(f"核心+边界区域样本: {core_and_shadow_samples}")
            # print(f"核心+边界区域样本数: {num_core_and_shadow_samples}")
            # print(f"簇中心: {cluster_center}\n")n

        # Append class-level clustering results
        eachclass_clustering_results.append({
            'label': label,
            'n_clusters': n_clusters,
            'results': class_results
        })

    return eachclass_clustering_results, all_clusters
## 对all_clusters 的结果画图
## 对eachclass_clustering_results 进行同质和异质粒球进行合并









###########################################################
#产生不同形式的GB，进行画图

def generate_granules(eachclass_clustering_results):
    """
    Generates different types of granules for each cluster based on the results from perform_subset_clustering.

    GB_type_1: samples = cluster['core_and_shadow_samples'], radius = np.max(distances)
    GB_type_2: samples = cluster['core_and_shadow_samples'], radius = np.mean(distances)
    GB_type_3: samples = cluster['core_samples'], radius = np.max(distances)
    GB_type_4: samples = cluster['core_samples'], radius = np.mean(distances)

    Returns:
        granules: A dictionary where each type (GB_type_1, GB_type_2, etc.) contains a dictionary of classes.
                  Each class has a list of granules where each granule contains cluster_id, cluster_center, radius, samples.
                  The class labels will be in the format: "label 0", "label 1", etc.
    """

    granules = {
        'GB_type_1': {},  # core_and_shadow_samples with max distance
        'GB_type_2': {},  # core_and_shadow_samples with mean distance
        'GB_type_3': {},  # core_samples with max distance
        'GB_type_4': {}   # core_samples with mean distance
    }

    # Iterate over each class's clustering results
    for class_result in eachclass_clustering_results:
        label = f"label {class_result['label']}"  # Create a key with "label X"

        # Initialize an empty list for each class under each granule type
        if label not in granules['GB_type_1']:
            granules['GB_type_1'][label] = []
            granules['GB_type_2'][label] = []
            granules['GB_type_3'][label] = []
            granules['GB_type_4'][label] = []

        for cluster_result in class_result['results']:
            cluster_id = cluster_result['cluster_id']  # Get the cluster ID
            cluster_center = cluster_result['cluster_center']

            # 1. GB_type_1: core_and_shadow_samples, radius = np.max(distances)
            core_and_shadow_samples_max = np.array(cluster_result['core_and_shadow_samples'])
            distances_core_and_shadow = np.linalg.norm(core_and_shadow_samples_max - cluster_center, axis=1)
            radius_max_core_and_shadow = np.max(distances_core_and_shadow) if len(distances_core_and_shadow) > 0 else 0

            # 2. GB_type_2: core_and_shadow_samples, radius = np.mean(distances)
            radius_mean_core_and_shadow = np.mean(distances_core_and_shadow) if len(
                distances_core_and_shadow) > 0 else 0
            core_and_shadow_samples_mean = core_and_shadow_samples_max[
                distances_core_and_shadow <= radius_mean_core_and_shadow]

            # 3. GB_type_3: core_samples, radius = np.max(distances)
            core_samples = np.array(cluster_result['core_samples'])
            distances_core = np.linalg.norm(core_samples - cluster_center, axis=1)
            radius_max_core = np.max(distances_core) if len(distances_core) > 0 else 0

            # 4. GB_type_4: core_samples, radius = np.mean(distances)
            radius_mean_core = np.mean(distances_core) if len(distances_core) > 0 else 0
            core_samples_mean = core_samples[distances_core <= radius_mean_core]

            # Append the granules with their respective cluster IDs, centers, radii, and samples
            granules['GB_type_1'][label].append({
                'cluster_id': cluster_id,
                'cluster_center': cluster_center.flatten(),
                'radius': radius_max_core_and_shadow,
                'samples': core_and_shadow_samples_max.tolist()
            })
            granules['GB_type_2'][label].append({
                'cluster_id': cluster_id,
                'cluster_center': cluster_center.flatten(),
                'radius': radius_mean_core_and_shadow,
                'samples': core_and_shadow_samples_mean.tolist()
            })
            granules['GB_type_3'][label].append({
                'cluster_id': cluster_id,
                'cluster_center': cluster_center.flatten(),
                'radius': radius_max_core,
                'samples': core_samples.tolist()
            })
            granules['GB_type_4'][label].append({
                'cluster_id': cluster_id,
                'cluster_center': cluster_center.flatten(),
                'radius': radius_mean_core,
                'samples': core_samples_mean.tolist()
            })

    return granules




def plot_granules(granules, X, y):
    """
    Plots the granules for each type (GB_type_1, GB_type_2, GB_type_3, GB_type_4) as separate subplots.
    Each class is represented with a unique color for its data points (as circles).
    Granules (clusters) within each class are differentiated by different circle boundary colors.
    The cluster centers are also plotted as 'x' marks.

    Arguments:
    - granules: A dictionary containing four types of granules: GB_type_1, GB_type_2, GB_type_3, GB_type_4.
    - X: The original dataset samples.
    - y: The labels for the dataset samples.
    """

    fig, axs = plt.subplots(2, 2, figsize=(20, 15))

    # Define some colors for different classes and cluster circles
    class_colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']  # Colors for different classes (data points)
    circle_colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']  # Different colors for cluster circles

    # Plot original data points for each class as circles ('o')
    for class_label in np.unique(y):
        class_data = X[y == class_label]
        for ax in axs.flat:
            ax.scatter(class_data[:, 0], class_data[:, 1], label=f'Class {class_label}',
                       c=class_colors[class_label % len(class_colors)], marker='o', alpha=0.8)

    # Plot GB_type_1: Core + Shadow Samples, Max Distance
    axs[0, 0].set_title('GB_type_1: Core + Shadow Samples, Max Distance')
    for label, granules_list in granules['GB_type_1'].items():
        class_color = class_colors[int(label.split()[-1]) % len(class_colors)]  # Same color for class data points
        for i, granule in enumerate(granules_list):
            center = granule['cluster_center']
            radius = granule['radius']

            # Plot the cluster boundary
            circle_color = circle_colors[i % len(circle_colors)]  # Different color for cluster boundary
            circle = Circle(center, radius, color=circle_color, fill=False, linestyle='--', alpha=0.8)
            axs[0, 0].add_patch(circle)

            # Plot the cluster center as an 'x' mark
            axs[0, 0].scatter(center[0], center[1], c='black', marker='x', s=100, label='Center')

    axs[0, 0].set_aspect('equal', adjustable='box')

    # Plot GB_type_2: Core + Shadow Samples, Mean Distance
    axs[0, 1].set_title('GB_type_2: Core + Shadow Samples, Mean Distance')
    for label, granules_list in granules['GB_type_2'].items():
        class_color = class_colors[int(label.split()[-1]) % len(class_colors)]  # Same color for class data points
        for i, granule in enumerate(granules_list):
            center = granule['cluster_center']
            radius = granule['radius']

            # Plot the cluster boundary
            circle_color = circle_colors[i % len(circle_colors)]  # Different color for cluster boundary
            circle = Circle(center, radius, color=circle_color, fill=False, linestyle='--', alpha=0.8)
            axs[0, 1].add_patch(circle)

            # Plot the cluster center as an 'x' mark
            axs[0, 1].scatter(center[0], center[1], c='black', marker='x', s=100, label='Center')

    axs[0, 1].set_aspect('equal', adjustable='box')

    # Plot GB_type_3: Core Samples, Max Distance
    axs[1, 0].set_title('GB_type_3: Core Samples, Max Distance')
    for label, granules_list in granules['GB_type_3'].items():
        class_color = class_colors[int(label.split()[-1]) % len(class_colors)]  # Same color for class data points
        for i, granule in enumerate(granules_list):
            center = granule['cluster_center']
            radius = granule['radius']

            # Plot the cluster boundary
            circle_color = circle_colors[i % len(circle_colors)]  # Different color for cluster boundary
            circle = Circle(center, radius, color=circle_color, fill=False, linestyle='--', alpha=0.8)
            axs[1, 0].add_patch(circle)

            # Plot the cluster center as an 'x' mark
            axs[1, 0].scatter(center[0], center[1], c='black', marker='x', s=100, label='Center')

    axs[1, 0].set_aspect('equal', adjustable='box')

    # Plot GB_type_4: Core Samples, Mean Distance
    axs[1, 1].set_title('GB_type_4: Core Samples, Mean Distance')
    for label, granules_list in granules['GB_type_4'].items():
        class_color = class_colors[int(label.split()[-1]) % len(class_colors)]  # Same color for class data points
        for i, granule in enumerate(granules_list):
            center = granule['cluster_center']
            radius = granule['radius']

            # Plot the cluster boundary
            circle_color = circle_colors[i % len(circle_colors)]  # Different color for cluster boundary
            circle = Circle(center, radius, color=circle_color, fill=False, linestyle='--', alpha=0.8)
            axs[1, 1].add_patch(circle)

            # Plot the cluster center as an 'x' mark
            axs[1, 1].scatter(center[0], center[1], c='black', marker='x', s=100, label='Center')

    axs[1, 1].set_aspect('equal', adjustable='box')

    # Add labels and grid to all subplots
    for ax in axs.flat:
        ax.set(xlabel='Feature 1', ylabel='Feature 2')
        ax.grid(True)

    plt.tight_layout()

    # Save the figure with 300 dpi resolution in a "figures" folder
    if not os.path.exists("figures"):
        os.makedirs("figures")
    plt.savefig('figures/granules_plot.png', dpi=300)

    # Show the plot
    plt.show()


def plot_granules_with_samples(granules):
    """
    绘制粒球和粒球内的样本。

    参数：
    - granules: 包含不同类型的粒球的字典结构。

    要求：
    - 绘制每个粒球内的样本。
    - 针对不同类别下的粒球，使用相同的颜色。

    粒球结构示例：
    granules = {
        'GB_type_1': {
            'label 1': [granule_1, granule_2, ...],
            'label 2': [granule_3, granule_4, ...],
            ...
        },
        'GB_type_2': { ... },
        ...
    }
    每个 granule 是一个包含 'cluster_center'、'radius' 和 'samples' 的字典。
    """
    # 获取粒球类型列表
    granule_types = list(granules.keys())
    num_types = len(granule_types)

    # 创建子图，布局为 2 行 2 列（根据粒球类型数量调整）
    fig_rows = (num_types + 1) // 2
    fig_cols = 2 if num_types > 1 else 1
    fig, axs = plt.subplots(fig_rows, fig_cols, figsize=(10 * fig_cols, 7 * fig_rows))
    axs = axs.flatten() if num_types > 1 else [axs]

    # 收集所有唯一的类别标签
    all_labels = set()
    for granule_type in granules.values():
        for label in granule_type.keys():
            all_labels.add(label)

    # 定义颜色列表
    class_colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'orange', 'purple', 'brown']

    # 创建类别标签到颜色的映射
    label_to_color = {label: class_colors[i % len(class_colors)] for i, label in enumerate(sorted(all_labels))}

    for idx, granule_type in enumerate(granule_types):
        ax = axs[idx]
        ax.set_title(f'{granule_type}')

        # 获取该类型的粒球数据
        granule_data = granules[granule_type]

        # 绘制粒球
        for label, granules_list in granule_data.items():
            class_color = label_to_color[label]  # 使用映射的颜色

            for i, granule in enumerate(granules_list):
                center = granule['cluster_center']
                radius = granule['radius']

                if radius == 0 or len(granule['samples']) == 0:
                    continue  # 跳过无效的粒球

                # 只修改三个参数：
                # 1. 数据点的颜色和大小设置
                samples = np.array(granule['samples'])
                if samples.ndim == 1:
                    samples = np.expand_dims(samples, axis=0)
                ax.scatter(samples[:, 0], samples[:, 1], c=class_color, marker='o', alpha=1, s=5)  # 设置数据点大小为5

                # 2. 粒球边界的颜色和大小设置
                theta = np.linspace(0, 2 * np.pi, 100)
                x = center[0] + radius * np.cos(theta)
                y = center[1] + radius * np.sin(theta)
                ax.plot(x, y, color=class_color, linestyle='-', linewidth=0.8)  # 设置粒球边界线宽为0.8

                # 3. 粒球中心的颜色和标记大小
                ax.scatter(center[0], center[1], c='black', marker='x', s=30)  # 设置中心标记大小为100

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True)

    plt.tight_layout()

    # 保存图像
    if not os.path.exists("figures"):
        os.makedirs("figures")
    plt.savefig('figures/granules_with_samples_plot.png', dpi=300)

    # 显示图像
    plt.show()




###################################################################################
#合并规则，检查重叠

def update_granules(granules):
    """
    对粒球进行同质和异质的监测和处理，更新粒球集合。

    - 删除半径为 0 的粒球。
    - 删除粒球内只有一个样本的粒球。
    - 删除半径大于所有粒球半径均值（或中位数）2 倍的粒球。

    :param granules: 由 generate_granules 函数生成的粒球数据结构
    :return: 更新后的 granules，包含处理后的粒球
    """
    #print(f"granules = {granules}")
    # 处理同质粒球
    print("Starting homogeneous granule processing...")
    granules = process_homogeneous_granules(granules)
    #print("Homogeneous granule processing completed.\n")

    # 处理异质粒球
    print("Starting heterogeneous granule processing...")
    granules = process_heterogeneous_granules(granules)
    #print("Heterogeneous granule processing completed.\n")



    for granule_type in granules:
        granule = granules[granule_type]
        for label in granule:
            granules_list = granule[label]

            # 计算所有粒球的半径，用于计算均值和中位数
            all_radii = [g['radius'] for g in granules_list if g['radius'] > 0]

            if len(all_radii) == 0:
                print(f"No valid granules in {granule_type} {label} after processing.")
                granule[label] = []  # 清空该标签下的粒球
                continue

            mean_radius = np.mean(all_radii)
            median_radius = np.median(all_radii)
            threshold_radius = 2 * max(mean_radius, median_radius)


            # 删除不符合条件的粒球
            new_granules_list = []
            for g in granules_list:
                if g['radius'] == 0:
                    continue  # 删除半径为 0 的粒球
                if len(g['samples']) <= 1:
                    continue  # 删除样本数小于等于 1 的粒球
                if g['radius'] > threshold_radius:
                    continue  # 删除半径大于阈值的粒球
                new_granules_list.append(g)

            # 更新粒球列表
            granule[label] = new_granules_list
        print(len(granules))

    return granules




def update_granules_new(granules):
    """
    对粒球进行同质和异质的监测和处理，更新粒球集合。

    - 删除半径为 0 的粒球。
    - 删除粒球内只有一个样本的粒球。
    - 删除半径大于所有粒球半径均值（或中位数）2 倍的粒球。

    :param granules: 由 generate_granules 函数生成的粒球数据结构
    :return: 更新后的 granules，包含处理后的粒球
    """
    # 打印初始粒球的个数
    # print("Initial granule counts per type and label:")
    # total_initial_granules = 0
    # for granule_type in granules:
    #     granule = granules[granule_type]
    #     for label in granule:
    #         num_granules = len(granule[label])
    #         print(f"{granule_type} - Label {label}: {num_granules} granules")
    #         total_initial_granules += num_granules
    # print(f"Total initial granules: {total_initial_granules}\n")

    # 处理同质粒球
    print("Starting homogeneous granule processing...")
    granules = process_homogeneous_granules(granules)

    # 处理异质粒球
    print("Starting heterogeneous granule processing...")
    granules = process_heterogeneous_granules(granules)



    # 打印处理后粒球的个数
    # print("Granule numbers after processing:")
    # total_granules = 0
    # for granule_type in granules:
    #     granule = granules[granule_type]
    #     for label in granule:
    #         num_granules = len(granule[label])
    #         print(f"{granule_type} - Label {label}: {num_granules} granules")
    #         total_granules += num_granules
    # print(f"Total granules after processing: {total_granules}\n")





    # 删除不符合条件的粒球
    for granule_type in granules:
        granule = granules[granule_type]
        for label in granule:
            granules_list = granule[label]

            # 计算所有粒球的半径，用于计算均值和中位数
            all_radii = [g['radius'] for g in granules_list if g['radius'] > 0]

            if len(all_radii) == 0:
                print(f"No valid granules in {granule_type} {label} after processing.")
                granule[label] = []  # 清空该标签下的粒球
                continue

            mean_radius = np.mean(all_radii)
            median_radius = np.median(all_radii)
            threshold_radius = 2 * max(mean_radius, median_radius)

            # 删除不符合条件的粒球
            new_granules_list = []
            for g in granules_list:
                if g['radius'] == 0:
                    continue  # 删除半径为 0 的粒球
                if len(g['samples']) <= 1:
                    continue  # 删除样本数小于等于 1 的粒球
                if g['radius'] > threshold_radius:
                    continue  # 删除半径大于阈值的粒球
                new_granules_list.append(g)

            # 更新粒球列表
            granule[label] = new_granules_list

    # 打印异质粒球处理后粒球的个数
    # print("\nUpdated granule counts per type and label:")
    # total_final_granules = 0
    # for granule_type in granules:
    #     granule = granules[granule_type]
    #     for label in granule:
    #         num_granules = len(granule[label])
    #         #print(f"{granule_type} - Label {label}: {num_granules} granules")
    #         total_final_granules += num_granules
    #print(f"Total updated granules: {total_final_granules}\n")

    return granules







## 同质粒球的判断
def check_homogeneous_invasive_overlap_and_nested(granule_ball_overlap_coef):
    """判断是否满足同质侵入重叠条件"""
    c1 = granule_ball_overlap_coef['cluster_center_i']
    c2 = granule_ball_overlap_coef['cluster_center_j']
    r1 = granule_ball_overlap_coef['radius_i']
    r2 = granule_ball_overlap_coef['radius_j']
    distance = np.linalg.norm(c1 - c2)
    return distance <= max(r1, r2)

## 同质粒球的处理方式
def process_homogeneous_granules(granules):
    """
    对粒球进行同质处理，合并满足条件的同质粒球。
    """

    # 遍历每个粒球类型（训练集或测试集）
    for granule_type in granules:
        granule = granules[granule_type]
        # 遍历每个类别标签
        for label in granule:
            each_class_granule = granule[label]
            i = 0
            while i < len(each_class_granule):
                granule_ball_i = each_class_granule[i]
                center_i = np.array(granule_ball_i['cluster_center'])
                radius_i = granule_ball_i['radius']
                j = i + 1

                while j < len(each_class_granule):
                    granule_ball_j = each_class_granule[j]
                    center_j = np.array(granule_ball_j['cluster_center'])
                    radius_j = granule_ball_j['radius']

                    # 计算粒球之间的距离
                    distance = np.linalg.norm(center_i - center_j)

                    # 判断是否满足同质侵入重叠或嵌套条件
                    if distance <= max(radius_i, radius_j):

                        # 新增逻辑：判断半径倍数
                        if radius_i > 2 * radius_j or radius_j > 2 * radius_i:
                            # 调用 process_homogeneous_gbs 进行处理
                            new_granule_ball_i, new_granule_ball_j = process_homogeneous_gbs(granule_ball_i,granule_ball_j)

                        else:
                            # 满足条件，调用合并粒球函数
                            new_granule_ball_i, new_granule_ball_j = process_granules_merging(granule_ball_i,granule_ball_j, option=1)


                        # 更新粒球 i 和粒球 j 的信息
                        each_class_granule[i] = new_granule_ball_i  # 更新粒球 i 的信息
                        each_class_granule[j] = new_granule_ball_j  # 更新粒球 j 的信息

                        # 更新粒球 i 的中心和半径
                        granule_ball_i = new_granule_ball_i
                        center_i = new_granule_ball_i['cluster_center']
                        radius_i = new_granule_ball_i['radius']
                        j += 1
                    else:
                        # 不满足合并条件，继续检查下一个粒球
                        j += 1

                # 只有在没有合并后，才增加 i 的值
                i += 1

    return granules




## 处理同质粒球方式:不合并粒球，调整粒球范围
def process_homogeneous_gbs(granule_i, granule_j):
    """
    更新粒球 i 和粒球 j 的半径，较大半径的粒球的半径修改为两个粒球中心的距离，
    较小半径的粒球保持不变。

    参数：
        granule_i (dict): 粒球 i，包含 'cluster_center'、'radius'、'samples'。
        granule_j (dict): 粒球 j，结构同上。

    返回：
        result (tuple): 返回更新后的粒球 (new_granule_ball_i, new_granule_ball_j)。
    """
    # 获取两个粒球的半径和中心
    radius_i = granule_i['radius']
    radius_j = granule_j['radius']
    center_i = np.array(granule_i['cluster_center'])
    center_j = np.array(granule_j['cluster_center'])

    # 计算粒球 i 与粒球 j 的中心距离
    distance_ij = np.linalg.norm(center_i - center_j)

    # 判断哪个粒球的半径更大
    if radius_i > radius_j:
        # 粒球 i 半径较大，更新粒球 i 的半径为两球中心的距离
        radius_i = distance_ij
        # 计算粒球 i 内部样本到新中心的距离，选择在新半径内的样本
        samples_i = np.array(granule_i['samples'])
        distances_to_center_i = np.linalg.norm(samples_i - center_i, axis=1)
        samples_within_new_radius_i = samples_i[distances_to_center_i <= radius_i]

        # 更新粒球 i 的信息
        new_granule_ball_i = {
            'cluster_center': center_i.tolist(),  # 粒球 i 的中心保持不变
            'radius': radius_i,  # 更新后的半径
            'samples': samples_within_new_radius_i.tolist()  # 更新后的样本
        }
        # 粒球 j 不变
        new_granule_ball_j = granule_j
    else:
        # 粒球 j 半径较大，更新粒球 j 的半径为两球中心的距离
        radius_j = distance_ij
        # 计算粒球 j 内部样本到新中心的距离，选择在新半径内的样本
        samples_j = np.array(granule_j['samples'])
        distances_to_center_j = np.linalg.norm(samples_j - center_j, axis=1)
        samples_within_new_radius_j = samples_j[distances_to_center_j <= radius_j]

        # 更新粒球 j 的信息
        new_granule_ball_j = {
            'cluster_center': center_j.tolist(),  # 粒球 j 的中心保持不变
            'radius': radius_j,  # 更新后的半径
            'samples': samples_within_new_radius_j.tolist()  # 更新后的样本
        }
        # 粒球 i 不变
        new_granule_ball_i = granule_i

    # 返回更新后的粒球 i 和粒球 j
    return new_granule_ball_i, new_granule_ball_j



## 合并粒球
def process_granules_merging(granule_i, granule_j, option = 1):
    """
    处理粒球的合并，返回合并后的粒球 new_granule_ball 和粒球 j。

    参数：
        granule_i (dict): 粒球 i，包含 'cluster_center'、'radius'、'samples'。
        granule_j (dict): 粒球 j，结构同上。
        option (int): 合并方式，1 或 2。

    返回：
        result (tuple): 返回合并后的粒球 (new_granule_ball, granule_j)。
    """
    radius_i = granule_i['radius']
    radius_j = granule_j['radius']
    center_i = np.array(granule_i['cluster_center'])
    center_j = np.array(granule_j['cluster_center'])

    # 进行合并操作
    samples_i = np.array(granule_i['samples'])
    samples_j = np.array(granule_j['samples'])
    merged_samples = np.vstack((samples_i, samples_j))

    # 根据选择的半径计算方式计算新半径
    if option == 1:
        # 选项 1：中心为所有合并的样本的均值，半径为所有样本到新中心距离的均值
        new_center = np.mean(merged_samples, axis=0)
        distances = np.linalg.norm(merged_samples - new_center, axis=1)
        new_radius = np.mean(distances)
    elif option == 2:
        # 选项 2：中心为两个粒球中心的均值，半径为两个粒球中较大的半径
        new_center = (center_i + center_j) / 2
        new_radius = max(radius_i, radius_j)
        distances = np.linalg.norm(merged_samples - new_center, axis=1)
    else:
        raise ValueError("Invalid option. Must be 1 or 2.")

    # 选择距离新中心小于等于新半径的样本
    samples_within_radius = merged_samples[distances <= new_radius]

    # 更新粒球 i 的信息为合并后的粒球
    new_granule_ball = {
        'cluster_center': new_center.tolist(),
        'radius': new_radius,
        'samples': samples_within_radius.tolist()
    }

    # 更新粒球 j 的信息：半径为 0
    granule_j['radius'] = 0

    return new_granule_ball, granule_j  # 返回合并后的粒球和更新后的粒球 j




def process_heterogeneous_granules(granules):
    """
    对每个 granule_type 下的粒球进行异质粒球的监测和处理。
    处理不同标签（类别）之间的粒球重叠和嵌套。

    :param granules: 由 generate_granules 函数生成的粒球数据结构
    :return: 更新后的 granules，包含处理后的粒球
    """
    # 遍历每个 granule_type
    for granule_type in granules:
        #print(f"Processing heterogeneous granules in {granule_type}")
        granule = granules[granule_type]

        labels = list(granule.keys())
        num_labels = len(labels)

        # 遍历每个标签组合（两两组合）
        for idx_i in range(num_labels):
            label_i = labels[idx_i]
            granules_i = granule[label_i]

            for idx_j in range(idx_i + 1, num_labels):
                label_j = labels[idx_j]
                granules_j = granule[label_j]

                #print(f"Checking between {label_i} and {label_j}")

                # 遍历标签 i 和标签 j 下的粒球
                for cluster_id_i, granule_ball_i in enumerate(granules_i):
                    center_i = granule_ball_i['cluster_center']
                    radius_i = granule_ball_i['radius']

                    for cluster_id_j, granule_ball_j in enumerate(granules_j):
                        center_j = granule_ball_j['cluster_center']
                        radius_j = granule_ball_j['radius']

                        # 构造粒球重叠参数
                        granule_ball_overlap_coef = {
                            'cluster_center_i': center_i,
                            'cluster_center_j': center_j,
                            'radius_i': radius_i,
                            'radius_j': radius_j
                        }

                        # 检查异质粒球的情况
                        if check_heterogeneous_overlap(granule_ball_overlap_coef):
                            #print(f"{label_i} Granule {cluster_id_i} and {label_j} Granule {cluster_id_j} have heterogeneous overlap.")
                            new_granule_ball_i, new_granule_ball_j = process_heterogeneous_overlap_ori(granule_ball_i, granule_ball_j)
                            granules_i[cluster_id_i] = new_granule_ball_i
                            granules_j[cluster_id_j] = new_granule_ball_j

                        elif check_heterogeneous_invasive_overlap_and_nested(granule_ball_overlap_coef):
                            #print(f"{label_i} Granule {cluster_id_i} and {label_j} Granule {cluster_id_j} have heterogeneous invasive overlap.")
                            new_granule_ball_i, new_granule_ball_j = process_heterogeneous_nested(granule_ball_i, granule_ball_j)
                            granules_i[cluster_id_i] = new_granule_ball_i
                            granules_j[cluster_id_j] = new_granule_ball_j

                        # elif check_heterogeneous_nested(granule_ball_overlap_coef):
                        #     #print(f"{label_i} Granule {cluster_id_i} and {label_j} Granule {cluster_id_j} are heterogeneously nested.")
                        #     new_granule_ball_i, new_granule_ball_j = process_heterogeneous_nested(granule_ball_i, granule_ball_j)
                        #     granules_i[cluster_id_i] = new_granule_ball_i
                        #     granules_j[cluster_id_j] = new_granule_ball_j


    return granules






# 异质粒球的三种判断 和 处理方式
def check_heterogeneous_overlap(granule_ball_overlap_coef):
    """判断是否满足异质重叠条件"""
    c1 = np.array(granule_ball_overlap_coef['cluster_center_i'])
    c2 = np.array(granule_ball_overlap_coef['cluster_center_j'])
    r1 = granule_ball_overlap_coef['radius_i']
    r2 = granule_ball_overlap_coef['radius_j']
    distance = np.linalg.norm(c1 - c2)
    #print(f"判断结果为",distance)
    return max(r1, r2) < distance <= r1 + r2

def check_heterogeneous_invasive_overlap_and_nested(granule_ball_overlap_coef):
    """判断是否满足异质侵入重叠条件"""
    c1 = np.array(granule_ball_overlap_coef['cluster_center_i'])
    c2 = np.array(granule_ball_overlap_coef['cluster_center_j'])
    r1 = granule_ball_overlap_coef['radius_i']
    r2 = granule_ball_overlap_coef['radius_j']
    distance = np.linalg.norm(c1 - c2)
    return distance <= max(r1, r2)


# def check_heterogeneous_nested(granule_ball_overlap_coef):
#     """判断是否满足异质嵌套条件"""
#     c1 = np.array(granule_ball_overlap_coef['cluster_center_i'])
#     c2 = np.array(granule_ball_overlap_coef['cluster_center_j'])
#     r1 = granule_ball_overlap_coef['radius_i']
#     r2 = granule_ball_overlap_coef['radius_j']
#     distance = np.linalg.norm(c1 - c2)
#     return distance <= abs(r1 - r2)



## 处理方式


def process_heterogeneous_overlap_ori(granule_i, granule_j):
    '''
    对粒球进行去重叠
    '''
    granule_i_center = np.array(granule_i['cluster_center'])
    granule_j_center = np.array(granule_j['cluster_center'])
    radius_i = granule_i['radius']
    radius_j = granule_j['radius']
    granule_i_samples = granule_i['samples']
    granule_j_samples = granule_j['samples']

    # 确保samples始终是列表
    if not isinstance(granule_i_samples, list):
        granule_i_samples = [granule_i_samples]
    if not isinstance(granule_j_samples, list):
        granule_j_samples = [granule_j_samples]

    # 计算新半径
    distance = np.linalg.norm(granule_i_center - granule_j_center)
    new_radius_i = distance - radius_j if distance - radius_j > 0 else 0
    new_radius_j = distance - radius_i if distance - radius_i > 0 else 0

    # 过滤超出新半径的样本
    filter_granule_i_samples = [sample for sample in granule_i_samples if np.linalg.norm(np.array(sample) - granule_i_center) <= new_radius_i]
    filter_granule_j_samples = [sample for sample in granule_j_samples if np.linalg.norm(np.array(sample) - granule_j_center) <= new_radius_j]

    # 更新粒球信息
    granule_i['radius'] = new_radius_i
    granule_i['samples'] = filter_granule_i_samples

    granule_j['radius'] = new_radius_j
    granule_j['samples'] = filter_granule_j_samples

    # 返回去除重叠后的粒球
    return granule_i, granule_j



def process_heterogeneous_overlap(granule_i, granule_j):
    '''
    对两个重叠的粒球进行更新，处理粒球内样本数量小于等于 1 的情况，将其半径设为 0。

    步骤：
    1. 排除两个粒球中互相重叠的样本，得到新的粒球样本集。
        - 对于 granule_i，移除位于 granule_j 半径范围内的样本。
        - 对于 granule_j，移除位于 granule_i 半径范围内的样本。
    2. 分别计算两个新粒球样本的中心，即新样本集的均值。
    3. 分别计算两个新粒球的半径，即新样本集中的样本与新中心距离的均值。
    4. 更新粒球的样本集，包含距离新中心小于新半径的样本。
    5. 如果粒球的样本数量小于等于 1，将其半径设为 0。

    返回更新后的 granule_i 和 granule_j。
    '''

    # 将中心和样本转换为 NumPy 数组
    granule_i_center = np.array(granule_i['cluster_center'])
    granule_j_center = np.array(granule_j['cluster_center'])
    granule_i_samples = np.array(granule_i['samples'])
    granule_j_samples = np.array(granule_j['samples'])

    # 确保样本数组是二维的
    granule_i_samples = np.atleast_2d(granule_i_samples)
    granule_j_samples = np.atleast_2d(granule_j_samples)

    # 初始化新的样本集
    new_samples_i = np.array([])
    new_samples_j = np.array([])

    # 步骤 1：排除重叠的样本
    # 对 granule_i，移除位于 granule_j 半径范围内的样本
    if granule_i_samples.size > 0:
        distances_i_to_j_center = np.linalg.norm(granule_i_samples - granule_j_center, axis=1)
        new_samples_i = granule_i_samples[distances_i_to_j_center > granule_j['radius']]
    # else:
    #     print("Granule_i has no samples.")

    # 对 granule_j，移除位于 granule_i 半径范围内的样本
    if granule_j_samples.size > 0:
        distances_j_to_i_center = np.linalg.norm(granule_j_samples - granule_i_center, axis=1)
        new_samples_j = granule_j_samples[distances_j_to_i_center > granule_i['radius']]
    # else:
    #     print("Granule_j has no samples.")

    # 步骤 2：计算新的中心
    if new_samples_i.size > 0:
        new_center_i = np.mean(new_samples_i, axis=0)
    else:
        new_center_i = granule_i_center  # 或者设置为 None，根据需求
        #print("Granule_i has no samples after removing overlapping samples.")

    if new_samples_j.size > 0:
        new_center_j = np.mean(new_samples_j, axis=0)
    else:
        new_center_j = granule_j_center
        #print("Granule_j has no samples after removing overlapping samples.")

    # 步骤 3：计算新的半径
    if new_samples_i.shape[0] > 1:
        distances_to_new_center_i = np.linalg.norm(new_samples_i - new_center_i, axis=1)
        new_radius_i = np.mean(distances_to_new_center_i)
    else:
        new_radius_i = 0

    if new_samples_j.shape[0] > 1:
        distances_to_new_center_j = np.linalg.norm(new_samples_j - new_center_j, axis=1)
        new_radius_j = np.mean(distances_to_new_center_j)
    else:
        new_radius_j = 0

    # 步骤 4：更新粒球的样本集
    if new_samples_i.size > 0 and new_radius_i > 0:
        distances_to_new_center_i = np.linalg.norm(new_samples_i - new_center_i, axis=1)
        updated_samples_i = new_samples_i[distances_to_new_center_i <= new_radius_i]
    else:
        updated_samples_i = np.array([])

    if new_samples_j.size > 0 and new_radius_j > 0:
        distances_to_new_center_j = np.linalg.norm(new_samples_j - new_center_j, axis=1)
        updated_samples_j = new_samples_j[distances_to_new_center_j <= new_radius_j]
    else:
        updated_samples_j = np.array([])

    # 将样本数组转换为列表（如果需要）
    granule_i['cluster_center'] = new_center_i
    granule_i['radius'] = new_radius_i
    granule_i['samples'] = updated_samples_i.tolist()

    granule_j['cluster_center'] = new_center_j
    granule_j['radius'] = new_radius_j
    granule_j['samples'] = updated_samples_j.tolist()

    # 步骤 5：如果粒球的样本数量小于等于 1，将其半径设为 0
    if len(granule_i['samples']) <= 1:
        granule_i['radius'] = 0

    if len(granule_j['samples']) <= 1:
        granule_j['radius'] = 0

    return granule_i, granule_j




def process_heterogeneous_nested(granule_i, granule_j):
    '''
    ltx
    对粒球进行去重叠
    :param granule_i: 粒球i
    :param granule_j: 粒球j
    :return:
    '''

    # 将新的数据赋给粒球
    granule_i_center = granule_i['cluster_center']
    granule_j_center = granule_j['cluster_center']

    granule_i['cluster_center'] = granule_i_center
    granule_j['cluster_center'] = granule_j_center
    granule_i['radius'] = 0
    granule_j['radius'] = 0
    granule_i['samples'] = 0
    granule_j['samples'] = 0
    # 返回去除重叠后的粒球
    return granule_i, granule_j





####################################################################################

## 评价每个粒度结构

def evaluate_granules(granules):
    """
    计算不同GB_type下的粒球结构的覆盖度和特异度，并比较综合评价指标。

    综合评价指标 = 覆盖度 * 特异度
    """

    evaluation_results = {}

    # 遍历每个 GB_type
    for granule_type in granules:
        granule = granules[granule_type]

        # 覆盖度和特异度的总和
        total_coverage = 0
        total_specificity = 0

        # 遍历每个类别
        for label in granule:
            granules_list = granule[label]

            if len(granules_list) == 0:
                continue

            # 计算该类别下的最大半径
            max_radius = np.max([g['radius'] for g in granules_list])

            # 该类别的覆盖度和特异度
            coverage_sum = 0
            specificity_sum = 0

            for g in granules_list:
                # 计算粒球的覆盖度 = 样本数量
                coverage = len(g['samples'])
                coverage_sum += coverage

                # 计算粒球的特异度 = 1 - (粒球半径 / 该类最大半径)
                specificity = 1 - (g['radius'] / max_radius) if max_radius > 0 else 0
                specificity_sum += specificity

            # 累加该类别的覆盖度和特异度
            total_coverage += coverage_sum
            total_specificity += specificity_sum

        # 综合评价指标 = 总覆盖度 * 总特异度
        #print("total_coverage:", total_coverage)
        #print("total_specificity:", total_specificity)
        comprehensive_metric = total_coverage * total_specificity
        evaluation_results[granule_type] = comprehensive_metric

    # 找到综合评价指标最大的GB_type
    best_granule_type = max(evaluation_results, key=evaluation_results.get)

    # 输出每个GB_type的评价指标
    #print("Evaluation Results for Each GB_type:")
    for granule_type, score in evaluation_results.items():
        print(f"{granule_type}: {score}")

    # print(f"\nBest GB_type: {best_granule_type} with score: {evaluation_results[best_granule_type]}")

    return evaluation_results, best_granule_type



















