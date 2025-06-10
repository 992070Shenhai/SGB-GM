import warnings
import numpy as np
import pandas as pd
import time
import os
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

from SGB_generation import new_granule_knn_algorithm
from gb_accelerate import gb_accelerate_algorithm
from gb_adaptive_ori import gb_adaptive_algorithm
from gb_origin import gb_origin
from GBkNN_xie import GBkNN_plus_plus






def add_feature_noise(data, noise_ratio):
    """
    为数据集添加特征噪声，随机选择一定比例的样本，并对每个选定样本随机选择一定比例的特征进行扰动
    
    :param data: 输入的数据集，第一列为标签
    :param noise_ratio: 噪声样本比例 (0-1)，用于确定添加噪声的样本比例
    :param feature_ratio: 特征噪声比例 (0-1)，用于确定每个噪声样本中要扰动的特征比例，若为None则随机生成
    :return: 添加噪声后的数据
    """
    # 复制原始数据，避免修改原始数据
    noisy_data = data.copy()
    
    # 获取样本数量和特征数量
    sample_number = data.shape[0]
    feature_number = data.shape[1] - 1  # 减去标签列
    
    # 计算要添加噪声的样本数量
    noise_sample_size = int(noise_ratio * sample_number)
    
    # 随机选择要添加噪声的样本索引
    if noise_sample_size > 0:
        noise_indices = np.random.choice(sample_number, size=noise_sample_size, replace=False)
        
        # 对每个选定的样本添加特征噪声
        for idx in noise_indices:
            
            # 计算要扰动的特征数量
            noise_feature_size = 1
            
            # 随机选择要扰动的特征索引（注意特征从索引1开始，因为索引0是标签）
            feature_indices = np.random.choice(feature_number, size=noise_feature_size, replace=False) + 1
            
            # 对每个选定的特征添加噪声
            for feature_idx in feature_indices:
                # 计算该特征在整个数据集中的最小值和最大值
                feature_min = np.min(data[:, feature_idx])
                feature_max = np.max(data[:, feature_idx])
                
                # 在特征的有效范围内生成随机值
                random_value = np.random.uniform(feature_min, feature_max)
                
                # 将原始特征值替换为随机值
                noisy_data[idx, feature_idx] = random_value
    
    return noisy_data


##### 示例使用

# np.random.seed(42)  # 设置随机种子以保证结果可重复
# data = np.random.rand(20, 5)  # 生成[0,1]之间的随机数
# data[:, 0] = np.random.randint(0, 2, size=20)  # 第一列设置为二分类标签(0或1)

# # 打印原始数据
# print("原始数据:")
# print(data)
# print("\n" + "="*50 + "\n")

# # 使用add_feature_noise函数添加噪声
# noisy_data = add_feature_noise(data, noise_ratio=0.1)  # 对50%的样本添加噪声

# # 打印添加噪声后的数据
# print("添加噪声后的数据:")
# print(noisy_data)

# # 找出发生变化的位置
# changed = np.where(data != noisy_data)
# print("\n发生变化的位置(行,列):")
# for i in range(len(changed[0])):
#     print(f"位置({changed[0][i]}, {changed[1][i]}): "
#           f"原值={data[changed[0][i], changed[1][i]]:.4f}, "
#           f"噪声值={noisy_data[changed[0][i], changed[1][i]]:.4f}")





def run_algorithms(X, y, feature_ratios, algorithms,dataset_path, num_folds=10):
    """
    运行算法并记录每次划分数据集后的性能指标，使用十折交叉验证
    """
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True)
    summary_results_list = []

    for ratio in feature_ratios:
        print(f"\n特征噪声比率: {ratio}")
        
        fold_results = {name: {"accuracies": [], "precisions": [], "recalls": [], "f1_scores": [], "times": []} 
                       for name in algorithms.keys()}

        for fold_idx, (train_index, test_index) in enumerate(kf.split(X, y)):
            print(f"\nFold {fold_idx + 1}")

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 为训练集添加特征噪声
            train_data_noisy = add_feature_noise(
                np.hstack((y_train.reshape(-1, 1), X_train)), 
                noise_ratio = 0.1,  # 固定为1.0,表示对所有样本添加特征噪声
            )

            for name, algo in algorithms.items():
                start_time = time.perf_counter()

                if callable(algo):
                    # 自定义算法
                    accuracy, precision, recall, f1 = algo(train_data_noisy, np.hstack((y_test.reshape(-1, 1), X_test)))
                
                else:
                    # 调用的模型
                    algo.fit(X_train, y_train)
                    y_pred = algo.predict(X_test)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                end_time = time.perf_counter()
                elapsed_time = end_time - start_time

                fold_results[name]["accuracies"].append(accuracy)
                # fold_results[name]["precisions"].append(precision)
                # fold_results[name]["recalls"].append(recall)
                # fold_results[name]["f1_scores"].append(f1)
                # fold_results[name]["times"].append(elapsed_time)

                print(f"\nAlgorithm {name}: "
                      f"Accuracy: {accuracy:.4f}, "
                    #   f"Precision: {precision:.4f}, "
                    #   f"Recall: {recall:.4f}, "
                    #   f"F1: {f1:.4f}, "
                    #   f"Time: {elapsed_time:.4f} 秒"
                    )

        # 计算并保存每个算法的汇总结果
        for name in algorithms.keys():
            summary_results_list.append({
                "Feature Noise Ratio": ratio,
                "Algorithm": name,
                "Average Accuracy": np.mean(fold_results[name]["accuracies"]),
                "Max Accuracy": np.max(fold_results[name]["accuracies"]),
                "Min Accuracy": np.min(fold_results[name]["accuracies"]),
                # "Accuracy Std": np.std(fold_results[name]["accuracies"]),
                # "Average Precision": np.mean(fold_results[name]["precisions"]),
                # "Precision Std": np.std(fold_results[name]["precisions"]),
                # "Average Recall": np.mean(fold_results[name]["recalls"]),
                # "Recall Std": np.std(fold_results[name]["recalls"]),
                # "Average F1 Score": np.mean(fold_results[name]["f1_scores"]),
                # "F1 Score Std": np.std(fold_results[name]["f1_scores"]),
                # "Average Time (s)": np.mean(fold_results[name]["times"]),
                # "Time Std": np.std(fold_results[name]["times"])
            })

    # 创建results文件夹（如果不存在）
    results_dir = "Feature_noise_results_on_20_datasets"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 将结果保存到Excel文件
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    output_file = os.path.join(results_dir, f"{dataset_name}_feature_noise_results1.xlsx")
    summary_results_df = pd.DataFrame(summary_results_list)
    summary_results_df.to_excel(output_file, index=False)
    print(f"\n结果已保存到 {output_file}")


def main():
    warnings.filterwarnings("ignore")
    
    # 定义数据集文件路径列表
    datasets = [
        # 'dataset/echocardiogram.csv',
        # 'dataset/Iris.csv',
        # 'dataset/parkinsons.csv',
        #  'dataset/seeds.csv',
        # 'dataset/Glass.csv',
        # 'dataset/audiology.csv',
        # 'dataset/ecoli.csv',
        # 'dataset/wdbc.csv',
        # 'dataset/Balance Scale.csv',
        # 'dataset/breast_ori.csv',
        # 'dataset/Credit Approval.csv',
        # 'dataset/Pima Indians Diabetes Database.csv',
        # 'dataset/diabetes.csv',
        # 'dataset/fourclass.csv',
        'dataset/biodeg.csv',
        # 'dataset/data_banknote_authentication.csv',
        # 'dataset/cardio.csv',
        # 'dataset/thyroid.csv',
        # 'dataset/rice.csv',
        # 'dataset/waveform.csv',
        # 'dataset/page-blocks.csv',
        # 'dataset/eletrical.csv'
    ]

    # 定义特征噪声比例列表
    feature_ratios = [0.05, 0.1,0.15, 0.2, 0.25, 0.3, 0.35, 0.4 ]

#0.15, 0.2, 0.25, 0.3, 0.35, 0.4
    # 定义算法
    algorithms = {
        "GB_origin": gb_origin,
        "GB_accelerate": gb_accelerate_algorithm,
        "GB_adaptive": gb_adaptive_algorithm,
        "GBkNN++": GBkNN_plus_plus,
        "Our_model": new_granule_knn_algorithm,
    }
    for dataset_path in datasets:
        print(f"\n处理数据集: {dataset_path}")
        
        try:
            # 读取数据集
            data = pd.read_csv(dataset_path, header=None)
            data = np.array(data)

            data = np.unique(data, axis=0)
            data_temp = []
            data_list = data.tolist()
            data = []
            for data_single in data_list:
                if data_single[1:] not in data_temp:
                    data_temp.append(data_single[1:])
                    data.append(data_single)
            data = np.array(data)
            print("去重后数据量",data.shape)


            # 数据预处理
            # scaler = MinMaxScaler()
            X = data[:, 1:]
            y = data[:, 0]
            # X = scaler.fit_transform(X)
            
            # 运行实验
            run_algorithms(X, y, feature_ratios, algorithms, dataset_path, num_folds=10)
            
        except Exception as e:
            print(f"处理数据集 {dataset_path} 时出错: {str(e)}")
            continue
    

if __name__ == "__main__":
    main()







