import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from SGB_generation import new_granule_knn_algorithm
from gb_accelerate import gb_accelerate_algorithm
from gb_adaptive_ori import gb_adaptive_algorithm
from gb_origin import gb_origin
from GBkNN_xie import GBkNN_plus_plus
from GBGxie import GBList



from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from gb_knn import gb_knn_algorithm
import time

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.datasets import load_breast_cancer,load_iris, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import KFold, StratifiedKFold







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

    # 将新标r签与数据的特征部分拼接，返回添加噪声后的数据集
    data_with_noise = np.hstack((new_labels.reshape(sample_number, 1), data[:, 1:]))

    return data_with_noise





# def run_algorithms(X, y, noise_ratios, algorithms, num_folds=10):
#     """
#     运行算法并记录每次划分数据集后的运行时间、准确率、精确率、召回率和 F1 分数，使用十折交叉验证
#     :param X: 特征数据
#     :param y: 标签
#     :param noise_ratios: 噪声比例列表
#     :param algorithms: 字典形式的算法名称和对应的算法函数或模型
#     :param num_folds: 折叠数（默认为10，即十折交叉验证）
#     """

#     kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

#     for ratio in noise_ratios:
#         print(f"\n噪声比率: {ratio}")

#         # 用于记录每个算法的性能结果
#         results = {
#             name: {
#                 "times": [],
#                 "accuracies": [],
#                 "precisions": [],
#                 "recalls": [],
#                 "f1_scores": []
#             } for name in algorithms.keys()
#         }

#         for fold_idx, (train_index, test_index) in enumerate(kf.split(X, y)):
#             print(f"\nFold {fold_idx + 1}")

#             X_train, X_test = X[train_index], X[test_index]
#             y_train, y_test = y[train_index], y[test_index]

#             # 为训练集添加噪声，保留测试集原始数据
#             train_data_noisy = add_noise_to_label(np.hstack((y_train.reshape(-1, 1), X_train)), ratio)

#             # 运行算法并记录结果
#             for name, algo in algorithms.items():
#                 start_time = time.perf_counter()

#                 if callable(algo):
#                     # 自定义算法：调用函数
#                     accuracy, precision, recall, f1 = algo(train_data_noisy, np.hstack((y_test.reshape(-1, 1), X_test)))
#                 else:
#                     # sklearn 模型：使用 fit 和 predict
#                     algo.fit(X_train, y_train)
#                     y_pred = algo.predict(X_test)

#                     # 计算性能指标
#                     accuracy = accuracy_score(y_test, y_pred)
#                     precision = precision_score(y_test, y_pred, average='macro')
#                     recall = recall_score(y_test, y_pred,  average='macro')
#                     f1 = f1_score(y_test, y_pred, average='macro')

#                 end_time = time.perf_counter()
#                 elapsed_time = end_time - start_time

#                 # 记录性能结果
#                 results[name]["times"].append(elapsed_time)
#                 results[name]["accuracies"].append(accuracy)
#                 results[name]["precisions"].append(precision)
#                 results[name]["recalls"].append(recall)
#                 results[name]["f1_scores"].append(f1)

#                 print(f"\nAlgorithm {name}: "
#                       f"Accuracy: {accuracy:.4f}, "
#                       f"Precision: {precision:.4f}, "
#                       f"Recall: {recall:.4f}, "
#                       f"F1: {f1:.4f}, "
#                       f"Time: {elapsed_time:.4f} 秒")

#         # 输出平均性能结果，格式为“平均值 ± 标准差”
#         for name in algorithms.keys():
#             avg_time = np.mean(results[name]["times"])
#             time_std = np.std(results[name]["times"])
#             max_accuracy = np.max(results[name]["accuracies"])
#             min_accuracy = np.min(results[name]["accuracies"])
#             mean_accuracy = np.mean(results[name]["accuracies"])
#             accuracy_std = np.std(results[name]["accuracies"])
#             mean_precision = np.mean(results[name]["precisions"])
#             precision_std = np.std(results[name]["precisions"])
#             mean_recall = np.mean(results[name]["recalls"])
#             recall_std = np.std(results[name]["recalls"])
#             mean_f1 = np.mean(results[name]["f1_scores"])
#             f1_std = np.std(results[name]["f1_scores"])

#             print(f"\n噪声比率: {ratio}")
#             print(f"\n算法: {name}")
#             print(f"平均运行时间: {avg_time:.4f} ± {time_std:.4f} 秒")
#             print(f"最大准确率: {max_accuracy:.4f}")
#             print(f"最小准确率: {min_accuracy:.4f}")
#             print(f"平均准确率: {mean_accuracy:.4f} ± {accuracy_std:.4f}")
#             print(f"平均精确率: {mean_precision:.4f} ± {precision_std:.4f}")
#             print(f"平均召回率: {mean_recall:.4f} ± {recall_std:.4f}")
#             print(f"平均F1分数: {mean_f1:.4f} ± {f1_std:.4f}")







def run_algorithms(X, y, noise_ratios, algorithms, num_folds=10):
    """
    运行算法并记录每次划分数据集后的运行时间、准确率、精确率、召回率和 F1 分数，使用十折交叉验证
    :param X: 特征数据
    :param y: 标签
    :param noise_ratios: 噪声比例列表
    :param algorithms: 字典形式的算法名称和对应的算法函数或模型
    :param num_folds: 折叠数（默认为10，即十折交叉验证）
    """

    kf = StratifiedKFold(n_splits=num_folds, shuffle=True,random_state=0)

    for ratio in noise_ratios:
        print(f"\n噪声比率: {ratio}")

        # 用于记录每个算法的性能结果
        results = {
            name: {
                "train_times": [],
                "test_times": [],
                "total_times": [],
                "accuracies": [],
                "precisions": [],
                "recalls": [],
                "f1_scores": []
            } for name in algorithms.keys()
        }

        for fold_idx, (train_index, test_index) in enumerate(kf.split(X, y)):
            print(f"\nFold {fold_idx + 1}")

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # 为训练集添加噪声，保留测试集原始数据
            train_data_noisy = add_noise_to_label(np.hstack((y_train.reshape(-1, 1), X_train)), ratio)

            # 运行算法并记录结果
            for name, algo in algorithms.items():
                # 记录总时间开始
                total_start_time = time.perf_counter()
                
                if callable(algo):
                    # 自定义算法：调用函数
                    # 对于自定义算法，我们假设其内部已经包含了训练和测试的时间计算
                    # 这里我们只能记录总时间
                    train_start_time = time.perf_counter()
                    accuracy, precision, recall, f1 = algo(train_data_noisy, np.hstack((y_test.reshape(-1, 1), X_test)))
                    train_end_time = time.perf_counter()
                    
                    # 对于自定义算法，我们假设训练时间占总时间的80%，测试时间占20%
                    # 这是一个估计值，实际情况可能不同
                    train_time = train_end_time - train_start_time
                    test_time = 0  # 对于自定义算法，我们无法分离测试时间
                else:
                    # sklearn 模型：使用 fit 和 predict
                    # 记录训练时间
                    train_start_time = time.perf_counter()
                    algo.fit(X_train, y_train)
                    train_end_time = time.perf_counter()
                    train_time = train_end_time - train_start_time
                    
                    # 记录测试时间
                    test_start_time = time.perf_counter()
                    y_pred = algo.predict(X_test)
                    test_end_time = time.perf_counter()
                    test_time = test_end_time - test_start_time

                    # 计算性能指标
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='macro')
                    recall = recall_score(y_test, y_pred,  average='macro')
                    f1 = f1_score(y_test, y_pred, average='macro')

                # 记录总时间结束
                total_end_time = time.perf_counter()
                total_time = total_end_time - total_start_time

                # 记录性能结果
                results[name]["train_times"].append(train_time)
                results[name]["test_times"].append(test_time)
                results[name]["total_times"].append(total_time)
                results[name]["accuracies"].append(accuracy)
                results[name]["precisions"].append(precision)
                results[name]["recalls"].append(recall)
                results[name]["f1_scores"].append(f1)

                print(f"\nAlgorithm {name}: "
                      f"Accuracy: {accuracy:.4f}, "
                      f"Precision: {precision:.4f}, "
                      f"Recall: {recall:.4f}, "
                      f"F1: {f1:.4f}, "
                      f"Train Time: {train_time:.4f} 秒, "
                      f"Test Time: {test_time:.4f} 秒, "
                      f"Total Time: {total_time:.4f} 秒")

        # 输出平均性能结果，格式为"平均值 ± 标准差"
        for name in algorithms.keys():
            avg_train_time = np.mean(results[name]["train_times"])
            train_time_std = np.std(results[name]["train_times"])
            avg_test_time = np.mean(results[name]["test_times"])
            test_time_std = np.std(results[name]["test_times"])
            avg_total_time = np.mean(results[name]["total_times"])
            total_time_std = np.std(results[name]["total_times"])
            
            max_accuracy = np.max(results[name]["accuracies"])
            min_accuracy = np.min(results[name]["accuracies"])
            mean_accuracy = np.mean(results[name]["accuracies"])
            accuracy_std = np.std(results[name]["accuracies"])
            mean_precision = np.mean(results[name]["precisions"])
            precision_std = np.std(results[name]["precisions"])
            mean_recall = np.mean(results[name]["recalls"])
            recall_std = np.std(results[name]["recalls"])
            mean_f1 = np.mean(results[name]["f1_scores"])
            f1_std = np.std(results[name]["f1_scores"])

            print(f"\n噪声比率: {ratio}")
            print(f"\n算法: {name}")
            print(f"平均训练时间: {avg_train_time:.4f} ± {train_time_std:.4f} 秒")
            print(f"平均测试时间: {avg_test_time:.4f} ± {test_time_std:.4f} 秒")
            print(f"平均总运行时间: {avg_total_time:.4f} ± {total_time_std:.4f} 秒")
            print(f"最大准确率: {max_accuracy:.4f}")
            print(f"最小准确率: {min_accuracy:.4f}")
            print(f"平均准确率: {mean_accuracy:.4f} ± {accuracy_std:.4f}")
            print(f"平均精确率: {mean_precision:.4f} ± {precision_std:.4f}")
            print(f"平均召回率: {mean_recall:.4f} ± {recall_std:.4f}")
            print(f"平均F1分数: {mean_f1:.4f} ± {f1_std:.4f}")











def main():
    # 忽略警告
    warnings.filterwarnings("ignore")
    # 按 数据集样本个数 排序
    # 'dataset/echocardiogram.csv',
    # 'dataset/Iris.csv',
    # 'dataset/parkinsons.csv',
    # 'dataset/seeds.csv',
    # 'dataset/Glass.csv',
    # 'dataset/audiology.csv',
    # 'dataset/wdbc.csv',
    # 'dataset/Balance Scale.csv',
    # 'dataset/Credit Approval.csv',
    # 'dataset/Pima Indians Diabetes Database.csv',
    #
    # 'dataset/diabetes.csv',
    # 'dataset/fourclass1.csv',
    # 'dataset/biodeg.csv',
    # 'dataset/data_banknote_authentication.csv',
    # 'dataset/cardio.csv',
    # 'dataset/thyroid.csv',
    # 'dataset/rice.csv',
    # 'dataset/waveform.csv',
    # 'dataset/page-blocks.csv',
    # 'dataset/eletrical.csv',

    data = pd.read_excel('dataset/DDSM.xlsx', header=0)



    data = np.array(data)  # 转换为 numpy 数组
    print("shuju", data.shape)
    




# #
# # 分离特征和标签
    X_initial = data[:, 1:]
    y_initial = data[:, 0]
# #
    # 特征选择，选择最重要的50个特征
    selector = SelectKBest(score_func=f_classif, k=20)
    X_selected = selector.fit_transform(X_initial, y_initial)
    data = np.hstack((y_initial.reshape(-1, 1), X_selected))











    # # 训练随机森林分类器
    # forest = RandomForestClassifier(n_estimators=100, random_state=42)
    # forest.fit(X_initial, y_initial)
    #
    # # 获取每个特征的重要性分数
    # importances = forest.feature_importances_
    #
    # # 获取前 52 个特征的索引（按重要性降序）
    # indices = np.argsort(importances)[::-1][:20]
    #
    # # 选择对应的特征
    # X_selected = X_initial[:, indices]
    #
    # # 拼接标签和特征作为最终数据
    # data = np.hstack((y_initial.reshape(-1, 1), X_selected))
    # print("特征选择后的数据形状:", data.shape)
#
# #

    # # # # 标准化
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    data_array_normalized = scaler.fit_transform(data[:, 1:])
    data = np.hstack((data[:, [0]], data_array_normalized))

    #
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


    X = data[:, 1:]  # 特征数据
    print(X.shape)
    y = data[:, 0]  # 标签列 (第一列)
    print(np.unique(y))



    # 噪声比率列表
    noise_ratios = [0]  # 可以修改噪声比例

    # 定义四个算法及其对应的函数
    algorithms = {
        # "GB_SCM": GB_scm_algorithm,  # 接受 X_train, y_train
        #"GB_SCM": gb_scm_algorithm,  # 接受 X_train, y_train
        # "GB_KNN": gb_knn_algorithm,  # 接受 train 和 test

        #"Our_model_DS": new_granule_knn_algorithm_DS,

        # "GBkNN": gb_origin,
        # "ACC-GBkNN": gb_accelerate_algorithm,
        # "ADP-GBkNN": gb_adaptive_algorithm,
        # "GBkNN++": GBkNN_plus_plus,
        "AGBkNN": new_granule_knn_algorithm,
        # # #
        # #
        # # # 新增对比算法
        # "KNN": KNeighborsClassifier(n_neighbors=1),
        # "Decision Tree": DecisionTreeClassifier(),
        # "SVC": SVC(),
        # 'AdaBoost': AdaBoostClassifier(
        #     n_estimators=100,
        #     random_state=42,
        #     learning_rate=0.1,
        #     algorithm='SAMME.R')


        # # "GaussianNB": GaussianNB(),
        # "MLP Classifier": MLPClassifier( hidden_layer_sizes=(128, 64),  activation='relu',solver='adam',alpha=1e-4,  learning_rate_init=1e-3,max_iter=200,random_state=42),
        # # # "Logistic Regression": LogisticRegression(),

        # #"Gaussian Process": GaussianProcessClassifier(),
        # "AdaBoost": AdaBoostClassifier(algorithm='SAMME'),
        # #"Extra Trees": ExtraTreesClassifier(),
        # "Random Forest": RandomForestClassifier(),
        # "XGBoost": XGBClassifier(),
        # # "LightGBM": LGBMClassifier(verbose=-1),
        # "CatBoost": CatBoostClassifier(verbose=False)

    }

    # 运行算法并输出结果
    run_algorithms(X, y, noise_ratios, algorithms, num_folds=5)

# 运行主函数
if __name__ == "__main__":
    main()


