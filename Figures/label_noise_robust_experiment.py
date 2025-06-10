import warnings
import numpy as np
import pandas as pd
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from SGB_generation import new_granule_knn_algorithm
from gb_accelerate import gb_accelerate_algorithm
from gb_adaptive_ori import gb_adaptive_algorithm
from gb_origin import gb_origin
from GBkNN_xie import GBkNN_plus_plus


def add_noise_to_label(data, noise_ratio):
    """
    为数据集添加噪声，随机替换一定比例的标签，适合多类情形
    :param data: 输入的数据集，第一列为标签
    :param noise_ratio: 噪声比例 (0-1)，用于确定添加噪声的样本比例
    :return: 添加噪声后的数据
    """
    sample_number = data.shape[0]
    labels = data[:, 0].astype(int)
    unique_labels = np.unique(labels)
    noise_sample_size = int(noise_ratio * sample_number)
    noise_indices = np.random.choice(sample_number, size=noise_sample_size, replace=False)
    new_labels = labels.copy()

    for idx in noise_indices:
        current_label = labels[idx]
        possible_labels = unique_labels[unique_labels != current_label]
        new_label = np.random.choice(possible_labels)
        new_labels[idx] = new_label

    data_with_noise = np.hstack((new_labels.reshape(sample_number, 1), data[:, 1:]))
    return data_with_noise


def run_algorithms(X, y, noise_ratios, algorithms, num_folds=10):
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    summary_results_list = []

    for ratio in noise_ratios:
        print(f"\n噪声比率: {ratio}")

        fold_results = {name: {"accuracies": [], "precisions": [], "recalls": [], "f1_scores": [], "times": []} for name in algorithms.keys()}

        for fold_idx, (train_index, test_index) in enumerate(kf.split(X, y)):
            print(f"\nFold {fold_idx + 1}")

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            train_data_noisy = add_noise_to_label(np.hstack((y_train.reshape(-1, 1), X_train)), ratio)

            for name, algo in algorithms.items():
                start_time = time.perf_counter()

                if callable(algo):
                    accuracy, precision, recall, f1 = algo(train_data_noisy, np.hstack((y_test.reshape(-1, 1), X_test)))
                else:
                    algo.fit(X_train, y_train)
                    y_pred = algo.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                end_time = time.perf_counter()
                elapsed_time = end_time - start_time

                # Save fold results for summary calculation
                fold_results[name]["accuracies"].append(accuracy)
                fold_results[name]["precisions"].append(precision)
                fold_results[name]["recalls"].append(recall)
                fold_results[name]["f1_scores"].append(f1)
                fold_results[name]["times"].append(elapsed_time)

                print(f"\nAlgorithm {name}: "
                      f"Accuracy: {accuracy:.4f}, "
                      f"Precision: {precision:.4f}, "
                      f"Recall: {recall:.4f}, "
                      f"F1: {f1:.4f}, "
                      f"Time: {elapsed_time:.4f} 秒")

        # Calculate and save summary results for each algorithm
        for name in algorithms.keys():
            summary_results_list.append({
                "Noise Ratio": ratio,
                "Algorithm": name,
                "Average Accuracy": np.mean(fold_results[name]["accuracies"]),
                "Accuracy Std": np.std(fold_results[name]["accuracies"]),
                "Average Precision": np.mean(fold_results[name]["precisions"]),
                "Precision Std": np.std(fold_results[name]["precisions"]),
                "Average Recall": np.mean(fold_results[name]["recalls"]),
                "Recall Std": np.std(fold_results[name]["recalls"]),
                "Average F1 Score": np.mean(fold_results[name]["f1_scores"]),
                "F1 Score Std": np.std(fold_results[name]["f1_scores"]),
                "Average Time (s)": np.mean(fold_results[name]["times"]),
                "Time Std": np.std(fold_results[name]["times"])
            })

    # Convert summary results list to DataFrame and save to Excel
    summary_results_df = pd.DataFrame(summary_results_list)
    summary_results_df.to_excel("rice_robust_results1.xlsx", index=False)
    print("\n结果已保存到 algorithm_results_summary.xlsx")


def main():
    warnings.filterwarnings("ignore")
    data = pd.read_csv('./dataset/rice.csv',header=None)
    data = np.array(data)
    print("数据量", data.shape)

    # # # # 标准化
    # scaler =  StandardScaler()
    scaler = MinMaxScaler()
    data_array_normalized = scaler.fit_transform(data[:, 1:])
    data = np.hstack((data[:, [0]], data_array_normalized))


    data = np.unique(data, axis=0)
    data_temp = []
    data_list = data.tolist()
    data = []
    for data_single in data_list:
        if data_single[1:] not in data_temp:
            data_temp.append(data_single[1:])
            data.append(data_single)
    data = np.array(data)

    X = data[:, 1:]
    y = data[:, 0]

    noise_ratios = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]

    algorithms = {
        # "GB_origin": gb_origin,
        # "GB_accelerate": gb_accelerate_algorithm,
        # "GB_adaptive": gb_adaptive_algorithm,
        "GBkNN++": GBkNN_plus_plus,
         # "Our_model": new_granule_knn_algorithm,

    }

    run_algorithms(X, y, noise_ratios, algorithms, num_folds=10)


if __name__ == "__main__":
    main()
