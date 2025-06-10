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
import os



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






def main():
    # 忽略警告
    warnings.filterwarnings("ignore")
    
    # 定义所有数据集的路径
    datasets = [
        # 'dataset/echocardiogram.csv',
        # 'dataset/Iris.csv',
        # 'dataset/parkinsons.csv',
        # 'dataset/seeds.csv',
        # 'dataset/Glass.csv',
        # 'dataset/audiology.csv',
        # 'dataset/ecoli.csv',
        # 'dataset/wdbc.csv',
        # 'dataset/Balance Scale.csv',
        # 'dataset/breast_ori.csv',
        # 'dataset/Pima Indians Diabetes Database.csv',
        'dataset/fourclass.csv',
        # 'dataset/biodeg.csv',
        # 'dataset/data_banknote_authentication.csv',
        # 'dataset/cardio.csv',
        # 'dataset/thyroid.csv',
        # 'dataset/rice.csv',
        # 'dataset/waveform.csv',
        # 'dataset/page-blocks.csv',
        # 'dataset/eletrical.csv'
    ]

    # 定义算法
    algorithms = {
        "GBkNN": gb_origin,
        "ACC-GBkNN": gb_accelerate_algorithm,
        "ADP-GBkNN": gb_adaptive_algorithm,
        "GBkNN++": GBkNN_plus_plus,
        "SGBkNN": new_granule_knn_algorithm,
    }


    # 创建保存结果的文件夹
    output_dir = 'time_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历每个数据集
    for dataset_path in datasets:
        print(f"\n处理数据集: {dataset_path}")
        try:
            # 读取数据
            data = pd.read_csv(dataset_path, header=None)
            data = np.array(data)

            # 标准化
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
            print("去重后数据量", data.shape)

            X = data[:, 1:]  # 特征数据
            y = data[:, 0]   # 标签列

            # 运行算法并记录结果
            kf = StratifiedKFold(n_splits=10, shuffle=True)
            
            # 用于存储当前数据集的结果
            dataset_results = []

            for fold_idx, (train_index, test_index) in enumerate(kf.split(X, y)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                train_data = np.hstack((y_train.reshape(-1, 1), X_train))
                test_data = np.hstack((y_test.reshape(-1, 1), X_test))

                # 运行每个算法
                for algo_name, algo in algorithms.items():
                    start_time = time.perf_counter()
                    accuracy, precision, recall, f1, train_time, test_time = algo(train_data, test_data)
                    total_time = time.perf_counter() - start_time
                    print(f"{algo_name}: {train_time}")
                    print(f"{algo_name}: {test_time}")

                    # 记录结果
                    dataset_results.append({
                        'Algorithm': algo_name,
                        'Fold': fold_idx + 1,
                        'Train_Time': train_time,
                        'Test_Time': test_time,
                        'Total_Time': total_time
                    })

            # 将当前数据集的结果转换为DataFrame
            current_results_df = pd.DataFrame(dataset_results)
            
            # 计算当前数据集的统计结果
            summary_stats = current_results_df.groupby('Algorithm').agg({
                'Train_Time': ['mean', 'std'],
                'Test_Time': ['mean', 'std'],
                'Total_Time': ['mean', 'std']
            }).round(4)

            # 将多级索引转换为列
            summary_stats = summary_stats.reset_index()

            # 重命名列名
            summary_stats.columns = [
                'Algorithm',
                'Train_Time_Mean', 'Train_Time_Std',
                'Test_Time_Mean', 'Test_Time_Std',
                'Total_Time_Mean', 'Total_Time_Std'
            ]

            # 格式化结果为 "mean ± std" 的形式
            formatted_stats = pd.DataFrame()
            formatted_stats['Algorithm'] = summary_stats['Algorithm']
            formatted_stats['Training Time (s)'] = summary_stats.apply(
                lambda x: f"{x['Train_Time_Mean']:.4f} ± {x['Train_Time_Std']:.4f}", axis=1)
            formatted_stats['Testing Time (s)'] = summary_stats.apply(
                lambda x: f"{x['Test_Time_Mean']:.4f} ± {x['Test_Time_Std']:.4f}", axis=1)
            formatted_stats['Total Time (s)'] = summary_stats.apply(
                lambda x: f"{x['Total_Time_Mean']:.4f} ± {x['Total_Time_Std']:.4f}", axis=1)

            # 获取数据集名称（去除路径和扩展名）
            dataset_name = dataset_path.split('/')[-1].split('.')[0]
            
            # 保存当前数据集的结果到指定文件夹
            output_file = os.path.join(output_dir, f'{dataset_name}_time_summary.xlsx')
            formatted_stats.to_excel(output_file, index=False)
            print(f"已保存数据集 {dataset_name} 的结果到 {output_file}")

        except Exception as e:
            print(f"处理数据集 {dataset_path} 时出错: {str(e)}")
            continue

    print("所有数据集处理完成")

if __name__ == "__main__":
    main()