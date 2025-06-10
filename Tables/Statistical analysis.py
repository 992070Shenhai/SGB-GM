
import numpy as np
from scipy import stats
import pandas as pd
from scipy.stats import rankdata
from itertools import combinations
import Orange
from Orange.evaluation import graph_ranks
from scipy.stats import rankdata
from itertools import combinations
import matplotlib.pyplot as plt



# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'


# 提取无sd的平均值（20，9）
def extract_means_from_excel(input_excel_path, output_excel_path):
    """
    从Excel文件中提取平均值并保存到新的Excel文件

    参数:
    input_excel_path: 输入Excel文件路径
    output_excel_path: 输出Excel文件路径
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(input_excel_path,header=None)

        # 获取原始数据
        raw_data = df.values

        means = []
        for row in raw_data:
            row_means = []
            for cell in row:
                if isinstance(cell, str) and '±' in cell:
                    # 分割字符串并获取平均值部分，保持原始精度
                    mean_str = cell.split('±')[0].strip()
                    mean = float(mean_str)  # 直接转换，不进行格式化
                    row_means.append(mean)
                elif isinstance(cell, (int, float)):
                    # 如果已经是数值类型，直接添加，不进行格式化
                    row_means.append(float(cell))
                else:
                    # 处理其他可能的情况
                    try:
                        mean_str = str(cell).split('±')[0].strip()
                        mean = float(mean_str)  # 直接转换，不进行格式化
                        row_means.append(mean)
                    except:
                        print(f"警告: 无法处理的单元格值: {cell}")
                        row_means.append(np.nan)
            means.append(row_means)

        # 转换为numpy数组
        means_array = np.array(means)

        # 创建新的DataFrame
        df_means = pd.DataFrame(means_array)

        # 保存到新的Excel文件
        df_means.to_excel(output_excel_path, index=False, header=False)

        print(f"平均值已成功提取并保存到: {output_excel_path}")

        return means_array

    except Exception as e:
        print(f"错误: {str(e)}")
        return None

if __name__ == "__main__":
    # 定义输入和输出文件路径
    input_excel_path = 'extracted_F1_no_average.xlsx'
    output_excel_path = 'extracted_F1_no_sd.xlsx'

    # 提取平均值并保存
    accuracy_data = extract_means_from_excel(input_excel_path, output_excel_path)

    if accuracy_data is not None:
        print("\n提取的平均值数据形状:", accuracy_data.shape)



import numpy as np
from scipy import stats
import pandas as pd
from scipy.stats import rankdata
from itertools import combinations
import matplotlib.pyplot as plt

def run_detailed_friedman_analysis(data_array):
    """
    执行详细的Friedman检验分析
    """
    n_datasets, n_models = data_array.shape
    print("\n1. 基本信息:")
    print(f"数据集数量 (N): {n_datasets}")
    print(f"模型数量 (k): {n_models}")

    # 1. 计算每个数据集上的排名
    ranks = np.zeros_like(data_array)
    for i in range(n_datasets):
        ranks[i] = rankdata(-data_array[i])  # 使用-号使得较大的值获得较小的排名

    # 3. 计算Friedman统计量相关值
    k = n_models  # 模型数量
    N = n_datasets  # 数据集数量

    # 2. 计算每个模型的平均排名
    mean_ranks = np.mean(ranks, axis=0)
    print("\n2. 每个模型的平均排名:")
    for i in range(k):
        print(f"模型 {i+1}: {mean_ranks[i]:.4f}")


    sum_ranks_squared = np.sum(mean_ranks ** 2, axis=0)
    print(f"\n3. 平均排名的平方和: {sum_ranks_squared:.4f}")

    # 计算Friedman统计量 χ_F^2
    chi_squared = (12 * N) / (k * (k + 1)) * (sum_ranks_squared -  k * (k + 1)**2 / 4)

    # 计算F统计量
    f_statistic = ((N - 1) * chi_squared) / (N * (k - 1) - chi_squared)

    # 计算p值
    df1 = k - 1
    df2 = (k - 1) * (N - 1)
    p_value = 1 - stats.f.cdf(f_statistic, df1, df2)
    print("P值",p_value)


    print("\n4. 统计量:")
    print(f"Friedman统计量 (χ_F^2): {chi_squared:.4f}")
    print(f"F统计量: {f_statistic:.4f}")
    print(f"自由度: df1 = {df1}, df2 = {df2}")
    print(f"p值: {p_value:.6f}")

    # 判断是否拒绝原假设
    alpha = 0.05
    print(f"\n5. 假设检验结果 (α = {alpha}):")
    if p_value < alpha:
        print(f"在显著性水平 {alpha} 下，拒绝原假设")
        print("结论：各个模型的性能存在显著差异")
    else:
        print(f"在显著性水平 {alpha} 下，接受原假设")
        print("结论：未能证明各个模型的性能存在显著差异")

    return chi_squared, f_statistic, p_value, mean_ranks


def nemenyi_test(data_array):
    """
    执行Nemenyi后续检验
    """
    n_datasets, n_models = data_array.shape

    # 计算每个数据集上的排名
    ranks = np.zeros_like(data_array)
    for i in range(n_datasets):
        ranks[i] = rankdata(-data_array[i])  # 使用-号使得较大的值获得较小的排名

    # 计算平均排名
    mean_ranks = np.mean(ranks, axis=0)
    

    # 计算临界值
    k = n_models
    N = n_datasets
    q_alpha = 2.724  # 在α=0.05水平下的临界值
    critical_difference = q_alpha * np.sqrt((k * (k + 1)) / (6 * N))

    # 进行两两比较
    print("\nNemenyi测试结果:")
    print(f"临界差异值: {critical_difference:.4f}")
    print("\n平均排名:")
    for i in range(n_models):
        print(f"模型 {i+1}: {mean_ranks[i]:.4f}")

    print("\n显著性差异对比:")
    for i, j in combinations(range(n_models), 2):
        diff = abs(mean_ranks[i] - mean_ranks[j])
        is_significant = diff > critical_difference
        print(f"模型 {i+1} vs 模型 {j+1}: 差异 = {diff:.4f} {'(显著)' if is_significant else '(不显著)'}")

    # 创建CD图#
    names = ["GBkNN","ACC-GBkNN","ADP-GBkNN", "GBkNN++","SGBkNN", "kNN", "DT", "SVM", "AdaBoost"]
    cd = Orange.evaluation.compute_CD(mean_ranks, n_datasets, alpha='0.05', test='nemenyi')
    print("CD值", cd)
    Orange.evaluation.scoring.graph_ranks(mean_ranks, names, cd=cd, width=8, textspace=1.5, reverse=True)
    # 添加 CD 值文本（靠近 CD 条的位置自动计算）
    
    # plt.annotate(f"CD = {cd:.3f}", xy=(0.02, 0.02), xycoords='figure fraction', fontsize=12, fontweight='bold')
    # graph_ranks(mean_ranks, names, cd=cd, width=5, textspace=1.5)
    plt.savefig('CD-9-model.png', bbox_inches='tight', dpi=300)
    plt.show()
    return mean_ranks,cd



if __name__ == "__main__":
    # 读取Excel文件
    excel_path = 'extracted_recall_no_sd.xlsx'
    data = pd.read_excel(excel_path, header=None)
    accuracy_data = data.values
    
    # 运行详细的Friedman检验
    chi_squared, f_statistic, p_value, mean_ranks = run_detailed_friedman_analysis(accuracy_data)
    # 只调用一次nemenyi_test
    mean_ranks, cd = nemenyi_test(accuracy_data)









#

    
# if __name__ == "__main__":
#     # 保存准确率数据到Excel
#     df = pd.DataFrame(accuracy_data)
#     excel_path = '/Users/shenhai/Desktop/paper 4/返修实验/src/Tables/accuracy_data.xlsx'
#     df.to_excel(excel_path, index=False, header=False)
    
#     # 运行Friedman检验
#     run_friedman_analysis(accuracy_data)
    
#     # 运行Nemenyi后续检验
#     nemenyi_test(accuracy_data)

#     # 运行并可视化Nemenyi检验
#     mean_ranks, cd = visualize_nemenyi_test(accuracy_data)
#     print(f"\n临界差异值 (CD): {cd:.4f}")






# # import Orange  # version: Orange3==3.32.0
# # import matplotlib
# # # matplotlib.use('TkAgg')  # 不显示图则加上这两行
# # import matplotlib.pyplot as plt
# #
# # names = ['alg1', 'alg2', 'alg3', 'alg4', 'alg5', 'alg6', 'alg7']
# # avranks = [5.9, 4.37, 4.25, 5.39, 2.19, 2.85, 3.04]
# # datasets_num = 135
# # CD = Orange.evaluation.scoring.compute_CD(avranks, datasets_num, alpha='0.05')
# # Orange.evaluation.scoring.graph_ranks(avranks, names, cd=CD, width=8, textspace=1.5, reverse=True)
# # plt.show()





