## 给四个性能指标的表  计算平均排名


import pandas as pd

# 假设数据在一个名为'data.xlsx'的表格中，第一列是数据集名称，第一行是模型名称

data = pd.read_excel('extracted_accuracy_no_average.xlsx')
# 对每一行进行排序，获取排名，处理相同值的情况
rankings = data.apply(lambda row: row.rank(ascending=False, method='average'), axis=1)

# 计算每列的均值
column_means = rankings.mean(axis=0)
rankings.loc['Average'] = column_means

# 导出排名表到xlsx
rankings.to_excel('rankings_accuracy.xlsx')