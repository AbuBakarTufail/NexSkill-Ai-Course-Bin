import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data = pd.DataFrame({'x': np.arange(100), 'y': np.random.rand(100).cumsum()})
# sns.set_theme(style='darkgrid')
# sns.lineplot(x = 'x', y = 'y', data = data)
# plt.show()


# tips = sns.load_dataset("tips")
# # Create scatter plot
# sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time", style="time")
# # Show plot
# plt.show()

# Sample data
# x1 = [89, 43, 36, 36, 95, 10, 66, 34, 38, 20]
# y1 = [21, 46, 3, 35, 67, 95, 53, 72, 58, 10]
# x2 = [26, 29, 48, 64, 6, 5, 36, 66, 72, 40]
# y2 = [26, 34, 90, 33, 38, 20, 56, 2, 47, 15]
# # Create scatter plots
# plt.scatter(x1, y1, c="pink", linewidths=2, marker="s", edgecolor="green", s=50)
# plt.scatter(x2, y2, c="yellow", linewidths=2, marker="^", edgecolor="red", s=200)
# # Add labels
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# # Show plot
# plt.show()


# df = pd.read_csv('D:\Fullstack AI\Full-Stack-AI-Bootcamp-B-11\Week4\zameencom-property-data-By-Kaggle-short.csv',delimiter=";", parse_dates=[14],  date_format={'date_added': '%m-%d-%Y'} , index_col='property_id')
# top10 = df.head(10)
# sns.set_theme(style='darkgrid')
# g = sns.displot(data = top10, x = 'agency', y = 'price', hue = 'agent', kind = 'hist')
# g.figure.suptitle("sns.displot(data=dffilter, x=agency , y=price , hue=agent,  kind='hist'  )")
# g.figure.show()
# input("Press Enter to continue...")


df = pd.read_excel('Datasets/Marksheet.xlsx', sheet_name='Sheet1')
sns.set_theme(style='darkgrid')
g = sns.displot(data = df, x = 'Student', y = 'Marks', hue = 'Student', kind = 'hist')
g.figure.suptitle("sns.displot(data=dffilter, x=Student , y=Marks, hue=Student,  kind='hist'  )")
g.figure.show()
input("Press Enter to continue...")

