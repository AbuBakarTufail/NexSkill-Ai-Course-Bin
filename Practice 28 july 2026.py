import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('archive/housing.csv')
#print(df.head())
# print(df.columns.to_list())

# print(df[['total_bedrooms', 'total_rooms']].head(10))
df.plot.scatter(x='total_bedrooms', y='total_rooms', c='red', alpha=0.5)
# plt.title('Abdul Hanan')
# plt.xlabel('Total Bedrooms')
# plt.ylabel('Total Rooms')
plt.show()

XVal = df['total_bedrooms'].values.reshape(-1, 1)
YVal = df['total_rooms'].values.reshape(-1, 1)

# print("y :  " , YVal)
# print("X :   " , XVal)
SEED = 42
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(XVal, YVal, test_size=0.2, random_state=SEED)

from sklearn.linear_model import LinearRegression
regresssion = LinearRegression()

regresssion.fit(X_train, Y_train)
print(regresssion.intercept_)
print(regresssion.coef_)  

regresssion.predict([[1000]])

