import pandas as pd

df = pd.read_csv('Datasets/RealEstate-USA.csv', delimiter=',')
# print(df)
# print(df.dtypes)
# print(df.info())
# print(df.tail())
# print(df.head())
# print(df.describe())
# print(df.shape)

# price = df[['price', 'status']]
# print(price)

# rows = df.loc[1:5]
# print(rows)

# filterByCity = df.loc[df['city'] == 'Ponce']
# print(filterByCity)

rows = df.loc[df['city'] == 'Ponce', ['price', 'status']]
print(rows)
