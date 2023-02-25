import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('kc_house_data.csv')
# Y price
# X -> bedrooms , bathrooms , sqft_living ,sqft_lot ,floors,waterfront
df = df[["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront"]]
# df.info()
print(df[["sqft_living"]].describe())
# df['bedrooms'].hist()
# plt.show()
mixMAx = MinMaxScaler()

price = df[['price']] / 10000

beds = df['bedrooms'].values.reshape(-1, 1)
df['bedrooms'] = mixMAx.fit_transform(beds)

sqft_living = df['sqft_living'].values.reshape(-1, 1)
df['sqft_living'] =  mixMAx.fit_transform(sqft_living)

sqft_lot = df['sqft_lot'].values.reshape(-1, 1)
df['sqft_lot'] =  mixMAx.fit_transform(sqft_lot)

print(df[["sqft_living"]].describe())