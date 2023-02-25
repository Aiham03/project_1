import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('kc_house_data.csv')
# Y price
# X -> bedrooms , bathrooms , sqft_living ,sqft_lot ,floors,waterfront
df = df[["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront"]]
df.info()
print(df.describe())
df['bedrooms'].hist()
plt.show()
mixMAx = MinMaxScaler()