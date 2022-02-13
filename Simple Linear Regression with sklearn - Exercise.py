from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


data = pd.read_csv("real_estate_price_size.csv")
data.head()

data.describe()

y = data['price']
x = data['size']

plt.scatter(x, y)
plt.xlabel("Size", fontsize="21")
plt.ylabel("Price", fontsize="21")
plt.show()

x_matrix = x.values.reshape(-1, 1)
x_matrix.shape

# ### Regression itself

reg = LinearRegression()
reg.fit(x_matrix, y)

reg.score(x_matrix, y)

reg.intercept_

reg.coef_

reg.predict([[750]])
