import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


data = pd.read_excel('data.xlsx', sheet_name='Sheet1')
x = data['x']
curve1 = data['curve1']
curve2 = data['curve2']degree = 3  # Adjust the degree of the polynomial as needed
poly_features = PolynomialFeatures(degree=degree)
X = poly_features.fit_transform(x.values.reshape(-1, 1))


model = LinearRegression()
model.fit(X, curve1)


x_pred = np.linspace(x.min(), x.max(), num=100)
X_pred = poly_features.fit_transform(x_pred.reshape(-1, 1))
curve_pred = model.predict(X_pred)


plt.figure(figsize=(8, 6))
plt.scatter(x, curve1, color='red', label='Curve 1')
plt.scatter(x, curve2, color='blue', label='Curve 2')
plt.plot(x_pred, curve_pred, color='green', label='Predicted Curve')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Curve Prediction')
plt.show()
