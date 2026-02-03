# ======================================================
# Task 2: Salary Prediction using Polynomial Regression
# Corizo ML Assignment
# Author: Your Name
# ======================================================

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------
# Step 1: Create dataset (No CSV required)
# ------------------------------------------------------
data = pd.DataFrame({
    "YearsExperience": [1,2,3,4,5,6,7,8,9,10],
    "Salary": [45000,50000,60000,65000,75000,85000,95000,105000,115000,130000]
})

print("Dataset Preview:\n")
print(data)

# ------------------------------------------------------
# Step 2: Split features & target
# ------------------------------------------------------
X = data[['YearsExperience']]
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------
# Step 3: Polynomial Features
# ------------------------------------------------------
poly = PolynomialFeatures(degree=3)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# ------------------------------------------------------
# Step 4: Train Model
# ------------------------------------------------------
model = LinearRegression()
model.fit(X_train_poly, y_train)

# ------------------------------------------------------
# Step 5: Prediction
# ------------------------------------------------------
y_pred = model.predict(X_test_poly)

# ------------------------------------------------------
# Step 6: Evaluation Metrics
# ------------------------------------------------------
print("\n===== Model Performance =====")
print("MAE  :", mean_absolute_error(y_test, y_pred))
print("MSE  :", mean_squared_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2   :", r2_score(y_test, y_pred))

# ------------------------------------------------------
# Step 7: Visualization
# ------------------------------------------------------
plt.scatter(X, y)
plt.plot(X, model.predict(poly.transform(X)))
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Polynomial Regression - Salary Prediction")
plt.show()
