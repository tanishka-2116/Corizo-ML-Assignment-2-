#Question for Task3: Find the house price prediction using support vector regressor (SVR)

# ======================================================
# Task 3: House Price Prediction using SVR
# Corizo ML Assignment
# ======================================================

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------------------------
# Step 1: Create dataset (No CSV required)
# ------------------------------------------------------
data = pd.DataFrame({
    "Area": [500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300],
    "Bedrooms": [1,1,2,2,3,3,3,4,4,5],
    "Price": [100000,150000,180000,220000,260000,300000,340000,380000,420000,460000]
})

print("Dataset Preview:\n")
print(data)

# ------------------------------------------------------
# Step 2: Split features & target
# ------------------------------------------------------
X = data[['Area','Bedrooms']]
y = data['Price']

# ------------------------------------------------------
# Step 3: Scaling (VERY IMPORTANT for SVR)
# ------------------------------------------------------
sc_X = StandardScaler()
sc_y = StandardScaler()

X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y.values.reshape(-1,1))

# ------------------------------------------------------
# Step 4: Train test split
# ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# ------------------------------------------------------
# Step 5: Train SVR model
# ------------------------------------------------------
model = SVR(kernel='rbf')
model.fit(X_train, y_train.ravel())

# ------------------------------------------------------
# Step 6: Prediction
# ------------------------------------------------------
y_pred = model.predict(X_test)

# Convert back to original scale
y_pred = sc_y.inverse_transform(y_pred.reshape(-1,1))
y_test = sc_y.inverse_transform(y_test)

# ------------------------------------------------------
# Step 7: Metrics
# ------------------------------------------------------
print("\n===== Model Performance =====")
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2   :", r2_score(y_test, y_pred))

# ------------------------------------------------------
# Step 8: Visualization
# ------------------------------------------------------
plt.scatter(range(len(y_test)), y_test, label="Actual")
plt.scatter(range(len(y_pred)), y_pred, label="Predicted")
plt.xlabel("Test Samples")
plt.ylabel("House Price")
plt.title("SVR House Price Prediction")
plt.legend()
plt.show()
