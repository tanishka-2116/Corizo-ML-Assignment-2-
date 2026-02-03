# ======================================================
# Task 4: Iris Classification - Model Comparison
# Corizo ML Assignment
# ======================================================

# Import libraries
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------
# Step 1: Load dataset (built-in, no CSV required)
# ------------------------------------------------------
iris = load_iris()

X = iris.data
y = iris.target

print("Dataset shape:", X.shape)

# ------------------------------------------------------
# Step 2: Train Test Split
# ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------
# Step 3: Models
# ------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}

# ------------------------------------------------------
# Step 4: Train & Evaluate each model
# ------------------------------------------------------
print("\n===== Model Accuracies =====")

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# ------------------------------------------------------
# Step 5: Best Model
# ------------------------------------------------------
best_model = max(results, key=results.get)

print("\n✅ Best Model:", best_model)
print("✅ Best Accuracy:", results[best_model])
