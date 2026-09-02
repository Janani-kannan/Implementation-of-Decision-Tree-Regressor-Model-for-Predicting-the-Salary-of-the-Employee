# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas

2.Import Decision tree classifier

3.Fit the data in the model

4.Find the accuracy score

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

# 1) Load dataset
csv_path = "Salary.csv"    # <-- Change path if needed
try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    raise FileNotFoundError(f"File not found at: {csv_path}. Update the path.")

print("Dataset Loaded Successfully!\n")

# 2) Data exploration
print("Shape:", data.shape)
display(data.head())
print("\nInfo:")
display(data.info())
print("\nMissing Values:\n", data.isnull().sum())

# 3) Encode categorical column 'Position'
if "Position" in data.columns:
    le = LabelEncoder()
    data["Position"] = le.fit_transform(data["Position"])
    print("\nLabel Encoding Mapping (Position):")
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(mapping)

# 4) Select features and target
X = data[["Position", "Level"]]
y = data["Salary"]

print("\nFeature Sample:")
display(X.head())
print("\nTarget Sample:")
display(y.head())

# 5) Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)
print(f"\nTrain Size: {X_train.shape}, Test Size: {X_test.shape}")

# 6) Initialize and train Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=10)
dt.fit(X_train, Y_train)
print("\nModel Training Completed!")

# 7) Predict on test data
y_pred = dt.predict(X_test)
print("\nPredicted Salaries:", y_pred)

# 8) Evaluate model using R² Score
r2 = metrics.r2_score(Y_test, y_pred)
print(f"\nR² Score: {r2:.4f}")

# 9) Visualize Decision Tree
plt.figure(figsize=(12,8))
plot_tree(dt, feature_names=["Position", "Level"], filled=True)
plt.title("Decision Tree Regressor for Salary Prediction")
plt.show()

# 10) Feature Importances
importances = pd.Series(dt.feature_importances_, index=["Position", "Level"])
print("\nFeature Importances:")
display(importances)

```

## Output:

<img width="1451" height="850" alt="image" src="https://github.com/user-attachments/assets/cee6ef47-7b1e-493d-98e9-6d4af2808e88" />
<img width="547" height="388" alt="image" src="https://github.com/user-attachments/assets/0f860aa3-5725-4e9d-a03f-e9050f2bff11" />
<img width="968" height="637" alt="image" src="https://github.com/user-attachments/assets/fb1b4e0b-4ee8-4127-acb1-8ed82e8ff416" />
<img width="191" height="190" alt="image" src="https://github.com/user-attachments/assets/a94fdafc-43dc-4e61-9960-7b7e26b97456" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
