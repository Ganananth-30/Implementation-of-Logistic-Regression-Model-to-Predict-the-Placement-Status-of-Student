# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load the placement dataset and remove unnecessary columns such as serial number and salary.
2.Convert all categorical attributes into numerical values using label encoding.
3.Split the dataset into training and testing sets using train–test split.
4.Train a Logistic Regression model using the training data.
5.Predict placement status and evaluate the model using accuracy and confusion matrix.
 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: GANANANTH H
RegisterNumber:  25010984
*/


# Logistic Regression for Student Placement Prediction

# 1️ Import Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 2️ Load Dataset

data = pd.read_csv("C:/Users/acer/Desktop/ML ex/Placement_Data.csv")   

print("Dataset Preview:")
print(data.head())

# 3️ Drop Unnecessary Columns

data = data.drop(["sl_no", "salary"], axis=1)

# 4️ Convert Target Variable (status) to Binary

# Placed = 1, Not Placed = 0
data["status"] = data["status"].map({"Placed": 1, "Not Placed": 0})

# 5️ Separate Features and Target

X = data.drop("status", axis=1)
y = data["status"]

# 6️ One-Hot Encode Categorical Variables

X = pd.get_dummies(X, drop_first=True)

print("\nAfter Encoding:")
print(X.head())

# 7️ Feature Scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8️ Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 9️ Train Logistic Regression Model

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 10 Make Predictions

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 1️1 Model Evaluation

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Placement Prediction")
plt.show()  
```

## Output:

<img width="880" height="634" alt="Screenshot 2026-02-04 093949" src="https://github.com/user-attachments/assets/91429b95-46b2-4cca-99a9-1ac5ab8a2139" />


<img width="797" height="419" alt="Screenshot 2026-02-04 094009" src="https://github.com/user-attachments/assets/7d2f391e-6ad9-4776-a279-09d077119c3c" />


<img width="850" height="589" alt="Screenshot 2026-02-04 094018" src="https://github.com/user-attachments/assets/162e99bb-e05c-44ed-b303-aae5c5cb6997" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
