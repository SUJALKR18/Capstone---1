import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\sujal\OneDrive\Desktop\CAPSTONE PROJECT\Capstone---1\embedded_data_bert\dataset_numeric_final.csv")

# X: first 100 columns, y: last column
X = data.iloc[:, :100]
y = data.iloc[:, 100]

print(data.info())


print("Y values:", y.value_counts())

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LogisticRegression(max_iter=1000)
svc = SVC()
rf = RandomForestClassifier()

svc.fit(X_train, y_train)

pred = svc.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

print("Confusion Matrix:", confusion_matrix(y_test, pred))

print("Train accuracy:", svc.score(X_train, y_train))

print("Test accuracy:", svc.score(X_test, y_test))