import mlflow
import mlflow.sklearn
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import seaborn as sns
import os


import dagshub
dagshub.init(repo_owner='Muhammedshibili688', repo_name='Mlops-Experiments-MLFlow', mlflow=True)


mlflow.set_tracking_uri("https://dagshub.com/Muhammedshibili688/Mlops-Experiments-MLFlow.mlflow")
mlflow.set_experiment("YT-MLOPS-Exp1")

df = load_wine()
X = df.data
y = df.target

with mlflow.start_run():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

    max_depth = 5
    n_estimators = 10

    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

    print(accuracy)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,6))
    plt.title("Confusion matrix")
    sns.heatmap(cm, cmap='Blues', fmt='d', xticklabels=df.target_names, yticklabels=df.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig('Confusion-matrix.png')

    mlflow.log_artifact('Confusion-matrix.png')
    mlflow.log_artifact("src/file2.py")
    mlflow.sklearn.log_model(rf, "Random Forest Classifier")

    mlflow.set_tags({"Author": "Muhammed SHibili", "Project": "Wine Classification"})