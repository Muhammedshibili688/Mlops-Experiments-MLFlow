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

#mlflow.set_tracking_uri("http://127.0.0.1:5000")
BASE_DIR = os.path.abspath("mlruns")

mlflow.set_tracking_uri(f"file:///{BASE_DIR}")
# mention experiment name 
mlflow.set_experiment("YT-MLOPS-Exp1")

df = load_wine()
x = df.data
y = df.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.10)

max_depth = 3
n_estimators = 15

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth = max_depth, 
                            n_estimators = n_estimators, 
                            random_state = 42)
    
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

    print (accuracy)

    cm = confusion_matrix(y_test, y_pred)
    # plotting and savinng fig

    plt.figure(figsize=(6,6))
    plt.title("Confusion matrix")
    sns.heatmap(cm, cmap = 'Blues', fmt = 'd',xticklabels = df.target_names, yticklabels = df.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    # Save img
    plt.savefig('Confusion-matrix.png')

    mlflow.log_artifact('Confusion-matrix.png')
    mlflow.log_artifact("src/first1.py")
    mlflow.sklearn.load_model(rf, "Random Forest Classifier")

    mlflow.set_tag({"Author": "Muhammed SHibili", "Project": "Wine Classifiaction"})