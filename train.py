import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accurac_score
import os

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("assignment5-exp")

def train():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    with mlflow.start_run() as run:
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)

        print("Run ID:", run.info.run_id)
        print("Accuracy:", acc)

        # save run id
        with open("model_info.txt", "w") as f:
            f.write(run.info.run_id)

if __name__ == "__main__":
    train()
