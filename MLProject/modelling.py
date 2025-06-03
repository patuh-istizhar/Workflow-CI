import os
import sys

import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# Enable MLflow autologging for LightGBM
mlflow.lightgbm.autolog()


def main(train_path, test_path, n_estimators, max_depth):
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns="Diabetes_012")
    y_train = train_df["Diabetes_012"]
    X_test = test_df.drop(columns="Diabetes_012")
    y_test = test_df["Diabetes_012"]

    # Train a LightGBM model with specified parameters
    with mlflow.start_run(run_name="LightGBM_Baseline"):
        model = LGBMClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Log test accuracy
        preds = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, preds)
        mlflow.log_metric("test_accuracy", test_accuracy)


if __name__ == "__main__":
    # Expect train_path, test_path, n_estimators, and max_depth as command-line arguments
    if len(sys.argv) != 5:
        print(
            "Usage: python modelling.py <train_path> <test_path> <n_estimators> <max_depth>"
        )
        sys.exit(1)

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    n_estimators = int(sys.argv[3])
    max_depth = int(sys.argv[4])

    # Verify the files exist
    if not os.path.exists(train_path):
        print(f"Error: Training file {train_path} does not exist")
        sys.exit(1)
    if not os.path.exists(test_path):
        print(f"Error: Test file {test_path} does not exist")
        sys.exit(1)

    main(train_path, test_path, n_estimators, max_depth)
