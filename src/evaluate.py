import hydra
import mlflow
from omegaconf import DictConfig
from src.model import fetch_features
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from src.model import choose_champion


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    # Find, mark, and return champion
    champion, _ = choose_champion(cfg.model.model_name)

    # Download the champion model
    client = mlflow.tracking.MlflowClient()
    client.download_artifacts(
        champion.metadata.run_id, cfg.model.artifact_path, "models"
    )

    # Evaluate it
    evaluate(cfg, champion)


def evaluate(cfg, champion):
    print("Fetching features for testing...")
    X_test, y_test = fetch_features(
        name=cfg.data.artifact_name,
        version=cfg.test_data_version,
        cfg=cfg,
    )

    # Predict Test Data
    y_pred = champion.predict(X_test)

    # Convert probabilities to class labels
    class_labels = np.argmax(y_pred, axis=1)

    # Now compute metrics using these class labels
    accuracy = accuracy_score(y_test, class_labels)
    precision = precision_score(y_test, class_labels, average="weighted")
    recall = recall_score(y_test, class_labels, average="weighted")
    f1 = f1_score(y_test, class_labels, average="weighted")

    # For AUC, you need the probabilities of the positive class (assuming it's at index 1)
    auc = roc_auc_score(y_test, y_pred[:, 1])

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"AUC: {auc}")

    # Generate Classification Report
    report = classification_report(y_test, class_labels)
    print(report)


if __name__ == "__main__":
    main()
