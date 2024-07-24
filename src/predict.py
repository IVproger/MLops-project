import json

import hydra
import requests
from omegaconf import DictConfig

from src.data import extract_data, preprocess_data


@hydra.main(config_path="../configs", config_name="predict", version_base=None)
def main(cfg: DictConfig):
    predict_url = f"http://{cfg.hostname}:{cfg.port}/predict"
    version = cfg.example_version
    random_state = cfg.random_state

    df, _ = extract_data(version, cfg)
    sample = df.sample(n=1000, random_state=random_state)

    print(f"Predicting from {predict_url} with sample data:\n{sample.head()}")

    X_df, y_df = preprocess_data(cfg, sample)

    correct_count = 0
    for i in range(len(X_df)):
        X = X_df.iloc[[i]]
        y = y_df.iloc[i]

        # Convert it into JSON
        payload = X.iloc[0, :].to_dict()
        for key, value in payload.items():
            if not (
                key.endswith("cos")
                or key.endswith("sin")
                or key.startswith("CRSElapsedTime")
                or key == "Distance"
            ):
                payload[key] = int(value)
        payload = json.dumps(payload)

        response = requests.post(
            url=predict_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        result = response.json()
        y_pred = result["Cancelled"] > result["On-Time"]
        if y["Cancelled"] == y_pred:
            correct_count += 1
            print(
                f"[{i+1}/{len(X_df)}] Target: {y['Cancelled']}. Prediction: {y_pred} (Correct!)"
            )
        else:
            print(f"[{i+1}/{len(X_df)}] Target: {y['Cancelled']}. Prediction: {y_pred}")
    print(f"Prediction finished. Accuracy: {correct_count / len(X_df)}")


if __name__ == "__main__":
    main()
