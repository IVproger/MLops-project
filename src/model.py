import giskard  # noqa
import os
import pandas as pd  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import importlib  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402
from pandas import DataFrame  # noqa: E402
from zenml.client import Client
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")


def fetch_features(name: str, version: str, cfg: DictConfig):
    client = Client()
    lst = client.list_artifact_versions(name=name, tag=version, sort_by="version").items
    lst.reverse()

    X, y = lst[0].load()
    return X, y


def train(
    X_train: DataFrame,
    y_train: DataFrame,
    cfg: DictConfig,
):
    # Define the model hyperparameters
    params = cfg.model.hydra.sweeper.params

    # Train the model
    module_name = cfg.model.module_name
    class_name = cfg.model.class_name

    print(params, module_name, class_name)

    # We will create the estimator at runtime
    import importlib

    # Load "module.submodule.MyClass"
    class_instance = getattr(importlib.import_module(module_name), class_name)

    estimator = class_instance(**params)

    # Grid search with cross validation
    from sklearn.model_selection import StratifiedKFold

    # Define cross validation
    cv = StratifiedKFold(
        n_splits=cfg.model.folds,
        random_state=cfg.random_state,
        shuffle=True,
    )

    # Define param grid
    param_grid = dict(params)

    # Define metrics for scoring
    scoring = list(cfg.model.metrics.values())

    # Define evaluation metric
    evaluation_metric = cfg.model.evaluation_metric

    # Instantiate GridSearch
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=cfg.cv_n_jobs,
        refit=evaluation_metric,
        cv=cv,
        verbose=1,
        return_train_score=True,
    )

    # Fit GridSearch
    print("Starting GridSearch fit...")
    gs.fit(X_train, y_train.values.ravel())

    return gs


def plot_performance_charts(model, X_test, y_test, run_name):
    # Generate ROC Curve
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    roc_path = f"{run_name}_roc_curve.png"
    plt.savefig(roc_path)
    plt.close()

    # Generate Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    cm_path = f"{run_name}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    return roc_path, cm_path


def download_charts(run_id, destination_folder="results"):
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Set MLflow tracking URI if necessary
    # mlflow.set_tracking_uri("your_mlflow_tracking_uri")

    # Fetch artifacts
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)

    # Filter for specific chart types if necessary, e.g., PNG files
    chart_artifacts = [
        artifact for artifact in artifacts if artifact.path.endswith(".png")
    ]

    for artifact in chart_artifacts:
        local_path = os.path.join(destination_folder, artifact.path)
        artifact_path = artifact.path  # The artifact's path in MLflow

        # Download the artifact
        client.download_artifacts(run_id, artifact_path, destination_folder)
        print(f"Downloaded: {artifact_path} to {local_path}")


def log_metadata(
    cfg: DictConfig,
    gs: GridSearchCV,
    X_train: DataFrame,
    y_train: DataFrame,
    X_test: DataFrame,
    y_test: DataFrame,
):
    # Filter cv_results
    cv_results = (
        pd.DataFrame(gs.cv_results_)
        .filter(regex=r"std_|mean_|param_")
        .sort_index(axis=1)
    )
    print(cv_results, cv_results.columns)

    # Get best params
    best_metrics_values = [
        result[1][gs.best_index_] for result in gs.cv_results_.items()
    ]
    best_metrics_keys = [metric for metric in gs.cv_results_]
    best_metrics_dict = {
        k: v
        for k, v in zip(best_metrics_keys, best_metrics_values)
        if "mean" in k or "std" in k
    }

    params = best_metrics_dict

    # Join back train and test datasets (for logging)
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    # Define experiment
    experiment_name = cfg.model.model_name + "_" + cfg.experiment_name

    try:
        # Create a new MLflow Experiment
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(
            name=experiment_name
        ).experiment_id  # type: ignore

    print("experiment-id : ", experiment_id)

    cv_evaluation_metric = cfg.model.cv_evaluation_metric
    run_name = "_".join(
        [
            cfg.run_name,
            cfg.model.model_name,
            cfg.model.evaluation_metric,
            str(params[cv_evaluation_metric]).replace(".", "_"),
        ]
    )
    print("run name: ", run_name)

    # Stop any existing runs
    if mlflow.active_run():
        mlflow.end_run()

    # Parent run
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        df_train_dataset = mlflow.data.pandas_dataset.from_pandas(
            df=df_train,
            # targets=cfg.fize,
        )
        df_test_dataset = mlflow.data.pandas_dataset.from_pandas(
            df=df_test,
            # targets=cfg.data.target_cols[0],
        )
        mlflow.log_input(df_train_dataset, "training")
        mlflow.log_input(df_test_dataset, "testing")

        # Log the hyperparameters
        mlflow.log_params(gs.best_params_)

        # Log the performance metrics
        mlflow.log_metrics(best_metrics_dict)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag(cfg.model.tag_key, cfg.model.tag_value)

        # Infer the model signature
        signature = mlflow.models.infer_signature(X_train, gs.predict(X_train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=gs.best_estimator_,
            artifact_path=cfg.model.artifact_path,
            signature=signature,
            input_example=X_train.iloc[0].to_numpy(),
            registered_model_name=cfg.model.model_name,
            pyfunc_predict_fn=cfg.model.pyfunc_predict_fn,
        )

        client = mlflow.client.MlflowClient()
        client.set_model_version_tag(
            name=cfg.model.model_name,
            version=model_info.registered_model_version,
            key="source",
            value="best_Grid_search_model",
        )
        client.set_registered_model_alias(
            name=cfg.model.model_name,
            alias=f"challenger{model_info.registered_model_version}",
            version=model_info.registered_model_version,
        )

        # Log metrics for every fold
        for index, result in cv_results.iterrows():
            child_run_name = "_".join(["child", run_name, str(index)])
            with mlflow.start_run(
                run_name=child_run_name,
                experiment_id=experiment_id,
                nested=True,
            ) as children_run:
                run_id_children = children_run.info.run_id
                ps = result.filter(regex="param_").to_dict()
                ms = result.filter(regex="mean_").to_dict()
                stds = result.filter(regex="std_").to_dict()

                # Remove param_ from the beginning of the keys
                ps = {k.replace("param_", ""): v for (k, v) in ps.items()}

                mlflow.log_params(ps)
                mlflow.log_metrics(ms)
                mlflow.log_metrics(stds)

                # We will create the estimator at runtime
                module_name = cfg.model.module_name
                class_name = cfg.model.class_name

                # Load "module.submodule.MyClass"
                class_instance = getattr(
                    importlib.import_module(module_name), class_name
                )

                estimator = class_instance(**ps)
                estimator.fit(X_train, y_train)

                # Generate the performance charts
                roc_path, cm_path = plot_performance_charts(
                    estimator, X_test, y_test, child_run_name
                )

                # Log the performance charts
                mlflow.log_artifact(roc_path)
                mlflow.log_artifact(cm_path)

                # Delete the temporary files of charts
                os.remove(roc_path)
                os.remove(cm_path)

                # download charts from MLflow artifact store into results
                download_charts(run_id_children)

                signature = mlflow.models.infer_signature(
                    X_train, estimator.predict(X_train)
                )

                model_info = mlflow.sklearn.log_model(
                    sk_model=estimator,
                    artifact_path=cfg.model.artifact_path,
                    signature=signature,
                    input_example=X_train.iloc[0].to_numpy(),
                    registered_model_name=cfg.model.model_name,
                    pyfunc_predict_fn=cfg.model.pyfunc_predict_fn,
                )

                model_uri = model_info.model_uri
                loaded_model = mlflow.sklearn.load_model(model_uri=model_uri)

                predictions = loaded_model.predict(X_test)

                eval_data = pd.DataFrame(y_test)
                eval_data.columns = ["label"]
                eval_data["predictions"] = predictions

                results = mlflow.evaluate(
                    data=eval_data,
                    model_type="classifier",
                    targets="label",
                    predictions="predictions",
                    evaluators=["default"],
                )

                print(f"metrics:\n{results.metrics}")

                mlflow.end_run()

        mlflow.end_run()


def retrieve_model_with_alias(model_name, model_alias) -> mlflow.pyfunc.PyFuncModel:
    model: mlflow.pyfunc.PyFuncModel = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}@{model_alias}"
    )

    return model


def retrieve_model_with_version(
    model_name, model_version="v1"
) -> mlflow.pyfunc.PyFuncModel:
    best_model: mlflow.pyfunc.PyFuncModel = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{model_version}"
    )

    # best_model
    return best_model


def choose_champion(model_name: str):
    client = mlflow.tracking.MlflowClient()
    max_f1 = -10000000
    champion = None
    champion_version = None
    for registered_model in client.search_registered_models():
        if registered_model.aliases.get("champion"):
            return (
                retrieve_model_with_alias(model_name, "champion"),
                registered_model.aliases.get("champion"),
            )

        for model_alias in registered_model.aliases.keys():
            model = retrieve_model_with_alias(model_name, model_alias)
            metrics = client.get_metric_history(model.metadata.run_id, "mean_test_f1")

            if metrics[0].value > max_f1:
                max_f1 = metrics[0].value
                champion = model
                champion_version = registered_model.aliases.get(model_alias)

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=champion_version,
    )

    return champion, champion_version
