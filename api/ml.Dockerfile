# Docker image for running API for MLflow model
FROM python:3.11-slim

# Install Nginx
RUN apt-get update && apt-get install --no-install-recommends -y nginx && rm -rf /var/lib/apt/lists/*

# Install MLflow
RUN pip install --no-cache mlflow==2.14.2

# Copy model to image and install dependencies
WORKDIR /opt/mlflow
COPY api/model_dir/ /opt/ml/model
RUN python3 -c "from mlflow.models import container as C; C._install_pyfunc_deps('/opt/ml/model', install_mlflow=False, enable_mlserver=False, env_manager='local');"

# Configure MLFlow settings
ENV MLFLOW_DISABLE_ENV_CREATION=True
ENV ENABLE_MLSERVER=False
ENV GUNICORN_CMD_ARGS="--timeout 60 -k gevent"

# Set correct permissions
RUN chmod o+rwX /opt/mlflow/

CMD [ "python3", "-c", "from mlflow.models import container as C; C._serve('local')" ]
