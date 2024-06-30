#!/bin/bash

# Start Airflow scheduler in the background
airflow scheduler --log-file services/airflow/logs/scheduler.log > /dev/null 2>&1 &

# Start Airflow webserver in the background with host binding
airflow webserver -p 8080 --hostname 0.0.0.0 --log-file services/airflow/logs/webserver.log > /dev/null 2>&1 &

# Start Airflow triggerer in the background
airflow triggerer --log-file services/airflow/logs/triggerer.log > /dev/null 2>&1 &

echo "Airflow services started in parallel."