#!/bin/bash

mkdir -p $AIRFLOW_HOME/logs $AIRFLOW_HOME/dags
echo > $AIRFLOW_HOME/logs/scheduler.log
echo > $AIRFLOW_HOME/logs/triggerer.log
echo > $AIRFLOW_HOME/logs/webserver.log
