#!/bin/bash

mkdir -p $AIRFLOW_HOME/logs
echo > $AIRFLOW_HOME/logs/scheduler.log
echo > $AIRFLOW_HOME/logs/triggerer.log
echo > $AIRFLOW_HOME/logs/webserver.log
