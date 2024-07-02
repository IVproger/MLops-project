#!/bin/bash

# Kill process on port 8080
PID=$(lsof -ti:8080)
if [ ! -z "$PID" ]; then
  echo "Killing process on port 8080 with PID: $PID"
  kill -9 $PID
fi

# Kill Airflow scheduler, webserver, and triggerer processes
for service in scheduler webserver triggerer; do
  PIDS=$(ps aux | grep "airflow $service" | grep -v grep | awk '{print $2}')
  if [ ! -z "$PIDS" ]; then
    echo "Killing Airflow $service processes: $PIDS"
    kill -9 $PIDS
  fi
done

pkill airflow

echo "Cleanup complete."