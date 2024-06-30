#!/bin/bash

        
trap 'echo "Caught Ctrl+C, killing daemons..."; kill -TERM "$child1" "$child2" "$child3"; exit' SIGINT

airflow webserver --daemon --log-file services/airflow/logs/webserver.log > /dev/null &
child1=$!
echo Webserver starting... PID $child1
# sleep 6

airflow scheduler --daemon --log-file services/airflow/logs/scheduler.log > /dev/null &
child2=$!
echo Scheduler starting... PID $child2
# sleep 2

airflow triggerer --daemon --log-file services/airflow/logs/triggerer.log > /dev/null &
child3=$!
echo Triggerer starting... PID $child3
# sleep 2
