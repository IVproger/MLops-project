FROM apache/airflow:2.9.2-python3.11

# Install dependencies
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

# Add our project to the PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow/project"
