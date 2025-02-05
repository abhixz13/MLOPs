FROM python:3.10-slim

ENV AIRFLOW_HOME=/opt/airflow
ENV PYTHONPATH=/opt/airflow
ENV AIRFLOW_VERSION=2.8.1

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash -d ${AIRFLOW_HOME} airflow

# Install psycopg2 before Airflow
RUN pip install --no-cache-dir psycopg2-binary

# Install Airflow with constraints
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.10.txt"

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR ${AIRFLOW_HOME}

# Copy application code last since it changes most frequently
COPY src/ ${AIRFLOW_HOME}/src/
COPY airflow/dags/ ${AIRFLOW_HOME}/dags/

RUN chown -R airflow: ${AIRFLOW_HOME}

USER airflow

EXPOSE 8080

CMD ["airflow", "webserver"]