# MLOps Capstone project

## Getting started

1. Create virtual environment
    ```
    $ python -m venv .venv
    $ source .venv/bin/activate
    ```
2. Install dependencies
    ```
    $ bash scripts/install_requirements.sh
    ```

### Sample and validate data

Run

```
$ bash scripts/test_data.sh
```

## How to Airflow

### Setup Poetry

1. Install poetry
2. Add poetry plugin
   ```bash
   $ poetry self add poetry-plugin-export
   ```
3. Configure `.venv` location
   ```bash
   $ poetry config virtualenvs.in-project true
   ```
4. Create `.venv` with Python 3.11 (make sure you have it installed)
   ```bash
   $ poetry env use <path-to-python3.11>
   ```
5. Install dependencies
   ```bash
   $ poetry install
   ```
6. Set up pre-commit hooks
   ```bash
   $ poetry run pre-commit install --install-hooks -t pre-commit -t commit-msg
   ```

### Setup PostgreSQL

1. Install PostgreSQL
2. Run the PostgreSQL service
   ```bash
   $ systemctl start postgresql
   ```
    - If you want to have PostgreSQL running on startup:
        ```bash
        $ systemctl enable postgresql
        ```
3. Run `psql` to connect to your PostgreSQL local server. You might need to switch to a specific user (`postgres` on
   Arch, for example)
4. Create user
    ```
    postgres=# CREATE USER <username> WITH PASSWORD '<password>';
    ```
5. Check that you have created the user with:
    ```
    postgres=# \du
    ```
6. Create database called `airflow`
    ```
    postgres=# CREATE DATABASE airflow;
    ```
7. Grant all permission for the user to the database:
    ```
    postgres=# GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO <username>;
    ```
8. To be extra sure that we have all the rights, let's make `<username>` the owner of the database
    ```
    ALTER DATABASE airflow OWNER to <username>;
    ```
9. Check the database status with:
    ```
    postgres=# \l
    ```
10. Let's configure the PSQL a bit, so that we have absolutely all of the permissions. Let's find where the config
    folder is:
     ```
     postgres=# show config_file
     ```

    Output:
     ```
                   config_file
     ----------------------------------------
     /var/lib/postgres/data/postgresql.conf
     (1 row)
     ```
    So my config folder is `/var/lib/postgres/data`. Let's save it as an env
    var: `export PSQL_CONFIG=/var/lib/postgres/data`

11. Edit `$PSQL_CONFIG_FOLDER/pg_hba.conf` and at the bottom add:
    ```
    host all all 0.0.0.0/0 trust
    ```
12. Edit `$PSQL_CONFIG_FOLDER/postgresql.conf`, find the lines, and edit:
    ```
    # $PSQL_CONFIG_FOLDER/postgresql.conf
    . . .
    #------------------------------------------------------------------------------
    # CONNECTIONS AND AUTHENTICATION
    #------------------------------------------------------------------------------

    # - Connection Settings -

    listen_addresses = '*'
    . . .
    ```
13. Restart PostgreSQL
    ```bash
    $ systemctl restart postgresql
    ```

### Setup Airflow

1. Extend your `.venv/bin/activate` by running
   ```bash
   ./scripts/extend_activate.sh
   ```
2. Initialize the database
    ```bash
    $ airflow db init
    ```
   This should create files in `services/airflow`
3. Go to `services/airflow/airflow.cfg`, then change the following
    ```
    # services/airflow/airflow.cfg
    . . .
    executor = LocalExecutor
    . . .
    sql_alchemy_conn = postgresql+psycopg2://<db_username>:<db_pass>@localhost:5432/airflow
    . . .
    load_examples = False
    . . .
    ```
4. Reset & init the database:
    ```bash
    $ airflow db reset
    $ airflow db init
    ```
5. Setup airflow folders
   ```bash
   $ ./scripts/airflow_logs.sh
   ```
6. Add airflow user
   ```bash
   $ airflow users create --role Admin --username admin --email admin@example.org --firstname admin --lastname admin --password admin
   ```
7. Run the airflow services:
    ```bash
    $ ./scripts/airflow_activate.sh
    ```
