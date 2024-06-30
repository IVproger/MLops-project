# MLOps Capstone project

## Getting started
- Create virtual environment
```
$ python -m venv .venv
$ source .venv/bin/activate
```
- Install dependencies
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
0. Install poetry
1. Configure `.venv` location
```
$ poetry config virtualenvs.in-project true
```
2. Create `.venv` with Python 3.11 (make sure you have it installed) 
```
$ poetry env use <path-to-python3.11>
```
3. Install dependencies
```
$ poetry install
```

### Setup PostgreSQL
0. Install PostgreSQL
1. Run the PostgreSQL service 
   ```
   $ systemctl start postgresql
   ```
    - If you want to have PostgreSQL running on startup:
        ```
        $ systemctl enable postgresql
        ```
2. Run `psql` to connect to your PostgreSQL local server. You might need to switch to a specific user (`postgres` on Arch, for example)
3. Create user
    ```
    postgres=# CREATE USER <username> WITH PASSWORD '<password>';
    ```
4. Check that you have created the user with:
    ```
    postgres=# \du
    ```
5. Create database called `airflow`
    ```
    postgres=# CREATE DATABASE airflow;
    ```
6. Grant all permission for the user to the database:
    ```
    postgres=# GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO <username>;
    ```
7. To be extra sure that we have all the rights, let's make `<username>` the owner of the database
    ```
    ALTER DATABASE airflow OWNER to <username>;
    ```
8. Check the database status with:
    ```
    postgres=# \l
    ```
9. Let's configure the PSQL a bit, so that we have absolutely all of the permissions. Let's find where the config folder is:
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
    So my config folder is `/var/lib/postgres/data`. Let's save it as an env var: `export PSQL_CONFIG=/var/lib/postgres/data`

10. Edit `$PSQL_CONFIG_FOLDER/pg_hba.conf` and at the bottom add:
    ```
    host all all 0.0.0.0/0 trust
    ```
11. Edit `$PSQL_CONFIG_FOLDER/postgresql.conf`, find the lines, and edit:
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
12. Restart PostgreSQL
    ```
    $ systemctl restart postgresql
    ```
### Setup Airflow
1. Extend your `.venv/bin/activate` by running the `scripts/extend_activate.sh` script
2. Initialize the database
    ```
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
    ```
    $ airflow db reset
    $ airflow db init
    ```