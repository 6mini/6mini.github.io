---
title: '[에어플로우] 에어플로우(Airflow)의 CLI와 UI'
description: 에어플로우 설치 후 CLI 및 UI 간단한 실습과 기능 파악. 에어플로우에서의 스파크(Spark) 사용
categories:
 - Data Engineering
tags: [데이터 엔지니어링, 에어플로우]
---

# 에어플로우(Airflow) 설치
- 다음과 같은 명령어로 간단하게 설치를 진행한다.

```sh
$ pip install apache-airflow
```

- 에어플로우는 플라스크(Flask)를 기반으로 웹 서버를 구현한다.

```sh
$ airflow db init
```

- 데이터베이스를 이니셜라이즈(initialize)한 후 UI를 띄운다.

```sh
$ airflow webserver -p 8080
```

- 127.0.0.1:8000 으로 접속하면 에어플로우에 접속할 수 있고 로그인 창이 생긴다.

![image](https://user-images.githubusercontent.com/79494088/146643924-646d9edf-16a0-4244-a878-e2108f733a9a.png)

- 유저를 생성한다.

```sh
$ airflow users create --role Admin --username admin --email admin --firstname admin --lastname admin --password admin 
```

![image](https://user-images.githubusercontent.com/79494088/146643899-444448ab-f95b-4b8a-9db7-08c68fbb1c17.png)

- 예제 DAG를 확인할 수 있다.

# 에어플로우 CLI

```sh
$ airflow -h
'''
positional arguments:
  GROUP_OR_COMMAND

    Groups: 번들링 된 커맨드
      celery         Celery components
      config         View configuration
      connections    Manage connections
      dags           Manage DAGs: DAG를 관리
      db             Database operations: 데이터 베이스를 관리
      jobs           Manage jobs
      kubernetes     Tools to help run the KubernetesExecutor
      pools          Manage pools
      providers      Display providers
      roles          Manage roles
      tasks          Manage tasks
      users          Manage users
      variables      Manage variables

    Commands: 일회성으로 사용할 수 있는 커맨드
      cheat-sheet    Display cheat sheet: 일반적으로 사용되는 커맨드의 조합
      info           Show information about current Airflow and
                     environment: 현재 환경의 정보들
      kerberos       Start a kerberos ticket renewer
      plugins        Dump information about loaded plugins
      rotate-fernet-key
                     Rotate encrypted connection credentials and
                     variables
      scheduler      Start a scheduler instance: 스케쥴을 여는 커맨드
      standalone     Run an all-in-one copy of Airflow
      sync-perm      Update permissions for existing roles and
                     optionally DAGs
      triggerer      Start a triggerer instance
      version        Show the version
      webserver      Start a Airflow webserver instance: 웹 서버를 여는 커맨드
'''
```

## webserver
- 웹 서버를 열어본다.

```sh
$ airflow webserver
```

- 127.0.0.1:8000으로 접속하면 에어플로우에 접속할 수 있고 로그인 창이 생긴다.

![image](https://user-images.githubusercontent.com/79494088/146643924-646d9edf-16a0-4244-a878-e2108f733a9a.png)

## users

```sh
$ airflow users -h 
'''
positional arguments:
  COMMAND
    add-role   Add role to a user
    create     Create a user
    delete     Delete a user
    export     Export all users
    import     Import users
    list       List users
    remove-role
               Remove role from a user

$ airflow users list

id | username | email | first_name | last_name | roles
===+==========+=======+============+===========+======
1  | admin    | admin | admin      | admin     | Admin
'''
```

- 설치할 때 만든 `admin user`가 보인다.

## scheduler

![image](https://user-images.githubusercontent.com/79494088/146645742-ecc6dbc0-369a-4a0b-ada1-a67143df1970.png)

- 위와 같은 문구가 뜨기 때문에 스케쥴러를 실행시켜 준다.
- 스케쥴러가 없으면 DAG가 업데이터 되지 않고 새로운 테스크가 스케쥴되지 않는다.

```sh
$ airflow scheduler
```

- 실행시키면 `Warning`사항이 없어진다.

## db

```sh
$ airflow db -h
'''
positional arguments:
  COMMAND
    check           Check if the database can be reached
    check-migrations
                    Check if migration have finished
    init            Initialize the metadata database: 기본적인 데이터 파이프 라인이 생성된다.
    reset           Burn down and rebuild the metadata database: 초기화가 된다.
    shell           Runs a shell to access the database
    upgrade         Upgrade the metadata database to latest version
'''
```


## dags

```sh
$ airflow dags -h
"""
positional arguments:
  COMMAND
    backfill      Run subsections of a DAG for a specified date range: 망가졌을 때 고친 다음 데이터를 되돌려서 다시 실행시킨다.
    delete        Delete all DB records related to the specified DAG
    list          List all the DAGs: 현재 존재하는 DAGs
    list-jobs     List the jobs
    list-runs     List DAG runs given a DAG id
    next-execution
                  Get the next execution datetimes of a DAG
    pause         Pause a DAG
    report        Show DagBag loading report
    show          Displays DAG's tasks with their dependencies
    state         Get the status of a dag run
    test          Execute one single DagRun
    trigger       Trigger a DAG run
    unpause       Resume a paused DAG
"""

$ airflow dags list
'''
dag_id                    | filepath                 | owner   | paused
==========================+==========================+=========+=======
example_bash_operator     | /Users/6mini/opt/anacond | airflow | True  
                          | a3/envs/spark-flink/lib/ |         |       
                          | python3.8/site-packages/ |         |       
                          | airflow/example_dags/exa |         |       
                          | mple_bash_operator.py    |         |       
.
.
.
tutorial_taskflow_api_etl | /Users/6mini/opt/anacond | airflow | True  
                          | a3/envs/spark-flink/lib/ |         |       
                          | python3.8/site-packages/ |         |       
                          | airflow/example_dags/tut |         |       
                          | orial_taskflow_api_etl.p |         |       
                          | y                        |         |    
'''
```

- 여기서 나오는 리스트는 UI에 전시되는 것과 동일하다.
- `example_xcom`이라는 DAG를 파헤쳐본다.

## tasks

```sh
$ airflow tasks -h
"""
positional arguments:
  COMMAND
    clear             Clear a set of task instance, as if they never ran
    failed-deps       Returns the unmet dependencies for a task instance
    list              List the tasks within a DAG: DAG 안에 존재하는 테스크를 리스팅할 수 있다.
    render            Render a task instance's template(s)
    run               Run a single task instance
    state             Get the status of a task instance
    states-for-dag-run
                      Get the status of all task instances in a dag run
    test              Test a task instance
"""

$ airflow tasks list example_xcom 
'''
bash_pull
bash_push
puller
push
push_by_returning
python_pull_from_bash
'''
```

- `example_xcom`이라는 DAG 안의 테스크들이 리스팅(Listing)된다.
- DAG를 트리거(trigger)하는 방법을 알아볼 것이다.

{% include ad.html %}

```sh
$ airflow dags trigger -h 
"""
usage: airflow dags trigger [-h] [-c CONF] [-e EXEC_DATE] [-r RUN_ID]
                            [-S SUBDIR]
                            dag_id

Trigger a DAG run

positional arguments:
  dag_id                The id of the dag

optional arguments:
  -h, --help            show this help message and exit
  -c CONF, --conf CONF  JSON string that gets pickled into the DagRun's conf attribute
  -e EXEC_DATE, --exec-date EXEC_DATE: DAG가 실행되는 날짜 명시
                        The execution date of the DAG
  -r RUN_ID, --run-id RUN_ID
                        Helps to identify this run
  -S SUBDIR, --subdir SUBDIR
                        File location or directory from which to look for the dag. Defaults to '[AIRFLOW_HOME]/dags' where [AIRFLOW_HOME] is the value you set for 'AIRFLOW_HOME' config you set in 'airflow.cfg' 
"""
```

- 2022년 1월 1일에 실행되게끔 트리거를 설정해본다.

```sh
$ airflow dags trigger -e 2022-01-01 example_xcom 
```

![image](https://user-images.githubusercontent.com/79494088/146649072-e45b88c4-c65a-49b0-96a5-2e314e633fd8.png)

- UI 상에서 `queued`가 설정되었음을 확인할 수 있다.

![image](https://user-images.githubusercontent.com/79494088/146649086-32984311-7751-463c-a93f-24ba599209c4.png)

- UI에서도 관리할 수 있다.
- 이 외의 많은 커맨드가 있지만 간단하게 필요한 부분만 알아보았다.

# 에어플로우 UI
- UI에 대한 사용법과 DAG를 컨트롤하는 방법을 알아볼 것이다.

![image](https://user-images.githubusercontent.com/79494088/146649232-1e6cf8e5-cd66-47d1-b7c7-96b92a3c8209.png)

- DAGs라는 테이블은 현재 에어플로우가 인식하는 데이터 베이스 상에 존재하는 DAG들을 리스팅 해놓은 곳이다.
- 컬럼들을 하나하나 살펴본다.

![image](https://user-images.githubusercontent.com/79494088/146649271-44512e28-4532-473e-8066-9b88203a2921.png)

- 맨 왼쪽의 버튼은 DAG를 켜고 끌 수 있는 스위치이다.

![image](https://user-images.githubusercontent.com/79494088/146649300-ce093840-ec59-40c2-aa12-9bbbf09b1136.png)

- DAG: DAG명과 태그들을 갖게 된다.
- Owner: DAG를 만든 사람을 나타내는 컬럼이다.
- Runs: 실행중인 DAG의 상태를 나타낸다.
    - Queued
    - Success
    - Running
    - Failed
- Schedule: 대부분의 데이터 파이프 라인이 스케쥴을 갖고 있기 때문에 주기를 나타내는 설정을 명시한다.
    - [Crontab](https://crontab.guru/)
    - `@daily`나 `None`
- Last Run: 마지막으로 실행된 DAG의 상태이며, 링크가 뜨고 마지막에 돌렸던 DAG의 로그를 보고 모니터링을 할 수 있다.
- Next Run: 다음 실행이 언제 스케쥴링 되었는지 확인할 수 있다.
- Recent Tasks: 방금 전에 실행 된 테스크의 상태를 확인할 수 있다.
- Actions: 행동할 수 있는 버튼이다.

![image](https://user-images.githubusercontent.com/79494088/146649572-28994def-024e-465a-b509-170e472272e5.png)

- Link: 마우스를 호버하면 여러가지 링크가 전시된다.

![image](https://user-images.githubusercontent.com/79494088/146649644-65107b4f-21c8-4407-8d23-c6281cd7ac89.png)

- DAG를 클릭해보면 여러가지 뷰(View)를 볼 수 있다.

![image](https://user-images.githubusercontent.com/79494088/146649696-70df9a05-a917-4891-ad09-ee8930806c45.png)

- 그래프 뷰(Graph View)는 테스크 간의 의존성을 확인할 때 유용하다.

![image](https://user-images.githubusercontent.com/79494088/146649793-be93766b-e6f7-4efd-9ef1-353c2758e7be.png)

- 간트 뷰(Gantt View)에서는 각 테스크가 얼마의 시간을 소비하였는지 확인할 수 있다.
    - 병렬 처리를 확인할 수 있다.
    - 각 작업의 시간을 볼 수 있기 때문에 최적화가 용이하다.

![image](https://user-images.githubusercontent.com/79494088/146649887-4af0565f-8acf-4b66-a2ad-ca9648a06bb7.png)

- 그래프 뷰에서 테스크를 클릭하면 모달(modal)이 뜨게 되고 여기에서 테스크를 컨트롤할 수 있다.
- 강제로 실패와 성공을 마크할 수 있고, 로그를 클릭하여 테스크가 뱉어내는 로그를 확인할 수 있다.

# 에어플로우와 스파크(Spark)
- 에어플로우에 스파크 프로바이더(provider)를 설치해야 한다.

```
$ pip install apache-airflow-providers-apache-spark
```

- DAG를 생성한다.

```py
from datetime import datetime

from airflow import DAG
from airflow.providers.apache.spark.operators.spark_sql import SparkSqlOperator

default_args = {
  'start_date': datetime(2021, 1, 1),
}

with DAG(dag_id='spark-example',
         schedule_interval='@daily',
         default_args=default_args,
         tags=['spark'],
         catchup=False) as dag:

    sql_job = SparkSqlOperator(sql="SELECT * FROM bar", master="local", task_id="sql_job") # 스파크 SQL을 쓰는 것 처럼 쓸 수 있다.
```

- 하지만 위처럼 사용하면 헤비하니 지양하는 편이 좋다.
- 스파크 서브밋(submit)을 실행해주는 편이 좋다.
- 서브밋만 하고 에어플로우는 지켜보며 실행되는 지만 모니터링하는 것이 좋다.

```py
from datetime import datetime

from airflow import DAG
# from airflow.providers.apache.spark.operators.spark_sql import SparkSqlOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

default_args = {
  'start_date': datetime(2021, 1, 1),
}

with DAG(dag_id='spark-example',
         schedule_interval='@daily',
         default_args=default_args,
         tags=['spark'],
         catchup=False) as dag:
  
#   sql_job = SparkSqlOperator(sql="SELECT * FROM bar", master="local", task_id="sql_job") # 스파크 SQL을 쓰는 것 처럼 쓸 수 있다.

  submit_job = SparkSubmitOperator(
      application="/Users/6mini/spark/count_trips_sql.py", task_id="submit_job", conn_id="spark_local"
  )
```