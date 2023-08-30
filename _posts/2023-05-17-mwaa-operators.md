---
title: '[AWS] Amazon MWAA를 이용한 EC2 인스턴스 스케쥴링과 원격 Python 스크립트 실행(SSHOperators, AWSOperators 사용법)'
description: "Amazon Managed Workflow for Apache Airflow(MWAA)를 활용하여 EC2 인스턴스에서 Python 스크립트를 정기적으로 실행하는 방법을 설명한다. AWS EC2 인스턴스를 시작하고 종료하는 방법, IP 주소를 동적으로 가져오는 방법, 그리고 SSH를 이용한 원격 명령 실행에 대해 자세히 알아본다."
categories:
 - AWS
tags: [MWAA, AWS, EC2, SSHOperators, AWSOperators]
---

# 문제상황

Amazon Managed Workflowfor Apache Airflow(이하 "MWAA")로 데이터 수집 절차를 스케쥴링 하고자 한다. BeautifulSoup(이하 "bs4")로 웹 데이터를 수집하여 저장하는 프로그램의 실행 시간이 15분을 초과한다. 앞 포스팅에서 소개한 람다(Lambda) 함수로 구성하여 스케쥴링할 수 없기 때문에 AWS EC2인스턴스에 프로그램을 위치시켜 에어플로우의 `SSHOperators`를 통하여 파이썬 프로그램을 실행시키고자 한다.

EC2의 요금을 절감하기 위해 플로우 계획은 "프로그램이 위치한 EC2 인스턴스 실행 >> 파이썬 프로그램 실행 >> 인스턴스 종료"를 DAG 절차로 구현하고자 했고, MWAA에서 이를 구현하는 것이 처음이라 버거웠다.

일단 첫번째 문제는 EC2는 고정된 IP가 아니라 껐다 켤 때 마다 IP가 변경된다는 점이고, 고정 IP를 부여한다면, 또 추가 요금을 내야했다. 그래서 인스턴스를 켠 뒤 IP를 가져와서 SSH 연결하는 과정이 필요했다.

다음 문제는 `SSHOperators`, `AWSOperators`를 사용하는 일이었다. 그럼 이제부터 천천히 알아보자.

# 두 `Operators` 환경 설정

## `SSHOperators` 사용하기

### 필요 라이브러리 설치

#### 요구 사항 파일 수정

MWAA에서 요구 사항 파일인 `requirements.txt`을 아래와 같이 수정한다.

```
apache-airflow-providers-ssh
```

#### S3로 전송 및 설치

```sh
$ aws s3 cp requirements.txt s3://your-bucket/
```

전송만 한다고 바로 라이브러리가 설치되지 않는다. Amazon MWAA > 환경 > YourAirflowEnvironment 에서 "편집"을 클릭 후 방금 보낸 요구 사항 파일의 S3 링크와 버전을 선택한 뒤 저장해야 한다.

### 비밀 키 S3 dags 폴더에 복사

해당 인스턴스 SSH 연결에 사용되는 `.pem`형태의 비밀 키를 dags에 전송해야 에어플로우가 해당 인스턴스에 접근할 수 있다.

```sh
$ aws s3 cp your-secret-key.pem s3://your-bucket/dags/
```

에어플로우의 dag가 저장되는 `s3://your-bucket/dags/`에 전송하면 에어플로우는 이를 `/usr/local/airflow/dags/`로 인식하여 액세스한다.

### 에어플로우 새 연결 생성

1. 에어플로우 UI에 접속한다.
2. 상단 탐색의 "Admin" > "Connections"을 선택한다.
3. "+" 버튼을 클릭한다.
4. "연결 ID"를 입력한다.
    - 이 포스팅에서는 `ec2_ssh`를 사용할 것이다.
5. "연결 유형(Connection Type)"에 "SSH"를 확인한다.
    - 만약 드롭 다운 리스트에 "SSH"가 없다면, 제대로 설치 된 것이 아니니, 제대로 다시 시도한다.
6. "Host"에는 원래 IP를 입력하는데, 지금 필자의 상황에선 인스턴스가 중지되어있는 상황이기에 IP가 할당되지 않은 상태이다. 이 부분은 나중에 DAG에서 인스턴스를 시작하고, IP를 파싱하여 이 Connection을 업데이트해줄 것이다.
7. "Username"에 인스턴스에 따라 다른 사용자 이름을 입력한다.
    - ex) ec2-user, ubuntu (시작하려는 ec2 인스턴스 페이지에서 확인할 수 있다)
8. "Extra"에 아까 복사한 비밀 키 경로를 입력해준다.
    - `{ "key_file": "/usr/local/airflow/dags/your-secret-key.pem" }`
    - `your-secret-key.pem` 부분만 변경하면 된다. MWAA가 사용하는 dags 폴더에 넣었다면, 저 경로로 인식할 것이다.

## `AWSOperators` 사용하기

### 필요 라이브러리 설치

#### 요구 사항 파일 수정

`requirements.txt`를 수정하는데 기존에 다른 것이 있다면 줄바꿈으로 추가하면 된다.

```
apache-airflow-providers-ssh
apache-airflow-providers-amazon
```

#### S3로 전송 및 설치

```sh
$ aws s3 cp requirements.txt s3://your-bucket/
```

아까와 같이 업데이트 했으면 편집을 통해 버전을 업그레이드해준다.

### IAM 설정

MWAA가 AWS 서비스에 접근할 수 있도록 IAM 역할(Role)에서 권한을 추가해주어야한다.

1. Amazon MWAA > 환경 > YourAirflowEnvironment에서 "권한"의 "실행 역할"로 이동한다.
2. "권한 추가"를 누른 뒤, "인라인 정책 생성"을 진행한다.
3. "JSON"을 클릭 후, 아래의 json을 붙여 넣는다.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ec2:StartInstances",
                "ec2:StopInstances",
                "ec2:DescribeInstanceStatus",
                "ec2:DescribeInstances"
            ],
            "Resource": "*"
        }
    ]
}
```

4. "정책 검토"를 클릭 후, 정책 이름을 설정하고 "정책 생성"을 진행한다.
    - `EC2InstanceStartStopPolicy`

이제 사용하기 위한 준비는 모두 끝났다.

# DAG

대략 `start_ec2 >> wait_for_running >> ssh_execute_command >> stop_ec2 >> wait_for_stopped`과 같은 dag를 구성할 것이다. 인스턴스를 실행하고, 실행을 확인한다. 그 뒤 IP를 가져와 Connection을 업데이트 해준 뒤, SSH로 접속하여 프로그램을 실행한다. 실행이 끝나면 인스턴스를 종료하고, 종료가 확인되면 dag는 끝난다.

## 라이브러리

필요한 라이브러리를 불러온다. 참고로 `boto3`도 `requirements.txt`를 통해 설치해줘야한다.

```py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.amazon.aws.operators.ec2 import EC2StartInstanceOperator, EC2StopInstanceOperator
from airflow.providers.amazon.aws.sensors.ec2 import EC2InstanceStateSensor
from airflow.providers.ssh.operators.ssh import SSHOperator
import boto3
```

## 인스턴스의 퍼블릭 IP 가져오기

아까 언급했듯, 중지된 인스턴스였기 때문에 IP가 할당되지 않았던 상태이다. 인스턴스를 실행 후 인스턴스 ID 통해 할당된 IP를 가져오는 함수를 생성한다.

```py
def get_instance_ip(instance_id):
    ec2 = boto3.resource('ec2')
    instance = ec2.Instance(instance_id)
    return instance.public_ip_address
```


## `SSH Connection`의 IP 업데이트

아까 생성한 `SSH Connection`에 Host를 비워두었다. 가져온 인스턴스 IP를 `Conn`의 `Host`에 업데이트해주는 함수를 생성한다.

```py
def update_ssh_connection(instance_ip):
    from airflow.models import Connection
    from airflow.settings import Session

    session = Session()
    existing_connection = session.query(Connection).filter(Connection.conn_id == "ec2_ssh").first()

    if existing_connection:
        existing_connection.host = instance_ip
        session.add(existing_connection)
    else:
        new_connection = Connection(
            conn_id="ec2_ssh",
            conn_type="SSH",
            host=instance_ip,
            login="ubuntu",
            private_key_file="/usr/local/airflow/dags/your-secret-key.pem",
        )
        session.add(new_connection)

    session.commit()
    session.close()
```

위 두 함수의 실행을 `PythonOperators`로 구성할 DAG를 위해 함수를 생성한다.

```py
def update_ssh_connection_with_instance_id(instance_id):
    instance_ip = get_instance_ip(instance_id)
    update_ssh_connection(instance_ip)
```

## DAG 기본 구성

크롤러를 실행할 인스턴스 ID와 실행 커맨드를 작성하고 기본적으로 구성한다.

```py
INSTANCE_ID = 'your-instance-id'
REMOTE_COMMAND = 'python program-name.py'

default_args = {
    'owner': 'your-name',
    'depends_on_past': False,
    'start_date': datetime(2021, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'email': ['your@email.com'],
    'retries': 0,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'your-dag',
    default_args=default_args,
    description='your-dascription',
    schedule_interval='0 0 * * *',
    catchup=False,
    tags=[]
)
```

## DAG 구성

최종적으로 DAG를 구성한다.

```py
start_ec2 = EC2StartInstanceOperator(
    task_id='start_ec2',
    instance_id=INSTANCE_ID,
    dag=dag,
)

wait_for_running = EC2InstanceStateSensor(
    task_id='wait_for_running',
    instance_id=INSTANCE_ID,
    target_state='running',
    dag=dag,
)

get_ip_and_update_ssh = PythonOperator(
    task_id='get_ip_and_update_ssh',
    python_callable=update_ssh_connection_with_instance_id,
    op_args=[INSTANCE_ID],
    dag=dag,
)

ssh_execute_command = SSHOperator(
    task_id='ssh_execute_command',
    ssh_conn_id='ec2_ssh',
    command=REMOTE_COMMAND,
    conn_timeout=3600,
    cmd_timeout=3600,
    dag=dag,
)

stop_ec2 = EC2StopInstanceOperator(
    task_id='stop_ec2',
    instance_id=INSTANCE_ID,
    dag=dag,
)

wait_for_stopped = EC2InstanceStateSensor(
    task_id='wait_for_stopped',
    instance_id=INSTANCE_ID,
    target_state='stopped',
    dag=dag,
)

start_ec2 >> wait_for_running >> get_ip_and_update_ssh >> ssh_execute_command >> stop_ec2 >> wait_for_stopped
```

1. `start_ec2`: `EC2StartInstanceOperator`를 사용하여 지정된 `INSTANCE_ID`의 EC2 인스턴스를 시작한다.
2. `wait_for_running`: `EC2InstanceStateSensor를` 사용하여 인스턴스가 'running' 상태가 될 때까지 기다린다. 이 작업은 인스턴스가 완전히 시작되고 실행 가능한 상태가 될 때까지 다음 작업으로 진행하지 않는다.
3. `get_ip_and_update_ssh`: `PythonOperator`를 사용하여 u`pdate_ssh_connection_with_instance_id` 함수를 호출한다. 이 함수는 `get_instance_ip` 함수를 사용하여 인스턴스의 현재 public IP 주소를 가져온 다음, `update_ssh_connection` 함수를 사용하여 Airflow의 SSH 연결을 업데이트한다.
4. `ssh_execute_command`: `SSHOperator`를 사용하여 업데이트된 SSH 연결을 통해 원격 스크립트를 실행한다. 여기서는 `REMOTE_COMMAND` 변수에 지정된 명령어를 실행한다. 이 작업은 원격 인스턴스에서 지정된 명령어를 완료할 때까지 기다린다.
    - 참고로 `SSHOperator`의 경우 연결 대기 시간이 기본적으로 10초 밖에 되지 않아서, `conn_timeout`, `cmd_timeout`에 한시간(3600초)을 할당했다. 만약 10초 이상 돌아가는 프로그램이라면 `SSH command timed out` 에러를 뱉을 것이다.
5. `stop_ec2`: `EC2StopInstanceOperator`를 사용하여 원격 스크립트 실행이 완료된 후 EC2 인스턴스를 중지한다.
6. `wait_for_stopped`: `EC2InstanceStateSensor`를 사용하여 인스턴스가 'stopped' 상태가 될 때까지 기다린다. 이 작업은 인스턴스가 완전히 정지되어 비용을 절약할 수 있는 상태가 될 때까지 다음 작업으로 진행하지 않는다.
7. 이 작업을 차례대로 실행한다.

# 최종 코드

```py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.amazon.aws.operators.ec2 import EC2StartInstanceOperator, EC2StopInstanceOperator
from airflow.providers.amazon.aws.sensors.ec2 import EC2InstanceStateSensor
from airflow.providers.ssh.operators.ssh import SSHOperator
import boto3

def get_instance_ip(instance_id):
    ec2 = boto3.resource('ec2')
    instance = ec2.Instance(instance_id)
    return instance.public_ip_address

def update_ssh_connection(instance_ip):
    from airflow.models import Connection
    from airflow.settings import Session

    session = Session()
    existing_connection = session.query(Connection).filter(Connection.conn_id == "ec2_ssh").first()

    if existing_connection:
        existing_connection.host = instance_ip
        session.add(existing_connection)
    else:
        new_connection = Connection(
            conn_id="ec2_ssh",
            conn_type="SSH",
            host=instance_ip,
            login="ubuntu",
            private_key_file="/usr/local/airflow/dags/your-secret-key.pem",
        )
        session.add(new_connection)

    session.commit()
    session.close()

def update_ssh_connection_with_instance_id(instance_id):
    instance_ip = get_instance_ip(instance_id)
    update_ssh_connection(instance_ip)

INSTANCE_ID = 'your-instance-id'
REMOTE_COMMAND = 'python program-name.py'

default_args = {
    'owner': 'your-name',
    'depends_on_past': False,
    'start_date': datetime(2021, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'email': ['your@email.com'],
    'retries': 0,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'your-dag',
    default_args=default_args,
    description='your-dascription',
    schedule_interval='0 0 * * *',
    catchup=False,
    tags=[]
)

start_ec2 = EC2StartInstanceOperator(
    task_id='start_ec2',
    instance_id=INSTANCE_ID,
    dag=dag,
)

wait_for_running = EC2InstanceStateSensor(
    task_id='wait_for_running',
    instance_id=INSTANCE_ID,
    target_state='running',
    dag=dag,
)

get_ip_and_update_ssh = PythonOperator(
    task_id='get_ip_and_update_ssh',
    python_callable=update_ssh_connection_with_instance_id,
    op_args=[INSTANCE_ID],
    dag=dag,
)

ssh_execute_command = SSHOperator(
    task_id='ssh_execute_command',
    ssh_conn_id='ec2_ssh',
    command=REMOTE_COMMAND,
    conn_timeout=3600,
    cmd_timeout=3600,
    dag=dag,
)

stop_ec2 = EC2StopInstanceOperator(
    task_id='stop_ec2',
    instance_id=INSTANCE_ID,
    dag=dag,
)

wait_for_stopped = EC2InstanceStateSensor(
    task_id='wait_for_stopped',
    instance_id=INSTANCE_ID,
    target_state='stopped',
    dag=dag,
)

start_ec2 >> wait_for_running >> get_ip_and_update_ssh >> ssh_execute_command >> stop_ec2 >> wait_for_stopped
```

이로써 모든 작업이 끝났다. 필자는 굉장히 온갖 에러를 조우하며 굉장히 오랜 시간이 걸렸는데, 나와 같은 전략으로 DAG를 구성하는 이가 이 포스팅을 발견하여 더욱 빠르고 간단하게 구성하길 바라는 마음으로 포스팅을 마친다.