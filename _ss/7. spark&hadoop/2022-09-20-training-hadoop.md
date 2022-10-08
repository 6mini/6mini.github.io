---
title: "[빅데이터 처리 입문] 하둡(Hadoop) 설치 및 HDFS 실습"
description: 
categories:
 - Spark & Hadoop
tags: []
mathjax: enable
---

# 하둡(Hadoop) 설치 및 실습

## 자바 환경 변수 설정
- 먼저 맥용 패키지 관리 도구인 홈브류(Homebrew)와 자바(Java)의 설치가 필요하다.

```s
$ java -version
'''
java version "1.8.0_25"
Java(TM) SE Runtime Environment (build 1.8.0_25-b17)
Java HotSpot(TM) 64-Bit Server VM (build 25.25-b02, mixed mode)
'''
```

- 환경 변수를 설정한다.

```s
$ vim ~/.zshrc

export JAVA_HOME=$(/usr/libexec/java_home)

$ source ~/.zshrc

$ echo $JAVA_HOME
'''
/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home
'''
```

- 자바 설치는 완료되었다.
- 하둡을 실행할 때 `ssh` 명령어를 이용하여 로컬호스트에 접속할 수 있는 환경을 구성해야한다.

## `ssh` 로그인 설정

```s
$ ssh localhost 
'''
ssh: connect to host localhost port 22: Connection refused
'''
```

- 현재는 원격 로그인이 설정되어 있지 않아 위와 같은 에러 메세지가 전시된다.

![image](https://user-images.githubusercontent.com/79494088/190044355-4ececeb5-4f74-4f0b-aa02-c0c45c190f1e.png)

- 시스템 환경설정 -> 공유 -> 원격 로그인을 체크한다.

```s
$ ssh localhost
'''
(6mini@localhost) Password:
'''
```

- 에러 메세지는 사라지고 패스워드를 입력하는 메세지가 전시된다.
- 패스워드를 입력하지 않고 바로 접속할 수 있는 환경을 구성한다.

```s
$ ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
"""
Generating public/private rsa key pair.
Your identification has been saved in /Users/6mini/.ssh/id_rsa
Your public key has been saved in /Users/6mini/.ssh/id_rsa.pub
The key fingerprint is:
SHA256:+2MzbkwRRf5gGRRVzAqdAdbxLAtnfkVjR0nao+ckkeM 6mini@6miniui-MacBookPro.local
The key's randomart image is:
+---[RSA 3072]----+
|           +O*=%*|
|          .o.oO==|
|           .**+++|
|          ...B=oo|
|        S  . E+o.|
|         ..   =. |
|        .o     . |
|         .B      |
|         +o+     |
+----[SHA256]-----+
"""

$ cd ~/.ssh

.ssh$ ls
'''
id_rsa          id_rsa.pub      known_hosts     known_hosts.old
'''

.ssh$ cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys 

.ssh$ ssh localhost
'''
Last login: Wed Sep 14 11:28:55 2022
'''
```

- 위 과정을 통해 패스워드 없이 `pub key`로 로컬호스트에 접속할 수 있는 상태를 만들었다.
- 다음으로는 하둡 패키지를 다운로드한다.

## 하둡 다운로드

```s
$ wget https://dlcdn.apache.org/hadoop/common/hadoop-3.3.2/hadoop-3.3.2.tar.gz

$ tar zxvf hadoop-3.3.2.tar.gz
```

- 환경 변수를 설정한다.

```s
$ vim ~/.zshrc

export HADOOP_HOME=/Users/6mini/hadoop/hadoop-3.3.2

export PATH=$PATH:$HADOOP_HOME/bin 

$ source ~/.zshrc

$ hadoop
'''
Usage: hadoop [OPTIONS] SUBCOMMAND [SUBCOMMAND OPTIONS]
 or    hadoop [OPTIONS] CLASSNAME [CLASSNAME OPTIONS]
  where CLASSNAME is a user-provided Java class

  OPTIONS is none or any of:

--config dir                     Hadoop config directory
--debug                          turn on shell script debug mode
--help                           usage information
buildpaths                       attempt to add class files from build tree
hostnames list[,of,host,names]   hosts to use in slave mode
hosts filename                   list of hosts to use in slave mode
loglevel level                   set the log4j level for this command
workers                          turn on worker mode

  SUBCOMMAND is one of:


    Admin Commands:

daemonlog     get/set the log level for each daemon

    Client Commands:

archive       create a Hadoop archive
checknative   check native Hadoop and compression libraries availability
classpath     prints the class path needed to get the Hadoop jar and the
              required libraries
conftest      validate configuration XML files
credential    interact with credential providers
distch        distributed metadata changer
distcp        copy file or directories recursively
dtutil        operations related to delegation tokens
envvars       display computed Hadoop environment variables
fs            run a generic filesystem user client
gridmix       submit a mix of synthetic job, modeling a profiled from
              production load
jar <jar>     run a jar file. NOTE: please use "yarn jar" to launch YARN
              applications, not this command.
jnipath       prints the java.library.path
kdiag         Diagnose Kerberos Problems
kerbname      show auth_to_local principal conversion
key           manage keys via the KeyProvider
rumenfolder   scale a rumen input trace
rumentrace    convert logs into a rumen trace
s3guard       manage metadata on S3
trace         view and modify Hadoop tracing settings
version       print the version

    Daemon Commands:

kms           run KMS, the Key Management Server
registrydns   run the registry DNS server

SUBCOMMAND may print help when invoked w/o parameters or with -h.
'''
```

- 하둡 명령어를 사용할 수 있는 상태가 되었다.
- 하둡을 실행하기 전 환경설정 파일들을 설정해야한다.

## 설정 파일 수정

```s
$ cd etc/hadoop

$ ls
'''
capacity-scheduler.xml           kms-log4j.properties
configuration.xsl                kms-site.xml
container-executor.cfg           log4j.properties
core-site.xml                    mapred-env.cmd
hadoop-env.cmd                   mapred-env.sh
hadoop-env.sh                    mapred-queues.xml.template
hadoop-metrics2.properties       mapred-site.xml
hadoop-policy.xml                shellprofile.d
hadoop-user-functions.sh.example ssl-client.xml.example
hdfs-rbf-site.xml                ssl-server.xml.example
hdfs-site.xml                    user_ec_policies.xml.template
httpfs-env.sh                    workers
httpfs-log4j.properties          yarn-env.cmd
httpfs-site.xml                  yarn-env.sh
kms-acls.xml                     yarn-site.xml
kms-env.sh                       yarnservice-log4j.properties
'''
```

### `core-site.xml`

```xml
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://localhost:9000</value>
    </property>
</configuration>
```

### `hdfs-site.xml`

```xml
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>/Users/6mini/hadoop/hadoop-3.3.2/dfs/name</value>
    </property>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>/Users/6mini/hadoop/hadoop-3.3.2/dfs/data</value>
    </property>
</configuration>
```

- 노드 위치를 설정해주었는데 디렉토리가 없기 때문에 생성해준다.

```s
hadoop-3.3.2$ mkdir -p dfs/name
hadoop-3.3.2$ mkdir -p dfs/data
```

### `mapred-site.xml`

```xml
<configuration>
    <property>
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
    </property>
    <property>
        <name>mapreduce.application.classpath</name>
        <value>$HADOOP_MAPRED_HOME/share/hadoop/mapreduce/*:$HADOOP_MAPRED_HOME/share/hadoop/mapreduce/lib/*</value>
    </property>
</configuration>
```

### `yarn-site.xml`

```xml
<configuration>
    <property>
        <name>yarn.nodemanager.aux-services</name>
        <value>mapreduce_shuffle</value>
    </property>
    <property>
        <name>yarn.nodemanager.env-whitelist</name>
        <value>JAVA_HOME,HADOOP_COMMON_HOME,HADOOP_HDFS_HOME,HADOOP_CONF_DIR,CLASSPATH_PREPEND_DISTCACHE,HADOOP_YARN_HOME,HADOOP_HOME,PATH,LANG,TZ,HADOOP_MAPRED_HOME</value>
    </property>
</configuration>
```

## HDFS 실행

- 네임 노드를 포맷시킨다.

```s
$ hdfs namenode -format
```

- hdfs를 시작한다.

```s
$ sbin/start-dfs.sh
```

- hdfs web ui에 접속한다.
    - http://localhost:9870
- yarn을 실행한다.

```s
$ sbin/start-yarn.sh
```

- resource manage web ui에 접속한다.
    - http://localhost:8088/
- 예제로 아래의 코드를 실행한다.

```s
$ hadoop jar $HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-examples-3.3.2.jar pi 16 10000
```

# HDFS 실습

## HDFS CLI
- 보통 HDFS의 명령어 문서를 통해 확인할 수 있다.
    - [HDFS 명령어 문서](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/FileSystemShell.html)

### `mkdir`
- 디렉토리를 생성한다.

```s
hadoop fs -mkdir [-p] <paths>
$ hadoop fs -mkdir /user
$ hadoop fs -mkdir /user/fastcampus
$ hadoop fs -mkdir /user/fastcampus/input
```

### `ls`
- 디렉토리 파일 목록을 확인한다.

```s
$ hadoop fs -ls /
$ hadoop fs -ls /user
$ hadoop fs -ls -R /
```

### `put`, `copyFromLocal`
- 로컬 파일 시스템에 있는 데이터를 목적지 파일 시스템으로 복사한다.

```s
$ hadoop fs -put /path/to/hadoop/LICENSE.txt /user/fastcampus/input
$ hadoop fs -copyFromLocal /path/to/hadoop/LICENSE.txt /user/fastcampus/input
```

### `get`, `copyToLocal`
- HDFS에 있는 파일을 로컬 파일 시스템에 복사한다.

```s
$ hadoop fs -get /user/fastcampus/input/LICENSE.txt .
$ hadoop fs -copyToLocal /user/fastcampus/input/LICENSE.txt .
```

### `cat`
- 파일 내용을 출력한다.

```s
$ hadoop fs -cat /user/fastcampus/input/LICENSE.txt
```

### `mv`
- 파일을 옮긴다.

```s
$ hadoop fs -mv /user/fastcampus/input/LICENSE.txt /user/fastcampus
```

### `cp`
- 파일을 복사한다.

```s
$ hadoop fs -cp /user/fastcampus/LICENSE.txt /user/fastcampus/input/
```

### `rm`
- 파일을 삭제한다.

```s
$ hadoop fs -rm /user/fastcampus/LICENSE.txt
```

### `tail`
- 파일의 끝부분을 보여준다.

```s
$ hadoop fs -tail /user/fastcampus/input/LICENSE.txt
```

### help
- 명령어 도움말을 출력한다.

```s
# hadoop fs -help [command]
$ hadoop fs -help cp
```