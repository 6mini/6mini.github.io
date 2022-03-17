---
title: '[카프카] 주키퍼(Zookeeper) 콘솔에서 브로커(Broker) 시작 및 토픽(Topic) 생성'
description: 카프카 클러스터의 여러 요소들의 설정을 정하는데 사용되는 주키퍼 콘솔에서 데이터 스트림이 어디에 퍼블리시될 지 정하는 데 쓰이는 (브로커로부터 서빙되는) 토픽 생성 방법
categories:
 - Data Engineering
tags: [카프카, 데이터 엔지니어링, 주키퍼, 브로커, 토픽]
---

- 본격적으로 카프카(Kafka)의 CLI에 대해 알아볼 것이다.
- 먼저 카프카를 실행해야 하고, 그 전에 주키퍼(Zookeeper)를 실행해야한다.
- 주키퍼 서버를 실행해볼 것이다.

# 주키퍼(Zookeeper) 서버 실행

![image](https://user-images.githubusercontent.com/79494088/156485871-263dbcd2-1607-4e2f-b29b-e64d2f3bb15d.png){: width="80%"}

- 설치했던 카프카 폴더의 `bin` 폴더를 살펴보면, `kafka-server-start.sh`라는 파일이 존재하는데, 이것을 실행시켜주면 주키퍼가 실행된다.

```s
$ ./bin/zookeeper-server-start.sh
'''
USAGE: ./bin/zookeeper-server-start.sh [-daemon] zookeeper.properties
'''
```

- `config` 파일이 필요하다는 에러가 전시된다.

![image](https://user-images.githubusercontent.com/79494088/156486223-6d65f679-bb78-41c3-a11b-d862c567a430.png){: width="80%"}

- `config` 폴더를 살펴본다.
- 주키퍼를 위한 config 파일이 존재한다.

```s
$ vim ./config/zookeeper.properties
'''
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# the directory where the snapshot is stored.
dataDir=/tmp/zookeeper
# the port at which the clients will connect
clientPort=2181
# disable the per-ip limit on the number of connections since this is a non-production config
maxClientCnxns=0
# Disable the adminserver by default to avoid port conflicts.
# Set the port to something non-conflicting if choosing to enable this
admin.enableServer=false
# admin.serverPort=8080
'''
```

- `dataDir`: 주키퍼가 스냅션(Snaption)이나 로그(Log)를 저장하는 폴더를 지정한다.
- `clientPort`: 클라이언트(Client)가 어떤 포트(Port)를 사용하여 주키퍼와 연락을 할건 지 명시한다.
- `maxClientCnxns`: 0으로 설정되어있다면 이 프로퍼티(Property)를 사용하지 않는다는 뜻이다.
    - 로컬이나 개발 환경에서는 0으로 설정해도 되지만, 프로덕션(Production) 환경에서는 0 이상이 좋다.
    - 보안적인 이슈를 위해 지정해주는 것이 좋다.
- `admin.enableServer`: 어드민(Admin) 서버를 열 지 지정하는 프로퍼티이다.
- 커스터마이징(Customizing)은 진행하지 않고 디폴트(Default)로 진행한다.

```s
$ ./bin/zookeeper-server-start.sh -daemon config/zookeeper.properties
```

- 위와 같이 `-daemon`을 이용한 명령어로 프로퍼티와 연결하여 주키퍼 서버를 실행한다.

# 브로커(Broker) 시작

- 카프카 브로커를 시작하기에 앞서 주키퍼와 마찬가지로 프로퍼티를 먼저 확인한다.

```s
$ vim ./config/server.properties
'''
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# see kafka.server.KafkaConfig for additional details and defaults

############################# Server Basics #############################

# The id of the broker. This must be set to a unique integer for each broker.
broker.id=0

############################# Socket Server Settings #############################

# The address the socket server listens on. It will get the value returned from
# java.net.InetAddress.getCanonicalHostName() if not configured.
#   FORMAT:
#     listeners = listener_name://host_name:port
#   EXAMPLE:
#     listeners = PLAINTEXT://your.host.name:9092
#listeners=PLAINTEXT://:9092

# Hostname and port the broker will advertise to producers and consumers. If not set,
# it uses the value for "listeners" if configured.  Otherwise, it will use the value
# returned from java.net.InetAddress.getCanonicalHostName().
#advertised.listeners=PLAINTEXT://your.host.name:9092

# Maps listener names to security protocols, the default is for them to be the same. See the config documentation for more details
#listener.security.protocol.map=PLAINTEXT:PLAINTEXT,SSL:SSL,SASL_PLAINTEXT:SASL_PLAINTEXT,SASL_SSL:SASL_SSL

# The number of threads that the server uses for receiving requests from the network and sending responses to the network
num.network.threads=3

# The number of threads that the server uses for processing requests, which may include disk I/O
num.io.threads=8

# The send buffer (SO_SNDBUF) used by the socket server
socket.send.buffer.bytes=102400

# The receive buffer (SO_RCVBUF) used by the socket server
socket.receive.buffer.bytes=102400

# The maximum size of a request that the socket server will accept (protection against OOM)
socket.request.max.bytes=104857600


############################# Log Basics #############################

# A comma separated list of directories under which to store log files
log.dirs=/tmp/kafka-logs

# The default number of log partitions per topic. More partitions allow greater
# parallelism for consumption, but this will also result in more files across
# the brokers.
num.partitions=1

# The number of threads per data directory to be used for log recovery at startup and flushing at shutdown.
# This value is recommended to be increased for installations with data dirs located in RAID array.
num.recovery.threads.per.data.dir=1

############################# Internal Topic Settings  #############################
# The replication factor for the group metadata internal topics "__consumer_offsets" and "__transaction_state"
# For anything other than development testing, a value greater than 1 is recommended to ensure availability such as 3.
offsets.topic.replication.factor=1
transaction.state.log.replication.factor=1
transaction.state.log.min.isr=1

############################# Log Flush Policy #############################

# Messages are immediately written to the filesystem but by default we only fsync() to sync
# the OS cache lazily. The following configurations control the flush of data to disk.
# There are a few important trade-offs here:
#    1. Durability: Unflushed data may be lost if you are not using replication.
#    2. Latency: Very large flush intervals may lead to latency spikes when the flush does occur as there will be a lot of data to flush.
#    3. Throughput: The flush is generally the most expensive operation, and a small flush interval may lead to excessive seeks.
# The settings below allow one to configure the flush policy to flush data after a period of time or
# every N messages (or both). This can be done globally and overridden on a per-topic basis.

# The number of messages to accept before forcing a flush of data to disk
#log.flush.interval.messages=10000

# The maximum amount of time a message can sit in a log before we force a flush
#log.flush.interval.ms=1000

############################# Log Retention Policy #############################

# The following configurations control the disposal of log segments. The policy can
# be set to delete segments after a period of time, or after a given size has accumulated.
# A segment will be deleted whenever *either* of these criteria are met. Deletion always happens
# from the end of the log.

# The minimum age of a log file to be eligible for deletion due to age
log.retention.hours=168

# A size-based retention policy for logs. Segments are pruned from the log unless the remaining
# segments drop below log.retention.bytes. Functions independently of log.retention.hours.
#log.retention.bytes=1073741824

# The maximum size of a log segment file. When this size is reached a new log segment will be created.
log.segment.bytes=1073741824

# The interval at which log segments are checked to see if they can be deleted according
# to the retention policies
log.retention.check.interval.ms=300000

############################# Zookeeper #############################

# Zookeeper connection string (see zookeeper docs for details).
# This is a comma separated host:port pairs, each corresponding to a zk
# server. e.g. "127.0.0.1:3000,127.0.0.1:3001,127.0.0.1:3002".
# You can also append an optional chroot string to the urls to specify the
# root directory for all kafka znodes.
zookeeper.connect=localhost:2181

# Timeout in ms for connecting to zookeeper
zookeeper.connection.timeout.ms=18000


############################# Group Coordinator Settings #############################

# The following configuration specifies the time, in milliseconds, that the GroupCoordinator will delay the initial consumer rebalance.
# The rebalance will be further delayed by the value of group.initial.rebalance.delay.ms as new members join the group, up to a maximum of max.poll.interval.ms.
# The default value for this is 3 seconds.
# We override this to 0 here as it makes for a better out-of-the-box experience for development and testing.
# However, in production environments the default value of 3 seconds is more suitable as this will help to avoid unnecessary, and potentially expensive, rebalances during application startup.
group.initial.rebalance.delay.ms=0
'''
```

- `broker.id`: 브로커가 클러스터(Cluster) 안에서 여러 개가 쓰인다면 아이디를 유니크(Unique)하게 설정해야 한다.
    - 똑같은 아이디를 가진 브로커는 존재하지 않는다.
    - 브로커를 하나만 사용할 것이기 때문에 0으로 지정한다.
- `num.network.threads`: 네트워크(Network)를 주고 받을 때 스레드(Thread)의 갯수이다.
- `num.io.threads`: 리퀘스트(Request)가 왔을 때 처리하는 데 사용하는 스레드의 갯수이다.
    - 디스크에 읽고 쓰는데 사용하기 때문에 네트워크 스레드보다 많은 편이 좋다.
- 그 외 정보들은 사용 중 필요 시 확인한다.

```s
$ ./bin/kafka-server-start.sh -daemon config/server.properties
```

- 위 명령어로 프로퍼티를 연결하여 카프카 서버를 실행한다.
- 아래와 같이 카프카 서버가 구동되고 있는지 확인할 수 있다.

```s
$ netstat -an | grep 2181
'''
tcp4       0      0  127.0.0.1.2181         127.0.0.1.52270        ESTABLISHED
tcp4       0      0  127.0.0.1.52270        127.0.0.1.2181         ESTABLISHED
tcp46      0      0  *.2181                 *.*                    LISTEN 
'''
```

# 토픽(Topic) 생성

- 프로듀서(Producer)와 컨슈머(Consumer)가 소통할 매개체인 토픽을 생성할 것이다.

```s
$ bin/kafka-topics.sh 
'''
Create, delete, describe, or change a topic.
Option                                   Description                            
------                                   -----------                            
--alter                                  Alter the number of partitions,        
                                           replica assignment, and/or           
                                           configuration for the topic.         
--at-min-isr-partitions                  if set when describing topics, only    
                                           show partitions whose isr count is   
                                           equal to the configured minimum.     
--bootstrap-server <String: server to    REQUIRED: The Kafka server to connect  
  connect to>                              to.                                  
--command-config <String: command        Property file containing configs to be 
  config property file>                    passed to Admin Client. This is used 
                                           only with --bootstrap-server option  
                                           for describing and altering broker   
                                           configs.                             
--config <String: name=value>            A topic configuration override for the 
                                           topic being created or altered. The  
                                           following is a list of valid         
                                           configurations:                      
                                                cleanup.policy                        
                                                compression.type                      
                                                delete.retention.ms                   
                                                file.delete.delay.ms                  
                                                flush.messages                        
                                                flush.ms                              
                                                follower.replication.throttled.       
                                           replicas                             
                                                index.interval.bytes                  
                                                leader.replication.throttled.replicas 
                                                local.retention.bytes                 
                                                local.retention.ms                    
                                                max.compaction.lag.ms                 
                                                max.message.bytes                     
                                                message.downconversion.enable         
                                                message.format.version                
                                                message.timestamp.difference.max.ms   
                                                message.timestamp.type                
                                                min.cleanable.dirty.ratio             
                                                min.compaction.lag.ms                 
                                                min.insync.replicas                   
                                                preallocate                           
                                                remote.storage.enable                 
                                                retention.bytes                       
                                                retention.ms                          
                                                segment.bytes                         
                                                segment.index.bytes                   
                                                segment.jitter.ms                     
                                                segment.ms                            
                                                unclean.leader.election.enable        
                                         See the Kafka documentation for full   
                                           details on the topic configs. It is  
                                           supported only in combination with --
                                           create if --bootstrap-server option  
                                           is used (the kafka-configs CLI       
                                           supports altering topic configs with 
                                           a --bootstrap-server option).        
--create                                 Create a new topic.                    
--delete                                 Delete a topic                         
--delete-config <String: name>           A topic configuration override to be   
                                           removed for an existing topic (see   
                                           the list of configurations under the 
                                           --config option). Not supported with 
                                           the --bootstrap-server option.       
--describe                               List details for the given topics.     
--disable-rack-aware                     Disable rack aware replica assignment  
--exclude-internal                       exclude internal topics when running   
                                           list or describe command. The        
                                           internal topics will be listed by    
                                           default                              
--help                                   Print usage information.               
--if-exists                              if set when altering or deleting or    
                                           describing topics, the action will   
                                           only execute if the topic exists.    
--if-not-exists                          if set when creating topics, the       
                                           action will only execute if the      
                                           topic does not already exist.        
--list                                   List all available topics.             
--partitions <Integer: # of partitions>  The number of partitions for the topic 
                                           being created or altered (WARNING:   
                                           If partitions are increased for a    
                                           topic that has a key, the partition  
                                           logic or ordering of the messages    
                                           will be affected). If not supplied   
                                           for create, defaults to the cluster  
                                           default.                             
--replica-assignment <String:            A list of manual partition-to-broker   
  broker_id_for_part1_replica1 :           assignments for the topic being      
  broker_id_for_part1_replica2 ,           created or altered.                  
  broker_id_for_part2_replica1 :                                                
  broker_id_for_part2_replica2 , ...>                                           
--replication-factor <Integer:           The replication factor for each        
  replication factor>                      partition in the topic being         
                                           created. If not supplied, defaults   
                                           to the cluster default.              
--topic <String: topic>                  The topic to create, alter, describe   
                                           or delete. It also accepts a regular 
                                           expression, except for --create      
                                           option. Put topic name in double     
                                           quotes and use the '\' prefix to     
                                           escape regular expression symbols; e.
                                           g. "test\.topic".                    
--topic-id <String: topic-id>            The topic-id to describe.This is used  
                                           only with --bootstrap-server option  
                                           for describing topics.               
--topics-with-overrides                  if set when describing topics, only    
                                           show topics that have overridden     
                                           configs                              
--unavailable-partitions                 if set when describing topics, only    
                                           show partitions whose leader is not  
                                           available                            
--under-min-isr-partitions               if set when describing topics, only    
                                           show partitions whose isr count is   
                                           less than the configured minimum.    
--under-replicated-partitions            if set when describing topics, only    
                                           show under replicated partitions     
--version                                Display Kafka version.   
'''
```

- 실행시켜보면 옵션(Option)이 나온다.
- 토픽을 생성해본다.
    - 부트스트랩(Bootstrap) 서버와 토픽 이름, 파티션(Partition) 수와 레플리케이션 팩터(Replication Factor) 수를 입력해야한다.

```s
$ bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --topic first-topic --partitions 1 --replication-factor 1
'''
Created topic first-topic.
'''
```

- 만들어진 토픽의 정보를 확인해본다.

```s
$ bin/kafka-topics.sh --list --bootstrap-server localhost:9092
'''
first-topic
'''

$ kafka_2.13-3.1.0 bin/kafka-topics.sh --describe --bootstrap-server localhost:9092
'''
Topic: first-topic      TopicId: 9AZAxsacTKSOfYLmgYj7ig PartitionCount: 1       ReplicationFactor: 1      Configs: segment.bytes=1073741824
        Topic: first-topic      Partition: 0    Leader: 0       Replicas: 0     Isr: 0
'''
```

- 카프카의 토픽을 생성했으니, 잘 사용하는 방법을 알아야한다.
- 토픽에 메세지를 보내기 위해 다음 포스팅에서 프로듀서를 알아볼 것이다.