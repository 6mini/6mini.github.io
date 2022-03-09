---
title: '[카프카] 컨슈머(Consumer)와 파티션(Partitions)의 관계'
description: 카프카 프로그래밍으로 넘어가기 전 컨슈머 그룹과 파티션이 가지는 의미
categories:
 - Data Engineering
tags: [카프카, 데이터 엔지니어링, 컨슈머]
---

- 본격적으로 카프카 프로그래밍(Kafka Programing)으로 넘어가기 전, 컨슈머 그룹(Consumer Group)과 파티션(Partition)이 가지는 의미에 대해 더 알아보고 넘어갈 것이다.

# 두 가지 컨슈머(Consumer)에서 메세지 테스트

<img width="1538" alt="image" src="https://user-images.githubusercontent.com/79494088/157170533-9862e1b3-c44e-4d56-83ef-8863a544b199.png">

- 위와 같이 네 개의 터미널(Terminal)을 사용하는 환경으로 진행할 것이다.
- 왼쪽 두개의 터미널은 프로듀서(Producer), 오른쪽 두개의 터미널은 컨슈머(Consumer)가 될 것이다.
- 주키퍼(Zookeeper)와 카프카 브로커(Broker)가 돌아가고 있다는 가정하에 진행하고, 전 포스팅에 만들어둔 `first-topic`을 사용한다.
       - 파티션 하나를 가진 토픽이다.

```s
(sixat) 14:48:10 kafka_2.13-3.1.0 ./bin/kafka-console-producer.sh --bootstrap-server localhost:9092 --topic first-topic
>
```

- 위와 같이 메세지를 보낼 수 있는 프로듀서를 생성한다.

<img width="770" alt="image" src="https://user-images.githubusercontent.com/79494088/157171437-5fcb1fc3-be73-4651-8cf7-36aa31561fbf.png">

- 다음, 위 이미지와 같이 컨슈머를 동시에 두 가지 생성한다.

## 문제 상황

```s
(sixat) 14:48:10 kafka_2.13-3.1.0 ./bin/kafka-console-producer.sh --bootstrap-server localhost:9092 --topic first-topic
>1st msg         
>2nd msg
```

<img width="762" alt="image" src="https://user-images.githubusercontent.com/79494088/157173344-f962d187-2696-4f95-8abb-53daad62a425.png">

- 각각의 컨슈머 그룹을 가지는 컨슈머를 만들었을 땐 모두가 메세지를 받았는데, 하나의 컨슈머 그룹으로 묶어주게 되면 컨슈머 중 하나만 메세지를 받게 된다.

```s
(sixat) 14:48:10 kafka_2.13-3.1.0 ./bin/kafka-console-producer.sh --bootstrap-server localhost:9092 --topic first-topic
>hello from 2nd producer
>oing?
```

<img width="760" alt="image" src="https://user-images.githubusercontent.com/79494088/157173632-1c0f3e88-c88b-43d1-9742-d11b0c18718c.png">

- 프로듀서를 하나 더 추가하여 메세지를 날리는 경우에도 하나의 컨슈머로만 도착하는 것을 확인할 수 있다.
  - 토픽이 가지고 있는 파티션(Partiton)이 하나뿐이기 때문이다.
- 파티션 하나는 무조건 컨슈머 그룹 안에서 하나의 컨슈머와만 매핑(Mapping)된다.
- 조금 더 리소스를 효율적으로 관리하기 위해서는 파티션을 여러가지로 설정한다.

# 문제 해결
- 파티션을 만들기 위해 새로운 토픽을 만들 것이다.

```s
(sixat) 14:44:02 kafka_2.13-3.1.0 ./bin/kafka-topics.sh --bootstrap-server localhost:9092 --list
'''
__consumer_offsets
first-topic
'''
```

- 토픽을 확인한다.
- `second-topic`을 만들것이며, 두개의 파티션을 할당한다.


```s
(sixat) 14:44:17 kafka_2.13-3.1.0 ./bin/kafka-topics.sh --create --topic second-topic --bootstrap-server localhost:9092 --partitions 2 --replication-factor 1
'''
Created topic second-topic.
'''
(sixat) 14:47:22 kafka_2.13-3.1.0 ./bin/kafka-topics.sh --bootstrap-server localhost:9092 --list                                                             
'''
__consumer_offsets
first-topic
second-topic
'''
```

- 새로운 토픽을 생성했다.

```s
(sixat) 14:48:48 kafka_2.13-3.1.0 ./bin/kafka-console-producer.sh --bootstrap-server localhost:9092 --topic second-topic

(sixat) 14:50:07 kafka_2.13-3.1.0 ./bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic second-topic --group second-group
```

- 프로듀서와 컨슈머 모두 `second-topic`을 사용하게끔 교체하고, `second-group`으로 할당한다.

```s
(sixat) 14:48:10 kafka_2.13-3.1.0 ./bin/kafka-console-producer.sh --bootstrap-server localhost:9092 --topic second-topic 
>1
>2
>3
>4
>5
>6
>7
>8
>9
>10
>11
>12
>13
>14
>15
>16
>17
>18
>19 
>20
```

<img width="757" alt="image" src="https://user-images.githubusercontent.com/79494088/157175102-e3afe65c-b366-42ba-a308-3258797870f6.png">

- 위와 같은 명령어로 프로듀서에서 마구잡이로 메세지를 보내게 되면, 첫번째와 두번째 컨슈머가 균등하게 분배되어 받고 있는 것을 볼 수 있다.
- 분산 처리를 할 때 리소스가 균등하게 분배되어 더 효율적으로 사용할 수 있다.
- 파티션의 수는 분산처리를 할 때 중요하며 다음 포스팅부터 본격적인 카프카 프로그래밍을 실시할 것이다.