---
title: '[효율적인 프로그램 운영] 동기와 비동기, 블락과 논블락'
description: "[회사에서 공부하면 좋을 개발 지식인 효율적인 프로그램 운영] "
categories:
 - Development Knowledge 
tags: [개발 지식, 동기와 비동기, 블락과 논블락]
---

# 동기와 비동기
- 동기(Synchronous)와 비동기(Asyncronous)는 "두 작업의 작동 방식"에 대한 내용이다.

![image](https://user-images.githubusercontent.com/79494088/177232247-9dd48b1d-9e1e-420b-afd1-5af6607e0293.png)

## 동기(Synchronous)
- 동기 방식은 작업 A가 작업 B에게 작업 요청을 하고 작업 A가 작업 B가 작업을 끝낼 때까지 관심을 가지고 기다리는 방식이다.
- 요청을 했을 때 시간이 오래 걸리더라도 요청한 자리에서 결과를 줘야 한다.

- 예를 들어 웹 게시판 서버를 운영한다고 가정한다.
- 어떤 작업에서 서버에 게시글을 생성하라는 요청을 보냈고, 우리는 서버가 게시글을 생성하기까지 기다린다.
- 그리고 마침내 서버로부터 게시글 생성 처리가 완료되었다는 메시지를 받았다.
- 우리는 이제 이후에 필요한 작업을 마저 진행한다.
- 이런 작업 방식을 동기적이라고 표현한다.

- 우리가 만드는 대부분의 코드는 동기 방식의 코드라고 볼 수 있다.
- 동기 방식은 직관적이고 이해가 쉽다.
- 또한 설계가 비교적 단순하다.

## 비동기(Asynchronous)
- 비동기 방식은 작업 A가 작업 B에게 작업 요청을 하고 작업 A가 작업 B가 작업을 끝낼 때까지 관심을 버리고 기다리지 않는다.
- 즉, 요청과 결과가 동시에 일어나지 않는 것이다.

- 마찬가지로 웹 게시판 서버를 운영한다고 가정한다.
- 서버에 게시글 생성 요청을 보냈는데, 응답으로 요청을 잘 받았고, 처리 중이라는 메시지를 받았다.
- 이제 우리는 필요한 작업을 마저 한 뒤, 게시글이 잘 생성되었는지 확인하기 위해 서버에 한 번 더 게시글 생성이 완료되었는지 확인하는 요청을 보내야 한다.
- 이런 작업 방식을 비동기적이라고 표현한다.

- 비동기 방식은 동기보다 비직관적이고 이해하기도 어렵다.
- 또한 설계도 다소 복잡하다.
- 하지만 요청 결과를 받을 때까지 기다리지 않고 다른 작업을 수행할 수 있어 효율적이다.

- 동기, 비동기는 프로그래밍 언어의 특성이 아니다.
- 예를 들어, 파이썬은 기본적으로 대부분 동기 기반의 코드지만, 내장 지원 라이브러리를 통해 비동기 코드를 작성할 수 있다.
- 반면 자바스크립트는 대표적인 비동기 언어로 소개되곤 하지만, 실제 동기 코드로도 많이 작성한다.

- 비동기 방식으로 코드를 작성하면, 응답에 대해서 처리를 하는 코드를 따로 작성해줘야 한다.
- 이때 콜백(Callback) 방식을 많이 활용한다.

# 블락과 논블락
- 블락(Block)과 논블락(Nonblock)은 "작업의 상태"에 대한 내용이다.

## 블락
- 일반적으로 함수 A가 함수 B를 호출하면, 프로세스의 제어권은 함수 B로 넘어가게 된다.
- 함수 B가 프로세스의 제어권을 가지고 있는 동안 함수 A는 아무것도 하지 않게 되는데 이 상태를 블락 상태에 있다고 말한다.
- 또 이런 함수 B를 블락킹 함수라고 말할 수 있다.
- 함수 B가 모두 실행되고, 프로세스의 제어권이 다시 함수 A로 오게 되면 함수 A의 "블락" 상태는 풀리게 된다.

## 논블락
- 이번에도 마찬가지로 두 함수 A, B가 있다고 가정한다.
- 함수 A에서 함수 B를 스레드로 생성하는 함수를 호출했다.
- 스레드를 생성하는 함수는 함수 B를 별도의 스레드로 생성하고, 특정 객체를 바로 리턴한다.
- 함수 A가 있는 스레드는 함수 호출 이후의 일을 계속해서 하게 된다.
- 즉 이 과정에서 함수 A는 "블락" 상태를 가지지 않는다.
- 이렇게 "블락" 상태를 가지지 않는 상태를 논블락 상태라고 한다.
- 또 이런 함수 B를 논블락킹 함수라고 부를 수 있다.
- 블락/논블락을 접하는 가장 대표적인 사례가 I/O 관련 코드를 작성할 때이다.


# 동기/비동기 vs 블락/논블락의 차이

![image](https://user-images.githubusercontent.com/79494088/177233206-b3d9d4e1-b92a-4499-9689-4bb2b6efa32d.png)

- 동기는 블락과, 비동기는 논블락과 비슷한 개념처럼 보인다.
- 동기/비동기는 한 작업에서 다른 작업의 작업 완료 여부에 관심이 있느냐에 있다.
- 즉 관심이 있다면 동기 작업이고, 관심이 없다면 비동기 작업이다.

- 한편 블락/논블락은 한 함수에서 호출한 다른 함수가 바로 리턴을 하여, 현재 진행 중인 함수의 프로세스 제어권을 가져가느냐 아니냐에 있다.
- 호출한 함수가 바로 리턴하지 않아, 프로세스 제어권을 뻇기게 되면 블락상태에 있게 되는 것이다.
- 반면 바로 리턴하게 된다면 논블락 상태에 있게 되는 것이다.

- 블락, 논블락과 달리 동기, 비동기는 추상적인 개념이다.
- 어떤 맥락에서는 블락, 논블락을 동기, 비동기라고 부를 수도 있다.

# 예시

## 동기 / 블락
- 동기이면서 블락인 상황은 가장 일반적이고, 흔하게 볼 수 있는 상황이다.

```py
# sync / block

import time

def a():
    print("start in a()")
    time.sleep(2)
    print("finished in a()")

def b():
    print("start in b()")
    time.sleep(2)
    print("finished in b()")

def task():
    print("start in task()")
    a()
    b()
    print("finished in task()")

task()
```

- 코드를 실행하면 다음과 같다.

```s
start in task()
start in a()
finished in a()
start in b()
finished in b()
finished in task()
```

- `task`는 `a`를 먼저 실행한 후 `b`를 순차적으로 실행한다.
- 두 함수가 실행되는 동안 `task`는 블락 상태에 놓이게 된다.
- 또한 `task`는 `a`, `b`의 작업 완료 여부에 의존적이므로 동기적이라고 볼 수 있다.

## 비동기 / 논블락
- 비동기이면서 논블락인 상황 역시 일반적이고, 흔하게 볼 수 있는 상황이다.

```py
# async / non-block

import asyncio

async def a():
    print("start in a()")
    await asyncio.sleep(2)
    print("finished in a()")

async def b():
    print("start in b()")
    await asyncio.sleep(2)
    print("finished in b()")

async def task():
    print("start in task()")
    asyncio.create_task(a())
    asyncio.create_task(b())
    print("finished in task()")
    await asyncio.sleep(3)


async def main():
    await task()

asyncio.run(main())
```

- 코드를 실행하면 다음과 같다.

```s
start in task()
finished in task()
start in a()
start in b()
finished in a()
finished in b()
```

- `task`는 `a`와 `b`를 실행했지만, 두 함수가 실행되고 끝나기까지 기다리지 않는다.
- `task`는 호출 후 논블락 상태로 본인의 로직을 막힘없이 실행한다.
- 또한 `task`는 `a`와 `b`의 작업 종료 여부에 관심이 없으므로 비동기이다.

## 동기 / 논블락
- 작업 A가 작업 B을 실행시키지만, 프로세스 제어권을 놓치지 않는다. -> 논블락
- 작업 A가 어느 정도 자신의 작업 이후, 작업 B의 작업 완료 여부에 관심이 있다. -> 동기
- 흔한 경우는 아니지만 이렇게 쓰이는 경우가 종종 있다.

```py
# sync / non-block

import asyncio

global a_task_success
a_task_success = False

async def a():
    print("doing ... in a()")
    await asyncio.sleep(3)
    print("finished a !")
    global a_task_success
    a_task_success = True


async def task():
    print("doing task ...")
    asyncio.create_task(a())

    print("doing something ...")
    global a_task_success
    while a_task_success is False:
        print("waiting a to be finished ...")
        await asyncio.sleep(1)

asyncio.run(task())
```

- 코드를 실행하면 다음과 같다.

```s
start in task()
doing something ... in task()
waiting a() to be finished ... in task()
start in a()
waiting a() to be finished ... in task()
waiting a() to be finished ... in task()
finished in a()
finished in task()
```

- `task`는 `a`를 실행하지만 논블락 상태를 가지고 본인의 로직을 막힘없이 실행한다.
- 하지만 마지막 `while`문에서 `a` 테스크가 끝나길 기다리고 있다.
- `a` 테스크가 완료될 때 `a_task_success`를 `True`로 바꿔주어야 비로소 `task`도 끝나게 된다.
- 즉 `task`는 `a`의 작업 완료 시점에 의존적이므로 동기적이다.

## 비동기 / 블락
- 작업 A는 작업 B를 실행시켰지만, 작업 A는 작업 B의 작업 완료 여부에 관심이 없다. -> 비동기
- 작업 A가 작업 B를 실행시켰을 때, 작업 A는 프로세스 제어권을 잃는다. -> 블락
- 일반적으로 좋은 경우는 아니지만, 코드로 구현하면 다음과 같은 모양새이다.

```py
# sync / non-block

import asyncio

async def a():
    print("start in a()")
    await asyncio.sleep(3)
    print("finished in a()")

async def task():
    print("start in task()")
    value = await a()
    print("doing something ... in task()")
    print("finished in task()")

asyncio.run(task())
```

- 코드를 실행하면 다음과 같다.

```s
start in task()
start in a()
finished in a()
doing something ... in task()
finished in task()
```

- 비동기 / 블락 결과는 동기 / 블락과 결과와 같은 걸 확인할 수 있다.
- 보통 비동기 처리 로직에서 데이터베이스의 결과 값을 받아와야만 하는 경우에 이런 방식으로 코드를 작성할 수 있다.


# 정리
- 동기와 비동기는 작업 완료에 관심이 있느냐에 관한 작동 방식이다.
- 작업 A가 작업 B의 작업 완료에 관심이 있다면 동기이다.
- 관심이 없다면 비동기이다.
- 블락과 논블락은 프로세스 제어권을 뺏기는 상태에 대한 내용이다.
- 함수 A가 함수 B를 호출하고 함수 B가 실행되는 동안 프로세스 제어권을 뺏겨 본인 로직을 실행하지 못하는 경우 블락이다.
- 반면 프로세스 제어권을 뺏기지 않고 바로 리턴 받아 본인의 로직을 실행하면 논블락이다.
- 일반적으로 동기/블락과 비동기/논블락 방식이 쓰인다.
- 동기/블락 방식은 이해하기 쉽고 직관적이지만 일반적으로 느리다.
- 비동기/논블락 방식은 이해하기 어렵고, 프로그램 흐름도 어려워지지만 일반적으로 빠르다.
- 동기, 비동기, 블락, 논블락의 차이점을 외우려고 하기보단, 맥락을 파악하는 정도면 충분하다.