---
title: '[클린 코드] 에러 핸들링'
description: "깔끔한 코드를 위한 에러 핸들링"
categories:
 - Clean Code
tags: [클린 코드]
---

# 오류 코드보다는 예외 사용
- 오류 코드를 사용하게 되면 상단에 오류인지 확인하는 불필요한 로직이 들어가게 된다.
- 오류의 범주에 들어가지 않은 상태를 나타내는 것이 아니라면, 예외(Exception)로 명시적으로 에러 처리를 표현해주는 게 좋다.

## as-is

```py
from enum import Enum 

class ErrorCodes(Enum):
    VALUE_ERROR="VALUE_ERROR"

def we_can_raise_error():
    ...
    return ERROR_CODES.VALUE_ERROR

def use_ugly_function():
    result = we_can_occur_error()
    if result == ErrorCodes.VALUE_ERROR:
        # 처리 코드
    ...
```

## to-be

```py
def we_can_raise_error():
    if ...
        raise ValueError("에러 발생")

def use_awesome_function():
    try:
        we_can_occur_error()
        ...
    except ValueError as e:
        # 에러 처리 로직				
```

# 예외 클래스 잘 정의
- 기본 `Exception`만 쓰기 보단 내장된 `built in Exception`을 잘 활용하면 좋다.
    - 파이썬 : https://docs.python.org/ko/3/library/exceptions.html(opens new window)
    - 자바 : https://docs.oracle.com/javase/8/docs/api/java/lang/Exception.html(opens new window)
    - 자바스크립트 : https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Error(opens new window)
- 상황에 맞게 `Custom Exception`을 만들어 사용하는 것도 좋다.

```py
class CustomException(Exception):
    ...

class WithParameterCustomException(Exception):
    def __init__(self, msg, kwargs):
        self.msg = msg
        self.kwargs = kwargs
    
    def __str__():
        return f"message {self.msg} with parameter {self(self.kwargs)}"

raise WithParameterCustomException("문제가 있습니다", {"name": "grab"})
```

# 에러 핸들링 잘하기
- 에러를 포착했다면 잘 핸들링해줘야 한다.

```py
def we_can_raise_error():
    ...
    raise Exception("Error!")

# BAD: 에러가 났는지 확인할 수 없다.
def use_ugly_function1():
    try:
        we_can_raise_error()
        ...
    except:
        pass

# BAD: 로그만 남긴다고 끝이 아니다.
def use_ugly_function2():
    try:
        we_can_raise_error()
        ...
    except Exception as e:
        print(f"에러 발생{e}")

# GOOD
def use_awesome_function():
    try:
        we_can_raise_error()
        ...
    except Exception as e:
        logging.error(...) # Error Log 남기기
        notify_error(...) # 예측 불가능한 외부 I/O 이슈라면 회사 내 채널에 알리기(이메일, 슬랙 etc)
        raise OtherException(e) # 만약 이 함수를 호출하는 다른 함수에서 추가로 처리해야 한다면 에러를 전파하기
    finally:
        ... #에러가 발생하더라도 항상 실행되어야 하는 로직이 있다면 finally 문을 넣어주기
```

- 에러 핸들링을 모을 수 있으면 한곳으로 모은다.
- 보통 같은 수준의 로직을 처리한다면 한 곳으로 모아서 처리하는 게 더 에러를 포착하기 쉽다.

## as-is

```py
def act_1():
    try:
        we_can_raise_error1()
        ...
    except:
        # handling

def act_2():
    try:
        we_can_raise_error2()
        ...
    except:
        # handling

def act_3():
    try:
        we_can_raise_error3()
        ...
    except:
        # handling

# 에러가 날 지점을 한눈에 확인할 수 없다. 
# act_1이 실패하면 act_2가 실행되면 안 된다면? 핸들링하기 어려워진다.
def main():
    act_1()
    act_2()
    act_3()
```

## to-be

```py
def act_1():
    we_can_raise_error1()
    ...

def act_2():
    we_can_raise_error2()
    ...

def act_3():
    we_can_raise_error3()
    ...

# 직관적이며 에러가 날 지점을 확인하고 처리할 수 있다.
# 트랜잭션같이 한 단위로 묶여야하는 처리에도 유용하다.
def main():
    try:
        act_1()
        act_2()
        act_3()
    except SomeException1 as e1:
        ...
    except SomeException2 as e2:	
        ...
    except SomeException2 as e3
        ...
    finally:
        ...	
```