---
title: '[Computer Science] Python Problem Solving'
description: 프로그래밍 진행과정, 파이썬을 활용한 실습 및 예외처리, 알고리즘을 위한 논리적 방법
categories:
 - Computer Science
tags: [Computer Science, pseudocode, comprehension, assert]
mathjax: enable
---

# 문제해결
- 의사코드(pseudocode)를 작성하며 로직을 코드로 표현하는 방법을 배운다.
- 알고리즘과 같이 수식이나 익숙하지 않은 개념에 대해 접근하는 경우를 위해 의사코드를 작성한다.
- 프로그래밍을 통해 논리적으로 문제를 생각하는 방법에 대해 알아본다.

## 프로세스
- 문제를 단위별로 쪼갠다.
- 최소한의 시간을 활용하여 분석한다.
- 어렵다면 전체 문제 중 해결할 수 있는 부분을 찾는다.

## 의사코드(pseudocode)
- 실행되는 소스코드 작성 전, 자신이 이해할 수 있는 코드를 작성하는 연습을 한다.

# Simulation

## Comprehension
- 한 줄로 파이썬 기능을 구현할 수 있는 기능이다.
- 코드간소화, 직관적, 속도가 빠르다.
- 복잡해지면 직관성이 떨어지고, 메모리 사용량이 증가하여 사용하기 어렵다.

```py
numbers = [1, 2, 3, 4]
squares = []

for n in numbers:
  squares.append(n**2)

print(squares)
'''
[1, 4, 9, 16]
'''


numbers = [1, 2, 3, 4]
squares = [n**2 for n in numbers]

print(squares) 
'''
[1, 4, 9, 16]
'''
```

```py
list_a = [1, 2, 3, 4]
list_b = [2, 3, 4, 5]

common_num = []

for a in list_a:
  for b in list_b:
    if a == b:
      common_num.append(a)
      
print(common_num)
'''
[2, 3, 4]
'''


list_a = [1, 2, 3, 4]
list_b = [2, 3, 4, 5]

common_num = [a for a in list_a for b in list_b if a == b]

print(common_num)
'''
[2, 3, 4]
'''
```

```py
# 딕셔너리 컴프리헨션

test = {'A': 5, 'B': 7, 'C': 9, 'D': 6, 'E': 10} 

test = {na:test for na,test in test.items() if na != 'E'}
print(test)
'''
{'A': 5, 'B': 7, 'C': 9, 'D': 6}
'''


# 아래와 같이 조건을 반복문 대신 조건을 먼저 쓸 수 있다.
# 조건을 위해 if를 사용하는 경우 else를 작성해줘야된다.
numbers = {'amy': 7, 'jane': 9, 'sophia': 5, 'jay': 10}
pas = {name: 'PASS' if numbers > 8 else 'NO PASS' for name, numbers in numbers.items()}
print(pas)
'''
{'amy': 'NO PASS', 'jane': 'PASS', 'sophia': 'NO PASS', 'jay': 'PASS'}
'''


# 아래처럼 반복문을 연속으로 작성가능하다.
# set은 특성상 중복값을 제외한다.

print('list for loop : ',[n for n in range(1, 5+1) for n in range(1, 5+1)])

print('set for loop : ',{n for n in range(1, 5+1) for n in range(1, 5+1)})
'''
list for loop :  [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
set for loop :  {1, 2, 3, 4, 5}
'''


# 두개의 리스트를 하나의 딕셔너리로 합침. 
# 하나는 key, 또 다른 하나는 value로 사용한다
subjects = ['math', 'history', 'english', 'computer engineering']
scores = [90, 80, 95, 100]
score_dict = {key: value for key, value in zip(subjects, scores)}
print(score_dict)

# 튜플 리스트를 딕셔너리 형태로 변환
score_tuples = [('math', 90), ('history', 80), ('english', 95), ('computer engineering', 100)]
score_dict = {t[0]: t[1] for t in score_tuples}
print(score_dict)
'''
{'math': 90, 'history': 80, 'english': 95, 'computer engineering': 100}
{'math': 90, 'history': 80, 'english': 95, 'computer engineering': 100}
'''
```

## 지역변수와 전역변수
- 변수사용에 따라 함수 및 클래스의 접근이 달라진다.
- 변수이름 설정, 변수 활용도에 따라 변수설계가 중요하다.
- 지역변수: 해당 변수가 포함된 함수 안에서만 수정하고, 읽을 수 있다.
- 일반 전역변수: 하나의 파이썬 파일 전체에서 값을 읽을 수 있다.
    - 되도록 함수 안에서 사이드 이펙트 및 가독성을 위해 값을 수정하지 않도록 한다.
- Global 전역변수: 일반 전역변수와 다른 점은 변수가 생성되는 시점만 다르다.

```py
g_var = 'g_var'   # 전역변수
  
# 값 수정후(수정값)
def variables():
  
    global glo_var  # global 전역변수
    glo_var = 'glo_var' # global 전역변수에 새로운 값 할당
    lo_var = 'lo_var'   # 지역변수

    print()
    print('(값 수정후)함수 안에서 g_var(전역변수) : ', g_var)  # 수정되지 않고 초기값을 출력함
    print('(값 수정후)함수 안에서 glo_var(global 전역변수) : ', glo_var)  # 함수에서 수정된 후 값을 출력함
    print('함수 안에서 lo_var(지역변수) : ', lo_var)    # 특정 함수에서만 출력되는 지역변수
```

## 구문 및 예외 상황 처리
- 반복문 이후, else를 사용하는 경우 대부분의 else가 즉시 실행된다.

### assert
- assert 조건, '에러메시지'
    - 조건이 True인 경우 그대로 코드 진행, False인 경우 에러메시지(Assertion Error)를 띄워준다.
    - 에러메시지: 앞에 조건이 False인 경우 AssertionError와 함께 남길 메시지를 남겨줄 수 있다. 이 부분은 생략 가능한 부분이다.
- assert()는 방어적 프로그래밍(defensive programming) 방법 중 하나이며, 코드를 점검하는데 사용된다.

```py
# 조건에 맞는 경우, 함수에서 부울값을 반환해주는 방법
def bool_return(v1, v2):
  # 두 인자가 서로소가 아닐 경우 False를 반환하고,
  # 서로소일 경우 True를 반환한다.
  for i in range(2, min(v1, v2) +1):
    if v1 % i == 0 and v2 % i == 0:
      return False
  return True

# 4와 9는 공약수가 1뿐인 두 정수로 서로소이다. 그러므로 아래 함수는 True 를 반환한다.
print('bool_return(4,9): ',bool_return(4,9))
assert bool_return(4,9)

# 아래 주석 처리된 함수는 False 를 반환하고 AssertionError가 발생한다.
print('bool_return(3,6):',bool_return(3,6))
# assert bool_return(3,6), '서로소를 입력해주세요.'

# assert not False는 에러를 반환하지 않는다.
# not bool_return(3,6) = not False = True
print('not bool_return(3,6):', not bool_return(3,6))
assert not bool_return(3,6)
```

### try / except / raise / finally
- try : 처리하고자 하는 부분을 넣는다.
- except : try구문 안에서 발생할 것으로 예상되는 예외를 처리한다.
- raise : 예외상황일 때 명시적으로 처리한다.
- finally : 마지막으로 실행하는 코드

<img width="573" alt="스크린샷 2021-11-18 02 12 18" src="https://user-images.githubusercontent.com/79494088/142248683-ede0c57e-640e-41ff-9abf-3f3352a8d4fb.png">

# Reference
- [우아하게 준비하는 테스트와 리팩토링 - PyCon Korea 2018](https://www.slideshare.net/KennethCeyer/pycon-korea-2018-109833085)
- [Python 코드 리팩터링](https://docs.microsoft.com/ko-kr/visualstudio/python/refactoring-python-code?view=vs-2019)
- [string의 변화에 대해 예상해보자](https://leetcode.com/problemset/all/?topicSlugs=string)
- [숫자와 일상생활](https://www.acmicpc.net/step/8)
- [특이한 숫자를 다뤄보자](https://www.acmicpc.net/step/10)
- [프로그래밍에서의 문제해결전략](https://youtu.be/XVhYjmNbgRs)
- [예외처리에 대해서](https://youtu.be/g7dzMgrWFic)