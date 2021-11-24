---
title: '[Computer Science] DataStructure ADT, Linked List, Queue, Stack'
description: 자료구조와 알고리즘 반복학습, 자료구조의 코어가 되는 추상자료형(ADT)과 연결리스트, 큐, 스택의 파이썬 코드 구현
categories:
 - Computer Science
tags: [Computer Science, Linked List, Queue, Stack, Deque, 연결리스트, 큐, 뎈, 스택]
mathjax: enable
---

- 키워드: 추상자료형, 파이썬 내장함수, Data Structure
- [자료구조를 알아야 하는 이유](https://youtu.be/OH7prOt3vQA)
- [무조건 알아야하는 자료구조/ADT](https://youtu.be/Nk_dGScimz8)

# ADT
- 자료구조 핵심: ADT(Abstract Data Type)와 연결리스트, 큐, 스택
- 프로그래밍을 하면서 데이터 처리를 위한 자료형이 존재한다.
- 파이썬에서 프로그래밍을 위한 도구인 기본자료형(숫자, 문자열, 리스트, 딕셔너리 등)이 있다.
- **ADT는 추상적으로 필요한 기능을 나열한 일종의 명세서(로직)이다.**
  - 기본자료형을 활용하여 사용자에 의해 구현된다.
- abstract는 소프트웨어가 발전하면서 프로그램의 크기나 복잡도가 같이 증가하였고 프로그램의 핵심부분을 간단하게 설명하기 위해 생겨났다.
  - 참고: [추상화](https://ko.wikipedia.org/wiki/추상화_(컴퓨터_과학))

![ADT](https://user-images.githubusercontent.com/79494088/143015084-1226a32c-16b9-4145-83e5-ba86c12f1d16.png)

# linked-list(연결리스트)

<img width="688" alt="연결리스트_리스트" src="https://user-images.githubusercontent.com/79494088/143015177-c62c941a-f12a-4477-9756-f957a1e9962c.png">

- 데이터를 노드의 형태로 저장한다.
- 노드에는 데이터와 다음 노드를 가르키는 포인터를 담은 구조로 이루어져 있다.

![image](https://user-images.githubusercontent.com/79494088/143208265-5bd7a1c7-b197-4332-bd98-09cf5d097262.png)

- Linked list는 Array처럼 선형 데이터 자료구조이지만, Array는 물리적인 배치 구조 자체가 연속적으로 저장되어 있고, Linked Array는 위 노드의 Next 부분에 다음 노드의 위치를 저장함으로써 선형적인 데이터 자료구조를 가진다.
- List의 삽입과 삭제의 시간복잡도가 O(n)이 걸리는 것은 배열이 물리적인 데이터의 저장 위치가 연속적이어야 하므로 데이터를 옮기는 연산작업이 필요하기 때문이다.
- 하지만 Linked list는 데이터를 삽입, 샂게할 경우, 노드의 Next 부분에 저장한 다음 노드의 포인터만 변경해주면 되므로 배열과 비교했을 때 Linked list가 효율적으로 데이터를 삽입, 삭제할 수 있다.
- 하지만 특정 위치의 데이터를 탐색하기 위해서는 첫 노드부터 탐색을 시작해야한다.
- 그 시간이 O(n)만큼 걸리게 되므로 탐색에 있어서는 배열이나 트리 구조에 비해 상대적으로 느리다.

## 장점
- Linked list의 길이를 동적으로 조절 가능하다.
- 데이터의 삽입과 삭제가 쉽다.

## 단점
- 임의의 노드에 바로 접근할 수 없다.
- 다음 노드의 위치를 저장하기 위한 추가 공간이 필요하다.
- Cache locality를 활용해 근접 데이터를 사전에 캐시에 저장하기 어렵다.
- Linked list를 거꾸로 탐색하기 어렵다.

## 단일 연결 리스트
- 각 노드에 자료 공간에 한 개의 포인터 공간이 있고, 각 노드의 포인터는 다음 노드를 가리킨다.

### 삽입

![image](https://user-images.githubusercontent.com/79494088/143263489-f3268634-e6e5-482b-8488-c2495bca1fb7.png)

### 삭제

![image](https://user-images.githubusercontent.com/79494088/143263524-6c70b734-a914-4856-b916-aef1303e043b.png)

## 파이썬 코드 구현

```py
class Node:
    def __init__(self,value,next=None):
        """
        Linkedlist에서 사용할 Node의 생성자 함수
        input: value, next
            value: Node의 값
            next: 생성될 Node의 다음 Node, 기본값은 None
        output:
            반환값은 없다.
        """
        self.value = value
        self.next = next


class linked_list:
    def __init__(self, value):
        """
        Linkedlist의 생성자 함수
        input: value
            value: Linkedlist의 Head Value
        output:
            반환값은 없다.
        """
        self.head = Node(value)


    def add_node(self, value):
        """
        Linkedlist에 새로운 Node를 추가하는 메소드
        input: value
            value: Linkedlist에 들어올 새로운 Node Value
        output:
            반환값은 없다.
        """
        node = self.head
        if node == None:
            node = Node(value)
        else:
            while node.next:
                node = node.next
            node.next = Node(value)


    def del_node(self,value):
        """

        Linkedlist에 value값을 가지고 있는 Node를 삭제하는 메소드를 작성해주세요.        
        input: value
            value: Linkedlist에서 삭제할 Node Value
        output:
            값을 삭제하였다면 삭제한 Node의 value를 반환
            만약 LinkedList에 값이 없다면 None 반환
        """
        node = self.head
        if node == None:
            return
        elif node.value == value:
            self.head = node.next
            return value
        else:
            while node.next:
                if node.next.value == value:
                    node.next = node.next.next                
                    return value
                else:
                    node = node.next


    def ord_desc(self):
        """
        Linkedlist에 저장된 값들을 리스트 형태로 반환하는 메소드
        input: 
            input값은 없다.
        output:
            Linkedlist의 Head부터 시작하여 저장된 모든 Node의 Value들을 리스트 형태로 반환
        """
        node = self.head
        link_list = []
        while node:
            link_list.append(node.value)
            node = node.next
        return link_list


    def search_node(self,value):
        """
        Linkedlist에 value값이 어디에 위치하는지 찾는 메소드
        input: value
            value: Linkedlist내부에서 찾고자 하는 값
        output:
            value값의 인덱스 번호를 반환
            (head의 위치는 0)
            값이 없는 경우 -1을 반환
        """
        node = self.head
        while node:
            if node.value == value:
                return node
            else:
                node = node.next
```

# Queue
- 큐란 목록 한쪽 끝에서만 자료를 넣고 다른 한쪽 끝에서만 자료를 빼낼 수 있는 자료구조이다.
- 먼저 집어넣은 데이터가 먼저 나오는(FIFO: Fist In, First Out, 선입선출)구조로 데이터를 저장한다.
- 데이터가 입력한 순서대로 처리되어야 할 경우에 사용한다.
- 큐에 새로운 데이터가 들어오면 큐의 끝에 데이터가 추가되며(enqueue), 반대로 삭제될 때는 첫번째 위치의 데이터가 삭제된다(dequeue).

## 종류
- 선형큐
  - 문제점: 일반적인 선형 큐는 배열의 마지막 index를 가리키는 변수가 있고, dequeue가 일어날 때마다 비어 있는 공간이 생기면서 이를 활용할 수 없게 된다.
  - 이 방식을 해결하기 위해 front를 고정시킨 채 뒤에 남아있는 데이터를 앞으로 한 칸 씩 땡길 수 밖에 없다.
  - 이에 대한 대안으로 사용하는것이 원형큐이다.
- 환형큐
- 우선순위큐

## 파이썬 코드 구현

```py
class Queue():
    def __init__(self):
        """
        Queue의 생성자 함수
        """
        self.queue = []


    def enqueue(self,item):
        """
        queue에 item 매개변수에 들어온 값을 집어넣는 메소드
        input: item
            queue로 들어갈 값
        output: 
            반환값은 없다.
        """
        self.queue.append(item)


    def dequeue(self):
        """
        queue 동작에 알맞게 값을 queue에서 삭제하는 메소드
        input: 
            input은 없다.
        output: 
            queue동작에 맞게 queue에서 삭제된 값을 반환
            만약 삭제한 값이 없다면 None을 반환
        """
        if len(self.queue) == 0:
            return
        return self.queue.pop(0)


    def return_queue(self):
        """
        queue내부에 있는 값을 반환하는 메소드
        input: 
            input은 없다.
        output: 
            queue내부에 있는 값을 리스트 형태로 반환
        """
        return self.queue
```

# Stack
- 스택은 데이터의 삽입과 삭제가 저장소의 맨 윗 부분(the top of stack)에서만 일어나는 자료구조이다.
- 스택은 데이터가 순서대로 저장되고 스택의 마지막에 넣은 요소가 처음으로 꺼내진다(LIFO: Last In, First Out).
- 스택은 연속으로 저장된 데이터 구조를 가지고 있고 맨 위 요소에 대한 포인터(주소값)을 갖고 있는 Array나 singly linked list로 구현할 수 있다.
- 스택은 함수의 콜스택, 문자열을 역순으로 출력하거나 연산자 후위표기법 등에 사용된다.

## 장점
- 참조 지역성(한번 참조된 곳은 다시 참조될 확률이 높다)을 활용할 수 있다.

## 단점
- 데이터를 탐색하기 어렵다.

## Stack의 ADT
- push(None): 맨 위에 값 추가
- pop(data): 가장 최근에 넣은 맨 위의 값을 제거
- peak(data or -1): 스택의 변형 없이 맨 위에 값을 출력
- is_empty(bool): 스택이 비어있는지 확인

## 파이썬 코드 구현

```py
class Stack():
    def __init__(self):
        """
        Stack의 생성자 함수
        """
        self.stack = []


    def push(self, item):
        """
        Stack에 item 매개변수에 들어온 값을 집어넣는 메소드
        input: item
            Stack로 들어갈 값
        output: 
            반환값은 없다.
        """
        self.stack.append(item)


    def pop(self):
        """
        Stack 동작에 알맞게 값을 Stack에서 삭제하는 메소드
        input: 
            input은 없다.
        output: 
            Stack동작에 맞게 Stack에서 삭제된 값을 반환
            만약 삭제한 값이 없다면 None을 반환
        """
        if len(self.stack) == 0:
            return
        return self.stack.pop()


    def return_stack(self):
        """
        Stack내부에 있는 값을 반환하는 메소드
        input: 
            input은 없다.
        output: 
            Stack내부에 있는 값을 리스트 형태로 반환
        """
        return self.stack
```

# Deque
- 큐가 선입선출 방식으로 작동한다면, 양방향 큐가 있는데 그것이 바로 Deque이다.
- 앞, 뒤 양쪽에서 엘리먼트를 추가하거나 제거할 수 있다.
- 덱는 양 끝 엘리먼트의 append와 pop가 압도적으로 빠르다.
- 컨테이너의 양 끝 엘리먼트에 접근하여 삽입 또는 제거를 할 경우, 일반적인 리스트가 이러한 연산에 O(n)이 소요되는 데 반해, 데크는 O(1)로 접근이 가능하다.

## When? Why?
- 덱은 스택처럼 사용할 수도 있고, 큐처럼 사용할 수도 있다.
- 시작점의 값을 넣고 빼거나, 끝 점의 값을 넣고 빼는 데 최적화된 연산 속도를 제공한다.
- 대부분의 경우 덱은 리스트보다 월등한 옵션이다.
- 덱은 특히 push/pop 연산이 빈번한 알고리즘에서 리스트보다 월등한 속도를 자랑한다.

## 파이썬 코드 구현

```py
class Node:
    """
    Deque 클래스에서 사용할 Node 클래스
    """
    def __init__(self, value, next=None):
        self.value = value
        self.next = next


class Deque:
    def __init__(self):
        self.top = None
        self.bottom = None


    def append(self, item):
        """
        Deque에 item 매개변수로 들어온 값을 제일 마지막 노드로 집어넣는 메소드
        input: item
            Deque로 들어갈 값
        output: 
            반환값은 없다.
        """
        top = self.top
        if top == None:
            self.top = Node(item)
            self.bottom = top
        else:
            while top.next:
                top = top.next
            top.next = Node(item)
            self.bottom = top.next


    def appendleft(self, item):
        """
        Deque에 item 매개변수로 들어온 값을 제일 앞 노드로 집어넣는 메소드
        input: item
            Deque로 들어갈 값
        output: 
            반환값은 없다.
        """
        node = Node(item)
        node.next = self.top
        self.top = node


    def pop(self):
        """
        Deque에 가장 뒤에 있는 값을 삭제하는 메소드
        input: 
            input은 없다.
        output: 
            Deque에서 삭제된 값을 반환
            만약 삭제한 값이 없다면 None을 반환
        """
        top = self.top
        bot = self.bottom
        if top == None:
            return
        elif top == bot:
            self.top = None
            self.bottom = None
            return bot.value
        else:
            while top.next:
                if top.next == bot:
                    top.next = None
                    self.bottom = top
                    return bot.value
                top = top.next


    def popleft(self):
        """
        Deque에 가장 앞에 있는 값을 삭제하는 메소드
        input: 
            input은 없다.
        output: 
            Deque에서 삭제된 값을 반환
            만약 삭제한 값이 없다면 None을 반환
        """
        top = self.top
        bot = self.bottom
        if top == None:
            return
        elif top == bot:
            self.top = None
            self.bottom = None
            return top.value
        else:
            self.top = top.next
            return top.value


    def ord_desc(self):
        """
        queue내부에 있는 값을 반환하는 메소드
        input: 
            input은 없다.
        output: 
            queue내부에 있는 값을 리스트 형태로 반환
        """
        top = self.top
        deque_list = []
        while top:
            deque_list.append(top.value)
            top = top.next
        return deque_list
```

# Reference

- [Python - 데크(deque) 언제, 왜 사용해야 하는가?](https://leonkong.cc/posts/-python-deque.html)
- [Python으로 구현하는 자료구조: Stack](https://daimhada.tistory.com/105)
- [Python으로 구현하는 자료구조: Linked List](https://daimhada.tistory.com/72)
- [Python으로 구현하는 자료구조: Queue](https://daimhada.tistory.com/107?category=820522)
- [파이썬의 문법을 활용한 자료구조](https://docs.python.org/ko/3/tutorial/datastructures.html)
- [파이썬 복잡도 상세내용](https://www.ics.uci.edu/~pattis/ICS-33/lectures/complexitypython.txt)
- [데이터의 7V](https://3months.tistory.com/348)