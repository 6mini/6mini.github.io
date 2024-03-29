---
title: '[객체 지향] 프로그래밍 패러다임 흐름: 절차 지향'
description: "[객체 지향 프로그래밍] 절차 지향 프로그래밍 패러다임의 개념 및 예시, 장단점"
categories:
 - Object Oriented Programming
tags: [객체 지향 프로그래밍, 절차 지향 프로그래밍]
---

# 프로그래밍 패러다임
- **프로그래밍 패러다임은 프로그래밍할 때 관점을 제공해주고, 설계를 결정하도록 돕는 패러다임**이다.
- 예를 들어 프로그래밍을 함수적 호출 과정을 중심으로 바라보고 설계한다면 절차지향 프로그래밍이 되고,<br> 객체의 책임과 협력을 중심으로 설계한다면 객체 지향 프로그래밍이 된다.
- 프로그래밍 패러다임은 프로그래밍 언어에도 자연스럽게 반영되는데,<br>이를테면 C언어는 대표적인 절차지향 언어이며, 자바는 대표적인 객체 지향 언어이다.
- 이 두 패러다임을 모두 수용하기도 하는데, 파이썬이 이런 멀티 패러다임 언어라고 볼 수 있다.
- 현재 가장 대중적인 프로그래밍 패러다임은 객체 지향 프로그래밍이다.
- 다만 Silver Bullet은 없듯이 비판적인 관점에서 프래그래밍 패러다임을 바라보는 것도 중요하다.

# 절차 지향

![image](https://user-images.githubusercontent.com/79494088/173485631-b4ec494a-4bd6-4716-8a6c-35e8de6f6bca.png)

## 개념
- **절차지향(Procedure Oriented) 프로그래밍은 프로시저 콜(Procedure Call), 즉 함수 호출을 중심으로 프로그래밍을 생각하는 것**이다.
- 재사용 가능한 코드들은 별도의 함수로 분리하고 함수 간의 호출로 하고자 하는 일을 수행한다.
- 이런 프로세스는 주로 "함수"와 "조건문", "루프문"을 활용하여 코드를 구성한다.
- 또한 절차지향 프로그래밍은 데이터를 중앙 집중식으로 관리한다.
- 즉 프로세스 로직과 데이터가 별도의 위치에 분리되어 있다.
- 프로세스 로직 부분에서는 어떤 데이터가 들어오는지 모르기 때문에 로직 내에 조건문 사용이 많은 경향이 있다.

## 예시
- 절차지향 프로그래밍 관점으로 코드를 짜는 간단한 파이썬 코드 예시를 살펴볼 것이다.
- 사용자로부터 파일을 입력받아 파일을 파싱한 후, 이 내용을 저장소에 저장하는 코드이다.

```py
def read_input_file(file_path: str) -> str:
    if file_path.endswith(".txt"):
        reader = get_file_reader(file_type="txt")
        return reader.read(file_path)
    elif file_path.endswith(".csv"):
        reader = get_file_reader(file_type="csv")
        return reader.read(file_path)
    elif file_path.endswith(".xlsx"):
        reader = get_file_reader(file_type="xlsx")
        return reader.read(file_path)
    else:
        raise ValueError("파일 확장자는 txt, csv, xlsx 중 하나여야 합니다.")

def get_file_reader(file_type: str) -> Reader:
    if file_type == "txt":
   		...
    elif file_type == "csv":
        ...
    elif file_type == "xlsx":
        ...

def parse_input_data(data: str) -> List[str]:
    ...

def save_data(data: List[str]) -> None:
    ...

def main() -> None:
    data = read_input_file("input_file.txt")
    parsed_data = parse_input_data(data)
    save_data(parsed_data)
    
if __name__ == "__main__":
    main()
```

- `main` 함수는 하위 함수들의 호출로 이루어져 있다.
- `main` 함수만 보면 이 프로그램이 전체적으로 어떤 일을 하는지 쉽게 파악할 수 있다.
- 절차지향 코드는 이렇게 전체 로직을 작은 함수 단위로 나누고 프로그래밍 메인 로직이 시작되는 곳부터 하위 로직이 실행되는 곳 까지 TOP -> DOWN 식으로 구성되곤 한다.

## 장단점
- 절차지향 프로그래밍으로 작성된 코드는 일반적으로 이해하기 쉽다. TOP -> DOWN 식이고, 함수라는 작은 단위로 나눠져 있기 때문이다.
- 로직이 복잡한 것이나 계속해서 기능을 확장해나가야 하는 것이 아니라면, 유지보수도 용이하다.
- 보통 프로그래밍을 처음 배우면 보통 절차지향으로 배우는 이유이기도 하다.
- 그러나 절차지향은 전체 로직이 매우 복잡하거나 동적으로 로직을 바꿔야 하는 등의 기능 확장이 필요할 때 유지 보수하기가 어려워진다.
- 또한 데이터와 함수가 분리되어 있기에 함수가 많아질수록 데이터의 변경 사항을 추적하기도 어려워진다.
- 따라서 **절차지향은 프로그램이 수행하는 알고리즘이 명확하고, 기능 확장 등이 자주 일어나지 않는 상황에서 사용하기에 좋다.**