---
title: '[객체 지향] 객체 지향의 특성'
description: "[객체 지향 프로그래밍] 파이썬 코드를 통해 이해하는 객체 지향 특성. 책임과 협력, 추상화와 다형성, 캡슐화 등의 이해와 객체 지향적으로 소프트웨어를 설계하는 방법"
categories:
 - Object Oriented Programming
tags: [객체 지향 프로그래밍, 추상화, 다형성, 캡슐화]
---

# 책임과 협력

## 책임(Responsibility)
- **책임은 한 객체가 특정하게 수행해야 하는 범위와 기능**을 말한다.
- 사람이 "모닝" 차량을 운전해야 하는 상황을 프로그래밍해야 한다고 가정해본다.
- 먼저 사람이라는 객체를 정의한다.
- 사람은 "운전"이라는 행위를 하게 된다.
- 즉 "사람" 객체는 "운전하는 것"에 대한 "책임"이 있다.

```py
class User:
    def __init__(self) -> None:
        pass

    def drive(self) -> None:
        pass
```
        
- 이번에는 모닝 차량에 대한 객체를 만든다.
- 차량은 출발하고 멈추는 기능을 제공하는 게 일반적이다.
- 따라서 모닝 차량은 "가속"과 "감속"에 대한 책임이 있다고 말할 수 있다.

```py
class MorningCar:
    def __init__(self) -> None:
        pass

    def accelerate(self) -> None:
        pass

    def decelerate(self) -> None:
        pass
```

### SRP(Single Responsibility Principle)
- 하나의 객체는 하나의 책임만 가지도록 설계하는 것이 일반적으로 좋다고 알려져 있다.
- 객체의 정체성이 명확하고, 변경에 용이하며, 추후 재사용 가능하고, 높은 응집도와 낮은 결합도를 유지할 수 있기 때문이다.
<br>이를 SRP(Single Responsibility Principle)라고 한다.

## 협력
- 위에서 정의한 기능을 실제로 구현해본다.
- 먼저 `User`는 구체적으로 다음처럼 구현해볼 수 있다.

```py
class User:
    def __init__(self) -> None:
        self.car = MorningCar()  # User는 MorningCar를 의존

    def drive(self) -> None:
        self.car.accelerate()  # User는 MorningCar가 제공하는 공개 메서드를 사용
```

- 그리고 `MorningCar`는 다음처럼 구현할 수 있다.

```py
class MorningCar:
    def __init__(self):
         self.speed = 0 
         self._fuel = 0  # 파이썬 특성상 private을 _ prefix를 붙여 암묵적으로 사용
						
    def accelerate(self):
        self.speed += 1

    def decelerate(self):
        self.speed -= 1
```

- "사람이 운전하는 상황"은 결과적으로 `User` 객체와 `MorningCar` 객체를 각자 책임에 맞게 설계하고, `User`가 `MorningCar` 객체를 호출하여 구현해냈다.
- 이렇게 객체가 서로 필요에 따라 의존하는 것을 "협력"이라고 표현한다.

### 책임 주도 설계
- **객체 지향에서 상황에 필요한 객체들의 책임을 중심으로 시스템을 설계해나가는 방법을 책임 주도 설계**라고 한다.
- 즉, 하나의 책임이 곧 하나의 객체가 되고, 책임이 곧 객체의 정체성이 되는 것이다.
- 책임주도 설계로 시스템을 설계해나가면, 객체들의 정체성이 명확해진다.
- 그리고 높은 응집도와 낮은 결합도를 유지하며, 시스템은 객체들의 협력으로 로직을 진행하게 된다.


# 추상화
## 추상화가 없을 때
- 위의 예시에서, 사람이 만약 '모닝' 차량 말고 '포르쉐' 차량도 운전하고 싶다.
<br>그러면 다음처럼 코드를 수정해볼 수 있다.

```py
class PorscheCar:
    def __init__(self):
        self.speed = 0 
        self._fuel = 0
						
    def accelerate(self):
        self.speed += 3

    def decelerate(self):
        self.speed -= 3
		

class User:
    def __init__(self, car_name: str) -> None:
        if car_name == "morning":
            self.car = MorningCar()
        elif car_name == "porsche":
            self.car = PorscheCar()

    def drive(self):
        self.car.accelerate()
```

- 그런데 만약 이후에 '포르쉐' 말고도 '벤츠', 'BMW' 등 더 다양한 차를 운전하고 싶다면 어떻게 해야 할까?
- 새로운 차량이 추가될 때마다 `__init__()` 함수 안에 `if .. else` 문이 추가될 것이다.

## 추상화가 있을 때
- 사실 사람은 '특정 차'를 운전하는 게 아니라 그냥 '차'라는 개념 자체를 운전하는 것으로 생각해볼 수 있다.
- '모닝', '포르쉐', 'BMW' 등등.. 모두 공통적으로 '차'다.
- '차'는 모두 가속과 감속 기능을 제공하기 때문이다.
- 이렇게 구체적인 객체(물체)들로부터 공통점을 생각하여 한 차원 높은 개념을 만들어내는(생각해내는) 것을 추상화(abstraction)라고 한다.
- 이제 우리는 '차' 라는 객체를 다음처럼 추상 클래스로 구현할 수 있다.

```py
from abc import ABC

class Car(ABC):
    pass
```

- '차'는 공통으로 가속, 감속 기능을 제공하므로 다음처럼 추상 메서드도 추가할 수 있다.

```py
from abc import ABC

class Car(ABC):
    @abstractmethod
    def accelerate(self) -> None:
        pass

    @abstractmethod
    def decelerate(self) -> None:
        pass
```

- '차'라는 추상화된 역할의 구현체인 '모닝'과 '포르쉐' 차량은 해당 추상 클래스를 상속받아 구현할 수 있다.

```py
class MorningCar(Car):
    def __init__(self) -> None:
         self.speed = 0 
         self._fuel = 0  
						
    def accelerate(self) -> None:
        self.speed += 1

    def decelerate(self) -> None:
        self.speed -= 1

class PorscheCar(Car):
    def __init__(self) -> None:
        self.speed = 0 
        self._fuel = 0
						
    def accelerate(self) -> None:
        self.speed += 3

    def decelerate(self) -> None:
        self.speed -= 3
```

- 이제 사람 입장에서는 구체적인 차량이 아닌 추상적인 차라는 객체만 알면 된다.

```py
class User:
    def __init__(self, car: Car) -> None:
        self.car = car

    def drive(self):
        self.car.accelerate()
```

# 다형성
- 이제 사람은 다음처럼 상황에 따라 운전하고 싶은 차량을 설정할 수 있다.

```py
# 모닝 차량을 운전하고 싶을 때
user = User(MorningCar())
user.drive()

# 포르쉐 차량을 운전하고 싶을 때
user = User(PorscheCar())
user.drive()
```

- 현재 `User`는 생성자에서 추상 클래스인 `Car`를 의존하고 있지만, 실제로 이 User를 사용하는 클라이언트에서는 `Car`가 아니라 `Car`의 자식 클래스인 `MorningCar`와 `PorscheCar` 객체를 생성자에서 파라미터로 넘겨줄 수 있다.
- 이처럼 `User` 입장에서 `Car`가 상황에 따라 그 형태가 달라질 수 있는데, 이런 특성을 다형성(polymorphism)이라고 한다.
<br>또한, 이렇게 외부에서 실제로 의존하는 객체를 만들어 넘겨주는 패턴을 의존성 주입(dependency injection)이라고 부른다.
- 다형성은 객체 지향의 꽃이라 불릴 만큼 중요한 특성이다.
- 추상 클래스 혹은 인터페이스로 객체를 상위 타입으로 추상화하고(위 예에서는 "차"가 바로 이런 상위 타입), 그 객체의 하위 타입들(모닝, 포르쉐 등)은 이러한 상위 타입의 추상 클래스나 인터페이스를 구현하도록 하면, 코드 설계가 전반적으로 유연해져 수정과 확장이 매우 용이해진다.

## OCP(Open-Close Principle)
- OCP(Open-Closed Principle)는 "소프트웨어는 확장에 대해 열려 있어야 하고, 수정에 대해서는 닫혀 있어야 한다는 원칙"이다.
- 쉽게 말해, 요구사항이 바뀌어 기존 코드를 변경해야 할 때, 기존 코드를 수정하지 않고 새로운 코드를 추가하는 것이 좋다는 것이다.


# 캡슐화
- 캡슐화(encapsulation)는 객체 내부의 데이터나 메서드의 구체적인 로직을 외부에서 모르고 사용해도 문제가 없도록 하는 특성이다.

## 캡슐화하지 않을 때

```py
class MorningCar:
    def __init__(self, fuel: int) -> None:
        self.speed = 0
        self.current_fuel = fuel
        self.max_fuel = 50
        if fuel > self.max_fuel:
            raise Exception(f"최대로 넣을 수 있는 기름은 {max_fuel}L 입니다")
						
    def accelerate(self) -> None:
        if self.current_fuel == 0:
            raise Exception("남아있는 기름이 없습니다")
        self.speed += 1
        self._current_fuel -= 1

    def decelerate(self) -> None:
        if self.current_fuel == 0:
            raise Exception("남아있는 기름이 없습니다")
        self.speed -= 1
        self._current_fuel -= 1
```

- 이제 MorningCar 객체를 다음처럼 사용한다.

```py
# 차량 생성
car = MorningCar(30)

# 차량 주행
car.accelerate()

# 차량에 필요한 기름 주유
car.current_fuel += 50

# 차량의 남은 주유랑을 퍼센트로 확인
f"{car.current_fuel / car.max_fuel * 100} %"
```

- 이때 위 코드의 차량에 필요한 기름을 주유하기 위해 `car.fuel`에 직접 접근하여 `+` 연산을 하고있다.
<br>또한, 차량의 남은 주유량을 퍼센트로 확인하기 위해 `car_max_fuel`에도 접근하여 `/` 연산을 하고있다.
- 현재 `MorningCar`는 캡슐화를 지키지 않은 객체이다.
<br>왜냐하면 `MorningCar`를 사용하는 쪽에서 주유를 하기 위해 `car.current_fuel`에 직접 `+` 연산을 하고 있고, 남은 주유량 확인을 위해 직접 필요한 연산을 모두 하고 있기 때문이다.
- 특히 `car.current_fuel`에 직접 연산을 하게 되면, 자칫 `car.max_fuel`을 초과한 값이 `car.current_fuel`에 들어갈 수 있으므로, 이는 버그에 취약한 코드이다.
<br>이렇게 캡슐화되지 않은 코드는 본인의 책임을 다하고 있지도 않으며(SRP 위반), 추후 코드 변화에도 매우 취약하다.
- 만약 `MorningCar` 객체의 `fuel` 변수의 이름이 다른 이름으로 바뀌게 되면 `MorningCar`를 사용하는 모든 코드를 수정해야 한다.

### `Getter`, `Setter와` 캡슐화
- 보통 객체 지향은 Java로 처음 배우곤 하는데, 이때 캡슐화를 공부하는 과정에서 `Getter/Setter` 메서드를 접하는 경우가 많다.
- 객체의 인스턴스 변수는 모두 `private`으로 두고, 이를 `Getter/Setter` 메서드로 접근하라고 배우게 된다.
<br>그리고 "캡슐화 = `Getter/Setter` 메서드 추가해주기"로 생각하게 되기도 한다.
- 정확히 말하면, `Getter/Setter` 메서드는 캡슐화를 돕는 방법 중 하나이지 캡슐화 그 자체가 아니다.(더욱이 항상 모든 인스턴스 변수에 대해 `Getter/Setter`를 다는 것은 좋지 않다.)
- 캡슐화는 객체가 알고 있는 것을 외부에 알리지 않는 것이며, 필요한 경우에 한해 객체 내부가 알고있는 것의 일부를 `Getter` 메서드로 제공하거나, `Setter` 메서드로 변경하게 하는 기능을 제공하는 것이다.
- 캡슐화는 `Getter/Setter` 메서드의 존재 여부로 결정되는 것이 아니라, 객체가 외부에서 알아도 되지 않을 내용을 잘 숨기고, 알아야 할 내용이나 제공하는 행동들을 필요에 맞게 제공하는 행동을 이야기한다.
- 캡슐화가 잘된 객체는 사용하는 입장에서도 편하다.

## 캡슐화할 때
- 이제 `MorningCar`를 캡슐화 해본다.
- 우선 외부에서 이 객체에 필요로 하는 기능을 메서드로 제공하고, 객체의 속성을 직접 수정하거나 가져다 쓰지 않도록 해야한다.
- 객체 밖에서는 이러한 정보나 로직을 모르기 때문에, 이를 정보은닉이라고도 부른다.

```py
class MorningCar:
    def __init__(self, fuel: int) -> None:
        self.speed = 0
        self._current_fuel = fuel  # 비공개형 변수로 변경
        self._max_fuel = 50  # 비공개형 변수로 변경
        if fuel > self._max_fuel:
            raise Exception(f"최대로 넣을 수 있는 기름은 {max_fuel}L 입니다!")
						
    def accelerate(self) -> None:
        if self._current_fuel == 0:
            raise Exception("남아있는 기름이 없습니다!")
        self.speed += 1
        self._current_fuel -= 1

    def decelerate(self) -> None:
        if self._current_fuel == 0:
            raise Exception("남아있는 기름이 없습니다!")
        self.speed -= 1
        self._current_fuel -= 1
        
    # 차량에 주유할 수 있는 기능을 메서드로 제공하고, 구체적인 로직은 객체 외부에서 몰라도 되도록 메서드 내에 둔다.
    def refuel(self, fuel: int) -> None:
        if self._current_fuel + fuel > self._max_fuel:
            raise Exception(f"추가로 더 넣을 수 있는 최대 연료는 {self._max_fuel - self._current_fuel}L 입니다!")
        self._current_fuel += fuel
        
    # 차량의 남은 주유랑을 퍼센트로 확인하는 기능을 메서드로 제공한다.
    def get_current_fuel_percentage(self) -> str:
        return f"{self._current_fuel / self._max_fuel * 100} %" 
```

- 이제 `MorningCar` 객체를 다음처럼 사용할 수 있다.

```py
# 차량 생성
car = MorningCar(30)

# 차량 주행
car.accelerate()

# 차량에 필요한 기름 주유(예외 발생)
car.refuel(50)

# 차량의 남은 주유랑을 퍼센트로 확인
car.get_current_fuel_percentage()
```

- 최종적으로 `MorningCar`를 사용하는 클라이언트는 `MorningCar`의 속성에 직접 접근하여 연산할 필요가 없다.
<br>그저 `MorningCar`가 제공하는 퍼블릭 메서드를 사용하면 된다.
- 또한 `MorningCar`의 속성 `fuel` 이나 `max_fuel` 등의 값이 바뀌어도 클라이언트의 코드는 바뀌지 않는다.
- 따라서 수정에도 더 용이한 코드가 된다.

# 정리
- 객체 지향은 객체들의 책임과 협력으로 이루어진다.
- 추상화를 통해 객체들의 공통 개념을 뽑아내어 한 차원 더 높은 객체를 만들 수 있다.
- 객체를 사용하는 입장에서 과연 어떤 역할을 할 객체가 필요한지 생각해보고 추상회된 객체를 생각하면 된다.
- 객체를 추상화한 클래스를 만든 후, 이 클래스를 상속받아 실제 구체적인 책임을 담당하는 객체를 만들 수 있다.
- 다형성으로 코드는 수정과 확장에 유연해진다.
- 캡슐화를 통해 객체 내부의 정보와 구체적인 로직을 외부에 숨길 수 있다.
- 외부에선 그저 객체가 제공하는 공개 메서드를 사용하면 된다.
- 캡슐화로 코드는 수정과 확장에도 유연해진다.