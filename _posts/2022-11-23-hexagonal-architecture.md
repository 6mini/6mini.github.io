---
title: '[소프트웨어 아키텍처 패턴] 헥사고날(hexagonal) 아키텍처'
description: "[더 나은 설계를 위한 소프트웨어 아키텍처 기초와 패턴] 애플리케이션과 바깥의 모듈들을 자유롭게 탈착 가능하게 하는 헥사고날(hexagonal) 아키텍처"
categories:
 - Software Architecture Pattern
tags: [소프트웨어 아키텍처 패턴, 헥사고날 아키텍처]
---

- 애플리케이션은 데이터베이스에 있는 데이터를 그저 옮겨주는 수동적인 소프트웨어를 넘어 보다 적극적이고, 여러 일을 할 수 있어야한다.
- 데이터베이스 같은 외부 시스템이 애플리케이션의 중심이 아니라, 애플리케이션이 "사용하는" 일부가 되어야 한다.
- 상황에 따라 RDBMS에서 NoSQL을 쓸 수도 있고, API 서버와 통신 방식을 Rest에서 GraphQL로 바꿀 수 있어야 한다.

- 이렇듯 애플리케이션을 중심으로 보고, DB, 웹 프레임워크 등은 모두 애플리케이션이 사용하는 부품(언제든 갈아끼울 수 있는)으로 보는 아키텍처들이 주목받기 시작했다.
- 대표적인 아키텍처로 헥사고날 아키텍처와 클린 아키텍처가 있다.

# 개념

![image](https://user-images.githubusercontent.com/79494088/175447795-39614360-985c-4ef3-b3f5-0427cf35de5e.png)

- 헥사고날(포트 앤 어댑터) 아키텍처는 흔히 육각형의 이미지로 소개되는데, 애플리케이션과 바깥의 모듈들을 자유롭게 탈착 가능하게 하는 것이 골자이다.

- "탈착 가능하다"의 개념은 플레이스테이션 같은 게임기를 생각해보면 더 쉽게 이해가 된다.
- 게임기에서 가장 중요한 부분은 게임을 실행시키는 본체이다.
- 게임기에 연결된 입력 패드나, 게임 화면을 보여주는 모니터는 게임기 본체의 포트와 맞는 규격이면 언제든지 사용자 취향에 따라 바뀔 수 있다.
- 따라서 중요한 것은 게임기 본체 그 자체이고, 그 외부적인 것들은 언제든지 탈부착이 가능하다.

- 헥사고날 아키텍처의 핵심도 바로 이와 같다.
- 가운데 애플리케이션을 중심으로 애플리케이션 외의 모듈은 애플리케이션에서 제공하는 포트 모양에 맞다면 언제든 바꿀 수 있도록 하는 것이다.

- 헥사고날 아키텍처는 어댑터가 포트 모양만 맞으면 동작하는 것 같다고 해서 포트 앤 어댑터(Ports-and-Adapters)라고 부르기도 한다.
- 즉 포트만 맞으면, 어떤 어댑터든 이 포트에 끼울 수 있다.

## 구조
- 이제 헥사고날 아키텍처의 구성 요소를 다음처럼 정리해볼 수 있다.

### 도메인
- 레이어드 아키텍처에서의 도메인 레이어 개념과 같다.
- 애플리케이션의 핵심이 되는 도메인을 표현한다.
### 애플리케이션
- 도메인을 이용한 애플리케이션의 비즈니스 로직을 제공한다.
- 관용적으로 `Service`라고 표현하곤 한다.
- 애플리케이션은 포트를 가지고 있다.
  - 포트는 외부 어댑터를 끼울 수 있는 인터페이스이다.
  - 위 플레이스테이션 예시에서 게임기 본체가 애플리케이션이고, 본체에 있는 입출력, 혹은 모니터 단자를 끼울 수 있는 포트가 여기서 말하는 포트라고 이해하면 쉽다.
  - 애플리케이션으로 흐름이 들어오는 포트는 인바운드 포트, 애플리케이션에서 흐름이 나가는 포트는 아웃바운드 포트라고 한다.
  - 포트는 프로그래밍 문법에서 인터페이스로 구현할 수 있다.
### 어댑터
- 애플리케이션 내에 있는 포트에 끼울 수 있는 구현체이다.
- web이나 cli 등은 인바운드 포트에 끼울 수 있는 인바운드 어댑터에서 이용한다.
- db 등은 아웃바운드 포트에 끼울 수 있는 아웃바운드 어댑터를 통해 이용한다.
- 어댑터는 보통 포트를 나타내는 인터페이스를 상속받아 구현한다.

- 헥사고날 아키텍처에서 레이어 의존성이 다음처럼 흐른다.
  - 어댑터 -> 애플리케이션 -> 도메인
- 위의 의존성 흐름을 역행하면 안된다.
- 예를 들면, 비즈니스 로직 -> 어댑터로, 도메인 -> 어댑터로 흐르는 의존성이 없어야한다.

# 예시
- 헥사고날 아키텍처를 프로젝트 구조로 표현하면 다음과 같다.

```
src/
  adapter/
    inbound/
      api/
        product_controller.py
        user_controller.py
        ...
    outbound/
      repositories/
        product_repository.py
  	    user_repository.py
  application/
    service/
      product_service.py
      user_service.py
    port/
      inbound/
        product_port.py
        user_port.py
      outbound/
        product_repository.py
        user_repository.py
  domain/
    product.py
    user.py
```

- 먼저 프로젝트 최상단에서 크게 어댑터, 애플리케이션, 도메인으로 레이어를 디렉토리로 나눈다.
- 그리고 각 디렉토리 내에 해당 레이어에 포함되는 컴포넌트들을 배치한다.
- 레이어드 아키텍처에서는 다음처럼 서비스 레이어에 있는 모듈이 인프라스트럭쳐에 있는 모듈을 사용할 수 있었다.

## as-is

```py
# src/application_layer/product_service.py

from src.domain_layer import product
from src.infrastructure_layer.database import db
from src.infrastructure_layer.repositories import product_repository

def create_product(name: str, price: str) -> bool:
    ...
    product_repository = product_repository.ProductRepository(session)
    ...
```

- 그러나 헥사고날 아키텍처에서는 이런 흐름은 금지되므로, 다음처럼 코드를 수정해야 한다.

## to-be

```py
# src/application/service/product_service.py

from src.domain import product
from src.applicaiton.port.outbound.product_repository import ProductRepository  # 이 부분이 수정됨
# 이제 애플리케이션 레이어는 인프라스트럭쳐 레이어가 아닌 애플리케이션 레이어에 의존

def create_product(name: str, price: str, product_repository: ProductRepository) -> bool:
    ...
    product_repository.create(...)
    ...
```

- 포트 앤 어댑터의 의존성 원칙은 저수준이 아닌 고수준에 의존하라는 의존성 역전 원칙과도 동일 선상이라고 볼 수 있다.
- 보통 컴파일 의존성(코드 의존성)에서는 이렇게 고수준을 의존하게 한 후, 런타임에서 의존성을 주입해준다.(의존성 주입 프레임워크를 많이 활용함)

- 포트는 다음처럼 인터페이스(추상 클래스)로 구현한다.

```py
# src/application/port/outbound/product_repository.py

from abc import ABC, abstractmethod
from src.domain.product import Product

class ProductRepository(ABC):
    @abstractmethod
    def save(product: Product) -> None:
        pass
```

- 그리고 이 포트를 구현한 클래스는 `src/adapter/outbound/repositories/product_repository.py`에 구현한다.

- 도메인도 마찬가지로 레이어드 아키텍처에서는 인프라스트럭쳐 레이어의 컴포넌트에 의존했다.

## as-is

```py
# src/domain_layer/product.py

from sqlalchemy import Column, String, Integer
# DB와 연결하는 일은 인프라스트럭처 레이어에서의 일
from src.infrastructure_layer.database import Base  

# 도메인 레이어의 컴포넌트(Product)는 인프라스트럭쳐 레이어의 컴포넌트(Base)에 의존
class Product(Base):
    __tablename__ = 'product'
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    price = Column(Integer)
```

- 이 코드도 이제 도메인이 인프라스트럭쳐에 의존하지 않게 다음처럼 변경할 수 있다.

## to-be

```py
# src/domain/product.py

from dataclasses import dataclass

@dataclass
class Product:
    id: int
    name: str
    price: int
```

- 어댑터 코드의 경우는 외부 서비스와 연결해주는 인터페이스의 역할을 해주는 코드를 작성해주면 된다.
- 애플리케이션의 Port(고수준)를 구현한 저수준의 코드가 포함된다.

```py
# src/adapter/outbound/product_repository.py
...
from src.application.port.outbound.product_repository import ProductRepository

class MysqlProductRepository(ProductRepository):
    ...

    def save(self, name: str, price: int):
        product = Product(name, price)
        with self.db.Session() as session:
            ...
            session.commit()
        ...
```

# 나가며
- 헥사고날 아키텍처를 통해 프로젝트의 중요한 부분(애플리케이션과 도메인)과 덜 중요한 부분(어댑터)을 구분하고, 의존성의 방향을 중요한 것으로 흐르게 해서, 덜 중요한 부분은 언제든 바꿀 수 있도록 유연하게 코드를 설계했다.
- 이를 통해 인프라스트럭처 중심의 설계를 하지 않아도 되며, 코드 확장에 대해서도 더 열려있게 되었다.
- 다만 이전보다 어댑터, 포트 등 알아야 할 개념과 보일러 플레이트 코드가 늘어날 수 있다는 단점이 있다.