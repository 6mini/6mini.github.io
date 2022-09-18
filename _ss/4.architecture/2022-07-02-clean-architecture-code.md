---
title: '[소프트웨어 아키텍처 패턴] 코드로 알아보는 클린 아키텍처'
description: "[더 나은 설계를 위한 소프트웨어 아키텍처 기초와 패턴] 간단하게 유저를 생성하는 로직을 구현한 서버 코드로 알아보는 클린 아키텍처"
categories:
 - Software Architecture Pattern
tags: [소프트웨어 아키텍처 패턴]
---

- 클린 아키텍처를 적용하여 간단하게 유저를 생성하는 로직을 구현한 서버 코드를 살펴볼 것이다.

# 프로젝트 구조

```
domain/
  entities.py
application/
  use_cases/
    create_user.py
  interfaces/
    user_repository.py
interface_adapter/
  controller/
    create_user.py
framework_and_drvier/
  server.py
  db/
    user_repository.py
    orm.py 
```

## 중요
1. 네이밍을 꼭 레이어 이름대로 할 필요는 없다.
  - 좀 더 명확한 네이밍이나, 팀에서 협의가 된 네이밍 규칙이 있다면 그것을 쓰면 된다.
2. 완벽한 + 똑같은 아키텍처는 존재하지 않는다.
  - 상황에 따라 레이어 개수나 레이어별 의미는 달라질 수 있다.
  - 중요한 것은 레어어를 잘 나눌 수 있도록 경계를 설정하고 의존 흐름을 바깥에서 안쪽으로 가져가는 것이다.

# 의존성 다이어그램

![image](https://user-images.githubusercontent.com/79494088/175450729-e5efc8bb-0a11-4212-9492-87a7399f095b.png)

# 엔티티
- 먼저 다음처럼 엔티티를 정의한다.
- 엔티티는 도메인에 핵심을 표현하는 객체이다.

```py
# domain/entities.py

@dataclass
class User:
    id: str
    name: str
    password: str
```

# 유즈 케이스
- 유즈 케이스는 애플리케이션의 주요 정책과 비즈니스 로직이 들어있는 계층이다. 
- 우리는 "유저 생성하기" 관련 비즈니스 로직을 작성하고 있다.

```py
# application/use_cases/create_user.py
from domain.entities import User
from application.interfaces.user_repository import UserRepository

@dataclass
class CreateUserInputDto:
    user_name: str
    user_password: str

        
@dataclass
class CreateUserOutputDto:
    user_id: str
        
        
class CreateUser:
    def __init__(self, user_repository: UserRepository) -> None:
        # 의존성 역전을 위해 같은 레이어(applicaiton)에 있는 추상화된 UserRepository에 의존한다.
        # 다시 말해, 인프라스트럭쳐에 정의될 구체적인 UserRepositoryImpl 객체에 의존하지 않는다.
        # 실제 런타임에서는 UserRepository를 상속받은 세부 클래스를 주입해야 한다.
        # 세부 클래스는 인프라스트럭처 레이어에 정의되며, 이는 의존성 주입하는 부분에서 주입된다.
        self._user_repository = user_repository
    
    def execute(self, input_dto: CreateUserInputDto) -> CreateUserOutputDto:
        user_id = self._user_repository.get_next_user_id()
        user = User(id=user_id, name=input_dto.user_name, paassword=input_dto.user_password)
        self._user_repository.save(user)
        return CreateUserOutputDto(user_id=user_id)
```
## DTO(Data Transfer Object)란?
- DTO는 데이터를 주고받기 위해 사용하는 객체이다.
- 보통 레이어간 의존성을 끊고, 도메인 모델을 보호하기 위해서 유즈 케이스의 입출력으로 DTO를 사용한다.

```py
# application/interfaces/user_repository.py

class UserRepository(ABC):
    @abstractmethod
    def save(user: User) -> None:
        pass
```

# 인터페이스 어댑터
- 인터페이스 어댑터는 외부 영역(외부 DB, 웹 서버 등)과 내부 영역(유즈 케이스)의 인터페이스를 변환해주는 역할을 한다.
- 예를 들어 API 요청이 외부에서 들어왔을 때 유즈 케이스 입력으로 변환하여 유즈 케이스를 실행한 후 출력을 JSON 데이터로 내보낸다.
- 일반적으로 웹, API 서버에서 컨트롤러 객체가 바로 이 인터페이스 어댑터에 해당된다.

```py
# interface_adapter/controller/create_user.py

from application.use_cases.create_user import CreateUser, CreateUserInputDto
from framework_and_driver.repository.userRepository import UserRepositoryImpl
...


class CreateUserJSONRequest(BaseModel):
    name: str
    password: str

class CreateUserJSONResponse(BaseModel):
    user_id: str

def create_user(json_request: CreateUserJSONRequest) -> CreateUserJSONResponse:
    # 엄밀하게 보면 framework를 의존하고 있기에 위배된다. 보통 의존성 주입(DI) 프레임워크를 사용하거나 별도의 Factory를 둔다.
    user_repository = UserRepositoryImpl(...)    
    use_case = CreateUser(user_repository=user_repository)
    input_dto = CreateUserInputDto(user_name=json_request.name, user_password=json_request.password)
    output_dto = use_case.execute(input_dto)
    return CreateUserJSONResponse(user_id=output_dto.user_id)
```

# 프레임워크 & 드라이버
- 프레임워크 & 드라이버에는 웹서버나 외부 데이터베이스 등 구체적으로 사용하는 세부 기술들이 놓이게 된다.
- 웹 서버를 실행하는 프레임워크나 외부 데이터베이스와 직접적으로 통신하는 ORM 등의 설정 파일이 포함된다.

```py
# framework_and_drvier/server.py
from interface_adapter.controller.create_user import create_user
...

app = FastAPI()

app.add_api_route(
    path="/users", endpoint=create_user, methods=["POST"], status_code=201
)

if __name__ == "__main__":
    uvicorn.run(app)
```

- 또한 외부 데이터베이스에서 데이터를 받아온 후 유즈 케이스에 맞게 처리하는 레포지토리도 이에 해당된다.

```py
# framework_and_drvier/db/user_repository.py
from application.interfaces.user_repository import UserRepository
from domain.entities import User
...

class UserRepositoryImpl(UserRepository): 
    def __init__(self, session_factory: Callable[..., AbstractContextManager[Session]]) -> None:
        self.session_factory = session_factory 
        
    def save(user: User) -> User:
        with self.session_factory() as session:
            ...
        return user
```

# 좋은 아키텍처에 대한 고민
- 레이어드 아키텍처부터, 헥사고날 그리고 클린 아키텍처까지 알아보았다.
- 사실 아키텍처에 정답은 없다. 아키텍처는 아키텍처 자체로 남는 것이 아니라, 실제로 개발자들이 쉽고 지속적인 개발을 위해 존재한다.
- 아무리 유명한 아키텍처라고 하더라도, 당장 상황에 맞지 않은 아키텍처는 좋은 아키텍처가 아니다.

- 다만 "쉽고 지속적이며 생산성을 높이는 아키텍처"를 고려해본다면 아래를 떠올릴 수 있을 것 같다.
  - 프로젝트, 아키텍처 구조만 보고도 애플리케이션을 쉽게 파악할 수 있는가?
  - 추가 확장 및 수정사항에 용이한 구조를 가지고 있는가?
  - 개발자가 어떤 모듈을 어디에 두어야 할지에 대한 고민을 줄여주는가?
  - 쉽게 테스트가 가능한가?
- 이런 질문에 해답을 내놓을 수 있는 아키텍처라면, 개발 생산성에도 도움을 줄 수 있는 좋은 아키텍처라고 할 수 있다.