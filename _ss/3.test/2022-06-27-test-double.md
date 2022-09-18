---
title: '[테스트 코드] 의존성을 대체하는 테스트 더블 코드'
description: "[견고한 서비스를 위한 테스트 코드 작성] 테스트할 로직에서 의존하고 있는 객체를 대체해주는 테스트 더블 코드 작성 방법"
categories:
 - Test Code
tags: [테스트 코드]
---

# 기존 테스트의 문제
- 테스트 더블은 테스트할 로직에서 의존하고 있는 객체를 대체해주는 객체이다.
- 앞 포스팅의 예시 코드 중 `login()` 함수 내 로직을 다시 본다.

```py
def login(user_id: str, user_password: str) -> str:
    user_repository = UserRepository()  # DB와 연동되어 User 정보를 저장하고 불러오는 객체
    user = user_repository.find_by_id(user_id)
    if user.id == user_id and user.password == user_password:
        return create_token(user_id)
    else:
        raise Exception("로그인 인증에 실패했습니다.")
```
- 위 코드는 `UserRepository` 객체를 의존하고 있다.
- `UserRepository` 객체는 DB와 연결을 맺어 데이터를 저장하고, 불러오는 객체로 DB가 먼저 실행된 상태여야 정상적으로 작동한다.
- 즉, `UserRepository` 객체는 외부 DB에 의존성이 있다.
- 따라서 위 `login()` 함수를 정상적으로 테스트하려면 DB가 어딘가에 실행된 상태여야 하고, `UserRepository` 역시 문제없이 잘 작동하는 상태여야 한다.
- 이처럼 로직이 다른 객체들과 외부 컴포넌트(DB 등)을 의존하게 되면 테스트를 실행하는데 신경 써야 할 것들이 생기게 된다.
- 단적으로 DB가 어딘가에서 실행되어 있지 않으면 작성했던 통합 테스트 코드를 정상적으로 실행시킬 수 없다.

# 테스트 더블 적용하기
- 테스트 더블은 이런 의존성 객체들을 "대체"함으로써 테스트를 좀 더 원활하게 진행하기 위한 객체이다.
- 예를 들어 위에서 `UserRepository` 객체는 테스트 코드에서 다음과 같은 `FakeRepository` 라는 페이크 객체로 대체할 수 있다.

```py
class FakeRepository(Repository):
    """ DB를 이용하지 않고, 인메모리로 데이터를 저장하고 불러낸다."""
    
    def __init__(self, data: Dict[str, User]) -> None:
        self._data = data
        
    def find_by_id(id: str) -> Optional[User]:
        return self._data.get(id, None)
```

- `login()` 함수를 좀 더 테스트하기 쉽게 만들기 위해, 의존하는 객체를 함수 내부에서 직접 생성하지 않고, 외부에서 파라미터로 주입받도록 수정한다.

```py
def login(user_id: str, user_password: str, repository: Repository) -> str:  # repository 파라미터를 추가한다.
    user = repository.find_by_id(user_id)
    if user_id == user.id and user.password == user_password:
        return create_token(user_id)
    else:
        raise Exception("로그인 인증에 실패했습니다.")
```

- 이제 테스트 코드는 다음처럼 `FakeRepository`를 이용하여 작성할 수 있다.

```py
def test_login_successful():
    # given
    repository = FakeRepository(data={  # 테스트 더블 객체를 만든다.
        "6mini": {
            "id": "6mini",
            "password": "1234"
        }
    })
    user_id = "6mini"
    user_password = "1234"
    
    # when
    actual = login(user_id, user_password, repository)  # 테스트 더블 객체를 주입한다.
    
    # then
    assert actual == "grab_verified"
```

![image](https://user-images.githubusercontent.com/79494088/174696928-324eba25-66b1-4449-ba33-7f44271ecd8b.png)


- 이제 테스트 코드는 DB에 대한 의존성이 없는 상태로 테스트가 가능하다.
- 위 예시 코드에서 우리가 사용한 테스트 더블은 `fake object`이다.

# 테스트 더블의 종류
- 위 테스트에서는 외부 의존성을 대체하기 위해 테스트 더블 중 하나인 페이크 객체로 구현했다.
- 테스트 더블은 이 외에도 대표적으로 다음과 같은 종류가 있다.

## dummy
- 실제 내부 동작은 구현하지 않은 채, 객체의 인터페이스만 구현한 테스트 더블 객체이다.
- 메서드가 동작하지 않아도 테스트에 문제가 없을 때 사용한다.

```py
class DummyRepository(Repository):
    def insert(self, data):
        return True
    
    def find_by_id(self, user_id):
        return "6mini"
```

## stub
- `dummy` 테스트 더블 객체에서 테스트에 필요한 최소한의 구현만 해둔 테스트 더블 객체이다.
- 테스트에서 호출될 요청에 대해 미리 준비해둔 결과만을 반환한다.

```py
class StubUserRepositry(Repository):
    def insert(self, data):
        return "OK"

    def findById(self, user_id):
        return {"id": user_id, "name": "test_grab", ...}

    ...
```

## spy
- `stub`에서 테스트에 필요한 정보를 기록해두는 테스트 더블 객체이다.
- 보통 `stub`의 역할을 포함한다.
- 실제로 내부가 잘 동작했는지 등을 별도의 인스턴스 변수로 기록해둔다.

```py
class SpyUserRepositry(Repository):
    insert_called=0
   
    def insert(self, data):
        SpyUserRepositry.insert_called += 1
        return "OK"
   
    @property
    def get_insert_called(self):
        return SpyUserRepositry.insert_called

    ...
```

## fake
- 동작의 구현은 갖추고 있지만, 테스트에서만 사용할 수 있는 테스트 더블 객체이다.
- 대체할 객체가 복잡한 내부 로직이나 외부 의존성이 있을 때 사용한다.

```py
class FakeUserRepository(Repository): 
    def __init__(self): 
        self.users = []

    def insert(self, data):
        self.users.append(data)

    def find_by_id(self, user_id):
        return [user for user in self.users if user.id == user_id]
```

## mock
- 테스트에 필요한 인터페이스와 반환 값을 제공해주는 객체이다.
- 해당 메서드가 제대로 호출됐는지를 확인하는 행위 검증의 기능을 가진다.
- 다른 테스트 더블과 다르게 보통은 객체를 직접 정의하지 않고, 보통 `Mock` 객체로 반환 값을 미리 지정해둔다.
- 대부분의 테스트 프레임워크는 `Mocking`을 정밀하게 할 수 있도록 지원해준다.

```py
@mock.patch.object(UserRepository, 'insert')
def test(insert_method):
    insert_method.return_value = "OK" # stub처럼 기대값을 반환한다.
    insert_method({"id": 1, "name": "6mini"}) 
    insert_method.assert_called_once() # 해당 메서드가 호출되었는 지를 확인한다.(행위 검증)

# 서드 파티 라이브러리에 mocking하는 사례를 추가했다.
@patch("requests.get")
def test_get_user(mock_get):
    response = mock_get.return_value # 해당 mock 객체를 받아서 자유롭게 mocking한다.
    response.status_code = 200
    response.json.return_value = { 
        "name" :  "Test User",
        "email" : "user@test.com"
    }
    user = get_user(1)
	
    assert user["name"] == "Test User" 
    # 해당 메서드와 인자가 제대로 불렸는지 행위를 검증한다.
    mock_get.assert_called_once_with("https://api-server.com/users/1")  
```

- 테스트 더블의 종류를 외울 필요는 절대 없다.
- 서로 개념적으로 비슷한 부분들이 많기 때문에 현업에서도 여러 용어로 부르곤 한다.