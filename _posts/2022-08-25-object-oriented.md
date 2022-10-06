---
title: '[객체 지향] 프로그래밍 패러다임 흐름: 객체 지향'
description: "[객체 지향 프로그래밍] 객체 지향 프로그래밍 패러다임의 개념 및 예시, 장단점"
categories:
 - Object Oriented Programming
tags: [객체 지향 프로그래밍]
---

# 개념
- **객체 지향(Object Oriented) 프로그래밍은 객체라고 하는 단위에 책임을 명확히 하고 서로 협력하도록 프로그래밍을 하는 패러다임**이다.
- 모든 것을 객체로 나누어 생각하고, 필요할 때 객체들을 활용하고 서로 협력하여 일을 수행한다.
- 절차지향과 다르게 객체는 데이터와 함수(메서드)를 함께 가지고 있다.
- 객체 내부의 데이터는 외부에 공개할 필요가 없거나 해서는 안 되는 데이터라면 모두 자신 내부에 숨겨 외부에서 알지 못하도록 한다.

# 예시
- 객체 지향 프로그래밍 관점으로 코드를 짜는 간단한 파이썬 코드 예시를 살펴볼 것이다.
- 이전 예시와 똑같이 사용자로부터 파일을 입력받아 파일을 파싱한 후, 이 내용을 저장소에 저장하는 코드이다.

```py
class Processor:
    def __init__(self,
                 file_reader: FileReader,
                 data_parser: DataParser,
                 repository: Repository) -> None:
        self.file_reader = file_reader
        self.data_parser = data_parser
        self.repository = repository

    def execute(self, file_path: str) -> None:
        data = self.file_reader.read(file_path)
        parsed_data = self.data_parser.parse(data)
        self.repository.save(parsed_data)


class FileReader:
    def __init__(self) -> None:
        self.file_types = ["txt"]
        self.file_history = [] # 만약 절차 지향이라면 file_history 데이터를 중앙 집중으로 관리하게 된다. 

    def read(self, file_path: str) -> str:
        self._validate(file_path)
        ...

    def _validate(self, file_path: str) -> None:
        for file_type in self.file_types:
            if file_path.endswith(file_type):
                return
        raise ValueError("파일 확장자는 txt, csv, xlsx 중 하나여야 합니다.")

class DataParser:
    def parse(self, data: str) -> List[str]:
        ...

class Repository:
    def init(self, database_url: str, ...):
        ...
    
    def save(self, data: List[str]) -> None:
        ...

class Main:
    @staticmethod
    def run(self) -> None:
        processor = Processor(
            file_reader=FileReader(),
            data_parser=DataParser(),
            repository=Repository()
        )
        processor.execute("input_file.txt")


 if __name__ == "__main__":
    Main.run()
```

- 코드는 `Processor`, `FileReader` 등 여러 객체(문법적으로는 클래스)로 이루어진다.
- 그리고 각 객체는 각자 자신의 역할과 기능이 있다.
- 예를 들면 `FileReader`는 파일을 읽는 역할을, `DataParser`는 데이터를 파싱하는 역할을 한다.
- 프로그래밍은 전체적으로 객체와 객체 간의 메서드 호출로 이루어진다. 그리고 각 객체는 자신의 기능을 수행하는데 필요한 데이터를 직접 가지고 있다.
- 예를 들어, `FileReader`는 `file_types` 속성으로 자신이 파싱할 수 있는 파일 확장자인지 검증한다.
- 이 외에 다른 객체들도 본인의 역할을 수행하는 과정에서 발생하는 데이터를 전부 관리할 수 있다.
- 코드는 조금 더 복잡해졌지만, 객체 지향은 기능을 확장할 때 효과적이다.
- 위의 예시에서는 `input_file.txt` 처럼 `txt` 파일만 읽었는데, 이제는 csv 파일이나 xlsx 파일도 읽어야 하는 상황이 주어졌다고 가정해본다. 그럼 코드를 다음처럼 확장해볼 수 있다.

```py
# FileReader는 이제 추상 클래스다.
class FileReader(ABC):
    def read(self, file_path: str) -> str:
        self._validate(file_path)
        data = self._open_file(file_path)
        return self._read(data)

    @abstractmethod
    def _read(self, data: str) -> str:
        pass

    # 공통으로 사용하는 메서드이다.
    def _validate(self, file_path: str) -> None:
        if not file_path.endswith(self.file_type):
            raise ValueError(f"파일 확장자가 {self.file_type} 아닙니다.")

    @abstractmethod
    def _open_file(file_path: str) -> str:
        ...

# txt 파일을 읽는 책임을 가진 FileReader 파생 클래스이다.
class TxtFileReader(FileReader):
    def file_type(self) -> str:
        return "txt"

    def _read(self, data: str) -> str:
        ...
    
    ...


# csv 파일을 읽는 책임을 가진 FileReader 파생 클래스이다.
class CsvFileReader(FileReader):
    def file_type(self) -> str:
        return "csv"

    def _read(self, data: str) -> str:
        ...
    
    ...


# xlsx 파일을 읽는 책임을 가진 FileReader 파생 클래스입니다.
class XlsxFileReader(FileReader):
    def file_type(self) -> str:
        return "xlsx"

    def _read(self, data: str) -> str:
        ...

    ...
```

- 객체 지향을 지원하는 대부분의 프로그래밍 언어들은 클래스라는 문법을 제공한다.
- 객체의 강력한 기능인 상속을 이용하면 한 번 정의해놓은 메서드를 파생 클래스에서 재사용 가능하다.
- 또한 상속으로 객체간의 계층 구조를 만들고 데이터와 메서드를 재사용할 수 있다.
- 객체 지향의 가장 큰 특징은 같은 역할을 하는 객체를 쉽게 바꾸도록 설계할 수 있다는 것이다.
- 예를 들어 위의 경우, 우리가 txt 파일을 읽어야할 경우 다음처럼 `Main.run()` 함수 내에서 `TxtFileReader`를 사용하면 된다.

```py
class Main:
    def run(self) -> None:
        processor = Processor(
            file_reader=TxtFileReader(),
        	data_parser=DataParser(),
            repository=Repository()
        )
```

- 만약 `csv`나 `xlsx` 파일을 읽어야할 경우 다음처럼 코드 한줄만 바꾸면 된다.

```py
class Main:
    def run(self) -> None:
        processor = Processor(
            file_reader=CsvFileReader(), # 이 한줄만 바뀐다.
        	data_parser=DataParser(),  
            repository=Repository()
        )
```

- 이렇게 코드 한줄만으로 가능한 이유는 `TxtFileReader`, `CsvFileReader`, `XlsxFileReader` 클래스가 모두 `FileReader`의 파생 클래스이기 때문이다.
- 이런 객체 지향의 특성을 "다형성"이라고 하며, 어떤 객체에 필요한 객체를 때에 따라 다르게 주입해주는 것을 "의존성 주입"이라고 한다.

# 장단점
- 객체 지향은 여러명의 개발자들이 협력을 해야 하거나, 확장 가능하도록 코드를 설계해야 하는 경우에 적합하다.
- 하지만 확장이 가능하고 유연한 만큼, 처음 코드를 보는 사람들은 어렵고 헷갈릴 수 있다.
- 또한 실행 환경에서 입력에 따라 다양한 작업 흐름이 만들어지기 때문에 디버깅하기가 상대적으로 어렵다.