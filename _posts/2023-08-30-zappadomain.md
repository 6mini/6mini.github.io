---
title: '플라스크(Flask)를 자파(Zappa)로 배포하고 가비아(Gabia) 커스텀 도메인 연결'
description: "AWS 람다(Lambda)와 API Gateway와 같은 서비스를 통해 파이썬(Python) 웹 앱들을 서버리스로 쉽게 배포하는 훌륭한 도구인 자파(Zappa)에 대한 설명 및 사용방법과 가비아(Gabia)에서 구매한 도메인으로 연결하는 과정"
categories:
 - APP
tags: [플라스크, 자파, 가비아, Flask, Zappa, Gabia]
---

자파(Zappa)는 AWS 람다(Lambda)와 API Gateway와 같은 서비스를 통해 파이썬(Python) 웹 앱들을 서버리스(Serverless)로 쉽게 배포하는 훌륭한 도구이다. 이 포스팅에서는 자파를 사용하여 플라스크(Flask) 앱을 배포하고, 가비아(Gaiba)에서 구매한 도메인으로 연결하는 과정을 단계별로 설명할 것이다.

# 준비물

- AWS 계정
- 가비아(Gabia)에서 구매한 도메인
- 플라스크(Flask) 웹 앱
- 자파(Zappa)

# 1. 플라스크(Flask) 앱 생성

## 가상환경 설치

자파는 파이썬 가상환경을 만들어 그 가상환경을 통채로 람다에 배포한다. 필자는 가벼운 `virtualenv`로 진행할 것이다.

```sh
$ pip3 install virtualenv
$ virtualenv venv
$ source /Users/6mini/zappa/venv/bin/activate
```

가상환경 안에 필요한 라이브러리를 설치한다.

```sh
$ pip3 install Flask
$ pip3 install zappa
```

## 간단한 웹 앱 제작

굉장히 간단한 웹 앱을 미리 만든다. 필자는 `zappa`라는 폴더를 생성하고, `app.py`라는 파일을 만들어서 아래와 같이 작성했다.

```py
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(debug=True)
```

이제 플라스크 웹 애플리케이션을 실행한다.

```sh
$ export FLASK_APP=app.py  
$ flask run
```

`127.0.0.1:5000`으로 접속하면 `Hello, World!`가 출력될 것이다.

# 2. 자파(Zappa) 배포

## AWS IAM 설정

먼저 자파가 내 AWS 계정을 조작할 수 있도록 권한을 주어야 한다. AWS CLI를 설치한 뒤 아래와 같이 AWS 계정정보를 입력하면 된다. 이 때 중요한 점은, 추후 커스텀 도메인을 사용할 것이기 때문에 리전(region)을 버지니아 북부(us-east-1)로 설정할 것이다.

```sh
$ aws configure
AWS Access Key ID [None]: QWER********
AWS Secret Access Key [None]: asdf********
Default region name [None]: us-east-1
Default output format [None]: json
```

## 배포

```sh
$ zappa init
# 1. 람다 함수의 이름에 사용될 가상환경의 이름
What do you want to call this environment (default 'dev'): dev
# 2. 람다 함수 코드가 저장될 S3 버킷 자동 생성
Your Zappa deployments will need to be uploaded to a private S3 bucket.
If you don't have a bucket yet, we'll create one for you too.
What do you want to call your bucket? (default 'zappa-qwerty1234')
# 3. 플라스크 애플리케이션 인식
It looks like this is a Flask application.
What's the modular path to your app's function?
This will likely be something like 'app.app'.
We discovered: app.app
Where is your app's function? (default 'app.app'): app.app
# 4. us-east-1 외의 다른 곳에서 서비스할지 선택
Would you like to deploy this application globally? (default 'n') [y/n/(p)rimary]: y
# 5. 설정을 만들어 저장
Okay, here's your zappa_settings.json:
{
    "dev": {
        "aws_region": "us-east-1",
        "app_function": "app.app",
        "profile_name": "default",
        "project_name": "zappa",
        "runtime": "python3.8",
        "s3_bucket": "zappa-qwerty1234"
    }
    ...
}
Does this look okay? (default 'y') [y/n]: y
```

람다 함수의 이름은 `{프로젝트명}-{가상환경명}`이 된다. 프로젝트명은 프로젝트 폴더 이름, 여기서는 `zappa`이고, 가상환경명은 지정해준 값, 여기서는 `dev`이므로, 람다 함수 이름은 `zappa-dev`가 된다.

이제 아래와 같이 배포한다.

```sh
$ zappa deploy dev
Downloading and Installing dependencies..
...
Uploading flask-dev-******.zip (*.*MiB)..
...
Deploying API Gateway..
...
Deployment complete! https://q1w2e3r4t5y6.execute-api.ap-northeast-2.amazonaws.com/dev
```

이제 `https://q1w2e3r4t5y6.execute-api.us-east-1.amazonaws.com/dev`로 접속하면 잘 작동하는 것을 확인할 수 있다. 근데 URL을 보면 굉장히 못생겼다. 커스텀 도메인을 연결해서 예쁘게 만들어보자.

# 3. 가비아 커스텀 도메인 연결

가비아에서 도메인을 구입하는 과정은 생략한다.

## AWS Certificate Manager에서 인증서 생성

아래와 같이 진행한다.

1. AWS Management Console 로그인
2. 버지니아 북부(us-east-1) 리전으로 변경(자파는 이 리전의 인증서만 허용하기 때문)
3. Certificate Manager 접속
4. 인증서 요청
5. 퍼블릭 인증서 요청
6. 구매한 도메인 이름 입력(DNS 검증, RSA 2048)
7. 키 알고리즘 RSA 2048

생성 후 보이는 CNAME 이름과 CNAME 값을 복사한다.

## 가비아 DNS 인증

아래와 같이 진행한다.

1. 가비아 로그인
2. My 가비아 > 도메인
3. 해당 도메인 관리
4. DNS 정보 > 도메인 연결 설정
5. DNS 설정 > 레코드 추가
6. 타입은 CNAME, 호스트 이름에 CNAME 이름 중 `.com.`을 제외하고 입력
7. 값/위치에 CNAME 값 입력
8. 확인 > 저장

조금만 기다리면, 인증서 상태가 `발급됨`으로 전시된다. 발급된 인증서의 `ARN`을 복사해둔다.

## 도메인 연결

앱에서 생성된 `zappa_settings.json` 설정 파일을 아래와 같이 설정한다.

```json
{
    "dev": {
        ...
        "certificate_arn": "arn:aws:acm:us-east-1:your/arn",
        "domain": "your.domain"
        ...
    }
}
```

## Zappa로 도메인 인증

터미널에서 아래의 명령어를 실행하여 자파를 사용하여 도메인을 인증한다.

```sh
$ zappa certify
Calling certify for environment dev..
Are you sure you want to certify? [y/n] y
Certifying domain www.your.domain..
Created a new domain name with supplied certificate. Please note that it can take up to 40 minutes for this domain to be created and propagated through AWS, but it requires no further work on your part.
Certificate updated!
```

## 가비와와 연결할 주소 확인

1. AWS Route 53 접속
2. 호스팅 영역 > 등록한 도메인 접속
3. 레코드의 NS유형 값/트래픽 라우팅 대상 4가지 복사

![](https://github.com/6mini/6mini.github.io/assets/79494088/94126f74-9185-47e1-99c2-7ad6e957e70f)

## 가비아 설정

1. 가비아 로그인
2. My 가비아 > 도메인
3. 해당 도메인 관리
4. 네임서버 1~4차에 위 값 입력: 마지막 `.`은 제거

![](https://github.com/6mini/6mini.github.io/assets/79494088/85d6d06f-9e7e-466b-b8ee-2a1b684c861d)

## 완료

이제 커스텀 도메인으로 접속하여 정상적으로 동작하는지 확인한다.

# 참조

- [Flask Microservice 구축 - Zappa로 AWS Lambda에 Flask 띄우기]{https://panda5176.tistory.com/39}
- [Using a Custom Domain](https://romandc.com/zappa-django-guide/walk_domain/)
- [[AWS] EC2 와 도메인 연결하기 (feat. 가비아)](https://developer-ping9.tistory.com/320)