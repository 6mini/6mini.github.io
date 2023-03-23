---
title: '[데이터 수집] 셀레니움(Selenium) 크롤링(Crawling) 속도 향상 전략'
description: "[셀레니움 활용 동적 웹 데이터 수집(Data Collection)] 멀티 스레딩(Multi-threading)과 로딩(Loading) 최적화를 통한 크롤링 효율성 향상"
categories:
 - Data Engineering
tags: [데이터 수집, Data Collection, Data Engineering, 셀레니움, Selenium, 크롤링, Crawling, 멀티 스레딩, Multi-threading]
---

셀레니움(Selenium)을 사용해 웹 페이지를 크롤링(Crawling)할 때 속도와 효율성이 매우 중요하다. 현재 필자의 경우 제품 코드가 60,000여개나 되는 상품 페이지를 하나하나 돌며 크롤링을 진행해야하기 때문에, 속도를 최적화하는 전략이 굉장히 중요했다. 이 글에서는 셀레니움을 이용한 크롤링 속도를 높이는 방법을 알아본다. 주요 내용은 아래와 같다.

1. 로딩(Loading) 최적화
2. 멀티스레딩(Multi-threading) 활용

# 1. 로딩(Loading) 최적화

웹 페이지를 로딩할 때, 셀레니움의 기본 동작은 페이지의 모든 요소가 로딩되기를 기다리는 것이다. 그러나 이 기다림은 크롤링에 필요하지 않은 리소스들까지 기다리게 되어 속도 저하의 원인이 될 수 있다. 이 문제를 해결하기 위해 페이지 로드 전략을 `none`으로 설정하고, `WebDriverWait`와 `expected_conditions`를 사용하여 속도를 최적화한다.

## 페이지 로드 전략 설정

아래 코드에서 `DesiredCapabilities` 객체를 이용해 페이지 로드 전략을 설정한다.

```py
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

caps = DesiredCapabilities.CHROME
caps["pageLoadStrategy"] = "none"
```

위 코드를 포함 시킨 셀레니움 전체 클래스 코드는 아래와 같다.

```py
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

class SeleniumDriver:
    def __init__(self):
        self.display = None
        self.driver = None
    
    def setup(self, options=None):
        caps = DesiredCapabilities.CHROME
        caps["pageLoadStrategy"] = "none"

        chrome_options = webdriver.ChromeOptions()

        if options:
            for opt in options:
                chrome_options.add_argument(opt)
        
        # if run on ubuntu
        if DISPLAY_AVAILABLE:
            self.display = Display(visible=0, size=(1920, 1080))
            self.display.start()

        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options, desired_capabilities=caps)
    
    def teardown(self):
        if self.display:
            self.display.stop()
        if self.driver:
            self.driver.quit()
```

## `WebDriverWait`와 `expected_conditions` 사용

그리고 페이지 로딩을 최적화하기 위해 `WebDriverWait`와 `expected_conditions`를 사용한다. `WebDriverWait`는 지정된 시간 동안 특정 조건이 충족될 때까지 기다리는 역할을 한다. `expected_conditions`는 다양한 조건을 지정할 수 있는데, 이 예제에서는 페이지의 특정 요소가 로딩될 때까지 기다리도록 설정한다.

아래 코드에서 `WebDriverWait`와 `expected_conditions`를 사용하여 페이지 로딩을 최적화한다.

```py
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

def process_id(id, driver):
    driver.get(f"https://www.shop.com/{id}")

    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".product_title"))
    )

    product_details = extract_product_details(id, driver)

    return product_details
```

위 코드에서는 페이지의 `.product_title` 요소가 로딩될 때까지 최대 30초 동안 기다리도록 설정했다. 이렇게 하면 페이지가 완전히 로딩되기를 기다리지 않고 필요한 요소만 로딩될 때까지 기다리므로 불필요한 대기 시간을 줄일 수 있다.


# 2. 멀티스레딩(Multi-threading) 활용

멀티스레딩을 활용하면 여러 작업을 동시에 진행할 수 있어 크롤링 속도를 높일 수 있다. 여기서는 for문을 멀티스레드로 변환해 크롤링 작업을 병렬 처리한다. 먼저 `concurrent.futures` 모듈의 `ThreadPoolExecutor`를 사용한다.

## 기존

```py
def main():
    id_list = [...]

    driver = SeleniumDriver()
    driver.setup()
    
    results = []

    for id in id_list:
        result = process_id(id, driver.driver)
        results.append(result)

    driver.teardown()

    product_details_df = pd.DataFrame([r["product_details"] for r in results])
```

## 병렬 처리

```py
from concurrent.futures import ThreadPoolExecutor

def main():
    driver = SeleniumDriver()
    driver.setup()
    
    results = []

    with ThreadPoolExecutor() as executor:
        futures = []

        for id in id_list:
            futures.append(executor.submit(process_id, id, driver.driver))

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    driver.teardown()

    product_details_df = pd.DataFrame([r["product_details"] for r in results])
```

이를 통해 셀레니움 크롤링의 속도를 효과적으로 향상시키고, 효율적인 데이터 수집 작업을 진행할 수 있다. 이러한 최적화 전략을 적용함으로써 대량의 웹 페이지 데이터를 효과적으로 처리할 수 있으며, 개발자의 시간과 자원을 절약할 수 있다.

이상으로 셀레니움 크롤링 속도 향상 전략에 대한 내용을 마친다. 이 글을 통해 속도와 효율성을 개선하는 방법을 이해하고, 실제 크롤링 작업에 적용해 더욱 빠르고 정확한 데이터 수집이 가능하길 바란다.

# 정리

앞서 다룬 전략을 바탕으로 개선된 셀레니움 크롤링 예제 코드는 아래와 같다.

```py
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from concurrent.futures import ThreadPoolExecutor
import pandas as pd


class SeleniumDriver:
    def __init__(self):
        self.display = None
        self.driver = None
    
    def setup(self, options=None):
        caps = DesiredCapabilities.CHROME
        caps["pageLoadStrategy"] = "none"

        chrome_options = webdriver.ChromeOptions()

        if options:
            for opt in options:
                chrome_options.add_argument(opt)
        
        if DISPLAY_AVAILABLE:
            self.display = Display(visible=0, size=(1920, 1080))
            self.display.start()

        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options, desired_capabilities=caps)
    
    def teardown(self):
        if self.display:
            self.display.stop()
        if self.driver:
            self.driver.quit()

def process_id(id, driver):
    driver.get(f"https://www.shop.com/{id}")

    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".product_title"))
    )

    product_details = extract_product_details(id, driver)

    return product_details

def main():
    id_list = [...]

    driver = SeleniumDriver()
    driver.setup()
    
    results = []

    with ThreadPoolExecutor() as executor:
        futures = []

        for id in id_list:
            futures.append(executor.submit(process_id, id, driver.driver))

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    driver.teardown()

    product_details_df = pd.DataFrame([r["product_details"] for r in results])

if __name__ == '__main__':
    main()
```

이제 이 예제 코드를 사용하여 효율적인 셀레니움 크롤링을 수행할 수 있다. 로딩 최적화 및 멀티스레딩 전략을 적용함으로써 빠르고 정확한 데이터 수집이 가능해진다. 이를 바탕으로 웹 데이터를 분석하고 활용하는 다양한 분야에서 성공적인 결과를 얻을 수 있을 것이다. 이러한 전략들은 웹 크롤링 작업의 효율성을 높이는 데 도움이 되며, 향후 다른 데이터 수집 및 처리 작업에서도 활용될 수 있다.

추가적으로, 크롤링 작업의 안정성과 확장성을 고려해야 한다. 예를 들어, 크롤링 작업 중 발생할 수 있는 예외 처리를 적절히 구현하고, 크롤링 대상 웹 사이트의 구조 변경에 유연하게 대응할 수 있는 코드를 작성하는 것이 좋다. 또한, 대량의 데이터를 처리하기 위해 클라우드 서비스나 분산 컴퓨팅 기술을 활용하는 것도 고려할 수 있다.