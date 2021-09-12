---
title: '[conda] 콘다 가상환경 base 자동 activate 해제 방법'
description: conda 가상환경 설치 후 자동적으로 터미널을 열 때마다 (base)로 진입되는 것을 방지하는 방법
categories:
 - Did Unknown
tags: [Did Unknown, anaconda, conda, base, activate, auto, 아나콘다, 콘다, 베이스, 가상환경]
mathjax: enable
# 0️⃣1️⃣2️⃣3️⃣4️⃣5️⃣6️⃣7️⃣8️⃣9️⃣🔟
---

# 1️⃣ 증상
- Terminal 실행 시 conda 가상환경으로 자동으로 진입되는 현상

![스크린샷 2021-09-09 23 18 30](https://user-images.githubusercontent.com/79494088/132705869-ef090bca-ec87-4054-a79f-b00bb45ba7ed.png)

# 2️⃣ 해결

```
$ conda activate base # base 진입
$ conda config --set auto_activate_base false # 자동 진입 해제
```

![스크린샷 2021-09-09 23 18 50](https://user-images.githubusercontent.com/79494088/132705782-91a7622d-4c89-424c-8ff0-7ccfb48f142f.png)

# 3️⃣ 결과
- 재실행 시 자동 진입이 되지 않는 것을 알 수 있다.

![스크린샷 2021-09-09 23 38 33](https://user-images.githubusercontent.com/79494088/132706361-ea2e6dc4-43dd-41fc-a3f0-71dca7ef2789.png)