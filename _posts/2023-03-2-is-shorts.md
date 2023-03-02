---
title: '[유튜브] 영상 ID를 통해 일반영상 or 쇼츠영상 분류 in 파이썬(python)'
description: "유튜브(Youtube) 비디오(video) id를 입력하면, 일반영상인지 쇼츠영상인지 판단하여 알려주는 간단한 함수"
categories:
 - Data Engineering
tags: [유튜브, youtube, 쇼츠, shorts]
---

# 개요
- 쇼츠 비디오는 60초 이하의 짧은 동영상으로, 유튜브 앱에서 플레이할 때 수직 모드로 플레이되는 등 독특한 특징을 가지고 있다.
- 유튜브 쇼츠는 인기 있는 콘텐츠를 쉽게 발견할 수 있도록 도와주는 기능으로, 최근에는 많은 창작자들이 쇼츠를 활용하여 창작 활동을 하고 있다.
- 파이썬으로 유튜브 비디오의 ID를 받아서 해당 비디오가 쇼츠 영상인지 확인하는 코드를 짜볼 것이다.

# 코드
```py
def is_short(vid):
    url = 'https://www.youtube.com/shorts/' + vid
    ret = requests.head(url)
    if ret.status_code == 200:
        return True
    else:
        return False
```

- `is_short()` 함수는 requests 모듈을 사용하여 쇼츠의 URL로 HEAD 요청을 보내고, 응답 상태 코드가 200인지 확인하여 쇼츠 여부를 판단한다.
- 유튜브 쇼츠의 여부를 쉽게 확인할 수 있습니다. 예를 들어, 아래와 같이 함수를 호출하여 쇼츠인지 여부를 확인할 수 있다.

# 호출

```py
is_short('zljxhljjjvA')
'''
True
'''
```

- 위 예제에서는 비디오 ID 'zljxhljjjvA'를 입력하여 해당 비디오가 쇼츠인지 여부를 확인한다.
- 함수는 True를 반환하므로, 해당 비디오가 쇼츠임을 확인할 수 있다.