---
title: '[깃(Git)] 9. 다른 브랜치의 커밋을 가져오는 체리 픽(cheery-pick)'
description: "[상황별 깃(Git) 핸들링] 체리 픽(cheery-pick)을 통해 다른 브랜치에 있는 커밋을 내 브랜치로 가져오는 방법"
categories:
 - Git
tags: [깃, 체리 픽, cherry-pick]
---

# `git cherry-pick {커밋 ID}`
- 다음과 같이 `main` 브랜치로부터 `my-branch` 브랜치를 만들어 작업하고 있었다고 가정한다.

```s
$ git switch -c my-branch
'''
Switched to a new branch "my-branch"
'''

$ touch b
$ git add b
$ git commit -m "add b"

$ touch c
$ git add c
$ git commit -m "add c"
```

- 현재까지 작업 내역은 다음과 같다.

```s
$ git log --oneline
'''
4dcc536 (HEAD -> my-branch) add c
7ffa05e add b
89941f1 (main) add a
'''
```

- 이제 다시 `main` 브랜치로 간다.

```s
$ git switch main
'''
Switched to branch "main"
'''

# main 브랜치의 현재 작업 내역은 다음과 같다.
$ git log --oneline
'''
89941f1 (HEAD -> main) add a
'''
```

- 이제 `main` 브랜치에 `my-branch`의 `4dcc536` (c 파일을 추가한다) 커밋만 가져오고 싶다.
- 이때 사용하는 것이 `git cherry-pick` 명령어이다.
- 이 명령어는 다른 브랜치의 일부 커밋을 현재 브랜치로 가져오게 한다.

```s
# 4dcc536 커밋을 현재 브랜치로 가져옵니다.
$ git cherry-pick 4dcc536
'''
[main 3421b43] add c
 Date: Wed Jun 8 12:15:00 2022 +0900
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 c
'''
```

- 이제 로그를 확인하면 다음처럼 `4dcc536` 커밋의 작업내역을 가져온 것을 확인할 수 있다.
- 이때 변경 사항을 복사해서 새로운 커밋을 만드는 것이기에 커밋 해시는 변경된다.

```s
$ git log --oneline
'''
3421b43 (HEAD -> main) add c
89941f1 add a
'''

$ ls
'''
a c
'''
```