---
title: '[깃] 리스토어 및 리셋(restore & reset)'
description: "[개발자 실무 필수 기본기 강의] 상황별 깃(Git) 핸들링: 변경 사항, 커밋을 초기화"
categories:
 - Git
tags: [깃]
---

- 다음과 같이 두 개의 커밋이 있는 상황에서, `a 파일을 추가한다` 커밋 시점으로 초기화하고 싶다.

```s
$ git log --oneline
'''
c008c47 a 파일을 수정한다
b014111 a 파일을 추가한다
'''
```

- 이때 `git reset` 명령어를 사용하면 된다.
- `git reset` 명령어는 아래와 같은 옵션을 갖고 있다.

# git reset --hard {커밋 ID}
- 특정 커밋 시점으로 돌아갈 때, 해당 커밋 이후 만들어진 모든 작업물을 삭제한다.

```s
$ git reset --hard b014111 
'''
HEAD의 현재 위치는 b014111입니다 a 파일을 추가한다
'''

$ git log --oneline
'''
b014111 a 파일을 추가한다
'''

$ git status
'''
현재 브랜치 main
커밋할 사항 없음, 작업 폴더 깨끗함
'''
```

- 현재 작업한 파일을 모두 날리고 이전 커밋 상태로 완전히 돌아가고 싶을 때 사용하지만, 기존에 작성하던 변경사항들도 전부 날아가기 때문에 주의해야 한다.

# git reset --mixed {커밋 ID}
- 특정 커밋 시점으로 돌아갈 때, 해당 커밋 이후 모든 작업물은 `workspace` 공간에 `unstaged` 상태로 남게 된다.

```s
$ git reset b014111 --mixed
'''
리셋 뒤에 스테이징하지 않은 변경 사항:
M	a
'''

$ git log --oneline
'''
b014111 a 파일을 추가한다
'''

$ git status
'''
현재 브랜치 main
커밋하도록 정하지 않은 변경 사항:
  (무엇을 커밋할지 바꾸려면 "git add <파일>..."을 사용하십시오)
  (use "git restore <file>..." to discard changes in working directory)
	수정함:        a

커밋할 변경 사항을 추가하지 않았습니다 ("git add" 및/또는 "git commit -a"를 사용하십시오)
'''
```
- `--mixed` 옵션은 기본 `git reset`의 기본 옵션으로 `git reset`만 실행해도 똑같다.

# git reset --soft {커밋 ID}
- 특정 커밋 시점으로 돌아갈 때, 해당 커밋 이후 모든 작업물은 `index` 공간에 `staged` 상태로 남게 된다.

```s
$ git reset b014111 --soft

$ git log --oneline
'''
b014111 a 파일을 추가한다
'''

$ git status

현재 브랜치  main
커밋할 변경 사항:
  (use "git restore --staged <file>..." to unstage)
	수정함:        a
```

# git restore {파일 경로}
- 특정 파일의 변경사항을 제거하고 `HEAD` 기준으로 되돌리고 싶을 때, `restore`를 사용할 수 있다.
- 워크스페이스에 있는 변경 사항을 되돌릴 때: `git restore {파일경로}`

```s
# 아직 stage(index)에 올라가지 않은 README.md 파일을 되돌릴 때  
$ git restore README.md
```

- `git restore`는 `git reset --hard HEAD`와 비슷한 결과를 낸다.
- 다만 `restore`는 새 파일의 변경사항을 되돌리지 않지만, `reset`은 새 파일의 변경사항도 되돌린다.