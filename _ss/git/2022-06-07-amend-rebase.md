---
title: '[깃] 어맨드 커밋 및 리베이스(amend commit & rebase)'
description: "상황별 깃(Git) 핸들링 이전에 쌓인 커밋 변경"
categories:
 - Git
tags: [깃]
---

- 깃을 사용하다 보면 이전 커밋을 변경해야 하는 경우들이 종종 있다.
- 커밋 메시지를 수정하고 싶다거나 변경된 파일 일부를 되돌릴 수 있다.
- 현재 작업 중인 커밋(HEAD)을 간단하게 수정할 때 `git commit --amend`를 사용한다.
- 아래에 있는 커밋 중 일부를 수정하거나 변경할 때 `git rebase --interactive`를 사용한다.
- `git revert`의 경우 대상 커밋을 되돌리는 새로운 커밋을 만드는 기능이며, 커밋 자체를 변경하지 못한다.

# `git commit --amend`
- `--amend`는 현재 커밋(HEAD) 위에 변경사항을 덮어씌울 때 사용하는 옵션이다.
- 커밋을 한 후 추가적인 변경사항이 생겼거나 커밋 메시지를 변경하고 싶을 때 많이 사용한다.
- 만약 변경사항을 추가하고 싶다면 커밋을 할 때와 마찬가지로 스테이징(index)에 올린 후 명령어를 입력하면 된다.

```s
$ git add .

# 만약 커밋 메시지를 변경하고 싶다면 텍스트를 수정한 후 저장하면 된다.
# 변경이 필요 없다면 바로 :wq로 저장하면 된다. 
$ git commit --amend
'''
기존 커밋 메세지

# Please enter the commit message for your changes. Lines starting
# with "#" will be ignored, and an empty message aborts the commit.
#
# Date:      Wed Jun 8 11:01:30 2022 +0900
#
# On branch main
# Your branch is up to date with "origin/main".
#
# Changes to be committed:
#       modified:   a
#       modified:   b
'''
```

- 리모트 레포지토리로 푸시하고 싶다면, 포스푸시를 이용해야 한다.

```s
$ git push --force
```

- 만약 커밋 메시지만 수정하고 싶다면 변경사항 없이 바로 `git commit --amend`를 사용하면 된다.
- 커밋 메시지의 수정이 필요하지 않는 경우 `--no-edit` 옵션을 붙이면 된다.

```s
$ git commit --amend --no-edit
'''
[main 2dd97d6] 기존 커밋 메세지
 Date: Wed Jun 8 11:01:30 2022 +0900
 2 files changed, 2 insertions(+)
'''
```

# `git rebase --interactive {커밋 ID}`
- `git rebase` 명령어는 브랜치 병합 과정에서 자주 사용된다.
- 하지만 동시에 과거 커밋 히스토리를 변경할 수 있는 기능도 `--interactive` 옵션을 통해 제공한다.
- 해당 옵션은 `-i shortcut`으로 많이 사용한다.
- 커밋 히스토리의 수정 범위는 현재 최신 커밋부터 {커밋 ID} 바로 위 커밋까지 적용된다.

- 총 3개의 커밋이 존재하며 각각 a, b, c 파일을 추가했다고 가정한다.

```s
$ git log --oneline
'''
2dd97d6 (HEAD -> main, origin/main, origin/HEAD) a + b 파일 수정
3218ba7 b 파일 수정
455f883 Revert "a 파일 수정"
'''
```
- 이때 두 번째 커밋에 b2 파일을 추가하고 세 번째(마지막) 커밋에 a파일을 삭제해야 하는 상황이 생겼다.
- 그러면 아래 커밋인 `455f883`을 범위로 적용해줘야 한다.

```s
$ git rebase --interactive 455f883 # 혹은 HEAD^^ , HEAD~2 로도 표현할 수 있다.
'''
pick 3218ba7 b 파일 수정
pick 2dd97d6 a + b 파일 수정

# Rebase 455f883..198d118 onto 455f883 (3 commands)
#
# Commands:
# p, pick <commit> = use commit
# r, reword <commit> = use commit, but edit the commit message
# e, edit <commit> = use commit, but stop for amending
# s, squash <commit> = use commit, but meld into previous commit
# f, fixup <commit> = like "squash", but discard this commit"s log message
# x, exec <command> = run command (the rest of the line) using shell
# b, break = stop here (continue rebase later with "git rebase --continue")
# d, drop <commit> = remove commit
# l, label <label> = label current HEAD with a name
# t, reset <label> = reset HEAD to a label
# m, merge [-C <commit> | -c <commit>] <label> [# <oneline>]
# .       create a merge commit using the original merge commit"s
# .       message (or the oneline, if no original merge commit was
# .       specified). Use -c <commit> to reword the commit message.
'''
```

- 입력 후, vi 에디터를 확인할 수 있다.
- 대표적인 커맨드는 아래와 같다.
  - `pick`: 별다른 변경 사항없이 그냥 커밋으로 두겠다.
  - `edit`: 해당 커밋 내용을 변경할 것이며 커밋 메시지도 변경할 수 있게 하겠다.
  - `reword`: 해당 커밋의 메시지만 변경하겠다.
  - `drop`: 해당 커밋을 제거하겠다.
  - 이 외에도 커밋들을 합칠 때 사용하는 `squash`, 브랜치를 머지시키는 `merge Command` 등이 존재한다.

- 우선 커밋의 변경사항을 수정하기 위해 해당 커밋 라인의 `pick`을 `edit`으로 변경한 후 저장한다.(:wq)

```s
'''
edit 3218ba7 b 파일 수정
pick 2dd97d6 a + b 파일 수정
'''

# 저장 후 
'''
Stopped at 3218ba7...  b 파일 수정
You can amend the commit now, with

  git commit --amend 

Once you are satisfied with your changes, run

  git rebase --continue
'''
```

- 그러면 해당 커밋으로 HEAD가 옮겨지며, 자유롭게 코드를 추가/삭제/변경할 수 있다.

```s
$ touch b2 # b2파일을 추가하는 변경 사항을 줍니다.
$ git add b2
```

- 모든 코드 변경을 마친 후 `git commit --amend`를 입력하면, 현재 최신 커밋(HEAD)에 덮어씌우는 작업을 하게 된다.
- 커밋 메시지도 같이 수정할 것이다.

```s
$ git commit --amend
"""
b 파일 수정, b2 파일 추가

# Please enter the commit message for your changes. Lines starting
# with '#' will be ignored, and an empty message aborts the commit.
#
# Date:      Wed Jun 8 11:01:07 2022 +0900
#
# interactive rebase in progress; onto 455f883
# Last command done (1 command done):
#    edit 3218ba7 b 파일 수정
# Next commands to do (1 remaining commands):
#    pick 2dd97d6 a + b 파일 수정
# You are currently editing a commit while rebasing branch 'main' on '455f883'.
#
# Changes to be committed:
#       modified:   b
#       new file:   b2
"""
```

- 최종적으로 커밋을 마친 후 모든 변경 사항을 적용했다면 `git rebase --continue`로 다음 작업 대상으로 넘어가면 된다.
- 그렇게 되면 HEAD는 위에서 `edit`을 입력한 커밋으로 올라가게 된다. 
- 이후 변경 작업은 동일하다.

```s
$ git rebase --continue
```

- 만약 해당 커밋의 변경 사항을 주지 않고 다음으로 넘어가고 싶다면 `git rebase --skip`을 사용하면 된다.

```s
$ git rebase --skip 

# 다음 변경할 commit으로 HEAD가 옮겨간다.
```

- 만약 `rebase`하는 과정에서 전체를 취소하고 싶을 수 있다.
- 이때 `git rebase --abort`를 사용하면 된다.

```s
$ git rebase --abort

# rebase -i를 주기 전 원래 환경으로 돌아옵니다. 
```

- 중간에 있는 커밋을 변경하는 과정에서 상위 커밋과의 변경 사항이 충돌할 수도 있다.
- 그럴 때는 충돌난 부분을 수정한 후 `git add`로 `index`에 충돌을 수정한 부분을 올린 후 `git rebase --continue`를 사용하면 된다.