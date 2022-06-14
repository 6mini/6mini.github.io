---
title: '[깃] 스쿼시 및 리베이스 머지(squash & rebase merge)'
description: "상황별 깃(Git) 핸들링 브랜치 머지할 때 머지 커밋 남기지 않는 방법"
categories:
 - Git
tags: [깃]
---

- 두 브랜치를 합치는 방법에는 크게 3가지가 존재한다.
  - 기본 Merge
  - Squash & Merge
  - Rebase & Merge

![image](https://user-images.githubusercontent.com/79494088/172518144-a00adb6b-9b0f-43f5-9b69-701a08f26979.png)


# `git merge {브랜치 이름}`
- `git merge`는 가장 기본적인 머지 방식이다.
- 기존 `main` 브랜치로 부터 `feature-branch` 브랜치를 만들어 둔다.
- 현재 `main` 브랜치는 다음과 같은 커밋 기록이 있다.

```s
$ git switch -c feature-branch
'''
Switched to a new branch "feature-branch"
'''

$ git log --oneline
'''
1bf9fa9 (HEAD -> feature-branch, origin/main, origin/HEAD, main) add b
4dc0231 edit a
4f4d43d add a
'''
```

- 이때 `feature-branch`에 a 파일을 수정하는 새로운 커밋을 만든다.

```s
# 파일 수정 작업
$ git commit -m 'edit a2'
```

- 이제 `feature-branch` 브랜치에서 작업한 내용을 하나의 커밋으로 만들어 `main` 브랜치에 합칠 것이다.
- 이때 사용하는 명령어가 `git merge` 이다.
- 합치기 위해 먼저 `main` 브랜치로 이동한다.

```s 
$ git switch main
'''
Switched to branch "main"
Your branch is up to date with "origin/main".
'''
```

- 만약 `main` 브랜치로부터 `feature-branch` 브랜치를 만든 이후, `main` 브랜치에 추가 커밋이 없는 상태라면 다음처럼 `git merge` 시 `feature-branch`의 모든 커밋이 그대로 `main` 브랜치로 들어가게 된다.
- 이를 `fast-foward` 방식이라고 한다.

```s
$ git merge feature-branch
'''
Updating 1bf9fa9..3558bfb
Fast-forward
 a | 1 +
 1 file changed, 1 insertion(+)
'''

```

- 그러나 다음처럼 병합이 이뤄지기 전 `main` 브랜치에 새로운 커밋이 생겼다고 가정해본다.

```s
$ touch c
$ git add c
$ git commit -m "add c"
```

- 이제 이전처럼 `git merge` 명령어를 입력하면 다음처럼 머지를 위한 머지 커밋이 생기게 된다.

```s
$ git merge feature-branch
'''
Merge branch "feature-branch"
# Please enter a commit message to explain why this merge is necessary,
# especially if it merges an updated upstream into a topic branch.
#
# Lines starting with "#" will be ignored, and an empty message aborts
# the commit.
'''

# git log로 확인하면 Merge 내용을 나타내는 커밋이 생성된다.
$ git log --oneline
'''
a964aa7 (HEAD -> main) Merge branch "feature-branch"
148997e (feature-branch) edit a3
f612482 add c
3558bfb edit a2
1bf9fa9 (origin/main, origin/HEAD) add b
4dc0231 edit a
4f4d43d add a
'''
```

- 머지 커밋을 통해 명시적으로 브랜치의 병합이 있었다는 걸 표시해주고 싶을 때 `git merge` 방식을 많이 활용한다.

## `merge conflict`
- 머지할 때 두 브랜치가 다음과 같은 상황일 때 Git은 충돌이 발생하는데, 이를 `Merge Conflict`라고 한다.
  - 한 파일의 같은 라인을 고쳤을 때
  - 한 브랜치에서는 파일을 삭제하고 한 브랜치에서는 파일을 변경할 때
- 두 작업 내역을 합치는 Git의 입장에서는 같은 파일의 두 작업내역 중 어떤 사항으로 적용해야 할 지 모른다.
- 따라서 컨플릭트가 난 파일을 해결(Resolve)해준 후 머지를 진행해야 합니다.


# `git merge {브랜치 이름} --squash`
- 별다른 머지 커밋을 만들지 않고 변경 사항만 병합하고 싶은 경우가 있다.
- 이때 머지 커밋을 남기지 않으면서, 해당 브랜치에서 작업한 모든 내용을 하나의 커밋으로 묶어버릴 때 사용하는 머지 방식이 `Squash & Merge` 방식이다.
- 명령어는 `git merge --sqaush` 이다.
- `squash`는 여러 커밋을 하나의 커밋으로 만들 때 주는 옵션인데, 브랜치 간 합칠 때 이 옵션을 주겠다는 의미이다.

```s
$ git merge feature-branch --squash
'''
Squash commit -- not updating HEAD
Automatic merge went well; stopped before committing as requested
'''

$ git commit -m "add e"
'''
[main fcf0d11] add e
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 e
'''
```

- 이제 다음처럼 `git log`로 커밋 내역을 확인해보면 `feature-branch`에서 작업한 커밋들을 위에서 작성한 하나의 커밋으로 만들어져서 `main` 브랜치에 합쳐진 것을 확인할 수 있다.

```s
$ git log --oneline
'''
fcf0d11 (HEAD -> main) add e
a964aa7 Merge branch "feature-branch"
148997e edit a3
f612482 add c
3558bfb edit a2
1bf9fa9 (origin/main, origin/HEAD) add b
4dc0231 edit a
4f4d43d add a
'''
```

- `git merge --squash`을 통해 하나의 커밋으로 묶어서 병합을 하게되면, 브랜치의 커밋 구조를 깔끔하게 유지할 수 있다.
- 다만 나중에 롤백 처리를 할 때 커밋을 한 번에 처리하는 게 불가능해지는 문제가 있다.


# `git rebase {브랜치 이름}`
- `Rebase & Merge` 방식은 머지할 때 머지 커밋을 남기지 않으면서도, 머지되는 브랜치의 모든 커밋 내역을 그대로 가져오는 머지이다.
- 명령어는 `git rebase`이다.
- 위의 예시와 마찬가지로 `main` 브랜치로부터 생성된 `feature-branch` 브랜치에는 다음과 같은 작업내역이 있다고 가정한다.

```s
$ git log --oneline
'''
a772fe6 (HEAD -> feature-branch) edit f
43ec631 add f
05a6bcf add e
148997e edit a3
3558bfb edit a2
'''
```

- 위 두 머지 방식과 다르게 리베이스의 경우 병합될 브랜치에서 `git rebase {대상 브랜치}`를 사용하면 된다.

```s
$ git switch feature-branch
$ git rebase main
'''
Successfully rebased and updated refs/heads/feature-branch.
'''

$ git merge feature-branch
'''
Updating 927f883..c924b44
Fast-forward
 f | 1 +
 1 file changed, 1 insertion(+)
 create mode 100644 f
'''

$ git log --oneline
'''
c924b44 (HEAD -> main, feature-branch) edit f
dc98158 add f
927f883 edit a3
fcf0d11 add e
a964aa7 Merge branch 'feature-branch'
148997e edit a3
f612482 add c
3558bfb edit a2
'''
```

- `git rebase`는 별다른 커밋을 생성하지 않고 브랜치의 커밋 구조를 변경한다고 보면 된다.
- 코드를 보는 입장에서는 깔끔할 수 있지만, 브랜치의 병합 히스토리가 명시적으로 잘 남지 않아 히스토리를 추적할 때 불편할 수 있다.

- 개발 팀에서 브랜치 관리 전략에 따라 각기 다른 머지를 사용한다.
- 따라서 상황에 맞는 최적의 머지 방식을 사용하면 된다.

- 대표적인 원격 저장소(GitHub, Bitbucket)들은 브랜치 간 병합을 하기 전 코드를 리뷰할 수 있는 PR(Pull Request) 환경을 제공한다.
- 이때 `squash`, `rebase merge` 방식을 모두 지원한다.