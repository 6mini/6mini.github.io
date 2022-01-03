---
title: "[깃허브 블로그] 맥 OS Monterey 'bundle exec jekyll serve' 에러"
description: 맥(Mac) OS Monterey로 업그레이드 후 jekyll serve 에러를 해결하는 방법
categories:
 - Github Blog
tags: [Jekyll, 깃허브 블로그]
mathjax: enable
---

# 문제
- 자칭 얼리어답터인 나는 Mac Moterey OS를 잽싸게 설치 했다.
- 여느 때와 같이 블로그 포스팅을 하여 `bundle exec jekyll serve`를 했는데... 므ㅏ?

```
Could not find commonmarker-0.17.13 in any of the sources.
Run `bundle install` to install missing gems.
```

- 난생 처음 보는 오류였다.
- `bundle install`을 해보았지만,

```
Gem::Ext::BuildError: ERROR: Failed to build gem native extension.
```

- Jekyll 블로그를 셋팅한 지가 너무 오래돼서... 정말 루비(ruby)의 흐름 자체가 익숙하지 않아 몇시간을 날린 끝에...! 해결하는 데 성공했다.

# 해결

## 홈브류(Homebrew) 설치
- 이미 설치되어 있지만 재설치한다.

```
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## rbenv & helpers 설치

```
$ brew install rbenv ruby-build rbenv-gemset rbenv-vars 
$ rbenv init 
$ curl -fsSL https://github.com/rbenv/rbenv-installer/raw/main/bin/rbenv-doctor | bash
```

- 모두 진행된다면, `vi ~/.zshrc`를 통해 아래 셀을 추가한다.

### Modify shell startup script

```
export RUBY_CONFIGURE_OPTS="--with-openssl-dir=$(brew --prefix openssl@1.1)"
eval "$(rbenv init -)"
```

{% include ad.html %}

## 루비(Ruby) 설치
- 이 블로그의 경우 NexT Theme를 사용하는데, 최신의 루비 버전이 아닌 2.4.4로 설치해야 진행되었다.

```
rbenv install 2.4.4
rbenv global 2.4.4
```

### 루비 버전 확인

```
$ ruby -v
'''
ruby 2.4.4p296 (2018-03-28 revision 63013) [x86_64-darwin20]
'''
```

## 번들(bundle) 설치
- 블로그가 설치 된 디렉토리로 와서 번들을 설치한다.
- 설치 전 반드시 기존의 `Gemfile.lock`를 제거한다.

```
$ cd 6mini.github.io
$ bundle install
$ bundle exec jekyll serve
```

# 최종
- 엉엉... 역시 또 해결하고보니 굉장히 간단한 방법이었음을...
- 포기할뻔 했지만 성공해서 기분좋다. 열심히 또 포스팅해야징

![image](https://user-images.githubusercontent.com/79494088/142725753-052fee67-c45d-4c35-aa74-2a75dde49534.png)

# 참조
- [How to install the latest Ruby on MacOS Monterey (12.0)](https://luther.io/macos/how-to-install-latest-ruby-on-a-mac/)
- [Mac OS BigSur 11.2.3 "bundle exec jekyll serve" Error](https://yskim0.github.io/troubleshooting/2021/05/12/BigSur_Jekyll_TroubleShooting/)