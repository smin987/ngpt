# MyPython

파이썬 연습과 공부를 위해 만들어진 개인 레포지토리

깃허브 window 버전 업그레이드 명령어
git vash를 실행 후
`git update-git-for-windows\`

설치되지 않았으면 아래의 사이트에서 다운로드
https://git-scm.com/downloads

[ssh 비번 변경방법]

1. ssh key가 있는 곳으로 이동
   cd ~/.ssh
2. ssh key 암호 변경 명령어 입력
   ssh-keygen -f id_ed25519 -p
3. 기존 암호 입력 및 신규 암호 두번 입력 (암호를 없애고 싶을 시 그냥 Enter 입력)

# VSCODE

## 수동업데이트

CMD 관리자 모드로 실행하여 아래의 명령어를 입력

> 윈도우

    `winget upgrade --id Microsoft.VisualStudioCode`

> black format, organize import 설정

## 확장팩 설치

### Black Fommater, isort

- VSCODE에서 CTRL+SHIFT+P 를 눌러서 커맨드창을 띄우고 user settings.json 을 입력후 항목을 선택한다
- 아래와 같이 settings.json 파일을 수정한다

```
{
    "editor.formatOnSave": true,
    "[python]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        },
        "editor.formatOnType": true
    },
    "python.analysis.autoImportCompletions": true,
    "python.formatting.provider": "black",
    "python.formatting.blackPath": "black",
    "python.formatting.blackArgs": [
        "--line-length=120"
    ],
    "isort.args": [
        "--profile",
        "black"
    ],
}
```

- 주의사항(settings.json)
  - "files.autoSave": "off", // enter 키가 안먹으면 이 설정을 off 할 것
  - "editor.accesibilitySupport" : "off" // enter 키가 안먹으면 이 설정을 off 할 것
  - **vscode-styled-components** extension 확장팩이 설치되었다면 제거

### flake8 (https://www.flake8rules.com/ )

- 파이썬 정적유형검사기
- flake8 설치 후 settings(ctrl+shift+p 하고 입력하여 선택) 에서 flake8을 검색하면 나오는 세팅 항목 중 우선적으로 적용할 포매터로 black fommater를 선택한다.
- ignore 적용은 'ctrl+shift+p' > settings > '사용자설정' 선택 > flake8 검색 > flake8 :Args > 항목추가 >
- `--ignore=E501, W403` 형식으로 입력

이제 저장할 때마다 자동으로 Black fommating을 맞춰주면서 Isort가 Import순서를 정렬해줌

### PYTEST

성능프로파일링을 위한 코드테스트용 확장팩

### tabnine

AI 코드 자동완성 기능을 지원하는 확장팩

### AWS codeWhisperer

아마존에서 지원하는 AI코드 자동완성기능 및 코드 체크용 챗봇(한국어 X)

# 가상환경(venv) 생성

## python 현재 버전 적용

python -m venv ./env

python 특정버전 적용방법(시스템에 설치되어있어야 가능)

py -3.6 -m venv ./env

## 버추얼환경 적용(venv 폴더에 진입 후)

윈도우 : `<venv>\Scripts\activate.bat 또는 Activate.ps1`

리눅스 : source `<venv>/bin/activate`

## 가상환경 삭제

sudo -rm -rf 가상환경이름(가상환경폴더에서)

## 현재 설치된 패키지 확인

pip list

conda list

## 패키지 목록 설치

pip install -r requirements.txt

## 패키지제거

pip uninstall 패키지명

## 현재 패키지 출력

pip freeze > requirements.txt

# 환경변수설정

api-key 등의 민감한 값등의 별도 관리를 위해 다음을 설치

pip install python-dotenv

소스가 있는 폴더내에 다음의 파일생성 후 메모장으로 작성 후 api key 등 설정 값을 저장

.env

> API_KEY = "12345"
>
> 이런 식으로 저장

사용법

> import os
>
> os.getenv("API_KEY")

# Streamlit 사용

> 커맨드 실행창에서 아래의 명렁어 실행

    streamlit run [실행파일 경로]

ex ) 이렇게 실행

    streamlit run ./pages/03_QuizGPT.py

## 환경설정 방법

.streamlit 폴더내에

secrets.toml 파일을 생성하여 메모장으로 작성

서버 종료 단축키

ctrl + c

# tunnel 구축

내부 서버를 외부망(인터넷)에서 접속하기 위한 도구를 사용하기 위해 설치한다.

- 홈페이지 : [https://www.cloudflare.com/ko-kr/](https://www.cloudflare.com/ko-kr/)
- 가이드 : [https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)

* 참고용 : 보안네트워크에서 터널링 사용해서 외부서비스 접근방법[https://blog.selectfromuser.com/cloudflare-tunnel/](https://blog.selectfromuser.com/cloudflare-tunnel/)

## Cloudflare설치

다음의 명령어를 실행하여 설치한다. ()powershell 에서 실행)

`winget install --id Cloudflare.cloudflared`

## Tunnel 실행 명령어

tunnel을 적용할 IP와 포트를 입력 후 실행하면 제공되는 url로 외부에서 내부 서버의 접근이 가능하다.

`cloudflared tunnel --url http://127.0.0.1:8000`

# 문제해결

    OSError: [Errno 28] inotify watch limit reached

`streamlit run app.py` 뒤에 다음 코드를 붙여주면 해결 `--server.fileWatcherType none`

    사이트에 연결할 수 없음 --.--.--.-- 에서 연결을 거부했습니다.

`streamlit run app.py` 뒤에 다음 코드를 넣어주면 해결 `--server.port 30001`
OSError: [Errno28] 이후에 발생했다면, 전체코드는 다음과 같음
`streamlit run app.py --server.port 30001 --server.fileWatcherType none`

    raise OSError(errno.EMFILE, "inotify instance limit reached")

- 원인 :Linux 시스템에서 inotify인스턴스 제한도달
- 헤결 :
  1. sysctl.config 수정하여 인스턴스 추가 `sudo vi /etc/sysctl.conf`
  2. inotify인스턴스 늘리는 라인 추가 `fs.inotify.max_user_instances=1024`작성후 `:wq` 로 저장하고 종료
  3. 시스탬애 반영
     `sudo sysctl -p`

Git에서 최초 commit할 때 username, email 설정 방법
`$ git config --global user.name "John Doe"`
`$ git config --global user.email johndoe@example.com`
