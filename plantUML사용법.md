# Plant UML 사용법

PlantUML은 코드로 작성가능한 Diagram 도구로, class uml, ERD, Usecase 등 다양한 용도의 diagram을 작성할수 있다.

## 1. 설치 (windows)

### 가. chocolatey 설치 (링크 : [https://chocolatey.org/install](https://chocolatey.org/install) )

관리자 권한으로 Powershell을 실행하여 아래의 명령어를 입력하고 설치가 완료될 때 까지 기다린다.

```Set-ExecutionPolicy
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

### 나. VSCODE 설치 (링크 : [https://code.visualstudio.com/](https://code.visualstudio.com/))

위의 url을 찾아가 다운로드 받아서 설치 후 다음의 플러그인을 설치한다.

### 다. PlantUML 설치

- PlantUML : PlantUML 플러그인
- PlantUML Syntax : 문법검사
- PlantUML -Simple Viewer : UML뷰어

  플러그인 설치 후 cmd 콘솔에 다음의 명령어를 입력해서 Windows PC에 PlantUML을 설치한다.

```cmd-admin
choco install plantuml
```

## 2. 사용법

### 가. 파일 확장자

.wsd 확장자로 파일을 생성한다.

### 나. uml 코드를 작성한다

docs를 확인하여 작성한다.

바로가기 :[https://plantuml.com/ko/](%E2%80%B8https://plantuml.com/ko/)

vscode 에서 작성 후 키보드에서 'alt' + 'D'를 누르면 미리보기가 생성된다.

### 다. 내보내기

VSCODE 단축키 'ctrl' + 'shift' + 'p' 를 누르고 콘솔 화면에 plantuml을 입력하여 검색한다.

(여기서는 단축키 'alt' + 'v' 로 설정 하였다. - 톱니바퀴 아이콘 누르면 단축키 설정가능 )

plantUML 내보내기를 선택한 후 내보낼 이미지 확장자를 선택한다.

### 라. 온라인 서버

아래의 링크를 클릭하면 plantuml 온라인서버에 접근할수 있다.

링크 : [https://www.plantuml.com/plantuml/uml/SyfFKj2rKt3CoKnELR1Io4ZDoSa70000](https://www.plantuml.com/plantuml/uml/SyfFKj2rKt3CoKnELR1Io4ZDoSa70000)
