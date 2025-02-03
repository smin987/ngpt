from typing import Any

from fastapi import Body, FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# 서버의 역할을 설명
app = FastAPI(
    title="Nicolacus Maximus Quote Giver",
    description="Get a real quote said by Nicolacus Maximus himself.",
    servers=[
        {
            "url": "https://amsterdam-distance-jerusalem-categories.trycloudflare.com",
        }
    ],
)


# 서버가 하는 일 설정
class Quote(BaseModel):
    quote: str = Field(
        description="The quote said by Nicolacus Maximus.",
    )
    year: int = Field(
        description="The year when Nicolacus Maximus siad the quote.",
    )


# 서버가 어떻게 응답해야 하는지 지시
@app.get(
    "/quote",
    summary="Returns a random quote by Nicolacus Maximus",
    description="Upon receiving a Get request this endpoint will return a real quote said by Nicolacus Maximus himself.",
    response_description="A Quote object that contains the quote said by Nicolacus Maximus and the date when the quote was said.",
    response_model=Quote,
    # 사용자에게 데이터 전송여부를 묻는 팝업창
    openapi_extra={
        "x-openai-isConsequential": False,  # 선택버튼 : 항상허용하기, 허용하기, 취소
        # "x-openai-isConsequential": True, # 선택버튼 : 허용하기, 취소
    },
)
# 콘솔창에서 실행 : uvicorn main:app --reload
def get_quote(request: Request):
    print(request.headers)
    return {
        "quote": "Life is short so eat it all.",
        "year": 1950,
    }


user_token_db = {
    "123456": "nico",
}


@app.get(
    "/authorize",
    response__class=HTMLResponse,
    include_in_schema=False,
)
def handle_authorize(
    client_id: str,
    redirect_uri: str,
    state: str,
):
    # print(
    #     client_id,
    #     redirect_uri,
    #     state,
    # )
    # return {
    #     "ok": True,
    # }
    return f"""
    <html>
        <head>
            <title>Nicolacus Maximus Log In</title>
        </head>
        <body>
            <h1>Authorize</h1>
            <a href="{redirect_uri}?code=123456&state={state}">Nicolacus Maximus GPT</a>
        </body>
    </html>
    """


# @app.post("/token")
# def handle_token(payload: Any = Body(None)):
#     print(payload)
#     return {"x": "x"}


@app.post(
    "/token",
    include_in_schema=False,
)
def handle_token(code=Form(...)):
    return {
        "access_token": user_token_db[code],
    }


# 외부에서 접근하기 위해 콘솔창에 아래의 명령어를 입력
# cloudflared tunnel --url http://127.0.0.1:8000
