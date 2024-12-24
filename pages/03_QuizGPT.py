import json
from typing import Any

import streamlit as st

# 모델의 실시간 답변 출력 기능
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import PromptTemplate
from langchain.retrievers import WikipediaRetriever

# from langchain.schema import BaseOutputParser
from langchain.text_splitter import CharacterTextSplitter

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

prompt = PromptTemplate.from_template(
    """Make a {difficulty} level quiz about {context}.

    For Easy level:
    - Use simple and straightforward questions
    - Focus on basic facts and definitions
    - Use clear and simple language

    For Medium level:
    - Include some detailed questions
    - Mix basic and advanced concepts
    - Add some questions that require understanding relationships between concepts

    For Hard level:
    - Use complex analytical questions
    - Include questions that require deep understanding
    - Add questions that combine multiple concepts
    - Include some technical terminology

    Make sure the questions match the selected difficulty level.
    Questions at each {difficulty} level should not be identical to those at other levels.
    And if input language is Korean, Answer in Korean. Otherwise, answer in English.
    """
)


def format_docs(docs):
    # 문자열 입력도 처리할수 있도록 수정
    if isinstance(docs, str):
        return docs
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    # st.write(file)
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    # st.write(file_content, file_path)
    with open(file_path, "wb") as f:
        f.write(file_content)
    # separator를 사용하면 특정 문자열을 지정하여 문장을 분할할수 있다.
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,  # 최대 글자 개수
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    # top_k_results=1 적용할 문서의 수(1, 2, 3.. 순번까지)
    retriever = WikipediaRetriever(
        top_k_results=3,
        lang="en",  # 언어를 지정할수 있다. 한국어:ko, 영어:en
        wiki_client=Any,
    )
    docs = retriever.get_relevant_documents(term)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, difficulty: str = "Medium"):
    # docs가 리스트인 경우 format_docs 적용
    context = format_docs(_docs) if isinstance(_docs, list) else _docs
    response = formatting_chain.invoke({"context": context, "difficulty": difficulty})
    return json.loads(response.additional_kwargs["function_call"]["arguments"])


with st.sidebar:
    docs = None
    topic = None
    openai_api_key = ""

    # API KEY 입력받기
    if "openai_api_key" in st.session_state:
        st.success("API Key has been set.")
    else:
        openai_api_key = st.text_input(
            "Enter your OpenAI API Key",
            type="password",
            help="OpenAI에서 발급하는 API KEY를 입력하세요. API KEY를 입력해야만 AI에게 응답을 받을수 있습니다. API KEY가 없다면 다음의 Link를 방문해서 발급받으세요. https://platform.openai.com/docs/api-reference/introduction",
        )

    # API KEY is TRUE
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
        st.success("API Key has been set.")
    else:
        if "openai_api_key" in st.session_state:
            openai_api_key = st.session_state.openai_api_key
        else:
            st.warning("Please enter your OpenAI API Key.")

    # LLM 설정 시작----------
    if openai_api_key:
        llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-4o-mini",
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            api_key=openai_api_key,
        ).bind(function_call={"name": "create_quiz"}, functions=[function])
        # chain 세팅
        formatting_chain = prompt | llm
    else:
        llm = None
    # LLM 설정 끝------------

    # 시험 난이도 선택
    level = st.selectbox(
        "Choose the level of difficulty.",
        (
            "Choose the level",
            "Easy",
            "Medium",
            "Hard",
        ),
    )
    if level == "Easy":
        st.warning("You choose level is Easy")
    elif level == "Medium":
        st.warning("You choose level is Medium")
    elif level == "Hard":
        st.warning("You choose level is Hard")
    else:
        st.warning("Please select a difficulty level first. Default is Medium")

    # 이용목적 선택
    if openai_api_key:
        choice = st.selectbox(
            "Choose what you want to use.",
            (
                "File",
                "Wikipedia Article",
            ),
        )
    else:
        choice = st.selectbox(
            "Choose what you want to use.",
            ("API KEY Required",),
            disabled=True,
        )

    # 파일 입력칸
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)

    # wiki 검색칸
    elif choice == "Wikipedia Article":
        topic = st.text_input("Search Wikipedia..")
        if topic:
            docs = wiki_search(topic)
    else:
        st.text_input(
            label="Search Wikipedia..", value="API KEY Required", disabled=True
        )

    # github link
    st.write("**Github Repo:**   (https://github.com/smin987/ngpt)")

# 이용 안내문
if not docs:
    st.markdown(
        """
    Welcome to QuizGPT

    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    # 난이도가 선택되지 않았을 경우 처리
    current_level = level if level not in [None, "Choose the level"] else "Medium"

    response = run_quiz_chain(
        docs,
        topic if topic else getattr(file, "name", "default_topic"),
        difficulty=str(current_level),
    )

    # 점수 계산을 위한 변수 초기화
    total_questions = len(response["questions"])
    correct_answers = 0

    # st.write(response)
    with st.form("questions_form"):
        for i, question in enumerate(response["questions"]):
            st.write(f"Question {i+1}:{question['question']}")
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
                key=f"question_{i}",  # 각 라디오 버튼에 고유한 key 추가
            )
            # 답변이 선택되었을 때만 정답 체크
            if value is not None:
                if {"answer": value, "correct": True} in question["answers"]:
                    correct_answers += 1
                    # st.success("Correct!")
                # elif value is not None:
                #     st.error("Wrong..")
        submit_button = st.form_submit_button("Submit Quiz")

        if submit_button:
            # 모든 문제에 답했는지 확인
            answered_questions = sum(
                1
                for q in response["questions"]
                if st.session_state.get(f"question_{response['questions'].index(q)}")
                is not None
            )

            if answered_questions < total_questions:
                st.warning(
                    f"Please answer all questions before submitting. ({answered_questions}/{total_questions} answered)"
                )
            else:
                # 점수 계산 및 표시
                score = (correct_answers / total_questions) * 100
                st.write(f"Your score: {score:.1f}%")
                st.write(f"Correct answers: {correct_answers}/{total_questions}")

                # 만점인 경우
                if score == 100:
                    st.balloons()
                    st.success("🎉 Congratulations! Perfect Score! 🎉")
                else:
                    # 재시험 버튼
                    if st.form_submit_button("Try Again"):
                        st.experimental_rerun()

                # 상세결과 표시
                with st.expander("Show Detail Results"):
                    for i, question in enumerate(response["questions"]):
                        selected = st.session_state.get(f"question_{i}")
                        correct_answer = next(
                            answer["answer"]
                            for answer in question["answers"]
                            if answer["correct"]
                        )

                        if selected == correct_answer:
                            st.success(f"Question {i+1}: Correct! ✅")
                        else:
                            st.error(f"Question{i+1}: Wrong ❌")
                            st.write(f"Your answer: {selected}")
                            st.write(f"Correct answer: {correct_answer}")
