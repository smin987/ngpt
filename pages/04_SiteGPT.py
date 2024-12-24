import asyncio
import sys

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import SitemapLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.

    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        Use ONLY the following pre-existing answers to answer the user's question.

        Use the answers that have the highest score (more helpful) and favor the most recent ones.

        Return the sources of the answers as they are, do not change them. Give me the URL and date.

        Answers : {answers}

        """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    chose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"Answer:{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return chose_chain.invoke({"question": question, "answers": condensed})


# XML문서의경우 XML파서를 사용하는 것이 더 안정적. lxml패키지를 설치 후 features="xml" 키워드 인수를 BeautifulSoup 클래스 생성자에 전달.
def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


# 에러가 발생하므로 @st.cache_data를 아래의 데코레이터로 교체함
@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        # 필터로 가져올 url을 정할수 있다.
        filter_urls=[
            # "https://developers.cloudflare.com/ai-gateway/",
            # 정규식 적용가능
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/workers-ai\/).*",
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 1
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings(api_key=openai_api_key))
    return vector_store.as_retriever()


if "messages" not in st.session_state:
    st.session_state["messages"] = []


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


st.markdown(
    """
    # SiteGPT

    Ask questions about the content of a website.

    Start by writing the URL of the website on the sidebar.
"""
)

if "win32" in sys.platform:
    # Windows specific event-loop policy & cmd
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    cmds = [["C:/Windows/system32/HOSTNAME.EXE"]]
else:
    # Unix default event-loop policy & cmds
    cmds = [
        ["du", "-sh", "/home/sm987/Desktop"],
        ["du", "-sh", "/home/sm987"],
        ["du", "-sh", "/home/sm987/Pictures"],
    ]

with st.sidebar:
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

    if openai_api_key:
        llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-4o-mini",
            api_key=openai_api_key,
        )
    else:
        llm = None
    # LLM 설정 끝------------
    if openai_api_key:
        url = st.text_input(
            "Write down a sitemap URL",
            placeholder="https://example.com/sitemap.xml",
        )
    else:
        url = st.text_input(
            "Write down a sitemap URL",
            disabled=True,
            placeholder="API KEY Required",
        )

    # github link
    st.write("**Github Repo:**   (https://github.com/smin987/ngpt)")

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        retriever = load_website(url)
        # 이전 대화기록 표시
        paint_history()
        query = st.chat_input("Ask a question about the website.")

        if query:

            # 사용자 질문 저장 및 표시
            send_message(query, "human")

            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )

            result = chain.invoke(query)

            # AI의 답변 저장 및 표시
            send_message(result.content.replace("$", "\\$"), "AI")
