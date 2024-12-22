import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📄",
)


class ChatCallBackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


# llm = ChatOpenAI(
#     temperature=0.1,
#     streaming=True,
#     callbacks=[
#         ChatCallBackHandler(),
#     ],
# )

if "messages" not in st.session_state:
    st.session_state["messages"] = []


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    # st.write(file)
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    # st.write(file_content, file_path)
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    # separator를 사용하면 특정 문자열을 지정하여 문장을 분할할수 있다.
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,  # 최대 글자 개수
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    # .cache에 저장된 embedding문서가 있으면 cache값을 읽어오고, 없으면 cache를 새로 만든다.
    cached_embaddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embaddings)
    # -------신규작성 및 추가부분 시작 ---------------
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


st.title("DocumentGPT")

st.markdown(
    """
    Welcome! 환영합니다.

    Use this chatbot to ask questions to an AI about yout Files!

    Upload your files on the sidebar.
    """
)

with st.sidebar:

    # API KEY 입력받기
    openai_api_key = st.text_input(
        "Enter your OpenAI API Key",
        type="password",
        help="OpenAI에서 발급하는 API KEY를 입력하세요. API KEY를 입력해야만 AI에게 응답을 받을수 있습니다. API KEY가 없다면 다음의 Link를 방문해서 발급받으세요. https://platform.openai.com/docs/api-reference/introduction",
    )

    # API KEY Setting
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
            streaming=True,
            callbacks=[
                ChatCallBackHandler(),
            ],
            api_key=openai_api_key,
        )
    else:
        llm = None

    if openai_api_key:
        file = st.file_uploader(
            "Upload a.txt .pdf. or .docx file",
            type=["pdf", "txt", "docx"],
        )
    else:
        file = st.file_uploader(
            "API KEY Required",
            disabled=True,
        )

if file:
    retriever = embed_file(file)
    send_message("I'm ready?! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            chain.invoke(message)


else:
    st.session_state["messages"] = []


# 우리의 Chain에는 아직 메모리가 없다. 메모리를 추가해보자.
