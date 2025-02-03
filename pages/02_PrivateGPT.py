import streamlit as st

# from altair import LocalMultiTimeUnit
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOllama
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings  # OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter, OllamaEmbeddings
from langchain.vectorstores import FAISS

# from uuid import UUID


st.set_page_config(
    page_title="PrivateGPT",
    page_icon="��🔒",
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


#
#
llm = ChatOllama(
    model="mistral:latest",
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallBackHandler(),
    ],
)

#
#

if "messages" not in st.session_state:
    st.session_state["messages"] = []


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    # st.write(file)
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    # st.write(file_content, file_path)
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
    # separator를 사용하면 특정 문자열을 지정하여 문장을 분할할수 있다.
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,  # 최대 글자 개수
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    #
    #
    embeddings = OllamaEmbeddings(model="mistral:latest")
    #
    #

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
    """
        Answer the question using ONLY the following context and not your traning data.
        If you don't know the answer just say you don't know. DON'T make anything up.

        Context: {context}
        Question: {question}
    """
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
    file = st.file_uploader(
        "Upload a.txt .pdf. or .docx file",
        type=["pdf", "txt", "docx"],
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

# 도전과제 : 왼쪽사이드메뉴 하단에 셀렉트박스를 넣어서 모델을 선택할수 있게 해보자.add()
# 추천모델 : llama, mistral, falcon
