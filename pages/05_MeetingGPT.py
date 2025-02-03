import glob
import math
import os
import subprocess

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from openai import OpenAI
from pydub import AudioSegment

llm = ChatOpenAI(
    temperature=0.1,
)

has_transcript = os.path.exists("./.cache/SteveJobs.txt")

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,  # 문서크기 설정
    chunk_overlap=100,  # 마지막 문장의 오버랩 길이 설정
)


@st.cache_resource(show_spinner=False)
def embed_file(file_path):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    # .cache에 저장된 embedding문서가 있으면 cache값을 읽어오고, 없으면 cache를 새로 만든다.
    cached_embaddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embaddings)
    # -------신규작성 및 추가부분 시작 ---------------
    retriever = vectorstore.as_retriever()
    return retriever


# 텍스트 변수를 만들지 않고 읽는 즉시 text파일로 저장하도록 수정(메모리 절약)
@st.cache_data(show_spinner=False)
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:
        return
    client = OpenAI()
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()  # 정렬 작업 추가
    for file in files:
        # open(destination, "a") 중 "a"는 append 라는 의미
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
            text_file.write(transcript.text)


@st.cache_data(show_spinner=False)
def extract_audio_from_video(video_path):
    if has_transcript:
        return
    audio_path = video_path.replace(".mp4", ".mp3")
    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-vn",
        audio_path,
        "-y",
    ]
    subprocess.run(command)
    print("Audio file is created from original source.")


@st.cache_data(show_spinner=False)
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    # 파일 갯수 확인(전체 파일 길이 / 청크 길이)
    chunks = math.ceil(len(track) / chunk_len)
    print("The audio file is currently being split...")

    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(f"{chunks_folder}/chunk_{i+1}.mp3", format="mp3")
        print(f"to {i+1}/{chunks} files.")
    print("The audio file has been successfully split.")


st.set_page_config(
    page_title="MeetingGPT",
    page_icon="🕓",
)
st.title("MeetingGPT")

st.markdown(
    """
    Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any question about it.

    Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mkv", "mov"],
    )
if video:
    chunks_folder = "./.cache/chunks"
    with st.status("Loading video...") as status:
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace(".mp4", ".mp3")
        transcript_path = video_path.replace(".mp4", ".txt")
        with open(video_path, "wb") as f:
            f.write(video_content)
        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path)
        status.update(label="Cutting audio segments...")
        cut_audio_in_chunks(audio_path, 10, chunks_folder)
        status.update(label="Transcribing audio...")
        transcribe_chunks(chunks_folder, transcript_path)

    transcribe_tab, summary_tab, qa_tab = st.tabs(
        [
            "Transcript",
            "Summary",
            "Q&A",
        ]
    )

    with transcribe_tab:
        with open(transcript_path, "r") as file:
            st.write(file.read())
    # Refine Chain 적용
    with summary_tab:
        start = st.button("Generate Summary")

        if start:
            loader = TextLoader(transcript_path)

            docs = loader.load_and_split(text_splitter=splitter)

            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                Write a concise summary of the following: 
                "{text}"
                CONCISE SUMMARY:
                """
            )

            first_summary_chain = first_summary_prompt | llm | StrOutputParser()

            summary = first_summary_chain.invoke({"text": docs[0].page_content})

            refine_prompt = ChatPromptTemplate.from_template(
                """
                Your job is to produce a final summary.
                We have provided an existing summary up to a certain point:
                {existing_summary}
                We have the opportunity to refine the existing summary (only if needed) with some more context blow.
                -----------------
                {context}
                -----------------
                Given the new context, refine the original summary.
                If the context isn't useful, RETURN the original summary.
                """
            )

            refine_chain = refine_prompt | llm | StrOutputParser()

            with st.status("Summarizing...") as status:
                for i, doc in enumerate(docs[1:]):
                    status.update(label=f"Processing document {i+1}/{len(docs)-1}")
                    summary = refine_chain.invoke(
                        {
                            "existing_summary": summary,
                            "context": doc.page_content,
                        }
                    )
                    st.write(summary)  # 각 페이지당 요약본 출력
            st.write(summary)  # 전체 요약본 출력

    with qa_tab:
        retriever = embed_file(transcript_path)

        docs = retriever.invoke("do they talk about web service of future? ")

        st.write(docs)

        # TODO 추가 작업 필요
        # file에 대한 질문에 답변하기 위해서 Stuff chain, map-reduce chain, map-rerank chain 중에서 적용 검토
        # 1. Q&A 챗봇 생성
        # 2. 챗봇 생성 시 입력 문장 추가
