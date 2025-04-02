"""
Streamlit 웹 애플리케이션

이 모듈은 Langchain 기반 RAG 시스템과 LLM 모델을 통합한 Streamlit 웹 인터페이스를 구현합니다.
openwebui 디자인을 기반으로 스타일링되었습니다.
"""

import os
import streamlit as st
from dotenv import load_dotenv
import time
from rag_system import RAGSystem
from llm_api import DualModelChain
from styles import openwebui_css

# 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="LLM RAG 애플리케이션",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# openwebui 스타일 적용
st.markdown(f'<style>{openwebui_css}</style>', unsafe_allow_html=True)

# 세션 상태 초기화
if "rag_system" not in st.session_state:
    st.session_state.rag_system = RAGSystem()

if "llm_chain" not in st.session_state:
    st.session_state.llm_chain = DualModelChain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store_path" not in st.session_state:
    st.session_state.vector_store_path = "./vector_store"

# 사이드바 - 문서 업로드 및 관리
with st.sidebar:
    st.title("📚 문서 관리")
    
    # 문서 업로드
    st.header("문서 업로드")
    uploaded_files = st.file_uploader("텍스트 파일 업로드", type=["txt"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("문서 처리 및 임베딩"):
            with st.spinner("문서를 처리 중입니다..."):
                # 임시 디렉토리에 파일 저장
                temp_dir = "./temp_docs"
                os.makedirs(temp_dir, exist_ok=True)
                
                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                
                # 문서 처리 및 임베딩
                st.session_state.rag_system.add_documents_from_directory(
                    temp_dir, 
                    save_path=st.session_state.vector_store_path
                )
                
                st.success(f"{len(uploaded_files)}개의 문서가 성공적으로 처리되었습니다.")
    
    # 샘플 데이터 추가
    st.header("샘플 데이터")
    if st.button("샘플 데이터 추가"):
        with st.spinner("샘플 데이터를 추가 중입니다..."):
            sample_texts = [
                "인공지능(AI)은 인간의 학습, 추론, 지각, 문제 해결 능력 등을 컴퓨터 프로그램으로 구현한 기술입니다.",
                "머신러닝은 컴퓨터가 데이터로부터 학습하여 예측이나 결정을 내릴 수 있게 하는 인공지능의 한 분야입니다.",
                "딥러닝은 인공 신경망을 기반으로 하는 머신러닝의 한 종류로, 복잡한 패턴을 인식하는 데 뛰어난 성능을 보입니다.",
                "자연어 처리(NLP)는 컴퓨터가 인간의 언어를 이해하고 처리하는 기술로, 번역, 감정 분석, 텍스트 요약 등에 활용됩니다.",
                "RAG(Retrieval-Augmented Generation)는 검색 기반 생성 모델로, 외부 지식을 활용하여 더 정확하고 신뢰할 수 있는 응답을 생성합니다.",
                "벡터 데이터베이스는 임베딩된 벡터를 저장하고 효율적으로 검색할 수 있는 데이터베이스로, 유사도 검색에 최적화되어 있습니다.",
                "FAISS는 Facebook AI에서 개발한 벡터 유사도 검색 라이브러리로, 대규모 벡터 데이터셋에서 효율적인 검색을 지원합니다.",
                "Langchain은 LLM 애플리케이션 개발을 위한 프레임워크로, 다양한 컴포넌트를 조합하여 복잡한 AI 시스템을 구축할 수 있습니다."
            ]
            
            sample_metadatas = [
                {"source": "AI 개요", "author": "김인공"},
                {"source": "머신러닝 기초", "author": "이학습"},
                {"source": "딥러닝 소개", "author": "박신경망"},
                {"source": "NLP 기술", "author": "최언어"},
                {"source": "RAG 시스템", "author": "정검색"},
                {"source": "벡터 데이터베이스", "author": "강벡터"},
                {"source": "FAISS 라이브러리", "author": "임유사도"},
                {"source": "Langchain 프레임워크", "author": "오체인"}
            ]
            
            st.session_state.rag_system.add_texts(
                sample_texts, 
                sample_metadatas, 
                save_path=st.session_state.vector_store_path
            )
            
            st.success("샘플 데이터가 성공적으로 추가되었습니다.")
    
    # 시스템 정보
    st.header("시스템 정보")
    st.info("""
    **모델 정보:**
    - 추론 모델: deepseek-r1:7b
    - 출력 모델: exaone3.5
    - 임베딩 모델: all-MiniLM-L6-v2
    
    **벡터 저장소:**
    - FAISS (Facebook AI Similarity Search)
    """)

# 메인 화면 - 채팅 인터페이스
st.title("🤖 LLM RAG 애플리케이션")
st.markdown("""
이 애플리케이션은 Langchain 기반 RAG(Retrieval-Augmented Generation)를 활용한 질의응답 시스템입니다.
- **deepseek-r1:7b** 모델은 추론 과정에 사용됩니다.
- **exaone3.5** 모델은 최종 출력 생성에 사용됩니다.
- **FAISS**를 사용하여 문서 임베딩을 저장하고 검색합니다.
""")

# 채팅 기록 표시
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # 추론 과정 표시 (사용자가 요청한 경우)
        if message["role"] == "assistant" and "reasoning" in message:
            with st.expander("추론 과정 보기"):
                st.write(message["reasoning"])

# 사용자 입력
user_query = st.chat_input("질문을 입력하세요...")

if user_query:
    # 사용자 메시지 추가
    st.chat_message("user").write(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    # 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("응답을 생성 중입니다..."):
            # 관련 문서 검색
            try:
                retrieved_docs = st.session_state.rag_system.query(user_query, k=3)
                
                # 문서가 없는 경우 샘플 데이터 추가
                if not retrieved_docs:
                    st.warning("관련 문서가 없습니다. 샘플 데이터를 추가합니다.")
                    sample_texts = [
                        "인공지능(AI)은 인간의 학습, 추론, 지각, 문제 해결 능력 등을 컴퓨터 프로그램으로 구현한 기술입니다.",
                        "머신러닝은 컴퓨터가 데이터로부터 학습하여 예측이나 결정을 내릴 수 있게 하는 인공지능의 한 분야입니다."
                    ]
                    st.session_state.rag_system.add_texts(sample_texts, save_path=st.session_state.vector_store_path)
                    retrieved_docs = st.session_state.rag_system.query(user_query, k=2)
                
                # LLM 체인 실행
                result = st.session_state.llm_chain.run(user_query, retrieved_docs)
                
                # 응답 표시
                st.write(result["response"])
                
                # 추론 과정 표시
                with st.expander("추론 과정 보기"):
                    st.write(result["reasoning"])
                
                # 참조 문서 표시
                with st.expander("참조 문서 보기"):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**문서 {i+1}:** {doc.metadata.get('source', '알 수 없음')}")
                        st.markdown(f"**작성자:** {doc.metadata.get('author', '알 수 없음')}")
                        st.markdown(f"**내용:** {doc.page_content}")
                        st.markdown("---")
                
                # 채팅 기록에 추가
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": result["response"],
                    "reasoning": result["reasoning"]
                })
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
                st.info("문서를 먼저 업로드하거나 사이드바에서 샘플 데이터를 추가해주세요.")

# 푸터
st.markdown("---")
st.markdown("© 2025 LLM RAG 애플리케이션 | Powered by Langchain, FAISS, Streamlit")
