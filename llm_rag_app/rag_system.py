"""
FAISS 기반 경량 RAG 시스템 구현

이 모듈은 FAISS를 사용하여 문서 임베딩을 저장하고 검색하는 경량 RAG 시스템을 구현합니다.
"""

import os
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.schema import Document

class FAISSVectorStore:
    """FAISS 기반 벡터 저장소 클래스"""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        FAISS 벡터 저장소 초기화
        
        Args:
            embedding_model_name: 사용할 HuggingFace 임베딩 모델 이름
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """
        디렉토리에서 문서 로드
        
        Args:
            directory_path: 문서가 저장된 디렉토리 경로
            
        Returns:
            로드된 문서 리스트
        """
        try:
            loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader)
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"문서 로드 중 오류 발생: {e}")
            return []
    
    def load_documents_from_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[Document]:
        """
        텍스트 리스트에서 문서 생성
        
        Args:
            texts: 텍스트 리스트
            metadatas: 메타데이터 리스트 (선택 사항)
            
        Returns:
            생성된 문서 리스트
        """
        documents = []
        
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
        
        for i, text in enumerate(texts):
            documents.append(Document(page_content=text, metadata=metadatas[i]))
        
        return documents
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        문서를 청크로 분할
        
        Args:
            documents: 원본 문서 리스트
            
        Returns:
            분할된 문서 청크 리스트
        """
        return self.text_splitter.split_documents(documents)
    
    def create_vector_store(self, documents: List[Document], save_path: Optional[str] = None) -> None:
        """
        문서로부터 벡터 저장소 생성
        
        Args:
            documents: 문서 리스트
            save_path: 벡터 저장소 저장 경로 (선택 사항)
        """
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        if save_path:
            self.save_vector_store(save_path)
    
    def save_vector_store(self, save_path: str) -> None:
        """
        벡터 저장소 저장
        
        Args:
            save_path: 저장 경로
        """
        if self.vector_store:
            self.vector_store.save_local(save_path)
            print(f"벡터 저장소가 {save_path}에 저장되었습니다.")
    
    def load_vector_store(self, load_path: str) -> None:
        """
        벡터 저장소 로드
        
        Args:
            load_path: 로드 경로
        """
        if os.path.exists(load_path):
            self.vector_store = FAISS.load_local(load_path, self.embeddings)
            print(f"벡터 저장소가 {load_path}에서 로드되었습니다.")
        else:
            print(f"경로 {load_path}에 벡터 저장소가 존재하지 않습니다.")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        쿼리와 유사한 문서 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            
        Returns:
            검색된 문서 리스트
        """
        if not self.vector_store:
            raise ValueError("벡터 저장소가 초기화되지 않았습니다. 먼저 create_vector_store 또는 load_vector_store를 호출하세요.")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        기존 벡터 저장소에 문서 추가
        
        Args:
            documents: 추가할 문서 리스트
        """
        if not self.vector_store:
            self.create_vector_store(documents)
        else:
            self.vector_store.add_documents(documents)

class RAGSystem:
    """
    FAISS 기반 RAG 시스템 클래스
    """
    
    def __init__(self, vector_store_path: Optional[str] = None):
        """
        RAG 시스템 초기화
        
        Args:
            vector_store_path: 기존 벡터 저장소 경로 (선택 사항)
        """
        self.vector_store = FAISSVectorStore()
        
        if vector_store_path and os.path.exists(vector_store_path):
            self.vector_store.load_vector_store(vector_store_path)
    
    def add_documents_from_directory(self, directory_path: str, save_path: Optional[str] = None) -> None:
        """
        디렉토리에서 문서 추가
        
        Args:
            directory_path: 문서 디렉토리 경로
            save_path: 벡터 저장소 저장 경로 (선택 사항)
        """
        documents = self.vector_store.load_documents_from_directory(directory_path)
        chunks = self.vector_store.process_documents(documents)
        
        if not self.vector_store.vector_store:
            self.vector_store.create_vector_store(chunks, save_path)
        else:
            self.vector_store.add_documents(chunks)
            
            if save_path:
                self.vector_store.save_vector_store(save_path)
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, save_path: Optional[str] = None) -> None:
        """
        텍스트 리스트 추가
        
        Args:
            texts: 텍스트 리스트
            metadatas: 메타데이터 리스트 (선택 사항)
            save_path: 벡터 저장소 저장 경로 (선택 사항)
        """
        documents = self.vector_store.load_documents_from_texts(texts, metadatas)
        chunks = self.vector_store.process_documents(documents)
        
        if not self.vector_store.vector_store:
            self.vector_store.create_vector_store(chunks, save_path)
        else:
            self.vector_store.add_documents(chunks)
            
            if save_path:
                self.vector_store.save_vector_store(save_path)
    
    def query(self, query_text: str, k: int = 4) -> List[Document]:
        """
        쿼리 실행
        
        Args:
            query_text: 쿼리 텍스트
            k: 반환할 문서 수
            
        Returns:
            검색된 문서 리스트
        """
        if not self.vector_store.vector_store:
            raise ValueError("벡터 저장소가 초기화되지 않았습니다. 먼저 문서를 추가하세요.")
        
        return self.vector_store.similarity_search(query_text, k=k)

# 샘플 사용법
if __name__ == "__main__":
    # RAG 시스템 초기화
    rag_system = RAGSystem()
    
    # 샘플 텍스트 추가
    sample_texts = [
        "인공지능(AI)은 인간의 학습, 추론, 지각, 문제 해결 능력 등을 컴퓨터 프로그램으로 구현한 기술입니다.",
        "머신러닝은 컴퓨터가 데이터로부터 학습하여 예측이나 결정을 내릴 수 있게 하는 인공지능의 한 분야입니다.",
        "딥러닝은 인공 신경망을 기반으로 하는 머신러닝의 한 종류로, 복잡한 패턴을 인식하는 데 뛰어난 성능을 보입니다.",
        "자연어 처리(NLP)는 컴퓨터가 인간의 언어를 이해하고 처리하는 기술로, 번역, 감정 분석, 텍스트 요약 등에 활용됩니다."
    ]
    
    sample_metadatas = [
        {"source": "AI 개요", "author": "김인공"},
        {"source": "머신러닝 기초", "author": "이학습"},
        {"source": "딥러닝 소개", "author": "박신경망"},
        {"source": "NLP 기술", "author": "최언어"}
    ]
    
    # 벡터 저장소에 텍스트 추가
    rag_system.add_texts(sample_texts, sample_metadatas, save_path="./vector_store")
    
    # 쿼리 실행
    query = "인공지능이란 무엇인가요?"
    results = rag_system.query(query)
    
    print(f"쿼리: {query}")
    print("\n검색 결과:")
    for i, doc in enumerate(results):
        print(f"\n문서 {i+1}:")
        print(f"내용: {doc.page_content}")
        print(f"메타데이터: {doc.metadata}")
