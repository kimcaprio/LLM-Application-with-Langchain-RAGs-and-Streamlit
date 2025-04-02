"""
애플리케이션 테스트 스크립트

이 스크립트는 LLM RAG 애플리케이션의 주요 기능을 테스트합니다.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# 현재 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 테스트할 모듈 임포트
from rag_system import FAISSVectorStore, RAGSystem
from llm_api import DeepseekLLM, ExaoneLLM, DualModelChain
from langchain.schema import Document

class TestFAISSVectorStore(unittest.TestCase):
    """FAISS 벡터 저장소 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        self.vector_store = FAISSVectorStore()
        self.test_texts = [
            "인공지능(AI)은 인간의 학습, 추론, 지각, 문제 해결 능력 등을 컴퓨터 프로그램으로 구현한 기술입니다.",
            "머신러닝은 컴퓨터가 데이터로부터 학습하여 예측이나 결정을 내릴 수 있게 하는 인공지능의 한 분야입니다."
        ]
        self.test_metadatas = [
            {"source": "AI 개요", "author": "김인공"},
            {"source": "머신러닝 기초", "author": "이학습"}
        ]
    
    def test_load_documents_from_texts(self):
        """텍스트에서 문서 로드 테스트"""
        documents = self.vector_store.load_documents_from_texts(self.test_texts, self.test_metadatas)
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].page_content, self.test_texts[0])
        self.assertEqual(documents[0].metadata, self.test_metadatas[0])
    
    def test_process_documents(self):
        """문서 처리 테스트"""
        documents = self.vector_store.load_documents_from_texts(self.test_texts, self.test_metadatas)
        chunks = self.vector_store.process_documents(documents)
        self.assertGreaterEqual(len(chunks), len(documents))
    
    @patch('langchain.vectorstores.FAISS.from_documents')
    def test_create_vector_store(self, mock_from_documents):
        """벡터 저장소 생성 테스트"""
        mock_from_documents.return_value = MagicMock()
        documents = self.vector_store.load_documents_from_texts(self.test_texts, self.test_metadatas)
        self.vector_store.create_vector_store(documents)
        mock_from_documents.assert_called_once()

class TestRAGSystem(unittest.TestCase):
    """RAG 시스템 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        self.rag_system = RAGSystem()
        self.test_texts = [
            "인공지능(AI)은 인간의 학습, 추론, 지각, 문제 해결 능력 등을 컴퓨터 프로그램으로 구현한 기술입니다.",
            "머신러닝은 컴퓨터가 데이터로부터 학습하여 예측이나 결정을 내릴 수 있게 하는 인공지능의 한 분야입니다."
        ]
        self.test_metadatas = [
            {"source": "AI 개요", "author": "김인공"},
            {"source": "머신러닝 기초", "author": "이학습"}
        ]
    
    @patch.object(FAISSVectorStore, 'load_documents_from_texts')
    @patch.object(FAISSVectorStore, 'process_documents')
    @patch.object(FAISSVectorStore, 'create_vector_store')
    def test_add_texts(self, mock_create_vector_store, mock_process_documents, mock_load_documents_from_texts):
        """텍스트 추가 테스트"""
        mock_documents = [MagicMock(), MagicMock()]
        mock_chunks = [MagicMock(), MagicMock(), MagicMock()]
        
        mock_load_documents_from_texts.return_value = mock_documents
        mock_process_documents.return_value = mock_chunks
        
        self.rag_system.add_texts(self.test_texts, self.test_metadatas)
        
        mock_load_documents_from_texts.assert_called_once_with(self.test_texts, self.test_metadatas)
        mock_process_documents.assert_called_once_with(mock_documents)
        mock_create_vector_store.assert_called_once_with(mock_chunks, None)

class TestLLMModels(unittest.TestCase):
    """LLM 모델 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        self.deepseek_llm = DeepseekLLM()
        self.exaone_llm = ExaoneLLM()
        self.dual_model_chain = DualModelChain()
    
    def test_deepseek_llm_call(self):
        """Deepseek LLM 호출 테스트"""
        prompt = "인공지능이란 무엇인가요?"
        result = self.deepseek_llm(prompt)
        self.assertIsInstance(result, str)
        self.assertIn("Deepseek", result)
    
    def test_exaone_llm_call(self):
        """Exaone LLM 호출 테스트"""
        prompt = "인공지능이란 무엇인가요?"
        result = self.exaone_llm(prompt)
        self.assertIsInstance(result, str)
        self.assertIn("Exaone", result)
    
    def test_dual_model_chain(self):
        """듀얼 모델 체인 테스트"""
        query = "인공지능이란 무엇인가요?"
        documents = [
            Document(page_content="인공지능(AI)은 인간의 학습, 추론, 지각, 문제 해결 능력 등을 컴퓨터 프로그램으로 구현한 기술입니다.", 
                    metadata={"source": "AI 개요", "author": "김인공"})
        ]
        
        result = self.dual_model_chain.run(query, documents)
        
        self.assertIsInstance(result, dict)
        self.assertIn("query", result)
        self.assertIn("reasoning", result)
        self.assertIn("response", result)
        self.assertEqual(result["query"], query)

if __name__ == "__main__":
    unittest.main()
