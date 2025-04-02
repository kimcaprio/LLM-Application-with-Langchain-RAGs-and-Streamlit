"""
애플리케이션 테스트 스크립트 (모킹 기반)

이 스크립트는 LLM RAG 애플리케이션의 주요 기능을 모킹을 사용하여 테스트합니다.
실제 의존성 없이도 테스트가 가능하도록 설계되었습니다.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, Mock

# 현재 디렉토리를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# HuggingFaceEmbeddings 모킹
sys.modules['langchain_community.embeddings.huggingface'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()

# 테스트할 모듈 임포트
from llm_api import DeepseekLLM, ExaoneLLM, DualModelChain
from langchain.schema import Document

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

class TestRAGSystemMocked(unittest.TestCase):
    """RAG 시스템 모킹 테스트 클래스"""
    
    @patch('rag_system.FAISSVectorStore')
    def test_rag_system_initialization(self, mock_vector_store):
        """RAG 시스템 초기화 테스트"""
        from rag_system import RAGSystem
        
        # 모킹된 벡터 저장소 설정
        mock_instance = mock_vector_store.return_value
        
        # RAG 시스템 초기화
        rag_system = RAGSystem()
        
        # 벡터 저장소가 초기화되었는지 확인
        mock_vector_store.assert_called_once()
        self.assertEqual(rag_system.vector_store, mock_instance)
    
    @patch('rag_system.FAISSVectorStore')
    def test_add_texts(self, mock_vector_store):
        """텍스트 추가 테스트"""
        from rag_system import RAGSystem
        
        # 모킹된 벡터 저장소 설정
        mock_instance = mock_vector_store.return_value
        mock_instance.load_documents_from_texts.return_value = [MagicMock(), MagicMock()]
        mock_instance.process_documents.return_value = [MagicMock(), MagicMock(), MagicMock()]
        
        # RAG 시스템 초기화 및 텍스트 추가
        rag_system = RAGSystem()
        test_texts = ["테스트 텍스트 1", "테스트 텍스트 2"]
        test_metadatas = [{"source": "테스트 1"}, {"source": "테스트 2"}]
        
        rag_system.add_texts(test_texts, test_metadatas)
        
        # 메서드 호출 확인
        mock_instance.load_documents_from_texts.assert_called_once_with(test_texts, test_metadatas)
        mock_instance.process_documents.assert_called_once()
    
    @patch('rag_system.FAISSVectorStore')
    def test_query(self, mock_vector_store):
        """쿼리 테스트"""
        from rag_system import RAGSystem
        
        # 모킹된 벡터 저장소 설정
        mock_instance = mock_vector_store.return_value
        mock_docs = [
            Document(page_content="테스트 문서 내용", metadata={"source": "테스트 소스"})
        ]
        mock_instance.vector_store = MagicMock()  # vector_store 속성 모킹
        mock_instance.similarity_search.return_value = mock_docs
        
        # RAG 시스템 초기화 및 쿼리 실행
        rag_system = RAGSystem()
        query_text = "테스트 쿼리"
        
        results = rag_system.query(query_text)
        
        # 메서드 호출 및 결과 확인
        mock_instance.similarity_search.assert_called_once_with(query_text, k=4)
        self.assertEqual(results, mock_docs)

if __name__ == "__main__":
    unittest.main()
