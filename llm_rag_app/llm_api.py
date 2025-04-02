"""
LLM API 통합 모듈

이 모듈은 deepseek-r1:7b와 exaone3.5 모델에 대한 API 기반 통합을 제공합니다.
deepseek-r1:7b는 추론 과정에 사용되고, exaone3.5는 최종 출력 생성에 사용됩니다.
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import Document

# 환경 변수 로드
load_dotenv()

class DeepseekLLM(LLM):
    """Deepseek-r1:7b 모델을 위한 LLM 클래스"""
    
    api_key: str = os.getenv("DEEPSEEK_API_KEY", "demo_api_key")
    api_url: str = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
    
    @property
    def _llm_type(self) -> str:
        return "deepseek-r1:7b"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Deepseek API 호출 메서드"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": "deepseek-r1:7b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 1024)
        }
        
        # 실제 API 호출 시 아래 주석 해제
        # response = requests.post(self.api_url, headers=headers, json=payload)
        # response_json = response.json()
        # return response_json["choices"][0]["message"]["content"]
        
        # 데모 목적으로 API 호출 시뮬레이션
        return f"[Deepseek 추론 결과] 주어진 문서를 분석한 결과, 다음과 같은 핵심 정보를 추출했습니다: {prompt[:100]}..."

class ExaoneLLM(LLM):
    """Exaone3.5 모델을 위한 LLM 클래스"""
    
    api_key: str = os.getenv("EXAONE_API_KEY", "demo_api_key")
    api_url: str = os.getenv("EXAONE_API_URL", "https://api.exaone.ai/v1/chat/completions")
    
    @property
    def _llm_type(self) -> str:
        return "exaone3.5"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Exaone API 호출 메서드"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": "exaone3.5",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024)
        }
        
        # 실제 API 호출 시 아래 주석 해제
        # response = requests.post(self.api_url, headers=headers, json=payload)
        # response_json = response.json()
        # return response_json["choices"][0]["message"]["content"]
        
        # 데모 목적으로 API 호출 시뮬레이션
        return f"[Exaone 최종 응답] 사용자님의 질문에 대한 답변입니다: {prompt[:100]}..."

class DualModelChain:
    """
    Deepseek와 Exaone 모델을 결합한 체인
    Deepseek는 추론 과정에 사용되고, Exaone은 최종 출력 생성에 사용됩니다.
    """
    
    def __init__(self):
        self.reasoning_model = DeepseekLLM()
        self.output_model = ExaoneLLM()
    
    def run(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """
        쿼리와 검색된 문서를 기반으로 추론 및 응답 생성
        
        Args:
            query: 사용자 쿼리
            documents: 검색된 관련 문서 목록
            
        Returns:
            결과 딕셔너리 (추론 과정 및 최종 응답 포함)
        """
        # 문서 컨텍스트 준비
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # Deepseek 모델을 사용한 추론 과정
        reasoning_prompt = f"""
        다음은 사용자의 질문입니다:
        {query}
        
        다음은 관련 문서 정보입니다:
        {context}
        
        위 정보를 바탕으로 질문에 답변하기 위한 추론 과정을 단계별로 작성해주세요.
        """
        
        reasoning_result = self.reasoning_model(reasoning_prompt)
        
        # Exaone 모델을 사용한 최종 응답 생성
        output_prompt = f"""
        다음은 사용자의 질문입니다:
        {query}
        
        다음은 관련 문서 정보입니다:
        {context}
        
        다음은 추론 과정입니다:
        {reasoning_result}
        
        위 추론 과정을 바탕으로 사용자 질문에 대한 최종 답변을 작성해주세요.
        추론 과정은 포함하지 말고, 최종 답변만 작성해주세요.
        """
        
        final_output = self.output_model(output_prompt)
        
        return {
            "query": query,
            "reasoning": reasoning_result,
            "response": final_output
        }
