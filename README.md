# LLM-Application-with-Langchain-RAGs-and-Streamlit
# LLM RAG 애플리케이션 사용 설명서

## 개요
이 애플리케이션은 Langchain 기반 RAG(Retrieval-Augmented Generation)를 활용한 질의응답 시스템입니다. 
- **deepseek-r1:7b** 모델은 추론 과정에 사용됩니다.
- **exaone3.5** 모델은 최종 출력 생성에 사용됩니다.
- **FAISS**를 사용하여 문서 임베딩을 저장하고 검색합니다.
- **Streamlit**을 통해 웹 인터페이스를 제공합니다.
- **openwebui** 디자인을 기반으로 스타일링되었습니다.

## 설치 방법

### 필수 요구사항
- Python 3.10 이상
- pip 패키지 관리자

### 설치 단계
1. 저장소 클론
```bash
git clone https://github.com/yourusername/llm_rag_app.git
cd llm_rag_app
```

2. 필수 패키지 설치
```bash
pip install -r requirements.txt
```

3. 환경 변수 설정
`.env` 파일을 생성하고 다음 내용을 추가합니다:
```
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions
EXAONE_API_KEY=your_exaone_api_key
EXAONE_API_URL=https://api.exaone.ai/v1/chat/completions
```

## 사용 방법

### 애플리케이션 실행
```bash
streamlit run app.py
```

### 주요 기능

#### 1. 문서 관리
- **문서 업로드**: 사이드바에서 텍스트 파일(.txt)을 업로드할 수 있습니다.
- **샘플 데이터 추가**: 테스트를 위한 샘플 데이터를 추가할 수 있습니다.

#### 2. 질의응답
- 채팅 인터페이스에 질문을 입력하면 시스템이 관련 문서를 검색하고 응답을 생성합니다.
- 응답에는 추론 과정과 참조 문서를 확인할 수 있는 확장 패널이 포함되어 있습니다.

## 시스템 구성 요소

### 1. RAG 시스템 (rag_system.py)
- **FAISSVectorStore**: 문서 임베딩을 저장하고 검색하는 벡터 저장소 클래스
- **RAGSystem**: 문서 추가 및 쿼리 실행을 위한 고수준 인터페이스 제공

### 2. LLM 모델 통합 (llm_api.py)
- **DeepseekLLM**: deepseek-r1:7b 모델을 위한 LLM 클래스 (추론 과정용)
- **ExaoneLLM**: exaone3.5 모델을 위한 LLM 클래스 (최종 출력용)
- **DualModelChain**: 두 모델을 결합한 체인 클래스

### 3. 웹 인터페이스 (app.py)
- Streamlit 기반 웹 애플리케이션
- 문서 업로드 및 관리 기능
- 채팅 인터페이스
- openwebui 스타일 적용 (styles.py)

## 커스터마이징

### 임베딩 모델 변경
`rag_system.py` 파일에서 `FAISSVectorStore` 클래스의 `__init__` 메서드에 있는 `embedding_model_name` 매개변수를 수정합니다:

```python
def __init__(self, embedding_model_name: str = "your-preferred-model"):
    self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    # ...
```

### LLM 모델 변경
`llm_api.py` 파일에서 `DeepseekLLM` 및 `ExaoneLLM` 클래스의 API URL과 모델 이름을 수정합니다.

### UI 스타일 변경
`styles.py` 파일에서 CSS 스타일을 수정하여 UI 디자인을 변경할 수 있습니다.

## 문제 해결

### 일반적인 문제
- **문서가 검색되지 않음**: 문서를 먼저 업로드하거나 샘플 데이터를 추가해야 합니다.
- **API 오류**: API 키가 올바르게 설정되었는지 확인하세요.
- **메모리 부족**: 대용량 문서 처리 시 청크 크기를 줄이거나 문서 수를 제한하세요.

### 지원 및 문의
문제가 발생하면 GitHub 이슈를 통해 문의하거나 이메일로 연락해 주세요.

## 라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다.
