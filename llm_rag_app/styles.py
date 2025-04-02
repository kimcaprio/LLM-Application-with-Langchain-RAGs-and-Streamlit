"""
openwebui 스타일 CSS 파일

이 파일은 Streamlit 애플리케이션에 openwebui 스타일을 적용하기 위한 CSS를 포함합니다.
"""

openwebui_css = """
/* openwebui 스타일 CSS */

/* 전체 페이지 스타일 */
.main {
    background-color: #0e1117;
    color: #e0e0e0;
}

/* 사이드바 스타일 */
.sidebar .sidebar-content {
    background-color: #1a1c24;
    color: #e0e0e0;
}

/* 헤더 스타일 */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff;
    font-family: 'Inter', sans-serif;
}

/* 버튼 스타일 */
.stButton > button {
    background-color: #2d7ff9;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: 500;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background-color: #1a6eeb;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
}

/* 입력 필드 스타일 */
.stTextInput > div > div > input {
    background-color: #1e2029;
    color: #e0e0e0;
    border: 1px solid #3a3d4a;
    border-radius: 8px;
    padding: 0.5rem;
}

/* 채팅 메시지 스타일 */
.stChatMessage {
    background-color: #1e2029;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.stChatMessage.user {
    background-color: #2d7ff9;
    color: white;
}

.stChatMessage.assistant {
    background-color: #1e2029;
    color: #e0e0e0;
}

/* 확장 패널 스타일 */
.streamlit-expanderHeader {
    background-color: #1a1c24;
    color: #e0e0e0;
    border-radius: 8px;
    padding: 0.5rem;
}

.streamlit-expanderContent {
    background-color: #1e2029;
    color: #e0e0e0;
    border-radius: 0 0 8px 8px;
    padding: 1rem;
}

/* 알림 스타일 */
.stAlert {
    background-color: #1e2029;
    color: #e0e0e0;
    border-radius: 8px;
    border-left-color: #2d7ff9;
}

/* 성공 메시지 스타일 */
.stSuccess {
    background-color: #1e2029;
    color: #e0e0e0;
    border-radius: 8px;
    border-left-color: #00cc66;
}

/* 경고 메시지 스타일 */
.stWarning {
    background-color: #1e2029;
    color: #e0e0e0;
    border-radius: 8px;
    border-left-color: #ffcc00;
}

/* 오류 메시지 스타일 */
.stError {
    background-color: #1e2029;
    color: #e0e0e0;
    border-radius: 8px;
    border-left-color: #ff4d4d;
}

/* 정보 메시지 스타일 */
.stInfo {
    background-color: #1e2029;
    color: #e0e0e0;
    border-radius: 8px;
    border-left-color: #2d7ff9;
}

/* 로딩 스피너 스타일 */
.stSpinner > div > div {
    border-color: #2d7ff9 !important;
}

/* 파일 업로더 스타일 */
.stFileUploader > div {
    background-color: #1e2029;
    color: #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
}

/* 체크박스 스타일 */
.stCheckbox > div > label {
    color: #e0e0e0;
}

.stCheckbox > div > div > label > span {
    color: #e0e0e0;
}

/* 선택 박스 스타일 */
.stSelectbox > div > div {
    background-color: #1e2029;
    color: #e0e0e0;
    border: 1px solid #3a3d4a;
    border-radius: 8px;
}

/* 슬라이더 스타일 */
.stSlider > div > div {
    background-color: #2d7ff9;
}

/* 푸터 스타일 */
footer {
    color: #a0a0a0;
    font-size: 0.8rem;
    text-align: center;
    margin-top: 2rem;
}

/* 스크롤바 스타일 */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #1a1c24;
}

::-webkit-scrollbar-thumb {
    background: #3a3d4a;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #4a4d5a;
}

/* 모바일 반응형 스타일 */
@media (max-width: 768px) {
    .main {
        padding: 1rem;
    }
    
    .stChatMessage {
        padding: 0.75rem;
    }
}
"""
