import streamlit as st
import openai

st.title('글로윙(GLOWING): AI 피부 분석 - 스킨케어 봇 🤖')
st.subheader('피부 관련 궁금한 것 뭐든지 질문 해주세요!')

# Initialize OpenAI client
client = openai

# Define skincare related keywords in Korean
SKINCARE_KEYWORDS = [
    "피부", "스킨케어", "여드름", "모이스처라이저", "선크림", "안녕",
    "화장품", "뷰티", "클렌저", "세럼", "각질 제거", "물", "영양",
    "토너", "마스크", "뾰루지", "잡티", "주름", "라이프", "태양", "눈",
    "미세한 주름", "햇빛 손상", "건성 피부", "지성 피부", "복합성 피부",
    "보습", "수분", "유분", "피지", "화장", "팩", "앰플", "항산화",
    "피부 장벽", "재생", "트러블", "자극", "피부톤", "코", "피지",
    "민감성 피부", "피부관리", "프라이머", "립밤", "얼굴", "식품",
    "비타민 C", "비타민 E", "콜라겐", "히알루론산", "니아신아마이드",
    "엑스트라버진", "바셀린", "스크럽", "세안", "모공", "다이어트",
    "지성", "건조", "균형", "여드름 흉터", "광채", "케어", "여드름관리",
    "리프팅", "스팟", "차단제", "피부 질환", "스트레스", "모공관리",
    "재생 크림", "피부 톤업", "선케어", "재생력", "자외선", "피부관리",
    "피부과", "마사지", "염증", "필링", "검진", "식습관", "피부케어",
    "펩타이드", "에센스", "디톡스", "축소", "미백", "헬스", "헬스케어",
    "성분", "더모", "클렌징", "아크네", "더마", "건강", "SPF",
    "자연유래", "보호", "자연 성분", "인공 성분", "민감도", "운동",
    "알로에", "화장품 성분", "블랙헤드", "화이트헤드", "컨투어링",
    "자연 유래", "테스트", "바디케어", "치약", "메이크업", "스트레스",
    "비타민", "무기자차", "화장 전", "피부 보호제", "이빨",
    "피부 트러블", "피부 강화", "균형", "항염", "아침 루틴"
]

# Function to check if the question is skincare related
def is_skincare_related(question):
    question = question.lower()
    return any(keyword in question for keyword in SKINCARE_KEYWORDS)

if 'openai_model' not in st.session_state:
    GPT_MODEL = 'gpt-3.5-turbo'
    st.session_state['openai_model'] = GPT_MODEL

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# React to user input
prompt = st.chat_input('피부 관련 궁금한 것 뭐든지 질문 해주세요!')
if prompt:
    # Display user message in chat message container
    with st.chat_message('user'):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    if is_skincare_related(prompt):
        with st.spinner("생각 중..."):
            with st.chat_message('assistant'):
                message_placeholder = st.empty()  # Empty placeholder
                full_response = ""
                # Calling OpenAI API, stream=True for typing effect - interactive
                for response in client.chat.completions.create(
                        model=st.session_state['openai_model'],
                        messages=[
                            {'role': m['role'], 'content': m['content']}
                            for m in st.session_state.messages
                        ],
                        stream=True
                ):
                    if response.choices[0].delta.content is not None:  # Check if the content is not None
                        full_response += response.choices[0].delta.content
                        message_placeholder.markdown(full_response + " ")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({'role': 'assistant', 'content': full_response})
    else:
        st.warning("이 봇은 피부 관리 관련 질문에만 답변할 수 있습니다. 다른 질문을 해주세요.")
