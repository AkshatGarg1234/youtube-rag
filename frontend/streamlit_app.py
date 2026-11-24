
import json
import os
import re

import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NeetCode Interview Copilot",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* Feature cards */

    .feature-card {
        background: linear-gradient(
            145deg,
            #1B2433,
            #141B27
        );
        border: 1px solid #293346;
        border-radius: 15px;
        padding: 20px;
        min-height: 125px;
        transition: border-color 0.2s ease;
    }

    .feature-card:hover {
        border-color: #60A5FA;
    }

    .feature-icon {
        font-size: 1.5rem;
        margin-bottom: 10px;
    }

    .feature-title {
        font-weight: 700;
        color: #F9FAFB;
        font-size: 1rem;
    }

    .feature-description {
        color: #9CA3AF;
        font-size: 0.85rem;
        margin-top: 7px;
        line-height: 1.5;
    }

    /* Sidebar */

    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #293346;
    }

    /* Buttons */

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #60A5FA;
        color: #93C5FD;
    }

    /* Chat */

    [data-testid="stChatMessage"] {
        border: 1px solid #293346;
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ============================================================
# HELPER: RENDER ANSWER AND COPYABLE CODE
# ============================================================

def render_answer(answer: str):
    """
    Render Markdown normally and fenced code blocks
    using Streamlit's copyable code component.
    """

    pattern = re.compile(
        r"```([a-zA-Z0-9_+#.-]*)[ \t]*\n(.*?)```",
        re.DOTALL,
    )

    cursor = 0

    for match in pattern.finditer(answer):
        text_before = answer[cursor:match.start()]

        if text_before.strip():
            st.markdown(text_before)

        language_name = match.group(1).strip().lower()
        code = match.group(2).strip("\n")

        language_map = {
            "python": "python",
            "py": "python",
            "cpp": "cpp",
            "c++": "cpp",
            "java": "java",
            "c": "c",
        }

        display_language = language_map.get(
            language_name,
            "text",
        )

        st.code(
            code,
            language=display_language,
            line_numbers=True,
            wrap_lines=False,
        )

        cursor = match.end()

    remaining_text = answer[cursor:]

    if remaining_text.strip():
        st.markdown(remaining_text)


# ============================================================
# HELPER: STREAM ANSWER FROM FASTAPI
# ============================================================

def stream_answer(
    question: str,
    language: str,
    code_language: str | None,
    top_k: int,
    placeholder,
):
    """
    Send a question to FastAPI and stream the response.
    """

    payload = {
        "question": question,
        "language": language,
        "code_language": code_language,
        "top_k": top_k,
    }

    answer = ""

    with requests.post(
        f"{API_URL}/ask/stream",
        json=payload,
        stream=True,
        timeout=(10, 300),
    ) as response:

        response.raise_for_status()
        response.encoding = "utf-8"

        for line in response.iter_lines(decode_unicode=True):

            if not line:
                continue

            if not line.startswith("data: "):
                continue

            data = line[6:]

            if data == "[DONE]":
                break

            event = json.loads(data)

            if "error" in event:
                raise RuntimeError(event["error"])

            token = event.get("token", "")

            if token:
                answer += token
                placeholder.markdown(answer + " ▌")

    return answer


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 💻 NeetCode Copilot")
    st.caption("Your personal coding interview companion")

    st.divider()

    st.markdown("### ⚙️ Preferences")

    language = st.selectbox(
        "Explanation language",
        options=["hinglish", "english"],
        format_func=lambda value: {
            "hinglish": "🇮🇳 Hinglish",
            "english": "🇬🇧 English",
        }[value],
    )

    code_language = st.selectbox(
        "Programming language",
        options=[None, "python", "cpp", "java"],
        format_func=lambda value: {
            None: "No code",
            "python": "🐍 Python",
            "cpp": "⚡ C++",
            "java": "☕ Java",
        }[value],
    )

    top_k = st.slider(
        "Retrieved context chunks",
        min_value=1,
        max_value=10,
        value=5,
    )

    st.divider()

    st.markdown("### 🧠 Capabilities")

    st.markdown(
        """
        - Concept explanations
        - Algorithm intuition
        - Code generation
        - Complexity analysis
        - YouTube timestamp references
        - Interview preparation
        """
    )

    st.divider()

    st.caption("POWERED BY")

    st.markdown(
        """
        - FastAPI
        - Qdrant
        - Groq
        - Sentence Transformers
        - Streamlit
        """
    )

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.caption("Made with ❤️ by Akshat Garg")


# ============================================================
# HERO SECTION - NATIVE STREAMLIT
# ============================================================

st.markdown("")

hero_left, hero_center, hero_right = st.columns([1, 5, 1])

with hero_center:

    st.markdown(
        "<p style='text-align:center; color:#93C5FD; "
        "font-size:0.8rem; font-weight:600;'>"
        "✨ AI-POWERED CODING INTERVIEW ASSISTANT"
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h1 style='text-align:center; color:#FAFAFA; "
        "font-size:2.8rem; font-weight:800;'>"
        "NeetCode <span style='color:#60A5FA;'>"
        "Interview Copilot</span></h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align:center; color:#9CA3AF; "
        "font-size:1.05rem;'>"
        "Learn concepts. Understand algorithms. "
        "Generate code. Crack interviews."
        "</p>",
        unsafe_allow_html=True,
    )

st.markdown("")


# ============================================================
# FEATURE CARDS
# ============================================================

col1, col2, col3 = st.columns(3)

features = [
    (
        "📚",
        "Smart Explanations",
        "Understand programming concepts in English or Hinglish.",
    ),
    (
        "💻",
        "Code Generator",
        "Generate solutions in Python, C++ and Java.",
    ),
    (
        "🎯",
        "Interview Revision",
        "Learn approaches, dry runs and complexity analysis.",
    ),
]

for col, (icon, title, description) in zip(
    [col1, col2, col3],
    features,
):
    with col:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-description">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.divider()


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.markdown("### 🚀 Try an example")

st.caption(
    "Choose a topic to start learning from your YouTube playlist."
)

example_questions = [
    "Explain largest rectangle in histogram",
    "Explain binary search with an example",
    "Explain the two pointer technique",
]

example_columns = st.columns(3)

for index, example in enumerate(example_questions):

    with example_columns[index]:

        if st.button(
            example,
            use_container_width=True,
            key=f"example_{index}",
        ):
            st.session_state.pending_prompt = example
            st.rerun()


st.divider()


# ============================================================
# CHAT HISTORY
# ============================================================

st.markdown("### 💬 Your conversation")

if not st.session_state.messages:
    st.info(
        "👋 Welcome! Ask a question about your programming "
        "playlist to get started."
    )

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message["role"] == "assistant":
            render_answer(message["content"])
        else:
            st.markdown(message["content"])


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.chat_input(
    "Ask anything about your programming playlist..."
)

if st.session_state.pending_prompt:
    question = st.session_state.pending_prompt
    st.session_state.pending_prompt = None


# ============================================================
# GENERATE ANSWER
# ============================================================

if question:

    question = question.strip()

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            placeholder = st.empty()

            try:
                answer = stream_answer(
                    question=question,
                    language=language,
                    code_language=code_language,
                    top_k=top_k,
                    placeholder=placeholder,
                )

                placeholder.empty()

                if answer.strip():

                    render_answer(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                else:
                    st.warning(
                        "The AI returned an empty response."
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to FastAPI. "
                    "Please start your backend server."
                )

            except requests.exceptions.Timeout:
                st.error(
                    "Request timed out. Please try again."
                )

            except requests.exceptions.HTTPError as exc:
                st.error(
                    f"API error: {exc.response.text}"
                )

            except (
                requests.exceptions.RequestException,
                RuntimeError,
                ValueError,
            ) as exc:
                st.error(f"Request failed: {exc}")

            except Exception as exc:
                st.error(
                    f"Unexpected error: {exc}"
                )


# ============================================================
# FOOTER - NATIVE STREAMLIT
# ============================================================

st.divider()

footer_left, footer_center, footer_right = st.columns([1, 3, 1])

with footer_center:

    st.markdown(
        "<p style='text-align:center; color:#9CA3AF;'>"
        "Built with ❤️ by "
        "<strong style='color:#60A5FA;'>Akshat Garg</strong>"
        "</p>",
        unsafe_allow_html=True,
    )
