
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖"
)

st.title("🤖 AI FAQ Chatbot")
st.write("Ask me a question about Python programming!")

faqs = [
    {
        "question": "What is Python?",
        "answer": "Python is a popular programming language used for web development, AI, and data analysis."
    },
    {
        "question": "What is a variable?",
        "answer": "A variable stores a value that can be used later in a program."
    },
    {
        "question": "What is a function?",
        "answer": "A function is a reusable block of code that performs a specific task."
    },
    {
        "question": "What is a loop?",
        "answer": "A loop repeats a block of code multiple times."
    },
    {
        "question": "What is a list in Python?",
        "answer": "A list stores multiple items in a single variable."
    },
    {
        "question": "What is artificial intelligence?",
        "answer": "Artificial intelligence enables computers to perform tasks that usually require human intelligence."
    },
    {
        "question": "What is machine learning?",
        "answer": "Machine learning is a branch of AI that allows computers to learn patterns from data."
    },
    {
        "question": "How do I install Python?",
        "answer": "Download Python from python.org and follow the installation instructions for your operating system."
    }
]

questions = [faq["question"] for faq in faqs]
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(questions)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_question = st.chat_input("Type your question here...")

if user_question:
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.write(user_question)

    user_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vector, faq_vectors)[0]
    best_match = similarities.argmax()

    if similarities[best_match] >= 0.15:
        answer = faqs[best_match]["answer"]
    else:
        answer = (
            "Sorry, I couldn't find a suitable answer. "
            "Please try asking about Python, variables, functions, "
            "loops, lists, AI, or machine learning."
        )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.write(answer)
