
import streamlit as st
from deep_translator import MyMemoryTranslator

st.set_page_config(
    page_title="AI Language Translator",
    page_icon="🌐",
    layout="centered"
)

st.title("🌐 AI Language Translation Tool")
st.write("Translate English text into different languages.")

languages = {
    "Urdu": "ur-PK",
    "Hindi": "hi-IN",
    "Arabic": "ar-SA",
    "French": "fr-FR",
    "Spanish": "es-ES",
    "German": "de-DE",
    "Chinese": "zh-CN",
    "English": "en-GB"
}

user_text = st.text_area(
    "Enter English text:",
    placeholder="Type your text here..."
)

target_lang = st.selectbox(
    "Choose target language:",
    list(languages.keys())
)

if st.button("Translate"):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            translated_text = MyMemoryTranslator(
                source="en-GB",
                target=languages[target_lang]
            ).translate(user_text)

            st.subheader("Translated Result")
            st.success(translated_text)

        except Exception as e:
            st.error(
                f"Translation failed: {type(e).__name__}: {e}"
            )

