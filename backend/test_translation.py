from app.rag.translation import translate_text_with_google
import logging

logging.basicConfig(level=logging.INFO)

text = "Hello, how are you?"
languages = ["te", "kn", "ml", "hi", "ta"]

for lang in languages:
    result = translate_text_with_google(text, lang)
    print(f"{lang}: {result}")
