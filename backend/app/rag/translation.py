"""
Translation service for RAG responses
Handles translating retrieved content and answers into multiple Indian languages
"""

import logging
from typing import Optional, Dict
from app.core.config import settings

logger = logging.getLogger(__name__)

# Language code to language name mapping
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
}

# Key disability terms with translations
DISABILITY_TERMS_TRANSLATIONS = {
    "blindness": {
        "en": "Blindness",
        "hi": "अंधत्व",
        "ta": "கண்ணின்மை",
        "te": "కళ్లు చూపని స్థితి",
        "bn": "অন্ধত্व",
        "mr": "अंधत्व",
        "gu": "અંધતા",
        "kn": "ಅಂಧತ್ವ",
        "ml": "കാണാതായിരിക്കുക",
        "pa": "ਅੰਧਾਪਨ",
        "or": "ଅନ୍ଧତା",
        "as": "অন্ধতা",
        "ur": "بینائی"
    },
    "visual impairment": {
        "en": "Visual Impairment",
        "hi": "दृष्टि हानि",
        "ta": "பார்வை குறைபாடு",
        "te": "దృష్టి కमజోरी",
        "bn": "দৃষ্টিশক্তি হ্রাস",
        "mr": "दृष्टी कमजोरी",
        "gu": "દૃષ્ટિ અશક્તતા",
        "kn": "ದೃಷ್ಟಿ ದುರ್ಬಲತೆ",
        "ml": "ദൃഷ്ടി ബലഹീനത",
        "pa": "ਨਜ਼ਰ ਦੀ ਕમਜ਼ੋਰੀ",
        "or": "ଦୃଷ୍ଟି ଦୁର୍ବଳତା",
        "as": "দৃষ্টি দুর্বলতা",
        "ur": "بینائی میں کمی"
    },
    "hearing impairment": {
        "en": "Hearing Impairment",
        "hi": "सुनने में कठिनाई",
        "ta": "கேட்பு குறைபாடு",
        "te": "శ్రవణ కमજోरी",
        "bn": "শ্রবণ দুর্বলতা",
        "mr": "ऐकिक कमजोरी",
        "gu": "સ્રવણ અશક્તતા",
        "kn": "ಶ್ರವಣ ದುರ್ಬಲತೆ",
        "ml": "വിനയോദ്ധരണ ബലഹീനത",
        "pa": "ਸੁਣਨ ਦੀ ਕಮਜ਼ੋਰੀ",
        "or": "ଶ୍ରବଣ ଦୁର୍ବଳତା",
        "as": "শ্রবণ দুর্বলতা",
        "ur": "سماعت میں کمی"
    },
    "RPWD Act": {
        "en": "RPWD Act 2016",
        "hi": "RPWD अधिनियम 2016",
        "ta": "RPWD சட்டம் 2016",
        "te": "RPWD చట్టం 2016",
        "bn": "RPWD আইন 2016",
        "mr": "RPWD कायदा 2016",
        "gu": "RPWD કાયદો 2016",
        "kn": "RPWD ಕಾಯಿದೆ 2016",
        "ml": "RPWD നിയമം 2016",
        "pa": "RPWD ਅਧਿਨਿਅਮ 2016",
        "or": "RPWD ଆଇନ 2016",
        "as": "RPWD আইন 2016",
        "ur": "RPWD ایکٹ 2016"
    },
    "rights": {
        "en": "Rights",
        "hi": "अधिकार",
        "ta": "உரிமைகள்",
        "te": "హక్కులు",
        "bn": "অধিকার",
        "mr": "हक्क",
        "gu": "અધિકાર",
        "kn": "ಹಕ್ಕುಗಳು",
        "ml": "അവകാശങ്ങൾ",
        "pa": "ਅਧਿਕਾਰ",
        "or": "ଅଧିକାର",
        "as": "অধিকাৰ",
        "ur": "حقوق"
    },
    "education": {
        "en": "Education",
        "hi": "शिक्षा",
        "ta": "கல்வி",
        "te": "విద్య",
        "bn": "শিক্ষা",
        "mr": "शिक्षण",
        "gu": "શિક્ષણ",
        "kn": "ಶಿಕ್ಷಣ",
        "ml": "വിദ്യാഭ്യാസം",
        "pa": "ਸਿੱਖਿਆ",
        "or": "ଶିକ୍ଷା",
        "as": "শিক্ষা",
        "ur": "تعلیم"
    },
    "employment": {
        "en": "Employment",
        "hi": "रोजगार",
        "ta": "வேலைவாய்ப்பு",
        "te": "ఉద్యోగం",
        "bn": "কর্মসংস্থান",
        "mr": "नोकरी",
        "gu": "રોજગાર",
        "kn": "ಉದ್ಯೋಗ",
        "ml": "തൊഴിൽ",
        "pa": "ਮੁਲਾਜ਼ਮਤ",
        "or": "ନିଯୁକ୍ତି",
        "as": "নিয়োগ",
        "ur": "ملازمت"
    },
    "accessibility": {
        "en": "Accessibility",
        "hi": "पहुंचयोग्यता",
        "ta": "அணுகுமை",
        "te": "సుலభతা",
        "bn": "অ্যাক্সেসিবিলিটি",
        "mr": "प्रवेशाधिकार",
        "gu": "સુલભતા",
        "kn": "ಪ್ರವೇಶಾಧಿಕಾರ",
        "ml": "ആക്സസിബിലിറ്റി",
        "pa": "ਪਹੁੰਚ",
        "or": "ପ୍ରବେଶାଧିକାର",
        "as": "প্রবেশাধিকার",
        "ur": "رسائی"
    },
    "scheme": {
        "en": "Scheme",
        "hi": "योजना",
        "ta": "திட்டம்",
        "te": "పథకం",
        "bn": "স্কিম",
        "mr": "योजना",
        "gu": "યોજના",
        "kn": "ಯೋಜನೆ",
        "ml": "പദ്ധതി",
        "pa": "ਸਕੀਮ",
        "or": "ଯୋଜନା",
        "as": "আঁচনি",
        "ur": "اسکیم"
    },
}


def translate_term(term: str, target_language: str) -> Optional[str]:
    """
    Translate a disability-related term to the target language
    """
    if target_language == "en":
        return term
    
    term_key = term.lower().strip()
    if term_key in DISABILITY_TERMS_TRANSLATIONS:
        return DISABILITY_TERMS_TRANSLATIONS[term_key].get(target_language, term)
    
    return None


def should_use_translation_api(language: str) -> bool:
    """Check if we should try to use external translation API"""
    return language not in ["en"]


def translate_text_with_google(text: str, target_language: str) -> Optional[str]:
    """
    Attempt to translate text using Deep Translator (primary) or Google Translate API (fallback)
    Returns translated text or None if translation fails
    """
    if not settings.TRANSLATION_ENABLED:
        return None

    if target_language == "en" or not text:
        return None
    
    # Try Deep Translator first as it is more stable
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source='en', target=target_language).translate(text)
        if translated and translated.strip():
            logger.info(f"Successfully translated to {target_language} using deep-translator")
            return translated
    except Exception as e:
        logger.debug(f"Deep-translator failed: {e}, falling back to googletrans")

    try:
        from googletrans import Translator
        translator = Translator()
        
        # Retry logic for robustness
        max_retries = 2
        for attempt in range(max_retries):
            try:
                result = translator.translate(text, src='en', dest=target_language)
                translated = getattr(result, "text", None)
                if translated and translated.strip():
                    logger.info(f"Successfully translated to {target_language}")
                    return translated
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.5)  # Small delay before retry to avoid rate limiting
                    continue
                else:
                    logger.error(f"Translation to {target_language} failed after {max_retries} attempts: {e}")
                    return None
        return None
    except ImportError:
        logger.debug("googletrans not available, translation skipped")
        return None
    except Exception as e:
        logger.error(f"Translation failed unexpectedly: {e}")
        return None


def translate_text_to_english(text: str, source_language: Optional[str] = None) -> Optional[str]:
    """Translate arbitrary text to English using deep-translator (primary) or googletrans (fallback)."""
    if not settings.TRANSLATION_ENABLED:
        return None

    if not text:
        return None

    src_lang = "auto"
    if source_language:
        src_lang = language_code_to_google_code(source_language)

    # Try Deep Translator first
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source=src_lang, target='en').translate(text)
        if translated and translated.strip():
            return translated
    except Exception as e:
        logger.debug(f"Deep-translator to English failed: {e}, falling back to googletrans")

    try:
        from googletrans import Translator
        translator = Translator()
        result = translator.translate(text, src=src_lang, dest="en")
        translated = getattr(result, "text", None)
        if translated and translated.strip():
            return translated
        return None
    except ImportError:
        logger.debug("googletrans not available for query translation")
        return None
    except Exception as e:
        logger.warning(f"Query translation to English failed: {e}")
        return None


def language_code_to_google_code(lang_code: str) -> str:
    """Convert our language codes to Google Translate language codes"""
    mapping = {
        "hi": "hi",
        "ta": "ta",
        "te": "te",
        "bn": "bn",
        "mr": "mr",
        "gu": "gu",
        "kn": "kn",
        "ml": "ml",
        "pa": "pa",
        "or": "or",
        "as": "as",
        "ur": "ur",
        "en": "en",
    }
    return mapping.get(lang_code, lang_code)


def enhance_answer_with_language_context(answer: str, language: str) -> str:
    """
    Add language context and improve answer formatting for non-English languages
    """
    if language == "en":
        return answer
    
    # For non-English languages, ensure proper formatting
    lang_name = LANGUAGE_NAMES.get(language, language)
    
    # Add context about the language of content if it's not English
    if language != "en":
        answer = answer.strip()
        if not answer:
            answer = f"Information available in {lang_name}. Please refer to the sources above."
    
    return answer
