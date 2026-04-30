// Global speech state
let _isSpeaking = false;

// Language code mappings for Web Speech API
const LANGUAGE_CODES = {
  en: "en-US",
  hi: "hi-IN",    // Hindi
  ta: "ta-IN",    // Tamil
  te: "te-IN",    // Telugu
  kn: "kn-IN",    // Kannada
  ml: "ml-IN",    // Malayalam
};

/**
 * Get speech locale from language code
 */
export const getSpeechLocale = (lang = "en") => {
  const short = (lang || "en").toLowerCase().split("-")[0];
  return LANGUAGE_CODES[short] || "en-US";
};

/**
 * Check voice availability for a language
 */
export const getVoiceAvailability = (lang = "en") => {
  if (!window.speechSynthesis) return false;
  const locale = getSpeechLocale(lang);
  const voices = window.speechSynthesis.getVoices();
  return voices.some(v => v.lang.startsWith(locale.split("-")[0]));
};

/**
 * Detect language from text script
 * Returns language code (en, hi, ta, te, kn, ml) based on Unicode ranges
 */
export const detectLanguageFromText = (text) => {
  if (!text) return "en";

  // Telugu: U+0C00 - U+0C7F
  if (/[\u0C00-\u0C7F]/.test(text)) return "te";

  // Hindi/Devanagari: U+0900 - U+097F
  if (/[\u0900-\u097F]/.test(text)) return "hi";

  // Tamil: U+0B80 - U+0BFF
  if (/[\u0B80-\u0BFF]/.test(text)) return "ta";

  // Kannada: U+0C80 - U+0CFF
  if (/[\u0C80-\u0CFF]/.test(text)) return "kn";

  // Malayalam: U+0D00 - U+0D7F
  if (/[\u0D00-\u0D7F]/.test(text)) return "ml";

  return "en";
};

/**
 * Speak text in specified language using Web Speech API
 */
export const speakTextInLanguage = (text, language = "en", options = {}) => {
  if (!text || !window.speechSynthesis) return;

  const { interrupt = false } = options;

  // Interrupt if already speaking
  if (_isSpeaking && interrupt) {
    window.speechSynthesis.cancel();
  }

  // Don't speak if already speaking (unless interrupt is true)
  if (_isSpeaking && !interrupt) {
    return;
  }

  // Get language code (en, hi, ta, te, kn, ml)
  const detectedLang = detectLanguageFromText(text) || language || "en";
  const webSpeechLang = getSpeechLocale(detectedLang);

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = webSpeechLang;
  utterance.rate = 0.9;
  utterance.pitch = 1;
  utterance.volume = 1;

  // Track speaking state
  utterance.onstart = () => {
    _isSpeaking = true;
  };

  utterance.onend = () => {
    _isSpeaking = false;
  };

  utterance.onerror = (e) => {
    console.warn("Speech synthesis error:", e);
    _isSpeaking = false;
  };

  window.speechSynthesis.speak(utterance);
};

/**
 * Stop current speech
 */
export const stopSpeech = () => {
  if (window.speechSynthesis) {
    window.speechSynthesis.cancel();
    _isSpeaking = false;
  }
};

/**
 * Check if speech is currently active
 */
export const isSpeaking = () => _isSpeaking;

/**
 * Speak text with automatic language detection
 */
export const speakAutoDetect = (text, options = {}) => {
  const lang = detectLanguageFromText(text) || "en";
  speakTextInLanguage(text, lang, options);
};