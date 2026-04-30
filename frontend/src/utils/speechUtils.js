const LANGUAGE_TO_LOCALE = {
  en: "en-US",
  hi: "hi-IN",
  ta: "ta-IN",
  te: "te-IN",
  kn: "kn-IN",
  ml: "ml-IN",
};

const API_CANDIDATES = [
  (process.env.REACT_APP_API_URL || "").trim(),
  `${window.location.protocol === "https:" ? "https:" : "http:"}//${window.location.hostname || "localhost"}:3000`,
  "http://localhost:3000",
].filter(Boolean).filter((value, index, arr) => arr.indexOf(value) === index);

let workingApiBase = null;
let backendAudioPlayer = null;
let _isSpeaking = false;

async function fetchBackendTtsAudio(text, languageCode) {
  // Prioritize the last known working API base to skip dead candidates
  const candidates = workingApiBase ? [workingApiBase, ...API_CANDIDATES] : API_CANDIDATES;
  for (const base of candidates) {
    try {
      const res = await fetch(`${base}/api/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, language: languageCode }),
      });
      if (!res.ok) {
        if (workingApiBase === base) workingApiBase = null;
        continue;
      }
      const data = await res.json();
      if (data?.audio_base64) {
        workingApiBase = base;
        return `data:${data.mime_type || "audio/mpeg"};base64,${data.audio_base64}`;
      }
    } catch (_) {
      if (workingApiBase === base) workingApiBase = null;
    }
  }
  return null;
}

async function playBackendTts(text, languageCode, onEnd, interrupt = false) {
  const dataUrl = await fetchBackendTtsAudio(text, languageCode);
  if (!dataUrl) { if (onEnd) onEnd(); return; }

  if (!backendAudioPlayer) {
    backendAudioPlayer = new Audio();
  }

  // If already speaking and not allowed to interrupt, skip
  if (_isSpeaking && !interrupt) {
    if (onEnd) onEnd();
    return;
  }

  // If interrupting, stop any existing audio
  if (interrupt) {
    try { backendAudioPlayer.pause(); backendAudioPlayer.src = ""; } catch (e) {}
    try { window.speechSynthesis?.cancel(); } catch (e) {}
  }

  backendAudioPlayer.pause();
  backendAudioPlayer.src = dataUrl;
  _isSpeaking = true;
  backendAudioPlayer.onended = () => { _isSpeaking = false; if (onEnd) onEnd(); };
  backendAudioPlayer.oncancel = () => { _isSpeaking = false; if (onEnd) onEnd(); };
  try {
    await backendAudioPlayer.play();
  } catch (_) {
    _isSpeaking = false;
    if (onEnd) onEnd();
  }
}

export function getSpeechLocale(languageCode) {
  const raw = (languageCode || "en").toLowerCase();
  const shortCode = raw.split("-")[0];
  return LANGUAGE_TO_LOCALE[raw] || LANGUAGE_TO_LOCALE[shortCode] || raw || "en-US";
}

export function pickVoiceForLocale(locale) {
  if (!("speechSynthesis" in window)) return null;
  const voices = window.speechSynthesis.getVoices() || [];
  if (!voices.length) return null;

  const localeLower = (locale || "").toLowerCase();
  const prefix = localeLower.split("-")[0];
  const voiceNameHint =
    prefix === "te" ? /telugu|vani|heera/i :
    prefix === "ta" ? /tamil|valluvar/i :
    prefix === "kn" ? /kannada|hemant/i :
    prefix === "ml" ? /malayalam|midhun/i :
    prefix === "hi" ? /hindi|kalpana|devanagari/i :
    prefix === "en" ? /english/i :
    null;
    
  const allMatching = voices.filter(v => 
    (v.lang && v.lang.toLowerCase() === localeLower) ||
    (v.lang && v.lang.toLowerCase().startsWith(`${prefix}-`)) ||
    (voiceNameHint && voiceNameHint.test(v.name || ""))
  );

  // Prefer local (non-network) voices if available to improve reliability.
  // Network voices can sometimes return as "available" but fail to play without error.
  return allMatching.find(v => v.localService === true) || allMatching[0] || null;
}

export function detectLanguageFromText(text) {
  const value = text || "";
  if (!value.trim()) return null;

  const counts = {
    te: (value.match(/[\u0C00-\u0C7F]/g) || []).length,
    ta: (value.match(/[\u0B80-\u0BFF]/g) || []).length,
    kn: (value.match(/[\u0C80-\u0CFF]/g) || []).length,
    ml: (value.match(/[\u0D00-\u0D7F]/g) || []).length,
    hi: (value.match(/[\u0900-\u097F]/g) || []).length,
    en: (value.match(/[A-Za-z]/g) || []).length,
  };

  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  if (sorted[0][1] > 0) return sorted[0][0];

  return null;
}

export function getVoiceAvailability() {
  const availability = {
    en: false,
    hi: false,
    te: false,
    ta: false,
    kn: false,
    ml: false,
  };

  if (!("speechSynthesis" in window)) {
    return availability;
  }

  const voices = window.speechSynthesis.getVoices() || [];
  for (const voice of voices) {
    const lang = (voice.lang || "").toLowerCase();
    if (!lang) continue;
    if (lang === "en" || lang.startsWith("en-")) availability.en = true;
    if (lang === "hi" || lang.startsWith("hi-")) availability.hi = true;
    if (lang === "te" || lang.startsWith("te-")) availability.te = true;
    if (lang === "ta" || lang.startsWith("ta-")) availability.ta = true;
    if (lang === "kn" || lang.startsWith("kn-")) availability.kn = true;
    if (lang === "ml" || lang.startsWith("ml-")) availability.ml = true;
  }

  return availability;
}

export function stopSpeech() {
  _isSpeaking = false;
  if ("speechSynthesis" in window) {
    try { window.speechSynthesis.cancel(); } catch (e) {}
  }
  if (backendAudioPlayer) {
    try { backendAudioPlayer.pause(); backendAudioPlayer.src = ""; } catch (e) {}
  }
}

export function speakTextInLanguage(text, languageCode, options = {}, onEnd) {
  // Clean text: replace underscores/hyphens with spaces so they aren't pronounced
  const cleanText = (text || "").replace(/[_-]/g, " ").replace(/\s+/g, " ").trim();
  if (!("speechSynthesis" in window) || !cleanText) return;

  const interrupt = !!options.interrupt;

  // If already speaking and not interrupting, do not start a new utterance
  if (_isSpeaking && !interrupt) return;

  const detectedCode = detectLanguageFromText(cleanText);
  
  // Improved resolution logic:
  // 1. If text script is specifically Indian (te, hi, etc.), use that language.
  // 2. If text is Latin script (en), use English even if the UI is in Telugu mode.
  // 3. Fallback to the UI language preference (languageCode).
  let resolvedCode = "en";
  if (["hi", "te", "ta", "kn", "ml"].includes(detectedCode)) {
    resolvedCode = detectedCode;
  } else if (detectedCode === "en") {
    resolvedCode = "en";
  } else {
    resolvedCode = languageCode || "en";
  }

  const utter = new SpeechSynthesisUtterance(cleanText);
  const locale = getSpeechLocale(resolvedCode);
  utter.lang = locale;
  utter.onstart = () => { _isSpeaking = true; };
  utter.onend = () => { _isSpeaking = false; if (onEnd) onEnd(); };
  utter.onerror = () => { _isSpeaking = false; if (onEnd) onEnd(); };

  const voice = pickVoiceForLocale(locale);
  if (voice) {
    if (interrupt) {
      try { backendAudioPlayer?.pause(); backendAudioPlayer.src = ""; } catch (e) {}
      try { window.speechSynthesis.cancel(); } catch (e) {}
    }
    utter.voice = voice;
    window.speechSynthesis.speak(utter);
    return;
  }

  utter.rate = 1;
  utter.pitch = 1;

  // Fallback: If no local voice is found, use the Website's Cloud Engine
  playBackendTts(cleanText, resolvedCode, onEnd, interrupt);
}

export function extractWordAt(text, index) {
  if (!text || index < 0 || index >= text.length) return "";
  const separators = /[\s.,!?;:()[\]{}"'`|/\\<>\-+=*~@#$%^&،।]/;
  let left = index;
  let right = index;

  while (left > 0 && !separators.test(text[left - 1])) left -= 1;
  while (right < text.length && !separators.test(text[right])) right += 1;

  return text.slice(left, right).trim();
}

export function getWordUnderCursor(event) {
  if (!event) return "";

  if (document.caretPositionFromPoint) {
    const pos = document.caretPositionFromPoint(event.clientX, event.clientY);
    if (pos && pos.offsetNode && pos.offsetNode.nodeType === Node.TEXT_NODE) {
      return extractWordAt(pos.offsetNode.textContent || "", pos.offset);
    }
  }

  if (document.caretRangeFromPoint) {
    const range = document.caretRangeFromPoint(event.clientX, event.clientY);
    if (range && range.startContainer && range.startContainer.nodeType === Node.TEXT_NODE) {
      return extractWordAt(range.startContainer.textContent || "", range.startOffset);
    }
  }

  return "";
}
