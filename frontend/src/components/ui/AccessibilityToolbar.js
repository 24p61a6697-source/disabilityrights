import React, { useState, useEffect, useRef, useCallback } from "react";
import { useAuth } from "../../contexts/AuthContext";
import { stopSpeech, detectLanguageFromText } from "../../utils/speechUtils";

const isBrowser = typeof window !== "undefined";

const styles = {
  toolbar: {
    position: "fixed",
    left: 20,
    bottom: 20,
    display: "flex",
    flexDirection: "column",
    gap: 10,
    zIndex: 1000,
    background: "rgba(255, 255, 255, 0.8)",
    backdropFilter: "blur(12px)",
    padding: "12px",
    borderRadius: "20px",
    border: "1px solid rgba(255, 255, 255, 0.3)",
    boxShadow: "0 10px 25px -5px rgba(0,0,0,0.1)",
  },
  btn: {
    background: "#1a3a6b",
    color: "#fff",
    border: "none",
    padding: "10px 16px",
    borderRadius: 12,
    cursor: "pointer",
    fontSize: "13px",
    fontWeight: "700",
  },
  stop: { background: "#dc2626" }
};

const LANGUAGE_TO_LOCALE = {
  en: "en-US",
  hi: "hi-IN",
  te: "te-IN",
  ta: "ta-IN",
  ml: "ml-IN",
  kn: "kn-IN",
};

function getSpeechLocale(lang) {
  const raw = (lang || "en").toLowerCase();
  const short = raw.split("-")[0];
  return LANGUAGE_TO_LOCALE[raw] || LANGUAGE_TO_LOCALE[short] || "en-US";
}

function extractWordAt(text, index) {
  const separators = /[\s.,!?;:()[\]{}"'`|/\\<>\-+=*~@#$%^&]/;
  let left = index, right = index;

  while (left > 0 && !separators.test(text[left - 1])) left--;
  while (right < text.length && !separators.test(text[right])) right++;

  return text.slice(left, right).trim();
}

function getWordUnderCursor(e) {
  if (!isBrowser) return "";

  if (document.caretPositionFromPoint) {
    const pos = document.caretPositionFromPoint(e.clientX, e.clientY);
    if (pos?.offsetNode?.nodeType === Node.TEXT_NODE) {
      return extractWordAt(pos.offsetNode.textContent, pos.offset);
    }
  }

  if (document.caretRangeFromPoint) {
    const range = document.caretRangeFromPoint(e.clientX, e.clientY);
    if (range?.startContainer?.nodeType === Node.TEXT_NODE) {
      return extractWordAt(range.startContainer.textContent, range.startOffset);
    }
  }

  return "";
}

export default function AccessibilityToolbar() {
  const [speaking, setSpeaking] = useState(false);
  const { language: appLanguage, screenReaderEnabled, toggleScreenReader } = useAuth();
  const [hoverEnabled, setHoverEnabled] = useState(() => {
    try {
      const v = localStorage.getItem("disability_hover");
      return v === null ? true : v === "true";
    } catch (e) {
      return true;
    }
  });
  const [fontSize, setFontSize] = useState(100);
  const [highContrast, setHighContrast] = useState(false);

  const lastSpokenRef = useRef("");
  const keyboardBufferRef = useRef("");
  const keyboardTimerRef = useRef(null);
  const cursorThrottleRef = useRef(0);

  const preferredLanguage = isBrowser ? (appLanguage || localStorage.getItem("disability_language") || navigator.language) : "en";

  const speakText = useCallback((text) => {
    if (!isBrowser || !window.speechSynthesis || !screenReaderEnabled) return;

    const synth = window.speechSynthesis;

    const clean = text.replace(/\s+/g, " ").trim();
    if (!clean || clean === lastSpokenRef.current) return;

    let toSpeak = clean.length > 500 ? clean.slice(0, 400) : clean;

    if (synth.speaking) synth.cancel();

    // Detect language from text content itself, not just app preference
    const detectedLang = detectLanguageFromText(toSpeak) || appLanguage || "en";

    const utter = new SpeechSynthesisUtterance(toSpeak);
    utter.lang = getSpeechLocale(detectedLang);
    utter.rate = 0.9;
    utter.pitch = 1;
    utter.volume = 1;

    utter.onend = () => setSpeaking(false);
    utter.onerror = () => setSpeaking(false);

    lastSpokenRef.current = toSpeak;
    setSpeaking(true);

    synth.speak(utter);
  }, [appLanguage, screenReaderEnabled]);

  const stopSpeaking = () => {
    if (!isBrowser) return;
    window.speechSynthesis.cancel();
    setSpeaking(false);
    lastSpokenRef.current = "";
  };

  // Keyboard reading
  useEffect(() => {
    if (!screenReaderEnabled) return;

    const handler = (e) => {
        if (e.key.length === 1) {
          keyboardBufferRef.current += e.key;

          clearTimeout(keyboardTimerRef.current);

          keyboardTimerRef.current = setTimeout(() => {
            const text = keyboardBufferRef.current.trim();
            if (text.length > 0) {
              try { stopSpeech(); } catch (e) {}
              speakText(text);
            }
            keyboardBufferRef.current = "";
          }, 200);
        }

        if (e.key === "Enter") {
          const txt = keyboardBufferRef.current.trim();
          if (txt) {
            try { stopSpeech(); } catch (e) {}
            speakText(txt);
          }
          keyboardBufferRef.current = "";
        }
    };

    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [screenReaderEnabled, speakText]);

  // Persist hover preference for the global hook to read and use
  useEffect(() => {
    try { localStorage.setItem("disability_hover", hoverEnabled ? "true" : "false"); } catch (e) {}
  }, [hoverEnabled]);

  // Font scaling
  useEffect(() => {
    document.documentElement.style.setProperty("--font-scale", fontSize / 100);
  }, [fontSize]);

  // Contrast
  useEffect(() => {
    document.documentElement.classList.toggle("high-contrast", highContrast);
  }, [highContrast]);

  const readPage = () => {
    const text = document.body.innerText || "";
    speakText(text.slice(0, 2000));
  };

  return (
    <div style={styles.toolbar}>
      <button onClick={toggleScreenReader} style={styles.btn}>
        {screenReaderEnabled ? "🔊 ON" : "🔇 OFF"}
      </button>

      {screenReaderEnabled && (
        <>
          <button onClick={readPage} style={styles.btn}>📖 Read</button>

          <button
            onClick={() => setHoverEnabled(v => !v)}
            style={styles.btn}
          >
            Hover: {hoverEnabled ? "ON" : "OFF"}
          </button>

          <button
            onClick={() => setHighContrast(v => !v)}
            style={styles.btn}
          >
            Contrast
          </button>

          <button onClick={() => setFontSize(f => f + 20)} style={styles.btn}>A+</button>
          <button onClick={() => setFontSize(f => f - 20)} style={styles.btn}>A-</button>

          {speaking && (
            <button onClick={stopSpeaking} style={{ ...styles.btn, ...styles.stop }}>
              ⏹ Stop
            </button>
          )}
        </>
      )}
    </div>
  );
}