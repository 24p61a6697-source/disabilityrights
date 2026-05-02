import React, { useState, useEffect, useRef, useCallback } from "react";
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
    boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)",
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
    transition: "all 0.2s",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
  },
  stop: { background: "#dc2626" }
};

const LANGUAGE_TO_LOCALE = {
  en: "en-US",
  hi: "hi-IN",
  ta: "ta-IN",
  te: "te-IN",
  kn: "kn-IN",
  ml: "ml-IN",
};

function getSpeechLocale(languageCode) {
  const raw = (languageCode || "en").toLowerCase();
  const shortCode = raw.split("-")[0];
  return LANGUAGE_TO_LOCALE[raw] || LANGUAGE_TO_LOCALE[shortCode] || raw || "en-US";
}

function pickBestVoice(locale) {
  if (!("speechSynthesis" in window)) return null;
  const voices = window.speechSynthesis.getVoices() || [];
  if (!voices.length) return null;

  const localeLower = (locale || "").toLowerCase();
  const localePrefix = localeLower.split("-")[0];

  return (
    voices.find((v) => v.lang && v.lang.toLowerCase() === localeLower) ||
    voices.find((v) => v.lang && v.lang.toLowerCase().startsWith(`${localePrefix}-`)) ||
    voices.find((v) => v.default) ||
    voices[0]
  );
}

function extractWordAt(text, index) {
  if (!text || index < 0 || index >= text.length) return "";
  const separators = /[\s.,!?;:()[\]{}"'`|/\\<>\-+=*~@#$%^&،।]/;
  let left = index;
  let right = index;

  while (left > 0 && !separators.test(text[left - 1])) left -= 1;
  while (right < text.length && !separators.test(text[right])) right += 1;

  return text.slice(left, right).trim();
}

function getWordUnderCursor(event) {
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

export default function AccessibilityToolbar() {
  const [speaking, setSpeaking] = useState(false);
  const [hoverEnabled, setHoverEnabled] = useState(true);
  const [screenReaderEnabled, setScreenReaderEnabled] = useState(true);
  const [, setKeyboardBuffer] = useState("");
  const [fontSize, setFontSize] = useState(100);
  const [highContrast, setHighContrast] = useState(false);
  const [statusMessage, setStatusMessage] = useState("Screen reader and keyboard navigation enabled.");
  const [preferredLanguage, setPreferredLanguage] = useState(() => localStorage.getItem("disability_language") || navigator.language || "en");
  const lastSpokenRef = useRef("");
  const timerRef = useRef(null);
  const keyboardTimerRef = useRef(null);
  const cursorTimerRef = useRef(null);

  const speakText = useCallback((text) => {
    if (!screenReaderEnabled || !("speechSynthesis" in window)) return;
    const trimmed = (text || "").replace(/\s+/g, " ").trim();
    if (!trimmed || trimmed.length < 1) return;
    if (trimmed === lastSpokenRef.current) return;
    // Limit length for long blocks while preserving complete words.
    let toSpeak = trimmed;
    if (toSpeak.length > 500) {
      const firstSentence = toSpeak.split(/[.!?]\s/)[0];
      toSpeak = firstSentence.length > 30 ? firstSentence : toSpeak.slice(0, 400);
    }
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(toSpeak);
    const locale = getSpeechLocale(preferredLanguage);
    utter.lang = locale;
    const voice = pickBestVoice(locale);
    if (voice) utter.voice = voice;
    utter.rate = 1;
    utter.pitch = 1;
    utter.onend = () => setSpeaking(false);
    utter.onerror = () => setSpeaking(false);
    lastSpokenRef.current = toSpeak;
    setSpeaking(true);
    window.speechSynthesis.speak(utter);
  }, [preferredLanguage, screenReaderEnabled]);

  const stopSpeaking = () => {
    window.speechSynthesis.cancel();
    setSpeaking(false);
    lastSpokenRef.current = "";
    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
  };

  useEffect(() => {
    if (!("speechSynthesis" in window)) return;
    // Some browsers load voices asynchronously.
    const warmupVoices = () => window.speechSynthesis.getVoices();
    warmupVoices();
    window.speechSynthesis.addEventListener("voiceschanged", warmupVoices);
    return () => window.speechSynthesis.removeEventListener("voiceschanged", warmupVoices);
  }, []);

  useEffect(() => {
    const syncLanguage = () => {
      setPreferredLanguage(localStorage.getItem("disability_language") || navigator.language || "en");
    };
    window.addEventListener("storage", syncLanguage);
    return () => window.removeEventListener("storage", syncLanguage);
  }, []);

  useEffect(() => {
    if (!screenReaderEnabled) return;
    const keyboardHandler = (e) => {
      // Only process printable characters
      if (e.key && e.key.length === 1 && !e.ctrlKey && !e.altKey && !e.metaKey) {
        setKeyboardBuffer(prev => {
          const newBuffer = prev + e.key;
          // Clear previous timer
          if (keyboardTimerRef.current) clearTimeout(keyboardTimerRef.current);
          // Set new timer to speak after 1 second of no typing
          keyboardTimerRef.current = setTimeout(() => {
            if (newBuffer.trim()) {
              speakText(newBuffer.trim());
              setKeyboardBuffer("");
            }
          }, 1000);
          return newBuffer;
        });
      } else if (e.key === 'Backspace') {
        setKeyboardBuffer(prev => prev.slice(0, -1));
      } else if (e.key === 'Enter' || e.key === ' ') {
        // Speak immediately on Enter or Space
        setKeyboardBuffer(prev => {
          if (prev.trim()) {
            speakText(prev.trim());
            return "";
          }
          return prev;
        });
      }
    };

    document.addEventListener("keydown", keyboardHandler);
    return () => {
      document.removeEventListener("keydown", keyboardHandler);
      if (keyboardTimerRef.current) {
        clearTimeout(keyboardTimerRef.current);
        keyboardTimerRef.current = null;
      }
    };
  }, [screenReaderEnabled, speakText]);

  useEffect(() => {
    document.documentElement.style.setProperty('--font-scale', `${fontSize / 100}`);
    setStatusMessage(`Text size set to ${fontSize}%`);
  }, [fontSize]);

  // Initialize font scale on mount
  useEffect(() => {
    document.documentElement.style.setProperty('--font-scale', '1');
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("high-contrast", highContrast);
    setStatusMessage(highContrast ? "High contrast enabled." : "High contrast disabled.");
  }, [highContrast]);

  const adjustFontSize = (delta) => {
    setFontSize((prev) => Math.min(200, Math.max(50, prev + delta)));
  };

  const toggleHighContrast = () => {
    setHighContrast((prev) => !prev);
  };

  // Cursor and focus reading functionality
  useEffect(() => {
    if (!screenReaderEnabled || !hoverEnabled) return;
    const mouseMoveHandler = (e) => {
      if (cursorTimerRef.current) clearTimeout(cursorTimerRef.current);
      cursorTimerRef.current = setTimeout(() => {
        const wordUnderCursor = getWordUnderCursor(e);
        if (wordUnderCursor) {
          speakText(wordUnderCursor);
          return;
        }

        const el = e?.target;
        if (!el) return;

        const text = (
          el.getAttribute?.("aria-label") ||
          el.getAttribute?.("alt") ||
          el.innerText ||
          el.textContent ||
          ""
        ).replace(/\s+/g, " ").trim();

        if (!text) return;
        speakText(text.length > 140 ? text.slice(0, 140) : text);
      }, 100);
    };

    const focusHandler = (e) => {
      const el = e.target;
      if (!el) return;
      const labelText = el.getAttribute && (el.getAttribute("aria-label") || el.getAttribute("title") || el.getAttribute("alt") || "");
      const valueText = (typeof el.value === "string" ? el.value : "").trim();
      const nodeText = (el.innerText || el.textContent || "").replace(/\s+/g, " ").trim();
      const text = labelText || valueText || nodeText;
      if (text) speakText(text);
    };

    document.addEventListener("mousemove", mouseMoveHandler);
    document.addEventListener("focusin", focusHandler);
    return () => {
      document.removeEventListener("mousemove", mouseMoveHandler);
      document.removeEventListener("focusin", focusHandler);
      if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
      if (cursorTimerRef.current) { clearTimeout(cursorTimerRef.current); cursorTimerRef.current = null; }
    };
  }, [hoverEnabled, screenReaderEnabled, speakText]);

  // Button that reads the whole main region
  const readMain = () => {
    if (!screenReaderEnabled) return;
    const main = document.getElementById("main-content") || document.querySelector("main") || document.body;
    const text = (main && main.innerText) ? main.innerText.replace(/\s+/g, " ") : document.body.innerText;
    speakText(text.slice(0, 2000));
  };

  return (
    <div style={styles.toolbar} role="complementary" aria-label="Accessibility options">
      <div style={{display:"flex", flexDirection: "column", gap: 8}}>
        <button
          title={screenReaderEnabled ? "Disable screen reader" : "Enable screen reader"}
          onClick={() => {
            setScreenReaderEnabled((v) => {
              const next = !v;
              setStatusMessage(next ? "Screen reader enabled." : "Screen reader disabled.");
              return next;
            });
          }}
          aria-pressed={screenReaderEnabled}
          style={{...styles.btn, background: screenReaderEnabled ? "#1a3a6b" : "#666"}}
        >
          {screenReaderEnabled ? "🔊 Screen Reader: ON" : "🔇 Screen Reader: OFF"}
        </button>
        {screenReaderEnabled && (
          <div style={{display:"flex", flexWrap:"wrap", gap:8}}>
            <button
              title="Read aloud page"
              onClick={readMain}
              style={styles.btn}
            >📖 Read Page</button>
            <button
              title={hoverEnabled ? "Disable hover speak" : "Enable hover speak"}
              onClick={() => {
                setHoverEnabled((v) => {
                  const next = !v;
                  setStatusMessage(next ? "Hover narration enabled." : "Hover narration disabled.");
                  return next;
                });
              }}
              aria-pressed={hoverEnabled}
              style={{...styles.btn, fontSize:"12px", padding:"6px 8px"}}
            >{hoverEnabled ? "Hover: On" : "Hover: Off"}</button>
            <button
              title="Toggle high contrast mode"
              onClick={toggleHighContrast}
              aria-pressed={highContrast}
              style={{...styles.btn, fontSize:"12px", padding:"6px 8px", background: highContrast ? "#333" : "#1a3a6b"}}
            >{highContrast ? "High Contrast: ON" : "High Contrast: OFF"}</button>
            <button
              title="Increase text size"
              onClick={() => adjustFontSize(20)}
              style={{...styles.btn, fontSize:"12px", padding:"6px 8px"}}
            >A+</button>
            <button
              title="Decrease text size"
              onClick={() => adjustFontSize(-20)}
              style={{...styles.btn, fontSize:"12px", padding:"6px 8px"}}
            >A-</button>
            {speaking && (
              <button title="Stop speaking" onClick={stopSpeaking} style={{...styles.btn, ...styles.stop}}>⏹</button>
            )}
          </div>
        )}
        <div role="status" aria-live="polite" style={{position:"absolute", left:-9999, top:-9999}}>
          {statusMessage}
        </div>
      </div>
    </div>
  );
}
