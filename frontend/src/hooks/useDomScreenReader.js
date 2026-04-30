import { useEffect, useRef } from "react";
import { speakTextInLanguage, detectLanguageFromText } from "../utils/speechUtils";

/**
 * Custom hook that provides "speak-on-hover" functionality.
 * 
 * @param {boolean} enabled - Whether the screen reader is active.
 * @param {string} langCode - The 2-letter language code (e.g., 'hi', 'te', 'ta').
 */
const useDomScreenReader = (enabled, langCode = "en") => {
  const lastElementRef = useRef(null);
  const lastTextRef = useRef("");
  const lastTimeRef = useRef(0);

  const sanitizeForSpeech = (text) => {
    const raw = (text || "").trim();
    if (!raw) return "";

    // Remove emoji and symbol glyphs so decorative icons are never spoken.
    const noEmoji = raw
      .replace(/[\p{Extended_Pictographic}]/gu, " ")
      .replace(/[\u2600-\u27BF]/g, " ");

    const collapsed = noEmoji.replace(/\s+/g, " ").trim();
    return collapsed;
  };

  const isDecorativeOrIgnored = (element) => {
    if (!element) return true;

    if (element.closest('[aria-hidden="true"]')) return true;
    if (element.closest('[data-no-sr="true"]')) return true;

    const tag = element.tagName;
    if (["IMG", "SVG", "PATH", "CIRCLE", "RECT", "LINE", "POLYGON", "POLYLINE"].includes(tag)) {
      return true;
    }

    return false;
  };

  const getReadableText = (element) => {
    if (!element) return "";

    const ariaLabel = element.getAttribute("aria-label");
    if (ariaLabel && ariaLabel.trim()) return ariaLabel.trim();

    const alt = element.getAttribute("alt");
    if (alt && alt.trim()) return alt.trim();

    if (element.placeholder && element.placeholder.trim()) return element.placeholder.trim();

    // Prefer explicit native language label for language option tiles to avoid double-reading
    try {
      const langOption = element.classList && element.classList.contains("lang-option") ? element : element.closest?.(".lang-option");
      if (langOption) {
        const native = langOption.querySelector?.(".lang-native");
        if (native && native.innerText && native.innerText.trim()) return native.innerText.trim();
      }
    } catch (e) {}

    // Remove explicit no-sr descendants so parent innerText won't include hidden names.
    const clone = element.cloneNode(true);
    if (clone.querySelectorAll) {
      clone.querySelectorAll('[data-no-sr="true"]').forEach((node) => node.remove());
    }
    return (clone.innerText || "").trim();
  };

  const isUnwantedSpeech = (text) => {
    const value = (text || "").trim();
    if (!value) return true;

    // Skip pure symbols/emoji/punctuation.
    if (!/[\p{L}\p{N}]/u.test(value)) return true;

    // Skip clock-like time strings (e.g. 09:30, 09:30 AM, 21:45:10).
    if (/^\d{1,2}:\d{2}(:\d{2})?\s?(AM|PM|am|pm)?$/.test(value)) return true;

    return false;
  };

  useEffect(() => {
    // Read hover preference from localStorage (toolbar persists this)
    const hoverPref = window.localStorage.getItem("disability_hover");
    const hoverEnabled = hoverPref === null ? true : hoverPref === "true";

    // Disable functionality if not supported or not toggled on
    if (!enabled || !hoverEnabled || !window.speechSynthesis) {
      window.speechSynthesis?.cancel();
      return;
    }

    const handleMouseOver = (event) => {
      let element = event.target;
      try {
        const langOption = element.classList && element.classList.contains("lang-option") ? element : element.closest?.(".lang-option");
        if (langOption) element = langOption;
      } catch (e) {}

      if (isDecorativeOrIgnored(element)) return;

      // Prevent re-reading the same element immediately (by node)
      if (element === lastElementRef.current) return;
      lastElementRef.current = element;

      // Extract text content prioritizing accessibility labels
      const rawText = getReadableText(element);

      // Clean text: replace underscores and hyphens/minus with spaces, then strip decorative symbols.
      const text = rawText ? sanitizeForSpeech(rawText.replace(/[_-]/g, " ")) : "";

      if (isUnwantedSpeech(text)) return;

      // Prevent immediate duplicate speech for identical text
      const now = Date.now();
      if (text === lastTextRef.current && now - lastTimeRef.current < 1200) return;
      lastTextRef.current = text;
      lastTimeRef.current = now;

      // Read common interactive or structural text elements
      const readableTags = ["H1", "H2", "H3", "P", "SPAN", "LABEL", "BUTTON", "A", "LI", "OPTION"];

      if (text && (readableTags.includes(element.tagName) || element.tagName === "INPUT")) {
        // Determine best language for this text using script detection; fall back to provided langCode
        const detected = detectLanguageFromText(text) || langCode || "en";
        // speakTextInLanguage handles both local speech and backend fallback
        speakTextInLanguage(text, detected);
      }
    };

    document.addEventListener("mouseover", handleMouseOver);

    return () => {
      document.removeEventListener("mouseover", handleMouseOver);
      window.speechSynthesis.cancel();
      lastElementRef.current = null;
    };
  }, [enabled, langCode]);
};

export default useDomScreenReader;