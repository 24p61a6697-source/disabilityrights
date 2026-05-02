import { useEffect, useRef } from "react";
import { speakTextInLanguage, detectLanguageFromText } from "../utils/speechUtils";

const isBrowser = typeof window !== "undefined";

const useDomScreenReader = (enabled, langCode = "en") => {
  const lastElementRef = useRef(null);
  const lastTextRef = useRef("");
  const lastTimeRef = useRef(0);
  const throttleRef = useRef(0);

  const sanitizeText = (text) => {
    if (!text) return "";

    return text
      .replace(/[\p{Extended_Pictographic}]/gu, " ")
      .replace(/[\u2600-\u27BF]/g, " ")
      .replace(/[_-]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  };

  const isIgnoredElement = (el) => {
    if (!el) return true;

    if (el.closest?.('[aria-hidden="true"]')) return true;
    if (el.closest?.('[data-no-sr="true"]')) return true;

    const tag = el.tagName;
    if (["IMG", "SVG", "PATH", "CIRCLE"].includes(tag)) return true;

    return false;
  };

  const getReadableText = (el) => {
    if (!el) return "";

    return (
      el.getAttribute?.("aria-label") ||
      el.getAttribute?.("alt") ||
      el.placeholder ||
      el.innerText ||
      el.textContent ||
      ""
    ).trim();
  };

  const isValidText = (text) => {
    if (!text) return false;

    // must contain letters or numbers
    if (!/[\p{L}\p{N}]{2,}/u.test(text)) return false;

    // avoid time strings
    if (/^\d{1,2}:\d{2}/.test(text)) return false;

    return true;
  };

  useEffect(() => {
    if (!isBrowser) return;

    const hoverPref = localStorage.getItem("disability_hover");
    const hoverEnabled = hoverPref === null ? true : hoverPref === "true";

    if (!enabled || !hoverEnabled || !window.speechSynthesis) {
      window.speechSynthesis?.cancel();
      return;
    }

    const handler = (e) => {
      const now = Date.now();

      // 🔥 HARD THROTTLE
      if (now - throttleRef.current < 400) return;
      throttleRef.current = now;

      let el = e.target;

      if (isIgnoredElement(el)) return;

      if (el === lastElementRef.current) return;
      lastElementRef.current = el;

      const raw = getReadableText(el);
      const text = sanitizeText(raw);

      if (!isValidText(text)) return;

      // prevent duplicate rapid speech
      if (text === lastTextRef.current && now - lastTimeRef.current < 1200) return;

      lastTextRef.current = text;
      lastTimeRef.current = now;

      const detectedLang = detectLanguageFromText(text) || langCode || "en";

      // 🔥 Prevent overlap
      if (window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
      }

      speakTextInLanguage(text.slice(0, 120), detectedLang);
    };

    document.addEventListener("mouseover", handler);

    return () => {
      document.removeEventListener("mouseover", handler);
      window.speechSynthesis.cancel();
    };
  }, [enabled, langCode]);
};

export default useDomScreenReader;