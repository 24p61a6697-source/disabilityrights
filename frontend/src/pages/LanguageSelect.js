import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { INDIAN_LANGUAGES, t } from "../languages/languages";
import { motion, AnimatePresence } from "framer-motion";

export default function LanguageSelect() {
  const { selectLanguage, language, screenReaderEnabled, toggleScreenReader } = useAuth();
  const navigate = useNavigate();
  const [selected, setSelected] = useState(language || null);
  const currentLang = selected || language || "en";
  const tr = (k) => t(k, currentLang);

  const handleContinue = () => {
    if (!selected) return;
    selectLanguage(selected);
    navigate("/auth");
  };

  const selectedLang = INDIAN_LANGUAGES.find(l => l.code === selected);

  return (
    <div className="language-page">
      <header className="gov-header" role="banner">
        <div className="container gov-header-stack">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="brand"
          >
            <span className="emblem">♿</span>
            <div className="brand-text">
              <div className="govt-title">Disability Rights</div>
              <div className="govt-subtitle">{tr("appSubtitle")}</div>
            </div>
          </motion.div>

          <div className="header-tools">
            <button className="header-pill" onClick={() => navigate(-1)}>
              ← {tr("home")}
            </button>
            <button
              className="header-pill"
              onClick={toggleScreenReader}
              aria-pressed={screenReaderEnabled}
            >
              {screenReaderEnabled ? `🔊 ${tr("screenReaderOn")}` : `🔇 ${tr("screenReaderOff")}`}
            </button>
            <label className="header-lang-select" aria-label="Language selector">
              <span>{tr("changeLanguage")}:</span>
              <select
                value={currentLang}
                onChange={(e) => {
                  const lang = e.target.value;
                  setSelected(lang);
                  selectLanguage(lang);
                }}
              >
                {INDIAN_LANGUAGES.map((lang) => (
                  <option key={lang.code} value={lang.code}>
                    {lang.english} ({lang.name})
                  </option>
                ))}
              </select>
            </label>
          </div>
        </div>
        <div className="tricolor-bar">
          <div className="saffron" />
          <div className="white" />
          <div className="green" />
        </div>
      </header>

      <main id="main-content" className="main-content container">
        <div className="hero-section">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="hero-icon">
            ♿
          </motion.div>
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="hero-title"
          >
            Disability Rights
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="hero-subtitle"
          >
            {tr("appSubtitle")}
          </motion.p>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.35 }}
            className="hero-subtitle-native"
          >
            दिव्यांग अधिकार एवं सुगमता मार्गदर्शिका
          </motion.p>
        </div>

        <motion.section 
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
          className="selection-card premium-card"
        >
          <div className="card-header card-header-left">
            <h2 className="selection-title">🌐 {tr("selectLanguage")}</h2>
            <p className="selection-subtitle">{tr("selectLanguage")}</p>
          </div>

          <div className="language-grid" role="radiogroup">
            {INDIAN_LANGUAGES.map((lang, index) => (
              <motion.button
                key={lang.code}
                whileHover={{ scale: 1.03, y: -2 }}
                whileTap={{ scale: 0.98 }}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 + index * 0.05 }}
                className={`lang-option ${selected === lang.code ? 'selected' : ''}`}
                onClick={() => setSelected(lang.code)}
                aria-checked={selected === lang.code}
                role="radio"
              >
                <span className="lang-native">{lang.name}</span>
                <span className="lang-english">{lang.english}</span>
                {selected === lang.code && (
                  <motion.div layoutId="active-ring" className="active-glow" />
                )}
              </motion.button>
            ))}
          </div>

          <AnimatePresence>
            <motion.button
              initial={{ opacity: 0.5 }}
              animate={{ opacity: selected ? 1 : 0.6 }}
              whileHover={selected ? { scale: 1.01 } : {}}
              onClick={handleContinue}
              disabled={!selected}
              className="continue-btn"
            >
              {selected ? `${tr("continueBtn")} ${tr("responsesIn")} ${selectedLang?.english}` : tr("selectLanguage")}
            </motion.button>
          </AnimatePresence>
        </motion.section>

        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="a11y-footer-note"
        >
          <span className="a11y-icon">{tr("accessibility")}:</span>
          This portal supports NVDA (NonVisual Desktop Access), other screen readers, and full keyboard navigation.
        </motion.div>
      </main>

      <footer className="page-footer">
        <div className="container footer-content">
          <p>© 2024 Disability Rights Portal</p>
        </div>
      </footer>
    </div>
  );
}

